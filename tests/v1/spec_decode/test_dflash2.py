# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.speculative_config = SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: False
        )
        self.max_num_reqs = 2
        # The fork derives the query length from the step count; anchor
        # sampling is rejected at construction, so it is always False here.
        self.sample_from_anchor = False
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_refuses_dynamic_speculative_depths(monkeypatch):
    """Per-depth CUDA-graph buckets would trip the fixed-depth walk mid-boot."""

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.speculative_config = SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: True
        )

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)
    with pytest.raises(ValueError, match="fixed physical depth"):
        DFlash2Speculator(None, torch.device("cpu"))


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_top_k_shard_mapping_honors_added_vocab_regions():
    """Shard layout is [org | org_pad | added | added_pad]; the k-wide top-k
    must mask both pad slabs and map added-vocab winners to their global ids,
    exactly as get_top_tokens does."""
    from vllm.model_executor.layers.logits_processor import (
        _globalize_token_ids,
        _mask_vocab_padding,
    )

    shard = SimpleNamespace(
        num_org_elements=6,
        num_org_elements_padded=8,
        num_org_vocab_padding=2,
        num_added_elements=3,
        num_added_elements_padded=4,
        num_added_vocab_padding=1,
        org_vocab_start_index=100,
        added_vocab_start_index=1000,
    )
    logits = torch.zeros(2, 12)
    _mask_vocab_padding(logits, shard)
    assert torch.isinf(logits[:, 6:8]).all() and (logits[:, 6:8] < 0).all()
    assert torch.isinf(logits[:, 11:12]).all()
    assert torch.isfinite(logits[:, :6]).all()
    assert torch.isfinite(logits[:, 8:11]).all()

    ids = torch.tensor([0, 5, 8, 10], dtype=torch.int64)
    torch.testing.assert_close(
        _globalize_token_ids(ids, shard),
        torch.tensor([100, 105, 1000, 1002], dtype=torch.int64),
    )


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ring_remap_preserves_nulls_and_encodes_positions():
    """Ring remap: non-null position p becomes 1 + req_state*R + p%R, nulls
    stay null, and rows address distinct per-request ring ranges."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        ring_remap_draft_block_tables,
    )

    ring = 4
    block_table = torch.tensor(
        [
            [0, 0, 17, 23, 9, 41, 0],
            [5, 6, 7, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    ring_remap_draft_block_tables(block_table, idx_mapping, ring)
    base0, base1 = 1 + 3 * ring, 1 + 1 * ring
    expected = torch.tensor(
        [
            [0, 0, base0 + 2, base0 + 3, base0 + 0, base0 + 1, 0],
            [base1 + 0, base1 + 1, base1 + 2, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(block_table, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ring_remap_then_shift_composes():
    """After a restored-prefix left shift, table position p must hold the
    ring id of original position p+shift -- the page that logical block
    p+shift's data was written to via the (remapped) slot mappings."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        ring_remap_draft_block_tables,
        shift_draft_block_tables,
    )

    ring = 5
    block_size = 4
    block_table = torch.tensor(
        [[11, 12, 13, 14, 15, 16, 0, 0]], dtype=torch.int32, device="cuda"
    )
    idx_mapping = torch.tensor([2], dtype=torch.int32, device="cuda")
    num_cached = torch.zeros(8, dtype=torch.int32, device="cuda")
    num_cached[2] = 2 * block_size
    seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")

    ring_remap_draft_block_tables(block_table, idx_mapping, ring)
    shift_draft_block_tables(
        block_table,
        idx_mapping,
        num_cached,
        seq_lens,
        block_size,
        0,
        update_seq_lens=False,
    )
    base = 1 + 2 * ring
    assert block_table[0, :4].tolist() == [
        base + (p + 2) % ring for p in range(4)
    ]


def _fake_dflash_model_and_attn(weight: torch.Tensor, q_size: int, scales=None):
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    qkv_proj = SimpleNamespace(weight=weight)
    if scales is not None:
        qkv_proj.weight_scale = scales
    fake_model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    )
    attn = SimpleNamespace(qkv_proj=qkv_proj, q_size=q_size)
    return DFlashQwen3Model._kv_projection_rows, fake_model, attn


def test_kv_projection_rows_passthrough_for_plain_weights():
    weight = torch.randn(12, 64, dtype=torch.bfloat16)
    rows_fn, fake_model, attn = _fake_dflash_model_and_attn(weight, q_size=8)
    torch.testing.assert_close(rows_fn(fake_model, attn), weight[8:])


def test_kv_projection_rows_dequantizes_mxfp8():
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    torch.manual_seed(2)
    weight = torch.randn(12, 64).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (12, 2), dtype=torch.uint8)
    rows_fn, fake_model, attn = _fake_dflash_model_and_attn(
        weight, q_size=8, scales=scales
    )
    expected = dequant_mxfp8_to_bf16(weight[8:].contiguous(), scales[8:].contiguous())
    torch.testing.assert_close(rows_fn(fake_model, attn), expected)


def test_kv_projection_rows_rejects_fp8_without_mxfp8_scales():
    weight = torch.randn(12, 64).to(torch.float8_e4m3fn)
    rows_fn, fake_model, attn = _fake_dflash_model_and_attn(weight, q_size=8)
    with pytest.raises(ValueError, match="weight_scale"):
        rows_fn(fake_model, attn)
