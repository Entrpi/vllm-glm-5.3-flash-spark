# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-M dispatch of the b12x MXFP8 linear: rows above VLLM_B12X_MXFP8_MAX_M
take the FlashInfer fallback when the layer carries a fallback scale."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import b12x as b12x_mod


@pytest.fixture
def fake_layer():
    return SimpleNamespace(
        b12x_mxfp8_packed_weight=SimpleNamespace(out_features=16),
        b12x_mxfp8_fallback_scale=torch.zeros(1),
        weight=torch.zeros(16, 32),
    )


def _install_fakes(monkeypatch, calls):
    fake_mm = lambda a, w, bias=None, expected_m=None, stream=None: (  # noqa: E731
        calls.append(("b12x", int(a.shape[0]))) or torch.zeros(a.shape[0], 16)
    )
    monkeypatch.setattr(
        b12x_mod, "_import_b12x_mxfp8", lambda: SimpleNamespace(mm=fake_mm)
    )
    monkeypatch.setattr(
        b12x_mod,
        "_apply_flashinfer_mxfp8_fallback",
        lambda layer, x, bias, scale: (
            calls.append(("flashinfer", int(x.shape[0]))) or torch.zeros(x.shape[0], 16)
        ),
    )
    monkeypatch.setattr(b12x_mod, "current_stream", lambda: SimpleNamespace(cuda_stream=0))


@pytest.mark.parametrize("m,expected", [(8, "b12x"), (16, "b12x"), (17, "flashinfer"), (32, "flashinfer")])
def test_dispatch_by_rows(monkeypatch, fake_layer, m, expected):
    monkeypatch.setenv("VLLM_B12X_MXFP8_MAX_M", "16")
    calls: list[tuple[str, int]] = []
    _install_fakes(monkeypatch, calls)
    out = b12x_mod._apply_b12x_mxfp8_packed_linear(
        fake_layer, torch.zeros(m, 32), None
    )
    assert out.shape == (m, 16)
    assert calls == [(expected, m)]


def test_zero_threshold_disables_fallback(monkeypatch, fake_layer):
    monkeypatch.setenv("VLLM_B12X_MXFP8_MAX_M", "0")
    calls: list[tuple[str, int]] = []
    _install_fakes(monkeypatch, calls)
    b12x_mod._apply_b12x_mxfp8_packed_linear(fake_layer, torch.zeros(64, 32), None)
    assert calls == [("b12x", 64)]


def test_no_fallback_scale_stays_on_b12x(monkeypatch, fake_layer):
    monkeypatch.setenv("VLLM_B12X_MXFP8_MAX_M", "16")
    fake_layer.b12x_mxfp8_fallback_scale = None
    calls: list[tuple[str, int]] = []
    _install_fakes(monkeypatch, calls)
    b12x_mod._apply_b12x_mxfp8_packed_linear(fake_layer, torch.zeros(64, 32), None)
    assert calls == [("b12x", 64)]
