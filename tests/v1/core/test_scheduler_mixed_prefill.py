# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-prefill decode-floor policy (mixed_prefill_token_cap).

While any running request is decoding, prefill chunks are skipped
(cap == -1) or capped (cap > 0); solo prefills are untouched, unlike
long_prefill_token_threshold. An anti-starvation guard lifts the policy
for one step after mixed_prefill_max_defer_steps consecutive deferrals.
"""

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

BUDGET = 1024


def _output_for(requests, sampled):
    return ModelRunnerOutput(
        req_ids=[r.request_id for r in requests],
        req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _start_decode(scheduler, request):
    """Prefill the request fully and feed one sampled token."""
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == request.num_tokens
    scheduler.update_from_output(output, _output_for([request], [[0]]))


def _decode_step(scheduler, decode_req, scheduled_ids=None):
    """Run one schedule step; feed a sampled token to the decode lane."""
    output = scheduler.schedule()
    if scheduled_ids is not None:
        assert set(output.num_scheduled_tokens) == scheduled_ids
    sampled = [
        [0] if r == decode_req.request_id else []
        for r in output.num_scheduled_tokens
    ]
    runner_output = ModelRunnerOutput(
        req_ids=list(output.num_scheduled_tokens),
        req_id_to_index={
            r: i for i, r in enumerate(output.num_scheduled_tokens)
        },
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, runner_output)
    return output


def test_skip_mode_defers_prefill_until_guard_fires():
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        mixed_prefill_token_cap=-1,
        mixed_prefill_max_defer_steps=3,
    )
    decode_req = create_requests(
        num_requests=1, num_tokens=16, max_tokens=64, req_ids=["decode"]
    )[0]
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    _start_decode(scheduler, decode_req)

    scheduler.add_request(prefill_req)
    # Steps 1-3: the waiting prefill is skipped while the decode runs.
    for _ in range(3):
        _decode_step(scheduler, decode_req, {decode_req.request_id})
    # Step 4: defer streak reached 3 -> guard suspends the policy and the
    # prefill gets a full chunk next to the decode token.
    output = _decode_step(scheduler, decode_req)
    assert output.num_scheduled_tokens[prefill_req.request_id] == BUDGET - 1
    # Step 5: the (now running) prefill chunk is skipped again.
    output = _decode_step(scheduler, decode_req, {decode_req.request_id})
    assert prefill_req.request_id not in output.num_scheduled_tokens


def test_cap_mode_caps_waiting_and_running_chunks():
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        mixed_prefill_token_cap=64,
        mixed_prefill_max_defer_steps=0,
    )
    decode_req = create_requests(
        num_requests=1, num_tokens=16, max_tokens=64, req_ids=["decode"]
    )[0]
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    _start_decode(scheduler, decode_req)

    scheduler.add_request(prefill_req)
    # Admission chunk is capped to 64 (waiting-loop site).
    output = _decode_step(scheduler, decode_req)
    assert output.num_scheduled_tokens[prefill_req.request_id] == 64
    # Subsequent chunks of the running prefill are capped too
    # (running-loop site), and the decode lane keeps its token.
    for _ in range(3):
        output = _decode_step(scheduler, decode_req)
        assert output.num_scheduled_tokens[prefill_req.request_id] == 64
        assert output.num_scheduled_tokens[decode_req.request_id] == 1


def test_solo_prefill_untouched_by_skip_mode():
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        mixed_prefill_token_cap=-1,
        mixed_prefill_max_defer_steps=0,
    )
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    scheduler.add_request(prefill_req)
    # No decode in flight: full-size chunks, both at admission and while
    # the prefill continues as a running request.
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[prefill_req.request_id] == BUDGET
    scheduler.update_from_output(output, _output_for([prefill_req], [[]]))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[prefill_req.request_id] == 2000 - BUDGET


def test_policy_off_by_default():
    scheduler = create_scheduler(max_num_batched_tokens=BUDGET)
    decode_req = create_requests(
        num_requests=1, num_tokens=16, max_tokens=64, req_ids=["decode"]
    )[0]
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    _start_decode(scheduler, decode_req)
    scheduler.add_request(prefill_req)
    # Stock behavior: the prefill takes the rest of the budget.
    output = _decode_step(scheduler, decode_req)
    assert output.num_scheduled_tokens[prefill_req.request_id] == BUDGET - 1


def test_skip_mode_releases_after_decode_finishes():
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        mixed_prefill_token_cap=-1,
        mixed_prefill_max_defer_steps=0,
    )
    decode_req = create_requests(
        num_requests=1, num_tokens=16, max_tokens=64, req_ids=["decode"]
    )[0]
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    _start_decode(scheduler, decode_req)
    scheduler.add_request(prefill_req)
    # Guard disabled: the prefill is deferred as long as the decode runs.
    for _ in range(4):
        _decode_step(scheduler, decode_req, {decode_req.request_id})
    scheduler.finish_requests(
        decode_req.request_id, RequestStatus.FINISHED_ABORTED
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[prefill_req.request_id] == BUDGET


def _multi_decode_step(scheduler, decode_reqs, scheduled_ids=None):
    """One schedule step feeding a sampled token to every decode lane."""
    output = scheduler.schedule()
    if scheduled_ids is not None:
        assert set(output.num_scheduled_tokens) == scheduled_ids
    decode_ids = {r.request_id for r in decode_reqs}
    sampled = [
        [0] if r in decode_ids else []
        for r in output.num_scheduled_tokens
    ]
    runner_output = ModelRunnerOutput(
        req_ids=list(output.num_scheduled_tokens),
        req_id_to_index={
            r: i for i, r in enumerate(output.num_scheduled_tokens)
        },
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, runner_output)
    return output


def test_share_gate_paces_prefill_and_bypasses_streak_guard():
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        mixed_prefill_token_cap=-1,
        mixed_prefill_decode_weight=1.0,
        # Would force a chunk every 2 deferrals in legacy mode; the share
        # gate must bypass it (it bounds waiting by construction).
        mixed_prefill_max_defer_steps=2,
    )
    clock = [0.0]
    scheduler._mixed_prefill_clock = lambda: clock[0]
    decode_req = create_requests(
        num_requests=1, num_tokens=16, max_tokens=64, req_ids=["decode"]
    )[0]
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    _start_decode(scheduler, decode_req)
    scheduler.add_request(prefill_req)

    # First contended chunk is admitted ungated to seed the estimate.
    _decode_step(
        scheduler,
        decode_req,
        {decode_req.request_id, prefill_req.request_id},
    )
    # The chunk step took 3.0s -> chunk EMA 3.0; with one decoder and one
    # prefill, target credit = 3.0s of decode-only time.
    clock[0] = 3.0
    for expected_credit in (0.0, 1.0, 2.0):
        _decode_step(scheduler, decode_req, {decode_req.request_id})
        assert scheduler._mixed_prefill_decode_credit == expected_credit
        clock[0] += 1.0
    # credit reaches 3.0 -> the next chunk is admitted (streak guard at 2
    # never fired: three consecutive deferrals happened above).
    _decode_step(
        scheduler,
        decode_req,
        {decode_req.request_id, prefill_req.request_id},
    )


def test_share_gate_target_scales_with_decoder_count():
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        mixed_prefill_token_cap=-1,
        mixed_prefill_decode_weight=1.0,
    )
    clock = [0.0]
    scheduler._mixed_prefill_clock = lambda: clock[0]
    decode_reqs = create_requests(
        num_requests=2, num_tokens=16, max_tokens=64, req_ids=["d0", "d1"]
    )
    for r in decode_reqs:
        scheduler.add_request(r)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output, _output_for(decode_reqs, [[0], [0]])
    )
    prefill_req = create_requests(
        num_requests=1, num_tokens=2000, max_tokens=64, req_ids=["prefill"]
    )[0]
    scheduler.add_request(prefill_req)

    decode_ids = {r.request_id for r in decode_reqs}
    _multi_decode_step(
        scheduler, decode_reqs, decode_ids | {prefill_req.request_id}
    )
    clock[0] = 3.0
    # Two decoders double the target to 6.0s of decode-only time: with
    # 2.0s steps the credit runs 0, 2, 4 across three gated steps and the
    # fourth admits the chunk (one decoder would admit at credit 3.0 by
    # the same pacing as the single-decoder test).
    for expected_credit in (0.0, 2.0, 4.0):
        _multi_decode_step(scheduler, decode_reqs, decode_ids)
        assert scheduler._mixed_prefill_decode_credit == expected_credit
        clock[0] += 2.0
    _multi_decode_step(
        scheduler, decode_reqs, decode_ids | {prefill_req.request_id}
    )


def test_share_gate_requires_skip_mode():
    import pytest

    with pytest.raises(ValueError, match="mixed_prefill_decode_weight"):
        create_scheduler(
            max_num_batched_tokens=BUDGET,
            mixed_prefill_token_cap=0,
            mixed_prefill_decode_weight=1.0,
        )
