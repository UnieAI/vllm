# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the QAIC decode-priority scheduler's admission policy.

These exercise `_qaic_should_defer_prefill` (the decision that keeps decode steps
pure) without constructing a full Scheduler: we bypass __init__ and set just the
attributes the method reads. The budget-cap path in schedule() is validated on the
box (it calls the real Scheduler.schedule()).

Requires vLLM importable (the QAIC host). Skipped otherwise.
"""

import pytest

pytest.importorskip("vllm")

from vllm_qaic.scheduler import QaicDecodePriorityScheduler  # noqa: E402


class _FakeReq:
    """Minimal stand-in: decode phase iff num_computed_tokens >= num_prompt_tokens."""

    def __init__(self, decode: bool):
        self.num_prompt_tokens = 10
        self.num_computed_tokens = 12 if decode else 4  # >= prompt => decode phase


def _mk(decodes=0, prefills=0, steps_since=0, every_n=8, frac=0.5, max_seqs=96):
    obj = QaicDecodePriorityScheduler.__new__(QaicDecodePriorityScheduler)
    obj.running = [_FakeReq(True) for _ in range(decodes)] + \
                  [_FakeReq(False) for _ in range(prefills)]
    obj._qaic_steps_since_prefill = steps_since
    obj._qaic_prefill_every_n = every_n
    obj._qaic_resume_frac = frac
    obj.max_num_running_reqs = max_seqs
    return obj


def test_no_running_never_defers():
    # Nothing to protect -> prefills must flow.
    assert _mk(decodes=0, prefills=0)._qaic_should_defer_prefill() is False


def test_running_prefill_only_does_not_defer():
    # Fix #1: running holds only in-flight prefills (no decode) -> must NOT defer,
    # otherwise waiting prefills get needlessly delayed.
    assert _mk(decodes=0, prefills=90)._qaic_should_defer_prefill() is False


def test_decode_backlog_defers():
    # Decodes running, recently prefilled -> defer new prefill.
    assert _mk(decodes=90, steps_since=0)._qaic_should_defer_prefill() is True
    # In-flight prefills alongside decodes don't change the decision (decode count rules).
    assert _mk(decodes=90, prefills=4, steps_since=0)._qaic_should_defer_prefill() is True


def test_cadence_backstop_allows_prefill():
    # Deferred long enough -> force a prefill step (TTFT backstop).
    assert _mk(decodes=90, steps_since=8)._qaic_should_defer_prefill() is False
    assert _mk(decodes=90, steps_since=9)._qaic_should_defer_prefill() is False


def test_low_decode_load_allows_prefill():
    # decodes < frac * max_num_seqs -> headroom to prefill cheaply.
    assert _mk(decodes=10, steps_since=0, frac=0.5, max_seqs=96)._qaic_should_defer_prefill() is False
    # at/above the fraction -> still defer.
    assert _mk(decodes=48, steps_since=0, frac=0.5, max_seqs=96)._qaic_should_defer_prefill() is True


if __name__ == "__main__":
    test_no_running_never_defers()
    test_running_prefill_only_does_not_defer()
    test_decode_backlog_defers()
    test_cadence_backstop_allows_prefill()
    test_low_decode_load_allows_prefill()
    print("QAIC SCHEDULER POLICY: ALL PASS")
