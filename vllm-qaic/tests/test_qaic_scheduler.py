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
    """Minimal stand-in: decode phase iff num_computed_tokens >= num_prompt_tokens.

    Token fields are set so the stock running-token formula
    (num_tokens_with_spec + num_output_placeholders - num_computed_tokens) yields 1
    per decode req (a single generated token) — keeps budget math easy to assert.
    """

    def __init__(self, decode: bool):
        self.num_prompt_tokens = 10
        if decode:
            self.num_computed_tokens = 12       # >= prompt => decode phase
            self.num_tokens_with_spec = 13      # one new token
            self.num_output_placeholders = 0    # contribution = 13 + 0 - 12 = 1
        else:
            self.num_computed_tokens = 4        # still prefilling
            self.num_tokens_with_spec = 10
            self.num_output_placeholders = 0


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


def _patched_super_schedule(record):
    """Temporarily replace Scheduler.schedule to capture the budget it sees.

    Returns a (install, restore) pair. `record` dict gets 'budget' set to
    self.max_num_scheduled_tokens at the moment super().schedule() runs.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    orig = Scheduler.schedule

    def fake(self):
        record["budget"] = self.max_num_scheduled_tokens
        return "SENTINEL"

    return (lambda: setattr(Scheduler, "schedule", fake),
            lambda: setattr(Scheduler, "schedule", orig))


def test_budget_capped_to_running_needs_then_restored():
    # Defer step: budget seen by super().schedule() must be capped to the running
    # (decode) needs (90 reqs * 1 token = 90), then restored to the saved value.
    obj = _mk(decodes=90, steps_since=0)
    obj.max_num_scheduled_tokens = 2048
    rec = {}
    install, restore = _patched_super_schedule(rec)
    install()
    try:
        out = obj.schedule()
    finally:
        restore()
    assert out == "SENTINEL"
    assert rec["budget"] == 90, f"budget not capped to running needs: {rec}"
    assert obj.max_num_scheduled_tokens == 2048, "budget not restored after schedule()"


def test_non_defer_step_does_not_touch_budget():
    # No decode -> not a defer step -> super().schedule() sees the full budget.
    obj = _mk(decodes=0, prefills=50, steps_since=0)
    obj.max_num_scheduled_tokens = 2048
    rec = {}
    install, restore = _patched_super_schedule(rec)
    install()
    try:
        obj.schedule()
    finally:
        restore()
    assert rec["budget"] == 2048, f"non-defer step must not cap budget: {rec}"
    assert obj.max_num_scheduled_tokens == 2048


def test_budget_restored_on_exception():
    # If super().schedule() raises, the finally must still restore the budget.
    from vllm.v1.core.sched.scheduler import Scheduler

    obj = _mk(decodes=90, steps_since=0)
    obj.max_num_scheduled_tokens = 2048
    orig = Scheduler.schedule

    def boom(self):
        raise RuntimeError("boom")

    Scheduler.schedule = boom
    try:
        try:
            obj.schedule()
        except RuntimeError:
            pass
    finally:
        Scheduler.schedule = orig
    assert obj.max_num_scheduled_tokens == 2048, "budget not restored after exception"


if __name__ == "__main__":
    test_no_running_never_defers()
    test_running_prefill_only_does_not_defer()
    test_decode_backlog_defers()
    test_cadence_backstop_allows_prefill()
    test_low_decode_load_allows_prefill()
    test_budget_capped_to_running_needs_then_restored()
    test_non_defer_step_does_not_touch_budget()
    test_budget_restored_on_exception()
    print("QAIC SCHEDULER POLICY: ALL PASS")
