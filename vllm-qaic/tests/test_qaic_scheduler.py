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


def _mk(running, steps_since, every_n=8, frac=0.5, max_seqs=96):
    obj = QaicDecodePriorityScheduler.__new__(QaicDecodePriorityScheduler)
    obj.running = [object()] * running
    obj._qaic_steps_since_prefill = steps_since
    obj._qaic_prefill_every_n = every_n
    obj._qaic_resume_frac = frac
    obj.max_num_running_reqs = max_seqs
    return obj


def test_no_running_never_defers():
    # Nothing to protect -> prefills must flow.
    assert _mk(running=0, steps_since=0)._qaic_should_defer_prefill() is False


def test_decode_backlog_defers():
    # Many decodes running, recently prefilled -> defer new prefill.
    assert _mk(running=90, steps_since=0)._qaic_should_defer_prefill() is True


def test_cadence_backstop_allows_prefill():
    # Deferred long enough -> force a prefill step (TTFT backstop).
    assert _mk(running=90, steps_since=8)._qaic_should_defer_prefill() is False
    assert _mk(running=90, steps_since=9)._qaic_should_defer_prefill() is False


def test_low_running_load_allows_prefill():
    # running < frac * max_num_seqs -> headroom to prefill cheaply.
    assert _mk(running=10, steps_since=0, frac=0.5, max_seqs=96)._qaic_should_defer_prefill() is False
    # at/above the fraction -> still defer.
    assert _mk(running=48, steps_since=0, frac=0.5, max_seqs=96)._qaic_should_defer_prefill() is True


if __name__ == "__main__":
    test_no_running_never_defers()
    test_decode_backlog_defers()
    test_cadence_backstop_allows_prefill()
    test_low_running_load_allows_prefill()
    print("QAIC SCHEDULER POLICY: ALL PASS")
