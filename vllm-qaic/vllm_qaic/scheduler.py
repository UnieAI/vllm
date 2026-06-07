# SPDX-License-Identifier: Apache-2.0
"""QAIC decode-priority scheduler.

Why this exists
---------------
vLLM's v1 scheduler mixes prefill chunks and decode tokens into the SAME engine
step (chunked prefill, on by default). On a GPU that is good: one fused forward
processes the mixed batch. On QAIC it is pathological: the compiled QPC has
SEPARATE fixed-shape prefill and decode graphs, so a mixed step runs the decode
QPC AND the prefill QPC *sequentially*. TPOT (the inter-token interval) then equals
``decode_time + prefill_time`` on every step that carries a prefill — at high
concurrency that is almost every step, so TPOT explodes (observed ~5x).

What this does
--------------
Decode-priority admission: when there is a decode backlog, do NOT admit new prefills
(keep decode steps pure, TPOT low). Periodically (cadence) or when the running load
is low, allow a prefill "burst" with the full token budget (so a prompt prefills in
as few full-cost QPC forwards as possible, instead of many tiny chunks). This bounds
the TTFT regression.

How it stays safe
-----------------
It does NOT empty ``self.waiting`` and does NOT touch preemption. On a defer step it
simply caps the scheduler's per-step token budget (``max_num_scheduled_tokens``) to
exactly the running (decode) requests' needs, so the stock waiting-admission loop —
``while (self.waiting or self.skipped_waiting) and token_budget > 0`` — sees a zero
budget after decodes and admits no new prefill. Preempted requests still route to the
real ``self.waiting`` queue (preemption is KV-driven, independent of this budget).

Tunables (env)
--------------
- ``QAIC_PREFILL_EVERY_N_STEPS`` (default 8): force a prefill-allowed step at least
  this often while deferring (TTFT backstop / starvation guard).
- ``QAIC_PREFILL_RESUME_FRAC`` (default 0.5): if running reqs < frac * max_num_seqs,
  stop deferring (there is decode headroom to prefill cheaply).

Installed by ``platform.py`` via ``scheduler_config.scheduler_cls`` unless
``QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER=1``.

NOTE: residual mixing — a long prompt admitted on a burst step that does not finish
prefilling that step stays in ``running`` and its remaining chunk can co-run with
decodes on later steps. Mitigate by sizing ``max_num_batched_tokens`` so prompts
finish within a burst step. Validate/tune on the QAIC host.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)


class QaicDecodePriorityScheduler(Scheduler):
    """Scheduler that protects decode steps from prefill co-scheduling on QAIC."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._qaic_steps_since_prefill = 0
        self._qaic_prefill_every_n = max(
            1, int(os.environ.get("QAIC_PREFILL_EVERY_N_STEPS", "8")))
        self._qaic_resume_frac = float(
            os.environ.get("QAIC_PREFILL_RESUME_FRAC", "0.5"))
        logger.info(
            "QAIC decode-priority scheduler active (prefill_every_n=%d, resume_frac=%.2f). "
            "Disable with QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER=1.",
            self._qaic_prefill_every_n, self._qaic_resume_frac)

    def _qaic_num_decode_running(self) -> int:
        # `self.running` can also hold requests still PREFILLING (chunked). A request
        # is in the decode phase once it has consumed all prompt tokens. Only decodes
        # are a reason to protect the step from new prefills.
        return sum(
            1 for r in self.running
            if r.num_computed_tokens >= r.num_prompt_tokens)

    def _qaic_should_defer_prefill(self) -> bool:
        # No decodes to protect -> let prefills flow (don't delay waiting prefills
        # just because other prefills are in flight).
        num_decode = self._qaic_num_decode_running()
        if num_decode == 0:
            return False
        # Cadence backstop: allow a prefill step at least every N deferred steps.
        if self._qaic_steps_since_prefill >= self._qaic_prefill_every_n:
            return False
        # Decode headroom: few decodes running -> prefilling now costs little TPOT.
        if num_decode < self._qaic_resume_frac * self.max_num_running_reqs:
            return False
        # Otherwise there is a decode backlog -> defer new prefills.
        return True

    def schedule(self):
        if not self._qaic_should_defer_prefill():
            self._qaic_steps_since_prefill = 0
            return super().schedule()

        # Cap the per-step token budget to exactly what the RUNNING requests need so
        # the stock waiting-admission loop admits no new prefill this step. Must cover
        # ALL running reqs (decodes + any in-flight prefill chunks) or they'd be cut.
        # Use the stock running-token formula (scheduler.py): num_tokens_with_spec +
        # num_output_placeholders - num_computed_tokens (placeholders/spec matter, or
        # the cap would be too small and starve decode itself). Never raise the budget.
        running_budget = sum(
            max(0, r.num_tokens_with_spec + r.num_output_placeholders
                - r.num_computed_tokens)
            for r in self.running)
        if running_budget <= 0:
            self._qaic_steps_since_prefill = 0
            return super().schedule()

        saved = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = min(saved, running_budget)
        try:
            output = super().schedule()
        finally:
            self.max_num_scheduled_tokens = saved
        self._qaic_steps_since_prefill += 1
        return output
