# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        has_structured_output_requests = False
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        enable_spec_decode = scheduler_output.enable_spec_decode
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            has_structured_output_requests |= request.use_structured_output
            pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = (
                len(spec_decode_tokens.get(req_id, ())) if enable_spec_decode else 0
            )
            if (
                request.num_computed_tokens
                == request.num_tokens
                + request.num_output_placeholders
                + cur_num_spec_tokens
            ):
                # The request will generate a new token plus num_spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add placeholders for the new tokens in spec_token_ids.
                # We will update the actual spec token ids in the worker process.
                request.spec_token_ids = [-1] * cur_num_spec_tokens

        scheduler_output.has_structured_output_requests = has_structured_output_requests
        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens
        )

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        # In async + speculative fallback paths, accounting can transiently
        # drift and produce more returned tokens than placeholders.
        # Clamp to zero to keep the scheduler alive.
        request.num_output_placeholders -= len(new_token_ids)
        if request.num_output_placeholders < 0:
            logger.warning(
                "Clamp negative num_output_placeholders for req %s: %d "
                "(returned_tokens=%d).",
                request.request_id,
                request.num_output_placeholders,
                len(new_token_ids),
            )
            request.num_output_placeholders = 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            num_tokens_to_cache = (
                request.num_computed_tokens - request.num_output_placeholders
            )
            if num_tokens_to_cache < 0:
                logger.warning(
                    "Clamp negative num_tokens_to_cache for req %s: %d "
                    "(computed=%d placeholders=%d).",
                    request.request_id,
                    num_tokens_to_cache,
                    request.num_computed_tokens,
                    request.num_output_placeholders,
                )
                num_tokens_to_cache = 0
            elif num_tokens_to_cache > request.num_tokens:
                logger.warning(
                    "Clamp overflow num_tokens_to_cache for req %s: %d -> %d "
                    "(computed=%d placeholders=%d num_tokens=%d).",
                    request.request_id,
                    num_tokens_to_cache,
                    request.num_tokens,
                    request.num_computed_tokens,
                    request.num_output_placeholders,
                    request.num_tokens,
                )
                num_tokens_to_cache = request.num_tokens

            if num_tokens_to_cache:
                self.kv_cache_manager.cache_blocks(request, num_tokens_to_cache)
        return new_token_ids, stopped
