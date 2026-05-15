# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Dict, List, Optional

import numpy as np
from numba import jit, njit, prange, set_num_threads

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class NgramProposer:

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # --- Jeff's Optimizations ---
        self.search_window = getattr(
            vllm_config.speculative_config,
            "prompt_lookup_window",
            None,
        )
        self.default_search_window = max(
            0,
            int(os.environ.get("VLLM_NGRAM_DEFAULT_SEARCH_WINDOW", "1024")),
        )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k),
                                          dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs),
                                               dtype=np.int32)

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        if cpu_count:
            default_numba_threads = min(4, max(1, (cpu_count // 2) // max(1, tp_size)))
        else:
            default_numba_threads = 1
        default_tokens_threshold = 4096 if default_numba_threads > 1 else 16384
        self.num_numba_thread_available = max(
            1,
            int(os.environ.get("VLLM_NGRAM_NUMBA_THREADS", str(default_numba_threads))),
        )
        self.num_tokens_threshold = max(
            1,
            int(os.environ.get("VLLM_NGRAM_NUMBA_TOKENS_THRESHOLD", str(default_tokens_threshold))),
        )
        # Keep one thread for small batches and only switch when needed.
        self._current_numba_threads = 1
        set_num_threads(self._current_numba_threads)

        # Backoff Heuristics
        self.no_match_backoff_enabled = bool(
            int(os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF", "1")))
        self.no_match_backoff_min_len = int(
            os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF_MIN_LEN", "256"))
        self.no_match_backoff_max_steps = int(
            os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF_MAX_STEPS", "4"))
        self.no_match_backoff_max_steps = max(1, self.no_match_backoff_max_steps)
        self.no_match_backoff_long_ctx_threshold = max(
            0,
            int(os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF_LONG_CTX_THRESHOLD", "2048")),
        )
        self.no_match_backoff_max_steps_long_ctx = max(
            self.no_match_backoff_max_steps,
            int(os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF_MAX_STEPS_LONG_CTX", "8")),
        )
        self.no_match_backoff_window = max(
            0,
            int(os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF_WINDOW", "256")),
        )
        self.no_match_backoff_window_streak = max(
            1,
            int(os.environ.get("VLLM_NGRAM_NO_MATCH_BACKOFF_WINDOW_STREAK", "2")),
        )
        self.max_match_reqs_per_step = max(
            0,
            int(os.environ.get("VLLM_NGRAM_MAX_MATCH_REQS_PER_STEP", "8")),
        )
        self._match_rr_cursor = 0
        self._req_skip_match_steps = np.zeros(max_num_seqs, dtype=np.int32)
        self._req_no_match_streak = np.zeros(max_num_seqs, dtype=np.int32)
        self._req_active_mask = np.zeros(max_num_seqs, dtype=np.bool_)

        # Warmup
        self._warmup(max_num_seqs)

    def _warmup(self, max_num_seqs):
        warmup_num_reqs = min(8, max_num_seqs)
        warmup_model_len = min(
            self.max_model_len,
            max(64, self.max_n + self.k, self.min_n + self.k),
        )
        warmup_valid_ngram_requests = np.arange(warmup_num_reqs, dtype=np.int32)
        self.batch_propose(
            warmup_num_reqs,
            warmup_valid_ngram_requests,
            np.full(warmup_num_reqs, warmup_model_len, dtype=np.int32),
            np.zeros((warmup_num_reqs, warmup_model_len), dtype=np.int32),
        )
        self._req_skip_match_steps.fill(0)
        self._req_no_match_streak.fill(0)

    def _ensure_request_backoff_state(self, num_requests: int) -> None:
        current_size = self._req_skip_match_steps.shape[0]
        if num_requests <= current_size:
            return
        new_size = max(num_requests, current_size * 2)
        new_skip = np.zeros(new_size, dtype=np.int32)
        new_streak = np.zeros(new_size, dtype=np.int32)
        new_active = np.zeros(new_size, dtype=np.bool_)
        new_skip[:current_size] = self._req_skip_match_steps
        new_streak[:current_size] = self._req_no_match_streak
        new_active[:current_size] = self._req_active_mask
        self._req_skip_match_steps = new_skip
        self._req_no_match_streak = new_streak
        self._req_active_mask = new_active

    def _select_match_requests_with_budget(self, run_requests: np.ndarray) -> np.ndarray:
        budget = self.max_match_reqs_per_step
        num_candidates = int(run_requests.size)
        if budget <= 0 or num_candidates <= budget:
            return run_requests
        if num_candidates <= (budget << 1):
            return run_requests
        start = int(self._match_rr_cursor % num_candidates)
        select_idx = (start + np.arange(budget, dtype=np.int32)) % num_candidates
        self._match_rr_cursor = (start + budget) % num_candidates
        return run_requests[select_idx]

    def run_batch_match(self,
                        valid_ngram_requests: np.ndarray,
                        num_tokens_no_spec: np.ndarray,
                        token_ids_cpu: np.ndarray,
                        search_window: int | None = None,
                        draft_k: int | None = None) -> None:
        num_ngram_requests = len(valid_ngram_requests)
        if not num_ngram_requests:
            return

        desired_threads = 1
        if self.num_numba_thread_available > 1 and num_ngram_requests > 1:
            total_tokens = int(np.sum(num_tokens_no_spec[valid_ngram_requests]))
            if total_tokens >= self.num_tokens_threshold:
                desired_threads = min(self.num_numba_thread_available,
                                      num_ngram_requests)
        if desired_threads != self._current_numba_threads:
            set_num_threads(desired_threads)
            self._current_numba_threads = desired_threads

        batch_propose_numba(
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
            self.min_n,
            self.max_n,
            search_window if search_window is not None else
            (self.search_window if self.search_window is not None
             else self.default_search_window),
            self.max_model_len,
            (self.k if draft_k is None else draft_k),
            self.valid_ngram_draft,
            self.valid_ngram_num_drafts,
        )

    def propose_batch(self, req_ids: List[str],
                      token_ids: List[List[int]]) -> List[List[int]]:
        if not req_ids:
            return []

    def batch_propose(self, num_requests: int,
                      valid_ngram_requests: np.ndarray,
                      num_tokens_no_spec: np.ndarray,
                      token_ids_cpu: np.ndarray) -> list[list[int]]:
        self._ensure_request_backoff_state(num_requests)
        num_ngram_requests = len(valid_ngram_requests)
        if num_ngram_requests == 0:
            if self.no_match_backoff_enabled and num_requests > 0:
                self._req_active_mask[:num_requests] = False
            return [[] for _ in range(num_requests)]

        if self.no_match_backoff_enabled and num_requests > 0:
            active_mask = self._req_active_mask[:num_requests]
            was_active_for_valid = active_mask[valid_ngram_requests]
            active_mask[:] = False
            active_mask[valid_ngram_requests] = True
            newly_active = valid_ngram_requests[~was_active_for_valid]
            if newly_active.size > 0:
                self._req_skip_match_steps[newly_active] = 0
                self._req_no_match_streak[newly_active] = 0

        self.valid_ngram_num_drafts[valid_ngram_requests] = 0
        run_requests = valid_ngram_requests
        use_no_match_backoff = self.no_match_backoff_enabled
        
        if use_no_match_backoff:
            req_indices = valid_ngram_requests
            token_counts = num_tokens_no_spec[req_indices]
            short_mask = token_counts < self.no_match_backoff_min_len
            if short_mask.any():
                short_reqs = req_indices[short_mask]
                self._req_skip_match_steps[short_reqs] = 0
                self._req_no_match_streak[short_reqs] = 0

            remaining = self._req_skip_match_steps[req_indices]
            skip_now_mask = (~short_mask) & (remaining > 0)
            if skip_now_mask.any():
                skip_reqs = req_indices[skip_now_mask]
                self._req_skip_match_steps[skip_reqs] -= 1
            run_requests = req_indices[(~short_mask) & (~skip_now_mask)]

        if run_requests.size > 0 and self.max_match_reqs_per_step > 0:
            run_requests = self._select_match_requests_with_budget(run_requests)
        
        if run_requests.size > 0:
            default_window = self.search_window if self.search_window is not None \
                else self.default_search_window
            use_window_backoff = (
                use_no_match_backoff
                and self.no_match_backoff_window > 0
                and (default_window == 0
                     or default_window > self.no_match_backoff_window))
            
            if use_window_backoff:
                streaks = self._req_no_match_streak[run_requests]
                backoff_window_mask = streaks >= self.no_match_backoff_window_streak
                run_requests_default = run_requests[~backoff_window_mask]
                run_requests_backoff = run_requests[backoff_window_mask]
                if run_requests_default.size > 0:
                    self.run_batch_match(
                        run_requests_default,
                        num_tokens_no_spec,
                        token_ids_cpu,
                        search_window=default_window,
                        draft_k=self.k,
                    )
                if run_requests_backoff.size > 0:
                    self.run_batch_match(
                        run_requests_backoff,
                        num_tokens_no_spec,
                        token_ids_cpu,
                        search_window=self.no_match_backoff_window,
                        draft_k=self.k,
                    )
            else:
                self.run_batch_match(
                    run_requests,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    search_window=default_window,
                    draft_k=self.k,
                )

        # Materialize results
        draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]
        if valid_ngram_requests.size > 0:
            matched_mask = self.valid_ngram_num_drafts[valid_ngram_requests] > 0
            matched_requests = valid_ngram_requests[matched_mask]
            for i in matched_requests:
                draft_len = int(self.valid_ngram_num_drafts[i])
                draft_token_ids[i] = self.valid_ngram_draft[i, :draft_len].tolist()

        # Update backoff state
        if use_no_match_backoff:
            if run_requests.size > 0:
                matched_mask = self.valid_ngram_num_drafts[run_requests] > 0
                matched_reqs = run_requests[matched_mask]
                if matched_reqs.size > 0:
                    self._req_skip_match_steps[matched_reqs] = 0
                    self._req_no_match_streak[matched_reqs] = 0

                unmatched_reqs = run_requests[~matched_mask]
                if unmatched_reqs.size > 0:
                    new_streak = self._req_no_match_streak[unmatched_reqs] + 1
                    self._req_no_match_streak[unmatched_reqs] = new_streak
                    exp = np.minimum(new_streak - 1, 10)
                    skip_steps = np.left_shift(np.ones_like(exp, dtype=np.int32), exp)
                    max_steps = np.full_like(skip_steps, self.no_match_backoff_max_steps, dtype=np.int32)
                    
                    if self.no_match_backoff_long_ctx_threshold > 0:
                        unmatched_num_tokens = num_tokens_no_spec[unmatched_reqs]
                        long_ctx_mask = unmatched_num_tokens >= self.no_match_backoff_long_ctx_threshold
                        if long_ctx_mask.any():
                            max_steps[long_ctx_mask] = self.no_match_backoff_max_steps_long_ctx
                    
                    self._req_skip_match_steps[unmatched_reqs] = np.minimum(max_steps, skip_steps)
        
        return draft_token_ids

    def propose(self, context_token_ids: np.ndarray) -> Optional[np.ndarray]:
        # Legacy interface for single request
        num_tokens = context_token_ids.shape[0]
        token_ids_cpu = np.expand_dims(context_token_ids, axis=0)
        num_tokens_no_spec = np.array([num_tokens], dtype=np.int32)
        valid_ngram_requests = np.array([0], dtype=np.int32)
        
        results = self.batch_propose(1, valid_ngram_requests, num_tokens_no_spec, token_ids_cpu)
        if results[0]:
            return np.array(results[0], dtype=np.int32)
        return None

    def load_model(self, *args, **kwargs):
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: np.ndarray,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    search_window: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        search_start = 0
        if search_window > 0 and num_tokens > search_window:
            search_start = num_tokens - search_window
        context_token_ids = token_ids_cpu[idx, search_start:num_tokens]
        start_position, draft_len = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )
        valid_ngram_num_drafts[idx] = draft_len
        if draft_len > 0:
            valid_ngram_draft[idx, :draft_len] = context_token_ids[
                start_position:start_position + draft_len]


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> tuple[int, int]:
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return 0, 0

    k = min(k, max_model_len - total_token)
    if k <= 0:
        return 0, 0

    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return 0, 0

    start_position = total_token - 1 - position + longest_ngram
    draft_len = min(k, total_token - start_position)
    return start_position, draft_len
