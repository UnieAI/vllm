# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from vllm._rs import batch_ngram_propose as _rs_batch_ngram_propose
    _HAS_RUST_NGRAM = True
except ImportError:
    try:
        from _rs import batch_ngram_propose as _rs_batch_ngram_propose
        _HAS_RUST_NGRAM = True
    except ImportError:
        _HAS_RUST_NGRAM = False


# Maximum number of tree branches (candidates) per position.
_MAX_TREE_BRANCHES = 3


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

        # Enable tree-structured multi-candidate proposing.
        # When True, the proposer finds multiple n-gram matches and
        # selects the best one using a recency-weighted scoring function.
        # This prepares data for future tree attention verification.
        self.enable_tree = True

        # Pre-allocate buffers for numba batch propose.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # Tree candidate buffers: (max_seqs, max_branches, k).
        # Branch 0 = primary (longest), branch 1 = recency-biased,
        # branch 2 = min_n=1 fallback.
        self.tree_drafts = np.zeros(
            (max_num_seqs, _MAX_TREE_BRANCHES, self.k), dtype=np.int32,
        )
        self.tree_num_drafts = np.zeros(
            (max_num_seqs, _MAX_TREE_BRANCHES), dtype=np.int32,
        )
        self.tree_scores = np.zeros(
            (max_num_seqs, _MAX_TREE_BRANCHES), dtype=np.float64,
        )

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        # Max number of threads for numba parallel processing.
        if cpu_count:
            # Divide by 2 to use physical cores
            # and not logical cores (hyper-threading).
            # Cap the number of threads to 8 to avoid using too many threads
            # since other components like frontend (incl tokenization)
            # and Structured Outputs also use multiple threads.
            self.num_numba_thread_available = min(8, max(1, cpu_count // 2))
            # Divide by tp_size to ensure each tensor parallel rank
            # has some threads since all ranks will run this.
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        self.propose(
            [[]] * 1024,
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, self.max_model_len), dtype=np.int32),
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Batch version of ngram proposer using numba for acceleration.

        Args:
            valid_ngram_requests:
                Set of indices of requests that need ngram proposals.
            num_tokens_no_spec:
                Numpy array of shape (batch_size,) representing the number
                of tokens without speculative tokens for each request.
            token_ids_cpu:
                Numpy array of shape (batch_size, max_model_len)
                representing the token IDs for each request.

        Returns:
            list[list[int]]:
                A list where each element is a list of proposed
                token IDs for the corresponding request.
        """
        draft_token_ids: list[list[int]] = []

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            if self.enable_tree:
                self._run_tree_propose(
                    valid_ngram_requests, num_tokens_no_spec,
                    token_ids_cpu, num_requests,
                )
            else:
                self._run_propose_backend(
                    valid_ngram_requests, num_tokens_no_spec,
                    token_ids_cpu, num_requests, self.min_n, self.max_n,
                )
                # D4: Relaxed retry.
                if self.min_n > 1:
                    retry_indices = [
                        idx for idx in valid_ngram_requests
                        if self.valid_ngram_num_drafts[idx] == 0
                    ]
                    if retry_indices:
                        self._run_propose_backend(
                            retry_indices, num_tokens_no_spec,
                            token_ids_cpu, num_requests, 1, self.max_n,
                        )

        valid_set = set(valid_ngram_requests)
        for i in range(num_requests):
            if i in valid_set and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    def _run_tree_propose(
        self,
        valid_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        num_requests: int,
    ) -> None:
        """Find multiple n-gram candidates and select the best one.

        Tree-structured multi-candidate proposing:
        1. Branch 0: longest match in full context (primary)
        2. Branch 1: longest match in recent 50% of context (recency-biased)
        3. Branch 2: min_n=1 fallback (widest net)

        Candidates are scored by: ngram_length * recency_weight.
        The highest-scoring candidate is placed into valid_ngram_draft.

        NOTE: Currently selects the single best candidate (linear
        verification). When tree attention is supported for CPU-side
        proposers, all branches can be verified in one GPU forward pass.
        """
        # Branch 0: full context, original min_n/max_n.
        self._run_propose_backend(
            valid_requests, num_tokens_no_spec,
            token_ids_cpu, num_requests, self.min_n, self.max_n,
        )
        for idx in valid_requests:
            nd = self.valid_ngram_num_drafts[idx]
            self.tree_drafts[idx, 0, :nd] = self.valid_ngram_draft[idx, :nd]
            self.tree_num_drafts[idx, 0] = nd
            # Score: ngram_length (num drafts as proxy).
            self.tree_scores[idx, 0] = float(nd)

        # Branch 1: recency-biased — search only last 50% of context.
        recency_requests = [
            idx for idx in valid_requests
            if num_tokens_no_spec[idx] > 100  # Need enough context.
        ]
        if recency_requests:
            # Create a view with only the recent half of tokens.
            recency_token_ids = np.copy(token_ids_cpu)
            recency_num_tokens = np.copy(num_tokens_no_spec)
            for idx in recency_requests:
                nt = num_tokens_no_spec[idx]
                half = nt // 2
                # Shift recent tokens to the beginning of the row, keeping
                # the last `half` tokens plus the suffix the KMP needs.
                # Actually, we just set num_tokens to start from halfway.
                # KMP matches suffix of the token sequence, so we limit the
                # search window by reducing the token count passed in.
                # But we need the suffix (last tokens) intact, so instead
                # we zero out the first half to prevent early matches.
                recency_token_ids[idx, :half] = -1  # Poison early tokens.
            self._run_propose_backend(
                recency_requests, recency_num_tokens,
                recency_token_ids, num_requests, self.min_n, self.max_n,
            )
            for idx in recency_requests:
                nd = self.valid_ngram_num_drafts[idx]
                self.tree_drafts[idx, 1, :nd] = (
                    self.valid_ngram_draft[idx, :nd]
                )
                self.tree_num_drafts[idx, 1] = nd
                # Score: ngram_length * 1.3 (recency bonus).
                self.tree_scores[idx, 1] = float(nd) * 1.3

        # Branch 2: min_n=1 fallback (D4).
        if self.min_n > 1:
            fallback_requests = [
                idx for idx in valid_requests
                if self.tree_num_drafts[idx, 0] == 0
                and self.tree_num_drafts[idx, 1] == 0
            ]
            if fallback_requests:
                self._run_propose_backend(
                    fallback_requests, num_tokens_no_spec,
                    token_ids_cpu, num_requests, 1, self.max_n,
                )
                for idx in fallback_requests:
                    nd = self.valid_ngram_num_drafts[idx]
                    self.tree_drafts[idx, 2, :nd] = (
                        self.valid_ngram_draft[idx, :nd]
                    )
                    self.tree_num_drafts[idx, 2] = nd
                    self.tree_scores[idx, 2] = float(nd) * 0.5

        # Select best candidate per request.
        for idx in valid_requests:
            best_branch = -1
            best_score = 0.0
            for b in range(_MAX_TREE_BRANCHES):
                if (self.tree_num_drafts[idx, b] > 0
                        and self.tree_scores[idx, b] > best_score):
                    best_score = self.tree_scores[idx, b]
                    best_branch = b
            if best_branch >= 0:
                nd = self.tree_num_drafts[idx, best_branch]
                self.valid_ngram_draft[idx, :nd] = (
                    self.tree_drafts[idx, best_branch, :nd]
                )
                self.valid_ngram_num_drafts[idx] = nd
            else:
                self.valid_ngram_num_drafts[idx] = 0

            # Reset tree buffers for next step.
            self.tree_num_drafts[idx, :] = 0
            self.tree_scores[idx, :] = 0.0

    def _run_propose_backend(
        self,
        valid_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        num_requests: int,
        min_n: int,
        max_n: int,
    ) -> None:
        """Run the n-gram propose backend (Rust or Numba)."""
        if _HAS_RUST_NGRAM:
            draft_arr, ndrafts_arr = _rs_batch_ngram_propose(
                token_ids_cpu,
                num_tokens_no_spec,
                valid_requests,
                min_n,
                max_n,
                self.max_model_len,
                self.k,
            )
            # Only update slots that were processed.
            for idx in valid_requests:
                if idx < num_requests:
                    self.valid_ngram_draft[idx, :] = draft_arr[idx, :]
                    self.valid_ngram_num_drafts[idx] = ndrafts_arr[idx]
        else:
            original_num_numba_threads = get_num_threads()
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available,
                           len(valid_requests))
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            batch_propose_numba(
                valid_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                min_n,
                max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )
            set_num_threads(original_num_numba_threads)

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        # find which requests need ngram proposals
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
