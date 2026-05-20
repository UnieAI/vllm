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
        spec_cfg = vllm_config.speculative_config
        
        self.min_n = spec_cfg.prompt_lookup_min
        self.max_n = spec_cfg.prompt_lookup_max
        self.k = spec_cfg.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Configurable search window
        self.search_window = getattr(spec_cfg, "prompt_lookup_window", 1024)
        
        # Parallelism settings
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_numba_threads = int(os.environ.get("VLLM_NGRAM_NUMBA_THREADS", "8"))
        set_num_threads(self.num_numba_threads)

        # Buffers for results
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros(max_num_seqs, dtype=np.int32)

        # Warmup
        self._warmup(max_num_seqs)

    def _warmup(self, max_num_seqs):
        num = min(4, max_num_seqs)
        dummy_tokens = np.zeros((num, 128), dtype=np.int32)
        valid_reqs = np.arange(num, dtype=np.int32)
        self.batch_propose(num, valid_reqs, np.full(num, 128, dtype=np.int32), dummy_tokens)

    def batch_propose(self, num_requests: int,
                      valid_ngram_requests: np.ndarray,
                      num_tokens_no_spec: np.ndarray,
                      token_ids_cpu: np.ndarray) -> List[List[int]]:
        
        # Reset buffers
        self.valid_ngram_num_drafts.fill(0)
        
        # Call optimized kernel
        _batch_propose_kernel(
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
            self.min_n,
            self.max_n,
            self.search_window,
            self.max_model_len,
            self.k,
            self.valid_ngram_draft,
            self.valid_ngram_num_drafts
        )
        
        # Collect results
        results = [[] for _ in range(num_requests)]
        for req_idx in valid_ngram_requests:
            n = self.valid_ngram_num_drafts[req_idx]
            if n > 0:
                results[req_idx] = self.valid_ngram_draft[req_idx, :n].tolist()
        return results

@njit(parallel=True, cache=True)
def _batch_propose_kernel(valid_reqs, num_tokens, all_tokens, 
                         min_n, max_n, window, max_model_len, k,
                         out_draft, out_num):
    
    for idx in prange(len(valid_reqs)):
        req_idx = valid_reqs[idx]
        n_tokens = num_tokens[req_idx]
        if n_tokens < min_n:
            continue
            
        # Use a local window view if window is set
        start_pos = max(0, n_tokens - window)
        tokens = all_tokens[req_idx, start_pos:n_tokens]
        curr_len = len(tokens)
        
        # High-performance search logic (LPS-based)
        # We need to find the longest suffix of 'tokens' that matches a previous occurrence
        # This is equivalent to finding the longest prefix of reversed_tokens 
        # that matches somewhere else in reversed_tokens.
        
        # Optimized reverse for Numba
        rev_tokens = tokens[::-1]
        
        lps = np.zeros(max_n, dtype=np.int32)
        longest_ngram = 0
        match_pos = 0
        
        prev_lps = 0
        i = 1
        while i < curr_len:
            if rev_tokens[prev_lps] == rev_tokens[i]:
                prev_lps += 1
                if prev_lps >= longest_ngram:
                    longest_ngram = prev_lps
                    match_pos = i
                if i < max_n:
                    lps[i] = prev_lps
                if prev_lps == max_n:
                    prev_lps = lps[max_n - 1]
                i += 1
            elif prev_lps != 0:
                prev_lps = lps[prev_lps - 1]
            else:
                i += 1
        
        if longest_ngram >= min_n:
            # Found a match!
            # match_pos is the end of the match in rev_tokens
            # in original 'tokens' array:
            # match_end = curr_len - 1 - (match_pos - longest_ngram)
            # Actually simpler: start_in_tokens = curr_len - match_pos + longest_ngram - 1
            # Wait, let's re-calculate:
            # rev_tokens[0:longest_ngram] == rev_tokens[match_pos-longest_ngram+1 : match_pos+1]
            # In 'tokens' array, this match ends at: curr_len - 1 - (match_pos - longest_ngram + 1) + 1
            # Which is curr_len - 1 - match_pos + longest_ngram
            
            start_draft_in_window = curr_len - match_pos + longest_ngram - 1
            num_to_copy = min(k, curr_len - start_draft_in_window)
            # Also limit by max_model_len
            num_to_copy = min(num_to_copy, max_model_len - n_tokens)
            
            if num_to_copy > 0:
                out_num[req_idx] = num_to_copy
                for d in range(num_to_copy):
                    out_draft[req_idx, d] = tokens[start_draft_in_window + d]
