# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Rust-accelerated n-gram proposer."""
import numpy as np
import pytest

try:
    from vllm_rs import batch_ngram_propose
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


@pytest.mark.skipif(not HAS_RUST, reason="vllm_rs not installed")
class TestRustNgramProposer:
    def test_basic_match(self):
        token_ids = np.zeros((1, 100), dtype=np.int32)
        token_ids[0, :8] = [1, 2, 3, 4, 5, 1, 2, 3]
        draft, nd = batch_ngram_propose(
            token_ids, np.array([8], dtype=np.int32),
            [0], 2, 5, 100, 3,
        )
        assert nd[0] == 3
        assert list(draft[0, :3]) == [4, 5, 1]

    def test_no_match(self):
        token_ids = np.zeros((1, 100), dtype=np.int32)
        token_ids[0, :8] = [1, 2, 3, 4, 5, 6, 7, 8]
        _, nd = batch_ngram_propose(
            token_ids, np.array([8], dtype=np.int32),
            [0], 2, 5, 100, 3,
        )
        assert nd[0] == 0

    def test_min_ngram_filter(self):
        token_ids = np.zeros((1, 100), dtype=np.int32)
        token_ids[0, :4] = [1, 2, 3, 1]
        _, nd = batch_ngram_propose(
            token_ids, np.array([4], dtype=np.int32),
            [0], 2, 5, 100, 3,
        )
        assert nd[0] == 0  # only 1-gram match, min_n=2

    def test_batch_processing(self):
        B = 4
        token_ids = np.zeros((B, 100), dtype=np.int32)
        token_ids[0, :8] = [1, 2, 3, 4, 5, 1, 2, 3]
        token_ids[1, :5] = [10, 20, 30, 40, 50]
        token_ids[2, :8] = [5, 6, 7, 8, 9, 5, 6, 7]
        token_ids[3, :6] = [1, 2, 1, 2, 1, 2]
        nt = np.array([8, 5, 8, 6], dtype=np.int32)

        _, nd = batch_ngram_propose(token_ids, nt, [0, 1, 2, 3], 2, 5, 100, 3)
        assert nd[0] > 0
        assert nd[1] == 0
        assert nd[2] > 0
        assert nd[3] > 0

    def test_valid_indices_subset(self):
        B = 4
        token_ids = np.zeros((B, 100), dtype=np.int32)
        token_ids[0, :8] = [1, 2, 3, 4, 5, 1, 2, 3]
        token_ids[1, :8] = [5, 6, 7, 8, 9, 5, 6, 7]
        nt = np.array([8, 8, 0, 0], dtype=np.int32)

        _, nd = batch_ngram_propose(token_ids, nt, [0], 2, 5, 100, 3)
        assert nd[0] > 0
        assert nd[1] == 0  # not in valid_indices

    def test_max_model_len_limit(self):
        token_ids = np.zeros((1, 100), dtype=np.int32)
        token_ids[0, :8] = [1, 2, 3, 4, 5, 1, 2, 3]
        _, nd = batch_ngram_propose(
            token_ids, np.array([8], dtype=np.int32),
            [0], 2, 5, 9, 3,  # max_model_len=9, only room for 1
        )
        assert nd[0] == 1

    def test_cross_validate_with_numba(self):
        """Verify Rust matches Numba on random data."""
        from numba import jit, njit, prange

        @jit(nopython=True)
        def _kmp(t, mn, mx, ml, k):
            n = t.shape[0]
            if n < mn:
                return np.empty((0,), dtype=t.dtype)
            k = min(k, ml - n)
            if k <= 0:
                return np.empty((0,), dtype=t.dtype)
            r = t[::-1]
            lps = np.zeros(mx, dtype=np.int32)
            lg = 0; pos = 0; p = 0; i = 1
            while i < n:
                if r[p] == r[i]:
                    p += 1
                    if p >= lg:
                        lg = p; pos = i
                    if i < mx:
                        lps[i] = p
                    if p == mx:
                        p = lps[mx - 1]
                    i += 1
                elif p != 0:
                    p = lps[p - 1]
                else:
                    i += 1
            if lg < mn:
                return np.empty((0,), dtype=t.dtype)
            s = n - 1 - pos + lg
            k = min(k, n - s)
            return t[s:s + k]

        @njit(parallel=True)
        def batch_nb(v, nt, ti, mn, mx, ml, k, d, nd):
            for i in prange(len(v)):
                x = v[i]
                r = _kmp(ti[x, :nt[x]], mn, mx, ml, k)
                nd[x] = r.shape[0]
                if len(r):
                    d[x, :r.shape[0]] = r

        rng = np.random.default_rng(123)
        B, SL, K = 64, 500, 5
        MN, MX, ML = 3, 7, 10000
        ti = np.zeros((B, ML), dtype=np.int32)
        ti[:, :SL] = rng.integers(0, 30, size=(B, SL), dtype=np.int32)
        nt = np.full(B, SL, dtype=np.int32)
        vi = list(range(B))

        db = np.zeros((B, K), dtype=np.int32)
        nb_ = np.zeros(B, dtype=np.int32)
        batch_nb(vi, nt, ti, MN, MX, ML, K, db, nb_)

        rd, rn = batch_ngram_propose(ti, nt, vi, MN, MX, ML, K)

        for i in range(B):
            assert nb_[i] == rn[i], (
                f"req {i}: numba={nb_[i]} rust={rn[i]}"
            )
            if nb_[i] > 0:
                assert list(db[i, :nb_[i]]) == list(rd[i, :rn[i]]), (
                    f"req {i}: numba={list(db[i,:nb_[i]])} "
                    f"rust={list(rd[i,:rn[i]])}"
                )

    def test_performance(self):
        """Sanity check: Rust should be reasonably fast."""
        import time

        B, SL, K = 128, 1500, 5
        MN, MX, ML = 3, 7, 100000
        rng = np.random.default_rng(42)
        ti = np.zeros((B, ML), dtype=np.int32)
        ti[:, :SL] = rng.integers(0, 50, size=(B, SL), dtype=np.int32)
        nt = np.full(B, SL, dtype=np.int32)
        vi = list(range(B))

        # Warmup
        batch_ngram_propose(ti, nt, vi, MN, MX, ML, K)

        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            batch_ngram_propose(ti, nt, vi, MN, MX, ML, K)
        elapsed = (time.perf_counter() - t0) / iters

        # Should complete under 2ms for 128 reqs × 1500 tokens
        assert elapsed < 0.002, f"Too slow: {elapsed*1e3:.2f}ms"
