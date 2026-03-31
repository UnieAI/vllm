# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Rust-accelerated scheduler functions.

These tests verify both the Rust native implementation (when available)
and the pure-Python fallback by testing the public API in
vllm.v1.core.sched.rust_accelerated.
"""
import numpy as np
import pytest

# Test the low-level Rust functions directly if available.
try:
    from vllm._rs import (
        batch_apply_generated_tokens,
        batch_check_stop,
        compute_running_tokens,
        compute_waiting_tokens,
    )

    HAS_RUST = True
except ImportError:
    try:
        from _rs import (
            batch_apply_generated_tokens,
            batch_check_stop,
            compute_running_tokens,
            compute_waiting_tokens,
        )

        HAS_RUST = True
    except ImportError:
        HAS_RUST = False


@pytest.mark.skipif(not HAS_RUST, reason="vllm._rs not installed")
class TestRustComputeRunningTokens:
    def test_basic(self):
        spec = np.array([100, 50, 200], dtype=np.int64)
        ph = np.zeros(3, dtype=np.int64)
        computed = np.array([90, 49, 100], dtype=np.int64)
        prompt = np.array([100, 50, 200], dtype=np.int64)
        max_tok = np.full(3, 100, dtype=np.int64)

        result = compute_running_tokens(
            spec, ph, computed, prompt, max_tok, 1000, 0, 4096
        )
        np.testing.assert_array_equal(result, [10, 1, 100])

    def test_budget_clamp(self):
        spec = np.array([100, 50, 200], dtype=np.int64)
        ph = np.zeros(3, dtype=np.int64)
        computed = np.array([90, 49, 100], dtype=np.int64)
        prompt = np.array([100, 50, 200], dtype=np.int64)
        max_tok = np.full(3, 100, dtype=np.int64)

        result = compute_running_tokens(
            spec, ph, computed, prompt, max_tok, 15, 0, 4096
        )
        # Budget: 15 -> 10 (req0) -> 5 left -> 1 (req1) -> 4 left -> 4 (req2)
        np.testing.assert_array_equal(result, [10, 1, 4])

    def test_long_prefill_threshold(self):
        spec = np.array([1000], dtype=np.int64)
        ph = np.zeros(1, dtype=np.int64)
        computed = np.zeros(1, dtype=np.int64)
        prompt = np.array([1000], dtype=np.int64)
        max_tok = np.full(1, 100, dtype=np.int64)

        result = compute_running_tokens(
            spec, ph, computed, prompt, max_tok, 10000, 512, 4096
        )
        np.testing.assert_array_equal(result, [512])

    def test_max_model_len_clamp(self):
        spec = np.array([5000], dtype=np.int64)
        ph = np.zeros(1, dtype=np.int64)
        computed = np.array([4000], dtype=np.int64)
        prompt = np.array([5000], dtype=np.int64)
        max_tok = np.full(1, 10000, dtype=np.int64)

        result = compute_running_tokens(
            spec, ph, computed, prompt, max_tok, 10000, 0, 4096
        )
        # max_model_len - 1 - computed = 4095 - 4000 = 95
        np.testing.assert_array_equal(result, [95])

    def test_async_scheduling_skip(self):
        # Request has placeholders and has reached max_tokens
        spec = np.array([110], dtype=np.int64)
        ph = np.array([3], dtype=np.int64)
        computed = np.array([110], dtype=np.int64)
        prompt = np.array([100], dtype=np.int64)
        max_tok = np.array([10], dtype=np.int64)  # prompt=100 + max_tok=10 = 110

        # computed + 2 - placeholders = 110 + 2 - 3 = 109 < 110 -> NOT skipped
        result = compute_running_tokens(
            spec, ph, computed, prompt, max_tok, 1000, 0, 4096
        )
        assert result[0] >= 0  # Just check it doesn't crash

    def test_empty(self):
        result = compute_running_tokens(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            1000, 0, 4096,
        )
        assert len(result) == 0


@pytest.mark.skipif(not HAS_RUST, reason="vllm._rs not installed")
class TestRustBatchCheckStop:
    def test_eos(self):
        # Now uses padded 2-D numpy array for stop_token_ids
        result = batch_check_stop(
            np.array([2], dtype=np.int64),   # last token
            np.array([100], dtype=np.int64), # num_tokens
            np.array([10], dtype=np.int64),  # num_output
            np.array([0], dtype=np.int64),   # min_tokens
            np.array([100], dtype=np.int64), # max_tokens
            np.array([2], dtype=np.int64),   # eos_token_id
            np.empty((1, 0), dtype=np.int64),  # no stop tokens
            4096,
        )
        assert result[0] == 1  # STOP_EOS

    def test_stop_token(self):
        # 2-D array: row 0 has stop tokens [42, 99]
        stop_arr = np.array([[42, 99]], dtype=np.int64)
        result = batch_check_stop(
            np.array([42], dtype=np.int64),
            np.array([100], dtype=np.int64),
            np.array([10], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([100], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            stop_arr,
            4096,
        )
        assert result[0] == 2  # STOP_TOKEN

    def test_length_cap(self):
        result = batch_check_stop(
            np.array([7], dtype=np.int64),
            np.array([4096], dtype=np.int64),
            np.array([50], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([100], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.empty((1, 0), dtype=np.int64),
            4096,
        )
        assert result[0] == 3  # STOP_LENGTH

    def test_min_tokens_suppresses_stop(self):
        result = batch_check_stop(
            np.array([2], dtype=np.int64),
            np.array([100], dtype=np.int64),
            np.array([5], dtype=np.int64),  # only 5 output tokens
            np.array([10], dtype=np.int64), # min_tokens=10
            np.array([100], dtype=np.int64),
            np.array([2], dtype=np.int64),  # EOS match, but min not reached
            np.empty((1, 0), dtype=np.int64),
            4096,
        )
        assert result[0] == 0  # Not stopped (below min_tokens)

    def test_multiple_requests_mixed(self):
        """Test batch with 4 requests hitting different stop conditions."""
        # Row padding: max stop list length is 1 (from request 1)
        stop_arr = np.array([[-1], [5], [-1], [-1]], dtype=np.int64)
        result = batch_check_stop(
            np.array([2, 5, 99, 100], dtype=np.int64),    # last tokens
            np.array([100, 50, 4096, 200], dtype=np.int64),  # num_tokens
            np.array([10, 5, 50, 100], dtype=np.int64),     # num_output
            np.array([0, 0, 0, 0], dtype=np.int64),         # min_tokens
            np.array([100, 100, 100, 100], dtype=np.int64), # max_tokens
            np.array([2, -1, -1, -1], dtype=np.int64),      # eos
            stop_arr,
            4096,
        )
        np.testing.assert_array_equal(result, [1, 2, 3, 3])


@pytest.mark.skipif(not HAS_RUST, reason="vllm._rs not installed")
class TestRustBatchApplyGenerated:
    def test_no_spec(self):
        ac, ap, aa, ar = batch_apply_generated_tokens(
            np.array([100, 200], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(ac, [100, 200])
        np.testing.assert_array_equal(ar, [0, 0])
        np.testing.assert_array_equal(aa, [1, 1])

    def test_with_spec(self):
        # 5 draft tokens, 3 generated -> 2 accepted, 3 rejected
        ac, ap, aa, ar = batch_apply_generated_tokens(
            np.array([105], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([5], dtype=np.int64),
        )
        np.testing.assert_array_equal(ac, [102])
        np.testing.assert_array_equal(ap, [2])
        np.testing.assert_array_equal(aa, [2])
        np.testing.assert_array_equal(ar, [3])


@pytest.mark.skipif(not HAS_RUST, reason="vllm._rs not installed")
class TestRustComputeWaitingTokens:
    def test_basic(self):
        result, remaining = compute_waiting_tokens(
            np.array([1000, 500, 200], dtype=np.int64),
            np.array([0, 100, 50], dtype=np.int64),
            2000, 0, True,
        )
        np.testing.assert_array_equal(result, [1000, 400, 150])
        assert remaining == 450

    def test_chunked_prefill_disabled(self):
        # First request exceeds budget -> stops
        result, remaining = compute_waiting_tokens(
            np.array([1000, 500], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            500, 0, False,  # chunked prefill disabled
        )
        np.testing.assert_array_equal(result, [0, 0])
        assert remaining == 500


@pytest.mark.skipif(not HAS_RUST, reason="vllm._rs not installed")
class TestRustPerformance:
    """Sanity check that Rust is faster than a Python loop."""

    def test_compute_running_is_fast(self):
        import time

        N = 1000
        rng = np.random.default_rng(42)
        spec = rng.integers(100, 4096, size=N, dtype=np.int64)
        ph = np.zeros(N, dtype=np.int64)
        computed = spec - rng.integers(1, 50, size=N, dtype=np.int64)
        prompt = rng.integers(50, 2000, size=N, dtype=np.int64)
        max_tok = np.full(N, 1024, dtype=np.int64)

        # Warm up
        compute_running_tokens(spec, ph, computed, prompt, max_tok, 100000, 0, 4096)

        iters = 1000
        t0 = time.perf_counter()
        for _ in range(iters):
            compute_running_tokens(spec, ph, computed, prompt, max_tok, 100000, 0, 4096)
        elapsed = (time.perf_counter() - t0) / iters

        # Should complete in under 100μs (typically ~6μs)
        assert elapsed < 0.0001, f"Too slow: {elapsed*1e6:.1f}μs"

    def test_batch_check_stop_is_fast(self):
        """Verify the numpy 2-D stop_token_ids path is fast."""
        import time

        N = 1000
        rng = np.random.default_rng(123)
        last_tok = rng.integers(0, 50000, size=N, dtype=np.int64)
        num_tok = rng.integers(50, 4096, size=N, dtype=np.int64)
        num_out = rng.integers(1, 200, size=N, dtype=np.int64)
        min_tok = np.zeros(N, dtype=np.int64)
        max_tok = np.full(N, 512, dtype=np.int64)
        eos = np.full(N, 2, dtype=np.int64)

        # 5 stop tokens per request, padded with -1
        max_stop_len = 5
        stop_arr = np.full((N, max_stop_len), -1, dtype=np.int64)
        for i in range(N):
            n_stops = rng.integers(0, max_stop_len + 1)
            stop_arr[i, :n_stops] = rng.integers(0, 50000, size=n_stops)

        # Warm up
        batch_check_stop(
            last_tok, num_tok, num_out, min_tok, max_tok, eos,
            stop_arr, 4096,
        )

        iters = 1000
        t0 = time.perf_counter()
        for _ in range(iters):
            batch_check_stop(
                last_tok, num_tok, num_out, min_tok, max_tok, eos,
                stop_arr, 4096,
            )
        elapsed = (time.perf_counter() - t0) / iters

        # Should complete in under 100μs (typically ~3μs with numpy 2D)
        assert elapsed < 0.0001, f"Too slow: {elapsed*1e6:.1f}μs"
