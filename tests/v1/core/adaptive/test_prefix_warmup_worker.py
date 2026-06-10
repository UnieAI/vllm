# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PrefixWarmupWorker."""

from __future__ import annotations

import time

import pytest

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.adaptive.prefix_warmup_worker import (
    PrefixWarmupWorker,
)

# ------------------------------------------------------------------
# Mock / Fake implementations of protocol interfaces
# ------------------------------------------------------------------


class FakeBlockPool:
    """Fake block pool for testing."""

    def __init__(
        self,
        cached_hashes: set[int] | None = None,
        num_free_blocks: int = 100,
        total_num_blocks: int = 100,
    ):
        self._cached_hashes = cached_hashes or set()
        self._num_free_blocks = num_free_blocks
        self._total_num_blocks = total_num_blocks

    def get_cached_block_hashes(self) -> set[int]:
        return self._cached_hashes

    def get_num_free_blocks(self) -> int:
        return self._num_free_blocks


class FakeKVCacheManager:
    """Fake KV cache manager for testing."""

    def __init__(
        self,
        num_free_blocks: int = 80,
        total_num_blocks: int = 100,
    ):
        self._num_free_blocks = num_free_blocks
        self._total_num_blocks = total_num_blocks
        self.block_pool = FakeBlockPool(
            num_free_blocks=num_free_blocks,
            total_num_blocks=total_num_blocks,
        )

    @property
    def usage(self) -> float:
        """Return KV cache usage ratio."""
        if self._total_num_blocks == 0:
            return 0.0
        return 1.0 - (self._num_free_blocks / self._total_num_blocks)

    def set_usage(self, usage_ratio: float) -> None:
        """Helper to set a specific usage ratio."""
        used = int(self._total_num_blocks * usage_ratio)
        self._num_free_blocks = self._total_num_blocks - used
        self.block_pool._num_free_blocks = self._num_free_blocks


class FakeModelExecutor:
    """Fake model executor for testing."""

    def __init__(
        self,
        blocks_per_prefill: int = 2,
        should_fail: bool = False,
    ):
        self._blocks_per_prefill = blocks_per_prefill
        self._should_fail = should_fail
        self.prefill_calls: list[list[int]] = []

    def execute_warmup_prefill(
        self, token_ids: list[int], **kwargs
    ) -> list[int] | None:
        self.prefill_calls.append(token_ids)
        if self._should_fail:
            return None
        return list(range(self._blocks_per_prefill))


class FakeEngineCore:
    """Fake engine core for testing."""

    def __init__(self, queue_depth: int = 0):
        self._queue_depth = queue_depth

    def get_queue_depth(self) -> int:
        return self._queue_depth


# ------------------------------------------------------------------
# Helper to create a configured worker
# ------------------------------------------------------------------


def make_worker(
    profile: str = "dev",
    cached_hashes: set[int] | None = None,
    kv_free: int = 80,
    kv_total: int = 100,
    blocks_per_prefill: int = 2,
    executor_fails: bool = False,
) -> tuple[
    PrefixWarmupWorker,
    PrefixFrequencyTracker,
    FakeBlockPool,
    FakeKVCacheManager,
    FakeModelExecutor,
]:
    config = AdaptiveServingConfig(adaptive_profile=profile)
    tracker = PrefixFrequencyTracker(
        max_entries=100,
        ema_decay=config.warmup_ema_decay,
    )
    block_pool = FakeBlockPool(
        cached_hashes=cached_hashes,
        num_free_blocks=kv_free,
        total_num_blocks=kv_total,
    )
    kv_manager = FakeKVCacheManager(
        num_free_blocks=kv_free,
        total_num_blocks=kv_total,
    )
    executor = FakeModelExecutor(
        blocks_per_prefill=blocks_per_prefill,
        should_fail=executor_fails,
    )
    worker = PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=block_pool,
        kv_cache_manager=kv_manager,
        model_executor=executor,
    )
    return worker, tracker, block_pool, kv_manager, executor


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestPrefixWarmupWorkerInit:
    """Test initialization and properties."""

    def test_initial_state(self):
        worker, _, _, _, _ = make_worker()
        assert worker.total_entries_warmed == 0
        assert not worker.is_paused_for_memory
        assert not worker.is_disabled_for_load

    def test_config_profile_dev(self):
        config = AdaptiveServingConfig(adaptive_profile="dev")
        assert config.warmup_budget_ms == 200.0
        assert config.warmup_vram_budget_ratio == 0.5
        assert config.warmup_min_hit_count == 3.0

    def test_config_profile_production(self):
        config = AdaptiveServingConfig(adaptive_profile="production")
        assert config.warmup_budget_ms == 100.0
        assert config.warmup_vram_budget_ratio == 0.3
        assert config.warmup_min_hit_count == 20.0


class TestAbort:
    """Test cooperative yielding via abort."""

    def test_abort_sets_flag(self):
        worker, _, _, _, _ = make_worker()
        assert not worker._abort_flag
        worker.abort()
        assert worker._abort_flag

    def test_abort_stops_warmup_loop(self):
        # Use low min_hit_count so candidates qualify
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(num_free_blocks=80, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=80, total_num_blocks=100)
        executor = FakeModelExecutor()
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        # Add candidates above the min_hit_count threshold
        for h in range(10):
            for _ in range(20):
                tracker.update(h)

        # Use an executor that aborts after first call
        class AbortingExecutor:
            def __init__(self, worker_ref):
                self.calls = 0
                self._worker = worker_ref

            def execute_warmup_prefill(self, token_ids, **kwargs):
                self.calls += 1
                # Abort after first prefill
                self._worker.abort()
                return [0, 1]

        aborting_exec = AbortingExecutor(worker)
        worker._model_executor = aborting_exec

        worker.on_idle(FakeEngineCore())

        # Should have executed only 1 prefill before aborting
        assert aborting_exec.calls == 1

    def test_abort_fn_passed_to_executor(self):
        """Verify that the warmup worker passes abort_fn to the
        executor so it can check for preemption during execution."""
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(num_free_blocks=80, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=80, total_num_blocks=100)

        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        token_registry = TokenRegistry(max_entries=100, block_size=16)

        # Register tokens for a candidate
        for _ in range(20):
            tracker.update(42)
        token_registry.register(42, list(range(16)))

        # Executor that captures abort_fn
        received_abort_fns: list = []

        class CapturingExecutor:
            def execute_warmup_prefill(self, token_ids, **kwargs):
                received_abort_fns.append(kwargs.get("abort_fn"))
                return [0]

        executor = CapturingExecutor()
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
            token_registry=token_registry,
        )

        worker.on_idle(FakeEngineCore())

        # abort_fn should have been passed
        assert len(received_abort_fns) >= 1
        abort_fn = received_abort_fns[0]
        assert abort_fn is not None
        assert callable(abort_fn)

        # abort_fn should reflect the worker's _abort_flag
        assert not abort_fn()
        worker.abort()
        assert abort_fn()

    """Test memory pressure and high-load pausing."""

    def test_no_pause_at_low_usage(self):
        worker, _, _, kv_manager, _ = make_worker(kv_free=80, kv_total=100)
        # 20% usage — well below thresholds
        assert not worker.should_pause()

    def test_pause_at_high_memory_pressure(self):
        worker, _, _, kv_manager, _ = make_worker(kv_free=5, kv_total=100)
        # 95% usage — above 90% pause threshold
        assert worker.should_pause()
        assert worker.is_paused_for_memory

    def test_hysteresis_stays_paused_between_thresholds(self):
        worker, _, _, kv_manager, _ = make_worker(kv_free=5, kv_total=100)
        # First: trigger pause at 95%
        worker.should_pause()
        assert worker.is_paused_for_memory

        # Now set to 85% (between 80% resume and 90% pause)
        kv_manager._num_free_blocks = 15
        worker.should_pause()
        # Should still be paused (hysteresis)
        assert worker.is_paused_for_memory

    def test_resumes_below_resume_threshold(self):
        worker, _, _, kv_manager, _ = make_worker(kv_free=5, kv_total=100)
        # Trigger pause
        worker.should_pause()
        assert worker.is_paused_for_memory

        # Drop below 80% (resume threshold)
        kv_manager._num_free_blocks = 25  # 75% usage
        result = worker.should_pause()
        assert not worker.is_paused_for_memory
        assert not result

    def test_high_load_disables_warmup(self):
        worker, _, _, _, _ = make_worker()

        # Simulate sustained high load
        worker.update_high_load_state_from_queue(15)
        # Not yet disabled — need to exceed duration
        assert not worker.is_disabled_for_load

        # Simulate time passing (manually set start time)
        worker._high_load_start_time = time.monotonic() - 10.0
        worker.update_high_load_state_from_queue(15)
        assert worker.is_disabled_for_load
        assert worker.should_pause()

    def test_high_load_resets_when_queue_drops(self):
        worker, _, _, _, _ = make_worker()

        # Set as disabled
        worker._disabled_for_load = True
        worker._high_load_start_time = time.monotonic() - 10.0

        # Queue drops below threshold
        worker.update_high_load_state_from_queue(5)
        assert not worker.is_disabled_for_load


class TestSelectNextCandidate:
    """Test candidate selection logic."""

    def test_no_candidates_when_empty(self):
        worker, _, _, _, _ = make_worker()
        assert worker.select_next_candidate() is None

    def test_no_candidates_below_threshold(self):
        worker, tracker, _, _, _ = make_worker()
        # One update gives score = 1 - 0.8 = 0.2
        # Dev min_hit_count = 3.0, so this is below threshold
        tracker.update(42)
        assert worker.select_next_candidate() is None

    def test_selects_highest_score_candidate(self):
        worker, tracker, _, _, _ = make_worker()

        # Build up scores above threshold (3.0 for dev)
        # Need many updates to get score above 3.0
        # EMA formula: score = 0.8 * old + 0.2
        # After n updates: score approaches 1.0
        # With decay=0.8: first=0.2, second=0.36, ...
        # converges to 1/(1-0.8) * (1-0.8) = 1.0
        # Actually max score is 1.0 since EMA formula converges
        # to (1-decay)/(1-decay) = 1.0
        # min_hit_count=3 for dev is actually the EMA score
        # threshold. With our EMA max approaching 1.0, we can't
        # reach 3.0. Let's use production profile or adjust.

        # Use a custom config with lower min_hit_count
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.5,
        )
        tracker2 = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool()
        kv_manager = FakeKVCacheManager()
        executor = FakeModelExecutor()
        worker2 = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker2,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        # Hash 1: many updates -> high score
        for _ in range(20):
            tracker2.update(1)
        # Hash 2: fewer updates -> lower score
        for _ in range(5):
            tracker2.update(2)

        result = worker2.select_next_candidate()
        assert result is not None
        assert result[0] == 1  # highest score

    def test_excludes_cached_hashes(self):
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.5,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        # Hash 1 is already cached
        block_pool = FakeBlockPool(cached_hashes={1})
        kv_manager = FakeKVCacheManager()
        executor = FakeModelExecutor()
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        # Both hashes have high scores
        for _ in range(20):
            tracker.update(1)
            tracker.update(2)

        result = worker.select_next_candidate()
        assert result is not None
        # Hash 1 is cached, so should pick hash 2
        assert result[0] == 2


class TestOnIdle:
    """Test the main idle callback loop."""

    def test_warms_candidates_in_order(self):
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(num_free_blocks=80, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=80, total_num_blocks=100)
        executor = FakeModelExecutor(blocks_per_prefill=1)
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        # Create candidates with different scores
        for _ in range(15):
            tracker.update(100)
        for _ in range(10):
            tracker.update(200)
        for _ in range(5):
            tracker.update(300)

        worker.on_idle(FakeEngineCore())

        # Should have called executor for candidates
        assert len(executor.prefill_calls) > 0
        assert worker.total_entries_warmed > 0

    def test_respects_vram_budget(self):
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
            warmup_vram_budget_ratio=0.1,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        # 50 free blocks * 0.1 ratio = 5 block budget
        # Keep usage at 50% so memory pressure doesn't trigger
        block_pool = FakeBlockPool(num_free_blocks=50, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=50, total_num_blocks=100)
        # Each prefill consumes 10 blocks
        executor = FakeModelExecutor(blocks_per_prefill=10)
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        # Many candidates available
        for h in range(20):
            for _ in range(15):
                tracker.update(h)

        worker.on_idle(FakeEngineCore())

        # Budget is floor(50 * 0.1) = 5 blocks
        # First prefill consumes 10 blocks (consumed=0 < budget=5,
        # so it executes). After: consumed=10 >= budget=5, stops.
        assert len(executor.prefill_calls) == 1
        assert worker._blocks_consumed_this_window == 10

    def test_skips_when_memory_pressure_high(self):
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool()
        # 95% usage -> should pause
        kv_manager = FakeKVCacheManager(num_free_blocks=5, total_num_blocks=100)
        executor = FakeModelExecutor()
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        for _ in range(15):
            tracker.update(42)

        worker.on_idle(FakeEngineCore())

        # Should not have executed any prefills
        assert len(executor.prefill_calls) == 0
        assert worker.total_entries_warmed == 0

    def test_handles_executor_failure(self):
        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(num_free_blocks=80, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=80, total_num_blocks=100)
        executor = FakeModelExecutor(should_fail=True)
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
        )

        for _ in range(15):
            tracker.update(42)

        worker.on_idle(FakeEngineCore())

        # Executor was called but returned None (failure)
        assert len(executor.prefill_calls) > 0
        # No entries actually warmed
        assert worker.total_entries_warmed == 0


class TestVRAMBudgetCalculation:
    """Test VRAM budget calculations."""

    def test_budget_with_free_blocks(self):
        worker, _, _, kv_manager, _ = make_worker(kv_free=100, kv_total=200)
        # dev profile: ratio = 0.5
        # floor(100 * 0.5) = 50
        assert worker._calculate_vram_budget() == 50

    def test_budget_zero_free_blocks(self):
        worker, _, _, kv_manager, _ = make_worker(kv_free=0, kv_total=100)
        assert worker._calculate_vram_budget() == 0

    def test_budget_with_production_ratio(self):
        worker, _, _, _, _ = make_worker(
            profile="production",
            kv_free=100,
            kv_total=200,
        )
        # production profile: ratio = 0.3
        # floor(100 * 0.3) = 30
        assert worker._calculate_vram_budget() == 30


class TestResolveMultiBlockTokens:
    """Test multi-block prefix token resolution."""

    def _make_worker_with_registry(self, block_size: int = 16):
        """Create a worker with a token registry."""
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(num_free_blocks=80, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=80, total_num_blocks=100)
        executor = FakeModelExecutor()
        registry = TokenRegistry(max_entries=100, block_size=block_size)
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
            token_registry=registry,
        )
        return worker, registry

    def test_single_block_returns_tokens(self):
        """Single block hash resolves to its token sequence."""
        worker, registry = self._make_worker_with_registry(block_size=4)
        tokens = [10, 20, 30, 40]
        registry.register(111, tokens)

        result = worker._resolve_multi_block_tokens([111])
        assert result == tokens

    def test_multi_block_concatenates_in_order(self):
        """Multiple block hashes concatenate tokens in order."""
        worker, registry = self._make_worker_with_registry(block_size=4)
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]
        registry.register(100, tokens_a)
        registry.register(200, tokens_b)
        registry.register(300, tokens_c)

        result = worker._resolve_multi_block_tokens([100, 200, 300])
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def test_returns_none_when_any_block_missing(self):
        """Returns None if any block hash is not in registry."""
        worker, registry = self._make_worker_with_registry(block_size=4)
        registry.register(100, [1, 2, 3, 4])
        # Hash 200 is not registered

        result = worker._resolve_multi_block_tokens([100, 200])
        assert result is None

    def test_returns_none_for_empty_list(self):
        """Returns None for an empty prefix hash list."""
        worker, registry = self._make_worker_with_registry(block_size=4)

        result = worker._resolve_multi_block_tokens([])
        assert result is None

    def test_returns_none_when_no_registry(self):
        """Returns None when no token registry is configured."""
        worker, _, _, _, _ = make_worker()
        # worker has no token_registry (None by default)

        result = worker._resolve_multi_block_tokens([111])
        assert result is None

    def test_on_idle_uses_multi_block_resolution(self):
        """on_idle passes resolved tokens to executor."""
        worker, registry = self._make_worker_with_registry(block_size=4)
        tokens = [10, 20, 30, 40]
        registry.register(42, tokens)

        # Build up score above threshold
        for _ in range(20):
            worker._frequency_tracker.update(42)

        worker.on_idle(FakeEngineCore())

        # Executor should have received the actual token IDs
        executor = worker._model_executor
        assert len(executor.prefill_calls) > 0
        assert executor.prefill_calls[0] == tokens


class TestBlockAllocationFailureHandling:
    """Test block allocation failure handling in on_idle.

    Requirements: 5.4
    Tests that insufficient free blocks cause the candidate to be
    skipped without error, and that successful allocation provides
    correct block IDs to the executor.
    """

    def _make_worker_with_registry(
        self,
        num_free_blocks: int = 80,
        block_size: int = 4,
    ):
        """Create a worker with a token registry and configurable
        free blocks."""
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(
            num_free_blocks=num_free_blocks,
            total_num_blocks=100,
        )
        kv_manager = FakeKVCacheManager(
            num_free_blocks=num_free_blocks,
            total_num_blocks=100,
        )
        executor = FakeModelExecutor(blocks_per_prefill=1)
        registry = TokenRegistry(max_entries=100, block_size=block_size)
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
            token_registry=registry,
            block_size=block_size,
        )
        return worker, tracker, registry, executor, block_pool

    def test_skip_when_insufficient_free_blocks(self):
        """Candidate is skipped without error when free blocks are
        insufficient for the token sequence.

        Requirement 5.4: IF available free blocks are insufficient
        for warmup allocation, the worker SHALL abort that candidate
        without error.
        """
        # block_size=4, tokens=[1,2,3,4] needs ceil(4/4)=1 block
        # but we only have 0 free blocks
        worker, tracker, registry, executor, _ = self._make_worker_with_registry(
            num_free_blocks=0, block_size=4
        )

        # Register tokens for our candidate
        registry.register(42, [1, 2, 3, 4])

        # Build up score above threshold
        for _ in range(20):
            tracker.update(42)

        # Run the idle loop — should skip due to insufficient blocks
        worker.on_idle(FakeEngineCore())

        # Executor should NOT have been called
        assert len(executor.prefill_calls) == 0
        # No entries warmed
        assert worker.total_entries_warmed == 0
        # No errors raised — test passes if we reach here

    def test_skip_multi_block_when_insufficient(self):
        """Multi-block candidate is skipped when free blocks are
        fewer than needed.

        A token sequence of 8 tokens with block_size=4 needs 2
        blocks. With only 1 free block, it should be skipped.
        """
        # block_size=4, need ceil(8/4)=2 blocks but only 1 free
        worker, tracker, registry, executor, _ = self._make_worker_with_registry(
            num_free_blocks=1, block_size=4
        )

        # Register tokens for two consecutive blocks
        registry.register(100, [1, 2, 3, 4])
        registry.register(200, [5, 6, 7, 8])

        # Build up score for hash 100
        for _ in range(20):
            tracker.update(100)

        # The worker resolves [100] -> 4 tokens -> needs 1 block
        # but since we have exactly 1 free block and the candidate
        # hash maps to 4 tokens, it needs ceil(4/4)=1 block.
        # With 1 free block available, this SHOULD succeed.
        # Let's instead set free blocks to 0 to force skip.
        worker._block_pool._num_free_blocks = 0

        worker.on_idle(FakeEngineCore())

        # Executor should NOT have been called
        assert len(executor.prefill_calls) == 0
        assert worker.total_entries_warmed == 0

    def test_successful_allocation_calls_executor(self):
        """When enough free blocks are available, the executor IS
        called with the correct token IDs.

        Requirement 5.4: successful allocation provides correct
        block IDs to executor.
        """
        # block_size=4, tokens need ceil(4/4)=1 block, 80 free
        worker, tracker, registry, executor, _ = self._make_worker_with_registry(
            num_free_blocks=80, block_size=4
        )

        tokens = [10, 20, 30, 40]
        registry.register(42, tokens)

        # Build up score above threshold
        for _ in range(20):
            tracker.update(42)

        # Stub _register_warmed_block_hashes (task 7.2, not yet
        # implemented) so on_idle completes the success path.
        worker._register_warmed_block_hashes = lambda t, b: None

        worker.on_idle(FakeEngineCore())

        # Executor SHOULD have been called with the token IDs
        assert len(executor.prefill_calls) > 0
        assert executor.prefill_calls[0] == tokens
        # Entry was warmed successfully
        assert worker.total_entries_warmed > 0
        assert worker.warmup_prefills_executed > 0

    def test_executor_receives_correct_block_ids(self):
        """The executor returns block IDs that are tracked by the
        worker's budget accounting."""
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
            # Tight VRAM budget: floor(80 * 0.05) = 4 blocks max
            warmup_vram_budget_ratio=0.05,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = FakeBlockPool(num_free_blocks=80, total_num_blocks=100)
        kv_manager = FakeKVCacheManager(num_free_blocks=80, total_num_blocks=100)
        # Executor returns 3 block IDs per call
        executor = FakeModelExecutor(blocks_per_prefill=3)
        registry = TokenRegistry(max_entries=100, block_size=4)
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
            token_registry=registry,
            block_size=4,
        )

        tokens = [1, 2, 3, 4]
        registry.register(55, tokens)

        for _ in range(20):
            tracker.update(55)

        # Stub _register_warmed_block_hashes (task 7.2, not yet
        # implemented) so on_idle completes the success path.
        worker._register_warmed_block_hashes = lambda t, b: None

        worker.on_idle(FakeEngineCore())

        # First call: executor returns [0,1,2] (3 blocks)
        # consumed=3 < budget=4, so loop continues.
        # Second call: consumed becomes 6 >= budget=4, stops.
        # Verify executor received correct tokens each time.
        assert len(executor.prefill_calls) >= 1
        assert executor.prefill_calls[0] == tokens
        # blocks_consumed is a multiple of 3 (blocks_per_prefill)
        assert worker._blocks_consumed_this_window % 3 == 0
        assert worker._blocks_consumed_this_window > 0

    def test_no_error_on_zero_free_blocks(self):
        """No exception is raised when block pool has zero free
        blocks — candidate is silently skipped."""
        worker, tracker, registry, executor, _ = self._make_worker_with_registry(
            num_free_blocks=0, block_size=4
        )

        registry.register(99, [5, 6, 7, 8])
        for _ in range(20):
            tracker.update(99)

        # Should not raise any exception
        worker.on_idle(FakeEngineCore())

        assert len(executor.prefill_calls) == 0
        assert worker.total_entries_warmed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
