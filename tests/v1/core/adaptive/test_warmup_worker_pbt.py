# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for PrefixWarmupWorker -- data plane properties.

# Feature: adaptive-warmup-data-plane,
#   Property 5: Warmup Worker skip missing token candidates
# Feature: adaptive-warmup-data-plane,
#   Property 6: Warmup VRAM budget tracking
# Feature: adaptive-warmup-data-plane,
#   Property 7: Multi-block prefix token concatenation order
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.adaptive.prefix_warmup_worker import PrefixWarmupWorker
from vllm.v1.core.adaptive.token_registry import TokenRegistry

# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

BLOCK_SIZE = 16


class FakeBlockPool:
    """Minimal fake for BlockPoolProtocol.

    Tracks warmed hashes so candidates are not re-selected.
    """

    def __init__(self, num_free_blocks: int = 1000):
        self._num_free_blocks = num_free_blocks
        self._cached_hashes: set[int] = set()

    def get_cached_block_hashes(self) -> set[int]:
        return self._cached_hashes

    def get_num_free_blocks(self) -> int:
        return self._num_free_blocks

    def add_cached_hash(self, h: int) -> None:
        self._cached_hashes.add(h)


class FakeKVCacheManager:
    """Minimal fake for KVCacheManagerProtocol."""

    def __init__(self, total_num_blocks: int = 10000):
        self._total_num_blocks = total_num_blocks
        self._num_free_blocks = total_num_blocks
        self.block_pool = FakeBlockPool(self._num_free_blocks)

    @property
    def usage(self) -> float:
        if self._total_num_blocks == 0:
            return 0.0
        return 1.0 - (self._num_free_blocks / self._total_num_blocks)


class VaryingBlockExecutor:
    """Fake executor returning varying block list sizes."""

    def __init__(self, block_sizes: list[int]):
        self._block_sizes = list(block_sizes)
        self._call_index = 0
        self.calls: list[list[int]] = []

    def execute_warmup_prefill(
        self, token_ids: list[int], **kwargs
    ) -> list[int] | None:
        if self._call_index >= len(self._block_sizes):
            return None
        n = self._block_sizes[self._call_index]
        self._call_index += 1
        result = list(range(n))
        self.calls.append(result)
        return result


class FakeModelExecutorTracking:
    """Fake executor that tracks calls and always succeeds.

    Also registers warmed hashes in block_pool to prevent the
    warmup loop from re-selecting the same candidate indefinitely.
    """

    def __init__(
        self,
        blocks_per_prefill: int = 1,
        block_pool: FakeBlockPool | None = None,
        block_size: int = BLOCK_SIZE,
    ):
        self._blocks_per_prefill = blocks_per_prefill
        self._block_pool = block_pool
        self._block_size = block_size
        self.prefill_calls: list[list[int]] = []

    def execute_warmup_prefill(
        self, token_ids: list[int], **kwargs
    ) -> list[int] | None:
        self.prefill_calls.append(token_ids)
        # Derive hash from first token (tokens = range(h*bs, ...))
        if self._block_pool is not None and token_ids:
            h = token_ids[0] // self._block_size
            self._block_pool.add_cached_hash(h)
        return list(range(self._blocks_per_prefill))


class FakeEngineCore:
    """Fake engine core for property testing."""

    def get_queue_depth(self) -> int:
        return 0


# ------------------------------------------------------------------
# Property 5: Warmup Worker skip missing token candidates
# ------------------------------------------------------------------


@st.composite
def candidate_sequence_strategy(draw: st.DrawFn):
    """Generate a warmup candidate scenario.

    Returns:
        Tuple of (num_candidates, has_tokens_mask) where
        has_tokens_mask[i] indicates whether candidate i has
        tokens registered in the TokenRegistry.
    """
    num_candidates = draw(st.integers(min_value=1, max_value=10))
    has_tokens_mask = draw(
        st.lists(
            st.booleans(),
            min_size=num_candidates,
            max_size=num_candidates,
        )
    )
    return num_candidates, has_tokens_mask


@settings(max_examples=50, deadline=None)
@given(scenario=candidate_sequence_strategy())
def test_warmup_worker_skips_missing_tokens(
    scenario: tuple[int, list[bool]],
) -> None:
    """Property 5: Warmup Worker skip missing token candidates.

    **Validates: Requirements 4.2**

    For any warmup candidate sequence, where some block hashes have
    corresponding tokens in the TokenRegistry and some don't,
    PrefixWarmupWorker should skip all None-returning candidates
    and continue processing the next valid candidate without error.
    """
    num_candidates, has_tokens_mask = scenario

    all_hashes = list(range(num_candidates))

    registry = TokenRegistry(max_entries=num_candidates + 10, block_size=BLOCK_SIZE)

    num_valid = 0
    num_missing = 0
    for h, has_tokens in zip(all_hashes, has_tokens_mask):
        if has_tokens:
            tokens = list(range(h * BLOCK_SIZE, (h + 1) * BLOCK_SIZE))
            registry.register(h, tokens)
            num_valid += 1
        else:
            num_missing += 1

    config = AdaptiveServingConfig(
        adaptive_profile="dev",
        warmup_min_hit_count=0.01,
        warmup_budget_ms=60000.0,
        warmup_vram_budget_ratio=0.9,
    )

    tracker = PrefixFrequencyTracker(
        max_entries=1000, ema_decay=config.warmup_ema_decay
    )

    for h in all_hashes:
        for _ in range(20):
            tracker.update(h)

    kv_manager = FakeKVCacheManager(total_num_blocks=10000)
    kv_manager._num_free_blocks = 9000
    kv_manager.block_pool._num_free_blocks = 9000

    block_pool = FakeBlockPool(num_free_blocks=9000)
    executor = FakeModelExecutorTracking(blocks_per_prefill=1, block_pool=block_pool)

    worker = PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=block_pool,
        kv_cache_manager=kv_manager,
        model_executor=executor,
        token_registry=registry,
    )

    # Override select_next_candidate to return each candidate
    # exactly once in order, avoiding infinite re-selection of
    # skipped candidates (which is a known limitation before
    # block hash registration is implemented in Task 7.2).
    candidate_iter = iter([(h, 1.0) for h in all_hashes])

    def mock_select_next_candidate():
        return next(candidate_iter, None)

    worker.select_next_candidate = mock_select_next_candidate

    # Run on_idle -- should NOT raise any exception
    worker.on_idle(FakeEngineCore())

    total_processed = (
        worker.warmup_prefills_executed + worker.warmup_prefills_skipped_no_tokens
    )

    # All candidates should have been processed
    assert total_processed == len(all_hashes), (
        f"Expected all {len(all_hashes)} candidates processed, got {total_processed}"
    )

    # Skipped count equals missing entries
    assert worker.warmup_prefills_skipped_no_tokens == num_missing, (
        f"Skipped {worker.warmup_prefills_skipped_no_tokens} "
        f"but {num_missing} hashes lack tokens"
    )

    # Executed count equals valid entries
    assert worker.warmup_prefills_executed == num_valid, (
        f"Executed {worker.warmup_prefills_executed} prefills "
        f"but {num_valid} hashes have tokens"
    )

    # Each executor call received exactly BLOCK_SIZE tokens
    for call_tokens in executor.prefill_calls:
        assert len(call_tokens) == BLOCK_SIZE, (
            f"Executor received {len(call_tokens)} tokens, expected {BLOCK_SIZE}"
        )


# ------------------------------------------------------------------
# Property 6: Warmup VRAM budget tracking
# ------------------------------------------------------------------


@st.composite
def vram_budget_scenario(draw: st.DrawFn):
    """Generate a scenario for VRAM budget tracking."""
    num_candidates = draw(st.integers(min_value=1, max_value=30))
    block_return_sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=num_candidates,
            max_size=num_candidates,
        )
    )
    vram_budget_ratio = draw(st.floats(min_value=0.01, max_value=1.0))
    free_blocks = draw(st.integers(min_value=1, max_value=500))
    return (
        num_candidates,
        block_return_sizes,
        vram_budget_ratio,
        free_blocks,
    )


@settings(max_examples=200, deadline=None)
@given(scenario=vram_budget_scenario())
def test_vram_budget_tracking_equals_sum_of_returned_blocks(
    scenario: tuple[int, list[int], float, int],
) -> None:
    """Property 6: Warmup VRAM budget tracking.

    **Validates: Requirements 4.4**

    For any series of successful warmup prefill operations,
    _blocks_consumed_this_window should equal the sum of all
    returned block list lengths, and warmup stops when the budget
    is reached.
    """
    (
        num_candidates,
        block_return_sizes,
        vram_budget_ratio,
        free_blocks,
    ) = scenario

    total_num_blocks = free_blocks * 10

    config = AdaptiveServingConfig(
        adaptive_profile="dev",
        warmup_vram_budget_ratio=vram_budget_ratio,
        warmup_min_hit_count=0.01,
        warmup_budget_ms=60000.0,
    )

    tracker = PrefixFrequencyTracker(
        max_entries=1000, ema_decay=config.warmup_ema_decay
    )

    token_registry = TokenRegistry(max_entries=1000, block_size=BLOCK_SIZE)

    for h in range(num_candidates):
        for _ in range(20):
            tracker.update(h)
        token_registry.register(h, list(range(BLOCK_SIZE)))

    kv_manager = FakeKVCacheManager(total_num_blocks=total_num_blocks)
    kv_manager._num_free_blocks = free_blocks
    kv_manager.block_pool._num_free_blocks = free_blocks

    executor = VaryingBlockExecutor(block_return_sizes)

    worker = PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=FakeBlockPool(num_free_blocks=free_blocks),
        kv_cache_manager=kv_manager,
        model_executor=executor,
        token_registry=token_registry,
    )

    worker.on_idle(FakeEngineCore())

    # Invariant 1: consumed equals sum of returned blocks
    num_prefills_executed = len(executor.calls)
    expected_consumed = sum(block_return_sizes[i] for i in range(num_prefills_executed))
    assert worker._blocks_consumed_this_window == expected_consumed, (
        f"blocks_consumed mismatch: "
        f"expected={expected_consumed}, "
        f"actual={worker._blocks_consumed_this_window}"
    )

    # Invariant 2: warmup stops when budget reached OR memory pressure
    vram_block_budget = math.floor(free_blocks * vram_budget_ratio)

    if vram_block_budget <= 0:
        assert num_prefills_executed == 0
    else:
        if num_prefills_executed < num_candidates:
            # Warmup may also stop due to memory pressure (usage >=
            # pause_threshold). When free_blocks is small relative to
            # total_num_blocks, KV cache usage can be >= 0.9 causing
            # should_pause() to return True before budget is reached.
            budget_reached = worker._blocks_consumed_this_window >= vram_block_budget
            paused_for_memory = worker.is_paused_for_memory
            # Also check if free blocks were insufficient for the
            # next candidate (block allocation pre-check)
            insufficient_free = worker._block_pool.get_num_free_blocks() < 1
            assert budget_reached or paused_for_memory or insufficient_free, (
                f"Stopped early but no valid reason: "
                f"consumed={worker._blocks_consumed_this_window}, "
                f"budget={vram_block_budget}, "
                f"paused_for_memory={paused_for_memory}, "
                f"insufficient_free={insufficient_free}"
            )

        if num_prefills_executed > 0:
            max_single = max(block_return_sizes[:num_prefills_executed])
            max_overshoot = vram_block_budget - 1 + max_single
            assert worker._blocks_consumed_this_window <= max_overshoot, (
                f"Consumed too many: "
                f"{worker._blocks_consumed_this_window} > "
                f"{max_overshoot}"
            )


# ------------------------------------------------------------------
# Property 7: Multi-block prefix token concatenation order
# ------------------------------------------------------------------


@st.composite
def multi_block_prefix(
    draw: st.DrawFn,
    block_size: int = BLOCK_SIZE,
) -> tuple[list[int], list[list[int]]]:
    """Generate N unique block hashes and their token sequences."""
    n = draw(st.integers(min_value=1, max_value=10))
    hashes: list[int] = []
    tokens: list[list[int]] = []
    seen: set[int] = set()
    for _ in range(n):
        h = draw(
            st.integers(min_value=0, max_value=2**63 - 1).filter(
                lambda x, s=seen: x not in s
            )
        )
        seen.add(h)
        hashes.append(h)
        t = draw(
            st.lists(
                st.integers(min_value=0, max_value=32000),
                min_size=block_size,
                max_size=block_size,
            )
        )
        tokens.append(t)
    return hashes, tokens


def _make_worker_with_registry(
    token_registry: TokenRegistry | None = None,
) -> PrefixWarmupWorker:
    """Create a PrefixWarmupWorker with minimal fakes."""
    config = AdaptiveServingConfig(adaptive_profile="dev")
    tracker = PrefixFrequencyTracker(
        max_entries=1000, ema_decay=config.warmup_ema_decay
    )
    return PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=FakeBlockPool(),
        kv_cache_manager=FakeKVCacheManager(),
        model_executor=VaryingBlockExecutor(block_sizes=[]),
        token_registry=token_registry,
    )


class TestMultiBlockTokenConcatenationOrder:
    """Property 7: Multi-block prefix token concatenation order.

    **Validates: Requirements 4.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(data=multi_block_prefix())
    def test_concatenation_matches_prefix_order(
        self,
        data: tuple[list[int], list[list[int]]],
    ) -> None:
        """Resolved tokens equal concatenation in prefix order.

        **Validates: Requirements 4.5**
        """
        prefix_hashes, token_sequences = data

        registry = TokenRegistry(max_entries=1024, block_size=BLOCK_SIZE)
        for h, tokens in zip(prefix_hashes, token_sequences):
            registry.register(h, tokens)

        worker = _make_worker_with_registry(token_registry=registry)
        result = worker._resolve_multi_block_tokens(prefix_hashes)

        expected: list[int] = []
        for tokens in token_sequences:
            expected.extend(tokens)

        assert result == expected

    @settings(max_examples=100, deadline=None)
    @given(data=multi_block_prefix())
    def test_returns_none_when_any_hash_missing(
        self,
        data: tuple[list[int], list[list[int]]],
    ) -> None:
        """Returns None if any block hash is missing.

        **Validates: Requirements 4.5**
        """
        prefix_hashes, token_sequences = data

        registry = TokenRegistry(max_entries=1024, block_size=BLOCK_SIZE)
        for h, tokens in zip(prefix_hashes[:-1], token_sequences[:-1]):
            registry.register(h, tokens)

        worker = _make_worker_with_registry(token_registry=registry)
        result = worker._resolve_multi_block_tokens(prefix_hashes)
        assert result is None

    @settings(max_examples=100, deadline=None)
    @given(data=multi_block_prefix())
    def test_returns_none_when_no_registry(
        self,
        data: tuple[list[int], list[list[int]]],
    ) -> None:
        """Returns None when token_registry is None.

        **Validates: Requirements 4.5**
        """
        prefix_hashes, _ = data
        worker = _make_worker_with_registry(token_registry=None)
        result = worker._resolve_multi_block_tokens(prefix_hashes)
        assert result is None

    def test_returns_none_for_empty_prefix_hashes(self) -> None:
        """Returns None for an empty prefix hash list.

        **Validates: Requirements 4.5**
        """
        registry = TokenRegistry(max_entries=1024, block_size=BLOCK_SIZE)
        worker = _make_worker_with_registry(token_registry=registry)
        result = worker._resolve_multi_block_tokens([])
        assert result is None
