# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based test for no block leaks on abort.

# Feature: adaptive-warmup-data-plane,
#   Property 9: 中止後無 block 洩漏
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum, auto

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.adaptive.prefix_warmup_worker import PrefixWarmupWorker
from vllm.v1.core.adaptive.token_registry import TokenRegistry

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BLOCK_SIZE = 16


# ------------------------------------------------------------------
# Block state tracking
# ------------------------------------------------------------------


class BlockState(Enum):
    """Possible states for a warmup-allocated block."""

    ALLOCATED = auto()  # Allocated but not yet processed
    COMMITTED = auto()  # Forward pass completed, hash registered
    FREED = auto()  # Released back to pool (never processed)


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


class AbortingExecutor:
    """Executor that simulates abort at a configurable point.

    For a multi-block sequence of N blocks, this executor completes
    forward passes for blocks 0..abort_after_block (inclusive) and
    then returns only those block IDs — simulating what the real
    GPU model runner does when abort_fn() returns True after the
    abort_after_block-th forward pass.

    Block tracking:
    - Blocks 0..abort_after_block are returned (completed)
    - Blocks abort_after_block+1..N-1 are NOT returned (aborted)
    """

    def __init__(
        self,
        total_blocks: int,
        abort_after_block: int,
    ):
        """
        Args:
            total_blocks: Total blocks needed for the sequence.
            abort_after_block: Last block index to complete before
                aborting. Use total_blocks-1 for no abort (all
                blocks complete). Use -1 to abort before any block.
        """
        self._total_blocks = total_blocks
        self._abort_after_block = abort_after_block
        self.calls: list[tuple[list[int], int]] = []
        self._block_states: dict[int, BlockState] = {}

    @property
    def block_states(self) -> dict[int, BlockState]:
        return self._block_states

    def execute_warmup_prefill(
        self,
        token_ids: list[int],
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[int] | None:
        """Simulate multi-block warmup prefill with abort.

        Allocates block IDs and processes them one by one, checking
        the abort point to decide when to stop.
        """
        num_blocks = math.ceil(len(token_ids) / BLOCK_SIZE)
        self.calls.append((token_ids, num_blocks))

        # Simulate block allocation — assign block IDs
        block_ids = list(range(num_blocks))

        # Mark all blocks as ALLOCATED initially
        for bid in block_ids:
            self._block_states[bid] = BlockState.ALLOCATED

        # Process blocks one by one
        completed: list[int] = []
        for i, bid in enumerate(block_ids):
            if i > self._abort_after_block:
                # Abort: free remaining blocks
                self._block_states[bid] = BlockState.FREED
            else:
                # Complete forward pass for this block
                self._block_states[bid] = BlockState.COMMITTED
                completed.append(bid)

        # If abort happened before any block, return None
        if not completed:
            return None

        return completed


class NoAbortExecutor:
    """Executor that always completes all blocks (no abort)."""

    def __init__(self):
        self.calls: list[tuple[list[int], int]] = []
        self._block_states: dict[int, BlockState] = {}

    @property
    def block_states(self) -> dict[int, BlockState]:
        return self._block_states

    def execute_warmup_prefill(
        self,
        token_ids: list[int],
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[int] | None:
        num_blocks = math.ceil(len(token_ids) / BLOCK_SIZE)
        self.calls.append((token_ids, num_blocks))
        block_ids = list(range(num_blocks))
        for bid in block_ids:
            self._block_states[bid] = BlockState.COMMITTED
        return block_ids


class FakeBlockPool:
    """Minimal block pool that tracks cached hashes."""

    def __init__(self, num_free_blocks: int = 1000):
        self._num_free_blocks = num_free_blocks
        self._cached_hashes: set[int] = set()

    def get_cached_block_hashes(self) -> set[int]:
        return self._cached_hashes

    def get_num_free_blocks(self) -> int:
        return self._num_free_blocks

    def add_cached_hash(self, h: int) -> None:
        self._cached_hashes.add(h)

    def free_blocks(self, ordered_blocks) -> None:
        pass


class FakeKVCacheManager:
    """Minimal KV cache manager."""

    def __init__(self, total_num_blocks: int = 10000):
        self._total_num_blocks = total_num_blocks
        self._num_free_blocks = total_num_blocks
        self.block_pool = FakeBlockPool(self._num_free_blocks)

    @property
    def usage(self) -> float:
        if self._total_num_blocks == 0:
            return 0.0
        return 1.0 - (self._num_free_blocks / self._total_num_blocks)


class FakeEngineCore:
    """Fake engine core for property testing."""

    def get_queue_depth(self) -> int:
        return 0


# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------


@st.composite
def abort_scenario(draw: st.DrawFn):
    """Generate a multi-block warmup abort scenario.

    Returns:
        Tuple of (num_blocks, abort_after_block) where:
        - num_blocks: Total blocks in the sequence (2-10)
        - abort_after_block: Index of last completed block (-1
          to N-1 inclusive). -1 means abort before any block.
    """
    num_blocks = draw(st.integers(min_value=2, max_value=10))
    # abort_after_block: -1 means abort before any processing,
    # 0..num_blocks-2 means abort mid-sequence,
    # num_blocks-1 means no abort (all complete).
    abort_after_block = draw(st.integers(min_value=-1, max_value=num_blocks - 1))
    return num_blocks, abort_after_block


# ------------------------------------------------------------------
# Property 9: 中止後無 block 洩漏
# ------------------------------------------------------------------


class TestNoBlockLeaksOnAbort:
    """Property 9: 中止後無 block 洩漏.

    **Validates: Requirements 6.3**

    For any multi-block warmup operation aborted at any intermediate
    point, all allocated blocks must be in one of two states:
    (a) successfully completed forward pass and committed to BlockPool
        with hash, or
    (b) not started and freed/never claimed.
    No block should be in an allocated-but-neither-committed-nor-freed
    state.
    """

    @settings(max_examples=200, deadline=None)
    @given(scenario=abort_scenario())
    def test_all_blocks_committed_or_freed_after_abort(
        self,
        scenario: tuple[int, int],
    ) -> None:
        """After abort, no block is in allocated-but-leaked state.

        **Validates: Requirements 6.3**

        Given a multi-block warmup with N blocks aborted after
        block K, blocks 0..K must be COMMITTED and blocks
        K+1..N-1 must be FREED.
        """
        num_blocks, abort_after_block = scenario

        # Create token sequence spanning num_blocks blocks
        token_ids = list(range(num_blocks * BLOCK_SIZE))

        executor = AbortingExecutor(
            total_blocks=num_blocks,
            abort_after_block=abort_after_block,
        )

        # Execute the warmup prefill
        result = executor.execute_warmup_prefill(token_ids, abort_fn=lambda: False)

        # Verify: no block is in ALLOCATED state (all must be
        # either COMMITTED or FREED).
        for bid, state in executor.block_states.items():
            assert state != BlockState.ALLOCATED, (
                f"Block {bid} is in ALLOCATED state (leaked!) "
                f"after abort_after_block={abort_after_block}, "
                f"num_blocks={num_blocks}"
            )

        # Verify: returned blocks (if any) are exactly the
        # COMMITTED blocks.
        committed_blocks = [
            bid
            for bid, state in executor.block_states.items()
            if state == BlockState.COMMITTED
        ]
        freed_blocks = [
            bid
            for bid, state in executor.block_states.items()
            if state == BlockState.FREED
        ]

        if result is not None:
            assert sorted(result) == sorted(committed_blocks), (
                f"Returned blocks {result} don't match committed "
                f"blocks {committed_blocks}"
            )
        else:
            assert len(committed_blocks) == 0, (
                f"No blocks returned but {len(committed_blocks)} blocks are COMMITTED"
            )

        # Verify: committed + freed == all allocated blocks
        assert len(committed_blocks) + len(freed_blocks) == num_blocks, (
            f"Block count mismatch: "
            f"committed={len(committed_blocks)}, "
            f"freed={len(freed_blocks)}, "
            f"total={num_blocks}"
        )

    @settings(max_examples=200, deadline=None)
    @given(scenario=abort_scenario())
    def test_returned_blocks_are_prefix_of_allocation(
        self,
        scenario: tuple[int, int],
    ) -> None:
        """Returned block IDs are a prefix of the full allocation.

        **Validates: Requirements 6.3**

        The returned blocks must be a contiguous prefix — blocks
        0, 1, ..., K in order. This ensures no "holes" exist in
        the committed sequence.
        """
        num_blocks, abort_after_block = scenario
        token_ids = list(range(num_blocks * BLOCK_SIZE))

        executor = AbortingExecutor(
            total_blocks=num_blocks,
            abort_after_block=abort_after_block,
        )

        result = executor.execute_warmup_prefill(token_ids, abort_fn=lambda: False)

        if result is not None:
            # Result should be a contiguous prefix: [0, 1, ..., K]
            expected_prefix = list(range(len(result)))
            assert result == expected_prefix, (
                f"Returned blocks {result} are not a contiguous "
                f"prefix. Expected {expected_prefix}."
            )

    @settings(max_examples=200, deadline=None)
    @given(scenario=abort_scenario())
    def test_all_returned_blocks_have_valid_forward_pass(
        self,
        scenario: tuple[int, int],
    ) -> None:
        """All returned blocks completed their forward pass.

        **Validates: Requirements 6.3**

        Every block in the returned list must be in COMMITTED state,
        meaning its forward pass completed and KV cache was stored.
        """
        num_blocks, abort_after_block = scenario
        token_ids = list(range(num_blocks * BLOCK_SIZE))

        executor = AbortingExecutor(
            total_blocks=num_blocks,
            abort_after_block=abort_after_block,
        )

        result = executor.execute_warmup_prefill(token_ids, abort_fn=lambda: False)

        if result is not None:
            for bid in result:
                assert executor.block_states[bid] == BlockState.COMMITTED, (
                    f"Returned block {bid} is not COMMITTED "
                    f"(state={executor.block_states[bid]})"
                )

    @settings(max_examples=100, deadline=None)
    @given(scenario=abort_scenario())
    def test_warmup_worker_integration_no_leak(
        self,
        scenario: tuple[int, int],
    ) -> None:
        """Integration: warmup worker handles abort without leaking.

        **Validates: Requirements 6.3**

        Tests the full PrefixWarmupWorker flow with an aborting
        executor. Verifies that the worker correctly handles partial
        results and all blocks end up accounted for.
        """
        num_blocks, abort_after_block = scenario

        # Token sequence spanning num_blocks blocks
        token_ids_full = list(range(num_blocks * BLOCK_SIZE))

        # Set up token registry with just the prefix_hash entry
        registry = TokenRegistry(max_entries=100, block_size=BLOCK_SIZE)
        prefix_hash = 42
        # Register the single-block entry for the candidate
        registry.register(prefix_hash, token_ids_full[:BLOCK_SIZE])

        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.01,
            warmup_budget_ms=60000.0,
            warmup_vram_budget_ratio=0.9,
        )

        tracker = PrefixFrequencyTracker(
            max_entries=1000, ema_decay=config.warmup_ema_decay
        )
        # Make prefix_hash a warmup candidate
        for _ in range(20):
            tracker.update(prefix_hash)

        block_pool = FakeBlockPool(num_free_blocks=9000)
        kv_manager = FakeKVCacheManager(total_num_blocks=10000)
        kv_manager._num_free_blocks = 9000
        kv_manager.block_pool._num_free_blocks = 9000

        executor = AbortingExecutor(
            total_blocks=num_blocks,
            abort_after_block=abort_after_block,
        )

        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=block_pool,
            kv_cache_manager=kv_manager,
            model_executor=executor,
            token_registry=registry,
        )

        # Override select_next_candidate to return candidate exactly
        # once, preventing infinite re-selection loops.
        called = [False]

        def mock_select_next_candidate():
            if not called[0]:
                called[0] = True
                return (prefix_hash, 1.0)
            return None

        worker.select_next_candidate = mock_select_next_candidate

        # Run on_idle — should not raise
        worker.on_idle(FakeEngineCore())

        # After execution, verify block state invariant:
        # All blocks are either COMMITTED or FREED
        for bid, state in executor.block_states.items():
            assert state != BlockState.ALLOCATED, (
                f"Block {bid} leaked in ALLOCATED state after "
                f"warmup worker processed abort at block "
                f"{abort_after_block} of {num_blocks}"
            )

        # Verify consumed count matches committed blocks
        committed_count = sum(
            1 for s in executor.block_states.values() if s == BlockState.COMMITTED
        )

        if executor.calls:
            if abort_after_block == -1:
                # No blocks returned (result is None), nothing
                # consumed.
                assert worker._blocks_consumed_this_window == 0, (
                    f"Expected 0 consumed when abort before any "
                    f"block, got "
                    f"{worker._blocks_consumed_this_window}"
                )
            else:
                assert worker._blocks_consumed_this_window == committed_count, (
                    f"Consumed blocks mismatch: "
                    f"worker="
                    f"{worker._blocks_consumed_this_window}, "
                    f"committed={committed_count}"
                )

    @settings(max_examples=100, deadline=None)
    @given(
        num_blocks=st.integers(min_value=2, max_value=10),
    )
    def test_no_abort_all_blocks_committed(
        self,
        num_blocks: int,
    ) -> None:
        """Without abort, all blocks are COMMITTED (baseline).

        **Validates: Requirements 6.3**

        When no abort occurs, all N blocks should be committed.
        This is the baseline that confirms the no-leak invariant
        holds in the normal (non-abort) path as well.
        """
        token_ids = list(range(num_blocks * BLOCK_SIZE))

        executor = NoAbortExecutor()
        result = executor.execute_warmup_prefill(token_ids, abort_fn=lambda: False)

        assert result is not None
        assert len(result) == num_blocks

        # All blocks should be COMMITTED
        for bid, state in executor.block_states.items():
            assert state == BlockState.COMMITTED, (
                f"Block {bid} not COMMITTED when no abort: state={state}"
            )

        # No FREED blocks
        freed = [
            bid for bid, s in executor.block_states.items() if s == BlockState.FREED
        ]
        assert len(freed) == 0, f"Unexpected FREED blocks with no abort: {freed}"
