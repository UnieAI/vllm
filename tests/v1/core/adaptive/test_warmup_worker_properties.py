# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for PrefixWarmupWorker.

# Feature: adaptive-speculative-serving, Property 5: VRAM Budget Invariant
# Feature: adaptive-speculative-serving,
#   Property 8: Compressed Block Size Budget Calculation
# Feature: adaptive-speculative-serving, Property 9: Memory Pressure Hysteresis
# Feature: adaptive-speculative-serving, Property 10: High-Load Warmup Disable
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.adaptive.prefix_warmup_worker import (
    PrefixWarmupWorker,
    _get_compression_ratio,
)


class FakeBlockPool:
    """Minimal fake for BlockPoolProtocol."""

    def __init__(self, num_free_blocks: int = 100):
        self._num_free_blocks = num_free_blocks

    def get_cached_block_hashes(self) -> set[int]:
        return set()

    def get_num_free_blocks(self) -> int:
        return self._num_free_blocks


class FakeKVCacheManager:
    """Minimal fake for KVCacheManagerProtocol."""

    def __init__(self, total_num_blocks: int = 100):
        self._total_num_blocks = total_num_blocks
        self._num_free_blocks = total_num_blocks
        self.block_pool = FakeBlockPool(self._num_free_blocks)

    @property
    def usage(self) -> float:
        """Return KV cache usage ratio between 0.0 and 1.0."""
        if self._total_num_blocks == 0:
            return 0.0
        return 1.0 - (self._num_free_blocks / self._total_num_blocks)

    def set_usage(self, usage_ratio: float) -> None:
        """Set usage to a specific ratio in [0, 1]."""
        used = int(self._total_num_blocks * usage_ratio)
        self._num_free_blocks = self._total_num_blocks - used
        self.block_pool._num_free_blocks = self._num_free_blocks


class FakeModelExecutor:
    """Minimal fake for ModelExecutorProtocol."""

    def execute_warmup_prefill(
        self, token_ids: list[int], **kwargs
    ) -> list[int] | None:
        return None


class FakeModelExecutorWithBlocks:
    """Fake executor that returns a configurable number of blocks."""

    def __init__(self, blocks_per_prefill: int = 1):
        self._blocks_per_prefill = blocks_per_prefill
        self.prefill_calls: list[list[int]] = []

    def execute_warmup_prefill(
        self, token_ids: list[int], **kwargs
    ) -> list[int] | None:
        self.prefill_calls.append(token_ids)
        return list(range(self._blocks_per_prefill))


class FakeEngineCore:
    """Fake engine core for property testing."""

    def get_queue_depth(self) -> int:
        return 0


# ------------------------------------------------------------------
# Property 5: VRAM Budget Invariant
# ------------------------------------------------------------------


@settings(max_examples=100)
@given(
    free_blocks=st.integers(min_value=1, max_value=1000),
    warmup_vram_budget_ratio=st.floats(min_value=0.01, max_value=1.0),
    blocks_per_prefill=st.integers(min_value=1, max_value=50),
    num_candidates=st.integers(min_value=1, max_value=100),
)
def test_vram_budget_invariant(
    free_blocks: int,
    warmup_vram_budget_ratio: float,
    blocks_per_prefill: int,
    num_candidates: int,
) -> None:
    """Property 5: VRAM Budget Invariant.

    **Validates: Requirements 2.7**

    For any free block count and warmup_vram_budget_ratio, the
    worker SHALL NOT start a new prefill once blocks consumed
    reaches or exceeds floor(free_blocks * warmup_vram_budget_ratio).
    """
    expected_budget = math.floor(free_blocks * warmup_vram_budget_ratio)

    # Set total high enough so memory pressure doesn't trigger.
    # Usage = 1 - free/total must be < 0.9 (pause threshold).
    total_num_blocks = free_blocks * 10

    config = AdaptiveServingConfig(
        adaptive_profile="dev",
        warmup_vram_budget_ratio=warmup_vram_budget_ratio,
        warmup_min_hit_count=0.01,
        warmup_budget_ms=10000.0,
    )

    tracker = PrefixFrequencyTracker(
        max_entries=1000, ema_decay=config.warmup_ema_decay
    )

    # Create enough candidates with scores above threshold
    for h in range(num_candidates):
        for _ in range(20):
            tracker.update(h)

    kv_manager = FakeKVCacheManager(total_num_blocks=total_num_blocks)
    kv_manager._num_free_blocks = free_blocks
    kv_manager.block_pool._num_free_blocks = free_blocks
    executor = FakeModelExecutorWithBlocks(blocks_per_prefill=blocks_per_prefill)

    worker = PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=FakeBlockPool(),
        kv_cache_manager=kv_manager,
        model_executor=executor,
    )

    worker.on_idle(FakeEngineCore())

    actual_consumed = worker._blocks_consumed_this_window

    if expected_budget <= 0:
        # Budget is 0 — no prefills should have run
        assert actual_consumed == 0, (
            f"Expected 0 consumed with budget=0, got {actual_consumed}"
        )
    else:
        # The worker checks consumed >= budget BEFORE each prefill.
        # Last prefill started with consumed < budget (at most
        # budget - 1), then added blocks_per_prefill.
        max_allowed = expected_budget - 1 + blocks_per_prefill
        assert actual_consumed <= max_allowed, (
            f"VRAM budget violated: consumed {actual_consumed} "
            f"> max allowed {max_allowed} "
            f"(budget={expected_budget}, "
            f"blocks_per_prefill={blocks_per_prefill})"
        )

        # Verify no prefill started once consumed >= budget
        num_prefills = len(executor.prefill_calls)
        if num_candidates >= num_prefills:
            max_prefills = math.ceil(expected_budget / blocks_per_prefill)
            assert num_prefills <= max_prefills, (
                f"Too many prefills: {num_prefills} > ceil(budget/bpp) = {max_prefills}"
            )


# ------------------------------------------------------------------
# Property 10: High-Load Warmup Disable
# ------------------------------------------------------------------


@st.composite
def queue_depth_time_series(draw: st.DrawFn):
    """Generate a time series of (queue_depth, time_delta) pairs.

    Each entry represents calling update_high_load_state_from_queue
    at a certain time offset from the previous call.
    """
    n = draw(st.integers(min_value=1, max_value=50))
    series: list[tuple[int, float]] = []
    for _ in range(n):
        depth = draw(st.integers(min_value=0, max_value=50))
        # Time deltas between 0.1 and 10 seconds
        delta = draw(st.floats(min_value=0.1, max_value=10.0))
        series.append((depth, delta))
    return series


@settings(max_examples=100)
@given(
    high_load_queue_depth=st.integers(min_value=1, max_value=20),
    high_load_duration_seconds=st.floats(min_value=0.5, max_value=15.0),
    series=queue_depth_time_series(),
)
def test_high_load_warmup_disable(
    high_load_queue_depth: int,
    high_load_duration_seconds: float,
    series: list[tuple[int, float]],
) -> None:
    """Property 10: High-Load Warmup Disable

    For any time series of queue depths, warmup SHALL be disabled
    when queue depth exceeds high_load_queue_depth for longer than
    high_load_duration_seconds, and re-enabled when the sustained
    high-load condition no longer holds.

    **Validates: Requirements 9.5**
    """
    config = AdaptiveServingConfig(
        high_load_queue_depth=high_load_queue_depth,
        high_load_duration_seconds=high_load_duration_seconds,
        adaptive_profile="dev",
    )
    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
    worker = PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=FakeBlockPool(),
        kv_cache_manager=FakeKVCacheManager(),
        model_executor=FakeModelExecutor(),
    )

    # Simulate time progression by patching time.monotonic
    current_time = 1000.0

    # Track expected state manually
    expected_high_load_start: float | None = None
    expected_disabled: bool = False

    for queue_depth, delta in series:
        current_time += delta

        with patch(
            "vllm.v1.core.adaptive.prefix_warmup_worker.time.monotonic",
            return_value=current_time,
        ):
            worker.update_high_load_state_from_queue(queue_depth)

        # Replicate the expected logic
        if queue_depth > high_load_queue_depth:
            if expected_high_load_start is None:
                expected_high_load_start = current_time
            else:
                duration = current_time - expected_high_load_start
                if duration >= high_load_duration_seconds:
                    expected_disabled = True
        else:
            expected_high_load_start = None
            expected_disabled = False

        assert worker.is_disabled_for_load == expected_disabled, (
            f"High-load disable mismatch: "
            f"queue_depth={queue_depth}, "
            f"threshold={high_load_queue_depth}, "
            f"duration_threshold={high_load_duration_seconds}s, "
            f"expected_disabled={expected_disabled}, "
            f"actual_disabled={worker.is_disabled_for_load}, "
            f"current_time={current_time}"
        )


# ------------------------------------------------------------------
# Property 9: Memory Pressure Hysteresis
# ------------------------------------------------------------------


@settings(max_examples=100)
@given(
    usage_sequence=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=100,
    ),
)
def test_memory_pressure_hysteresis(
    usage_sequence: list[float],
) -> None:
    """Property 9: Memory Pressure Hysteresis

    For any sequence of KV cache usage readings, the
    PrefixWarmupWorker SHALL: (a) pause warmup when usage >=
    warmup_pause_threshold (0.9), (b) resume warmup only when
    usage drops below warmup_resume_threshold (0.8), and (c)
    remain paused for usage values between 0.8 and 0.9 if
    previously paused.

    **Validates: Requirements 9.1**
    """
    config = AdaptiveServingConfig(adaptive_profile="dev")
    pause_threshold = config.warmup_pause_threshold  # 0.9
    resume_threshold = config.warmup_resume_threshold  # 0.8

    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=config.warmup_ema_decay)
    kv_manager = FakeKVCacheManager(total_num_blocks=100)

    worker = PrefixWarmupWorker(
        config=config,
        frequency_tracker=tracker,
        block_pool=FakeBlockPool(),
        kv_cache_manager=kv_manager,
        model_executor=FakeModelExecutor(),
    )

    # Track expected paused state manually
    expected_paused = False

    for usage in usage_sequence:
        kv_manager.set_usage(usage)

        # Compute actual usage as the worker sees it
        actual_usage = kv_manager.usage

        # Apply hysteresis rules to expected state
        if actual_usage >= pause_threshold:
            expected_paused = True
        elif actual_usage < resume_threshold:
            expected_paused = False
        # Between thresholds: retain current state

        # Call should_pause which updates memory pressure state
        worker.should_pause()

        assert worker.is_paused_for_memory == expected_paused, (
            f"Memory pressure hysteresis violation: "
            f"usage={actual_usage:.4f}, "
            f"pause_threshold={pause_threshold}, "
            f"resume_threshold={resume_threshold}, "
            f"expected_paused={expected_paused}, "
            f"actual_paused={worker.is_paused_for_memory}"
        )


# ------------------------------------------------------------------
# Property 8: Compressed Block Size Budget Calculation
# ------------------------------------------------------------------

# Strategy for dtype configs that don't require TurboQuant imports
UNCOMPRESSED_DTYPES = ["auto", "float16", "bfloat16"]
FP8_DTYPES = ["fp8", "fp8_e4m3", "fp8_e5m2"]
TURBOQUANT_DTYPES = [
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]

dtype_strategy = st.sampled_from(UNCOMPRESSED_DTYPES + FP8_DTYPES + TURBOQUANT_DTYPES)
block_size_strategy = st.sampled_from([8, 16, 32, 64])
head_size_strategy = st.sampled_from([64, 128, 256])
num_kv_heads_strategy = st.integers(min_value=1, max_value=32)
free_blocks_strategy = st.integers(min_value=1, max_value=2000)


@settings(max_examples=100)
@given(
    kv_cache_dtype=dtype_strategy,
    block_size=block_size_strategy,
    head_size=head_size_strategy,
    num_kv_heads=num_kv_heads_strategy,
    free_blocks=free_blocks_strategy,
)
def test_compressed_block_size_budget_calculation(
    kv_cache_dtype: str,
    block_size: int,
    head_size: int,
    num_kv_heads: int,
    free_blocks: int,
) -> None:
    """Property 8: Compressed Block Size Budget Calculation.

    **Validates: Requirements 5.3**

    For any KV cache dtype configuration (including TurboQuant variants)
    and free block count, the warmup VRAM budget calculation SHALL use
    the actual compressed block size (not the uncompressed size) to
    determine how many prefixes can be warmed.

    Specifically:
    - For auto/bf16/fp16: compression_ratio == 1.0, full-size blocks
    - For fp8 variants: compression_ratio == 0.5, half-size blocks
    - For TurboQuant variants: compression_ratio < 1.0, smaller blocks
    - get_estimated_block_size_bytes reflects actual compressed size
    """
    # Mock TurboQuant config for turboquant dtypes since the package
    # may not be installed in the test environment.
    mock_tq_config = MagicMock()
    # TurboQuant slot_size_aligned is smaller than uncompressed.
    # Use a realistic value: e.g., for 4-bit with head_size=128,
    # slot_size_aligned would be much smaller than 2*head_size*2=512
    # We simulate a 4x compression (slot_size = head_size // 2)
    mock_tq_slot_size = head_size // 2
    mock_tq_config.slot_size_aligned = mock_tq_slot_size

    mock_tq_config_class = MagicMock()
    mock_tq_config_class.from_cache_dtype.return_value = mock_tq_config

    # Determine expected compression ratio
    if kv_cache_dtype.startswith("turboquant_"):
        # TQ: slot_size_aligned / (2 * head_size * 2)
        # = (head_size // 2) / (4 * head_size)
        expected_ratio = mock_tq_slot_size / (2 * head_size * 2)
    elif kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        expected_ratio = 0.5
    else:
        expected_ratio = 1.0

    # Test _get_compression_ratio with mocked TurboQuant
    with patch.dict(
        "sys.modules",
        {
            "vllm.model_executor.layers.quantization.turboquant": MagicMock(),
            "vllm.model_executor.layers.quantization.turboquant.config": MagicMock(
                TurboQuantConfig=mock_tq_config_class
            ),
        },
    ):
        ratio = _get_compression_ratio(kv_cache_dtype, head_size)

    assert abs(ratio - expected_ratio) < 1e-9, (
        f"Compression ratio mismatch for dtype={kv_cache_dtype}: "
        f"expected={expected_ratio}, got={ratio}"
    )

    # Verify properties of the compression ratio based on dtype class
    if kv_cache_dtype in UNCOMPRESSED_DTYPES:
        assert ratio == 1.0, (
            f"Uncompressed dtype {kv_cache_dtype} should have ratio 1.0, got {ratio}"
        )
    elif kv_cache_dtype in FP8_DTYPES:
        assert ratio == 0.5, (
            f"FP8 dtype {kv_cache_dtype} should have ratio 0.5, got {ratio}"
        )
    elif kv_cache_dtype in TURBOQUANT_DTYPES:
        assert ratio < 1.0, (
            f"TurboQuant dtype {kv_cache_dtype} should have ratio < 1.0, got {ratio}"
        )

    # Test get_estimated_block_size_bytes with the PrefixWarmupWorker
    total_num_blocks = free_blocks * 10
    config = AdaptiveServingConfig(adaptive_profile="dev")
    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=config.warmup_ema_decay)

    with patch.dict(
        "sys.modules",
        {
            "vllm.model_executor.layers.quantization.turboquant": MagicMock(),
            "vllm.model_executor.layers.quantization.turboquant.config": MagicMock(
                TurboQuantConfig=mock_tq_config_class
            ),
        },
    ):
        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=FakeBlockPool(),
            kv_cache_manager=FakeKVCacheManager(total_num_blocks=total_num_blocks),
            model_executor=FakeModelExecutor(),
            kv_cache_dtype=kv_cache_dtype,
            head_size=head_size,
        )

        # Verify compression_ratio property on the worker
        assert abs(worker.compression_ratio - expected_ratio) < 1e-9, (
            f"Worker compression_ratio mismatch: "
            f"expected={expected_ratio}, got={worker.compression_ratio}"
        )

        # Verify get_estimated_block_size_bytes uses compressed size
        estimated_bytes = worker.get_estimated_block_size_bytes(
            block_size, num_kv_heads
        )

    # Compute expected block size for each dtype class
    if kv_cache_dtype.startswith("turboquant_"):
        # TQ: block_size * num_kv_heads * slot_size_aligned
        expected_bytes = block_size * num_kv_heads * mock_tq_slot_size
    elif kv_cache_dtype in FP8_DTYPES:
        # FP8: 2 * block_size * num_kv_heads * head_size * 1 byte
        expected_bytes = 2 * block_size * num_kv_heads * head_size * 1
    else:
        # bf16/fp16: 2 * block_size * num_kv_heads * head_size * 2 bytes
        expected_bytes = 2 * block_size * num_kv_heads * head_size * 2

    assert estimated_bytes == expected_bytes, (
        f"Block size estimation mismatch for dtype={kv_cache_dtype}: "
        f"expected={expected_bytes}, got={estimated_bytes}"
    )

    # Verify uncompressed reference for comparison
    uncompressed_bytes = 2 * block_size * num_kv_heads * head_size * 2
    if kv_cache_dtype in FP8_DTYPES:
        assert estimated_bytes == uncompressed_bytes // 2, (
            "FP8 blocks should be half the uncompressed size"
        )
    elif kv_cache_dtype in TURBOQUANT_DTYPES:
        assert estimated_bytes < uncompressed_bytes, (
            f"TurboQuant blocks should be smaller than uncompressed: "
            f"compressed={estimated_bytes}, uncompressed={uncompressed_bytes}"
        )
    elif kv_cache_dtype in UNCOMPRESSED_DTYPES:
        assert estimated_bytes == uncompressed_bytes, (
            f"Uncompressed blocks should equal full size: "
            f"got={estimated_bytes}, expected={uncompressed_bytes}"
        )
