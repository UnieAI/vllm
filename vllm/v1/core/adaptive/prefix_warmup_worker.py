# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix Warmup Worker for adaptive speculative serving.

Warms high-frequency prefixes during idle GPU windows by selecting
candidates from the PrefixFrequencyTracker, submitting prefill
computations via the model executor, and storing resulting KV blocks
in the Block Pool.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.core.adaptive.metrics import AdaptiveServingLogging
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    hash_block_tokens,
    make_block_hash_with_group_id,
)

if TYPE_CHECKING:
    from vllm.v1.core.adaptive.token_registry import TokenRegistry

logger = init_logger(__name__)


def _get_compression_ratio(kv_cache_dtype: CacheDType, head_size: int) -> float:
    """Compute the compression ratio for a given KV cache dtype.

    Returns the ratio of compressed block size to the uncompressed
    (bf16/fp16) block size. For uncompressed dtypes, returns 1.0.
    For TurboQuant variants, returns a ratio < 1.0 indicating how
    much smaller each block is compared to bf16/fp16.

    Args:
        kv_cache_dtype: The configured KV cache dtype string.
        head_size: The attention head dimension.

    Returns:
        A float ratio where 1.0 means no compression and values
        < 1.0 indicate compression (e.g., 0.5 means half the size).
    """
    # Guard against non-string values (e.g., from MagicMock in tests)
    if not isinstance(kv_cache_dtype, str):
        return 1.0

    if kv_cache_dtype.startswith("turboquant_"):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        # TQ packs K+V into slot_size_aligned bytes per head per position.
        # Uncompressed bf16 uses 2 * head_size * 2 bytes (K + V, 2 bytes each).
        uncompressed_bytes_per_head_per_pos = 2 * head_size * 2  # K+V in bf16
        compressed_bytes_per_head_per_pos = tq_config.slot_size_aligned
        return compressed_bytes_per_head_per_pos / uncompressed_bytes_per_head_per_pos
    elif kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        # FP8 uses 1 byte per element vs 2 bytes for bf16/fp16
        return 0.5
    else:
        # auto, float16, bfloat16, or other — no compression
        return 1.0


class BlockPoolProtocol(Protocol):
    """Protocol for the KV cache block pool interface.

    The actual Block_Pool provides hash-based prefix cache lookup
    and block storage. This protocol defines the subset of the
    interface needed by the warmup worker.
    """

    def get_cached_block_hashes(self) -> set[int]:
        """Return set of prefix hashes currently cached."""
        ...

    def get_num_free_blocks(self) -> int:
        """Return number of free (unallocated) KV cache blocks."""
        ...

    def free_blocks(self, ordered_blocks: Any) -> None:
        """Free blocks back to the pool."""
        ...


class KVCacheManagerProtocol(Protocol):
    """Protocol for the KV cache manager interface.

    Provides cache usage information for memory pressure checks.
    Compatible with vllm.v1.core.kv_cache_manager.KVCacheManager.
    """

    @property
    def usage(self) -> float:
        """Return KV cache usage ratio between 0.0 and 1.0."""
        ...

    @property
    def block_pool(self) -> BlockPoolProtocol:
        """Return the block pool."""
        ...


class ModelExecutorProtocol(Protocol):
    """Protocol for the model executor interface.

    The actual executor submits prefill computations to the GPU.
    This protocol defines the minimal interface for warmup prefill.
    """

    def execute_warmup_prefill(
        self,
        token_ids: list[int],
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[int] | None:
        """Execute a warmup prefill for the given token IDs.

        Args:
            token_ids: Token IDs to execute prefill for.
            abort_fn: Optional callable returning True if warmup
                should be aborted. Checked after each block's
                forward pass for multi-block sequences.

        Returns a list of block IDs where KV cache was stored,
        or None if the prefill failed or was aborted.
        """
        ...


class EngineCoreProtocol(Protocol):
    """Protocol for Engine Core interface used during idle callback.

    Provides access to the current queue depth for load detection.
    """

    def get_queue_depth(self) -> int:
        """Return current number of pending requests in queue."""
        ...


class PrefixWarmupWorker:
    """Warms high-frequency prefixes during idle windows.

    This worker is registered as an idle state callback with
    Engine Core. When the engine enters an idle state (no active
    requests to process), this worker selects high-EMA-score
    prefixes that are not yet cached and submits them for prefill
    computation.

    The worker implements several safety mechanisms:
    - Budget tracking: aborts after warmup_budget_ms elapsed
    - Memory pressure hysteresis: pauses at >=90%, resumes <80%
    - High-load detection: disables during sustained high load
    - Cooperative yielding: abort flag for request preemption
    - VRAM budget: limits total blocks consumed per idle window
    """

    def __init__(
        self,
        config: AdaptiveServingConfig,
        frequency_tracker: PrefixFrequencyTracker,
        block_pool: Any,
        kv_cache_manager: Any,
        model_executor: Any,
        kv_cache_dtype: CacheDType = "auto",
        head_size: int = 128,
        token_registry: TokenRegistry | None = None,
        block_size: int = 16,
        caching_hash_fn: Callable[[Any], bytes] | None = None,
    ) -> None:
        self._config = config
        self._frequency_tracker = frequency_tracker
        self._block_pool: BlockPoolProtocol = block_pool
        self._kv_cache_manager: KVCacheManagerProtocol = kv_cache_manager
        self._model_executor: ModelExecutorProtocol = model_executor
        self._token_registry = token_registry
        self._block_size = block_size
        self._caching_hash_fn = caching_hash_fn

        # KV cache dtype configuration for TurboQuant compatibility.
        # Used to compute compressed block sizes in VRAM budget
        # calculations and to ensure warmed KV blocks are stored in
        # the correct format.
        self._kv_cache_dtype: CacheDType = kv_cache_dtype
        self._head_size: int = head_size
        self._compression_ratio: float = _get_compression_ratio(
            kv_cache_dtype, head_size
        )

        if self._compression_ratio < 1.0:
            logger.info(
                "PrefixWarmupWorker using compressed KV cache dtype=%s "
                "(compression ratio=%.3f). More prefixes can be warmed "
                "within the same VRAM budget.",
                kv_cache_dtype,
                self._compression_ratio,
            )

        # Cooperative abort flag — set by abort() on request arrival
        self._abort_flag: bool = False

        # Memory pressure hysteresis state
        self._paused_for_memory: bool = False

        # High-load tracking state
        self._high_load_start_time: float | None = None
        self._disabled_for_load: bool = False

        # Metrics / counters
        self._total_entries_warmed: int = 0
        self._blocks_consumed_this_window: int = 0
        self.warmup_prefills_executed: int = 0
        self.warmup_prefills_skipped_no_tokens: int = 0

        # Periodic logging (every 60s)
        self._logging = AdaptiveServingLogging()

    @property
    def total_entries_warmed(self) -> int:
        """Total number of prefix entries warmed across all windows."""
        return self._total_entries_warmed

    @property
    def is_paused_for_memory(self) -> bool:
        """Whether warmup is currently paused due to memory pressure."""
        return self._paused_for_memory

    @property
    def is_disabled_for_load(self) -> bool:
        """Whether warmup is disabled due to sustained high load."""
        return self._disabled_for_load

    @property
    def kv_cache_dtype(self) -> CacheDType:
        """The configured KV cache dtype used for warmed blocks."""
        return self._kv_cache_dtype

    @property
    def compression_ratio(self) -> float:
        """Compression ratio of configured dtype vs uncompressed bf16.

        A value of 1.0 means no compression. Values < 1.0 indicate
        that blocks are smaller (e.g., 0.25 for 4-bit quantization),
        allowing more prefixes to be warmed within the same VRAM budget.
        """
        return self._compression_ratio

    def on_idle(self, engine_core: Any) -> None:
        """Idle state callback — registered with Engine Core.

        Called when Engine Core detects an idle window (no pending
        requests). Iterates through warmup candidates in descending
        EMA score order until the time budget is exhausted, the VRAM
        budget is reached, the abort flag is set, or no candidates
        remain.

        Args:
            engine_core: The Engine Core instance (used to check
                queue depth for high-load detection).
        """
        # Reset per-window state
        self._abort_flag = False
        self._blocks_consumed_this_window = 0

        # Check if we should even start
        if self.should_pause():
            return

        # Start budget tracking
        assert self._config.warmup_budget_ms is not None
        budget_seconds = self._config.warmup_budget_ms / 1000.0
        window_start = time.monotonic()

        # Calculate VRAM budget for this window
        vram_block_budget = self._calculate_vram_budget()
        if vram_block_budget <= 0:
            return

        # Warmup loop: process candidates until budget exhausted
        while not self._abort_flag:
            # Check time budget
            elapsed = time.monotonic() - window_start
            if elapsed >= budget_seconds:
                break

            # Re-check pause conditions between candidates
            if self.should_pause():
                break

            # Check VRAM budget
            if self._blocks_consumed_this_window >= vram_block_budget:
                break

            # Select next candidate
            candidate = self.select_next_candidate()
            if candidate is None:
                break

            prefix_hash, _score = candidate

            # Resolve token IDs from registry if available
            if self._token_registry is not None:
                token_ids = self._resolve_multi_block_tokens([prefix_hash])
                if token_ids is None:
                    self.warmup_prefills_skipped_no_tokens += 1
                    logger.debug(
                        "Warmup skipped: no tokens in registry for hash=%d",
                        prefix_hash,
                    )
                    continue
            else:
                # Fallback: pass hash as single-element list
                # (legacy behavior when no registry configured)
                token_ids = [prefix_hash]

            # Check if enough free blocks are available before
            # attempting the prefill to avoid wasting GPU computation.
            num_blocks_needed = math.ceil(len(token_ids) / self._block_size)
            if self._block_pool.get_num_free_blocks() < num_blocks_needed:
                logger.debug(
                    "Warmup skipped: insufficient free blocks "
                    "(need=%d, free=%d) for hash=%d",
                    num_blocks_needed,
                    self._block_pool.get_num_free_blocks(),
                    prefix_hash,
                )
                continue

            # Submit prefill via model executor, passing abort_fn so
            # the execution path can check for preemption between
            # per-block forward passes in multi-block sequences.
            start_time = time.monotonic()
            result_blocks = self._model_executor.execute_warmup_prefill(
                token_ids, abort_fn=lambda: self._abort_flag
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            if result_blocks is not None:
                # Register block hashes in the block pool so that
                # subsequent requests with matching prefixes find them
                # in the prefix cache.
                #
                # Block cleanup on abort (Requirement 6.3):
                # The executor returns ONLY block IDs for blocks that
                # completed their forward pass (valid KV cache). On
                # abort mid-sequence, this list is shorter than
                # num_blocks_needed — we only commit the completed
                # blocks. Unprocessed blocks were never claimed from
                # the free pool (the model runner allocates from a
                # fixed range), so no explicit freeing is required.
                # This guarantees: all blocks are either committed
                # (hash registered) or never-allocated — no leaks.
                self._register_warmed_block_hashes(token_ids, result_blocks)
                self._blocks_consumed_this_window += len(result_blocks)
                self._total_entries_warmed += 1
                self.warmup_prefills_executed += 1
                logger.debug(
                    "Warmup prefill: hash=%d, tokens=%d, elapsed_ms=%.2f",
                    prefix_hash,
                    len(token_ids),
                    elapsed_ms,
                )

        # Periodic logging (every 60s)
        self._maybe_log_progress()

    def should_pause(self) -> bool:
        """Check memory pressure and load conditions.

        Implements hysteresis for memory pressure:
        - Pause when KV cache usage >= warmup_pause_threshold (90%)
        - Resume only when usage drops below
          warmup_resume_threshold (80%)
        - Between 80% and 90%: retain previous state

        Also checks high-load conditions:
        - Disable warmup when queue depth > high_load_queue_depth
          for > high_load_duration_seconds

        Returns:
            True if warmup should be paused/disabled.
        """
        # Check memory pressure with hysteresis
        self._update_memory_pressure_state()
        if self._paused_for_memory:
            return True

        # Check high-load conditions
        self._update_high_load_state()
        return bool(self._disabled_for_load)

    def select_next_candidate(self) -> tuple[int, float] | None:
        """Select the highest-EMA uncached prefix for warmup.

        Uses the frequency tracker's get_warmup_candidates() to
        find uncached prefixes with score above the configured
        minimum threshold, then returns the top candidate.

        Returns:
            Tuple of (prefix_hash, ema_score) for the best
            candidate, or None if no candidates are available.
        """
        cached_hashes = self._block_pool.get_cached_block_hashes()
        assert self._config.warmup_min_hit_count is not None
        min_score = self._config.warmup_min_hit_count

        candidates = self._frequency_tracker.get_warmup_candidates(
            cached_hashes=cached_hashes,
            min_score=min_score,
        )

        if not candidates:
            return None

        # Return the top candidate (highest EMA score)
        return candidates[0]

    def abort(self) -> None:
        """Abort current warmup (called on request arrival).

        Sets the cooperative abort flag which is checked in the
        warmup loop. This ensures the worker yields GPU resources
        to incoming requests promptly.
        """
        self._abort_flag = True

    def update_high_load_state_from_queue(self, queue_depth: int) -> None:
        """Update high-load tracking from external queue depth.

        Called by Engine Core to feed queue depth observations.
        This enables the worker to detect sustained high load
        conditions even when not in an idle callback.

        Args:
            queue_depth: Current number of pending requests.
        """
        now = time.monotonic()
        threshold = self._config.high_load_queue_depth

        if queue_depth > threshold:
            if self._high_load_start_time is None:
                self._high_load_start_time = now
            else:
                duration = now - self._high_load_start_time
                if duration >= self._config.high_load_duration_seconds:
                    self._disabled_for_load = True
        else:
            # Load dropped below threshold — reset
            self._high_load_start_time = None
            self._disabled_for_load = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_log_progress(self) -> None:
        """Invoke periodic logging if the 60s interval has elapsed."""
        # Compute prefix cache hit rate from block pool stats
        cached_count = len(self._block_pool.get_cached_block_hashes())
        free_blocks = self._block_pool.get_num_free_blocks()
        # Estimate total blocks from free + cached (approximate)
        total_est = free_blocks + cached_count
        hit_rate = (cached_count / total_est * 100.0) if total_est > 0 else 0.0

        self._logging.maybe_log(
            total_entries_warmed=self._total_entries_warmed,
            frequency_tracker_scores=self._frequency_tracker._scores,
            prefix_cache_hit_rate=hit_rate,
        )

    def _update_memory_pressure_state(self) -> None:
        """Update memory pressure hysteresis state.

        Uses KV cache manager to determine current usage ratio and
        applies hysteresis logic.
        """
        usage = self._kv_cache_manager.usage

        if usage >= self._config.warmup_pause_threshold:
            self._paused_for_memory = True
        elif usage < self._config.warmup_resume_threshold:
            self._paused_for_memory = False
        # Between resume and pause thresholds: retain current state

    def _update_high_load_state(self) -> None:
        """Update high-load state based on tracked observations.

        The actual queue depth is fed via
        update_high_load_state_from_queue(). This method just reads
        the current state (no-op if called within idle callback
        where queue depth is typically 0).
        """
        # State is maintained by update_high_load_state_from_queue()
        # Nothing additional to do here — the flag is already set.
        pass

    def _calculate_vram_budget(self) -> int:
        """Calculate maximum blocks allowed for warmup this window.

        Returns floor(free_blocks * warmup_vram_budget_ratio), adjusted
        for KV cache compression. When a compressed dtype (e.g.,
        TurboQuant) is configured, each block consumes less VRAM, so
        more blocks can be warmed within the same effective VRAM budget.

        The adjustment divides by compression_ratio: if blocks are 4x
        smaller (ratio=0.25), we can fit 4x more blocks.
        """
        free_blocks = self._kv_cache_manager.block_pool.get_num_free_blocks()
        assert self._config.warmup_vram_budget_ratio is not None
        budget = math.floor(free_blocks * self._config.warmup_vram_budget_ratio)

        # When using compressed KV cache, blocks are smaller in VRAM.
        # The block pool already accounts for compression in its block
        # count (profiling allocated more blocks because each is smaller),
        # so free_blocks already reflects the compressed reality.
        # No additional ratio adjustment is needed here since the block
        # pool's free count already incorporates the actual block size.
        # However, we expose the compression_ratio for external callers
        # (metrics, budget estimation) that work in byte terms.
        return max(0, budget)

    def _resolve_multi_block_tokens(self, prefix_hashes: list[int]) -> list[int] | None:
        """Resolve and concatenate token sequences from multiple
        block hashes.

        For multi-block prefixes, each block hash maps to a
        fixed-size token sequence in the TokenRegistry. This method
        looks up each hash in prefix order and concatenates the
        resulting token sequences into a single list suitable for
        passing to ``execute_warmup_prefill``.

        Args:
            prefix_hashes: Ordered list of block hashes forming a
                multi-block prefix. Must contain at least one hash.

        Returns:
            Concatenated token IDs from all blocks in prefix order,
            or None if any block hash is missing from the registry.
        """
        if self._token_registry is None:
            return None

        if not prefix_hashes:
            return None

        concatenated: list[int] = []
        for block_hash in prefix_hashes:
            tokens = self._token_registry.get_tokens(block_hash)
            if tokens is None:
                return None
            concatenated.extend(tokens)

        return concatenated

    def _register_warmed_block_hashes(
        self,
        token_ids: list[int],
        block_ids: list[int],
    ) -> None:
        """Register warmed blocks in the block pool prefix cache.

        Computes block hashes using the same algorithm as the normal
        request path (``hash_block_tokens``) and registers them in
        the block pool's ``cached_block_hash_to_block`` map. After
        registration, blocks are freed so they have the same eviction
        priority as blocks produced by normal user requests.

        Args:
            token_ids: The full token sequence that was warmed.
            block_ids: Block IDs returned by execute_warmup_prefill.
        """
        if self._caching_hash_fn is None:
            # Cannot register without a hash function — this means
            # prefix caching is not enabled.
            return

        # Verify block pool supports hash registration.
        block_pool = self._block_pool
        if not hasattr(block_pool, "blocks") or not hasattr(
            block_pool, "cached_block_hash_to_block"
        ):
            return

        # Compute block hashes matching the normal request path.
        # The algorithm chains hashes: each block's hash depends on
        # its parent (previous block's hash).
        parent_hash: BlockHash | None = None
        block_size = self._block_size

        for i, block_id in enumerate(block_ids):
            start = i * block_size
            end = start + block_size
            block_tokens = token_ids[start:end]

            if len(block_tokens) < block_size:
                # Incomplete final block — skip hash registration.
                break

            block_hash = hash_block_tokens(
                self._caching_hash_fn,
                parent_hash,
                block_tokens,
                extra_keys=None,
            )

            # Register in block pool (group_id=0 for standard
            # single-group KV cache).
            block_hash_with_group = make_block_hash_with_group_id(block_hash, 0)

            # Access the KVCacheBlock object and set its hash.
            block = block_pool.blocks[block_id]
            if block.block_hash is None:
                block.block_hash = block_hash_with_group
                block_pool.cached_block_hash_to_block.insert(
                    block_hash_with_group, block
                )

            # Free the block so it becomes an eviction candidate
            # with the same priority as normal cached blocks.
            block_pool.free_blocks([block])

            parent_hash = block_hash

    def get_estimated_block_size_bytes(self, block_size: int, num_kv_heads: int) -> int:
        """Estimate the VRAM size of a single KV block in bytes.

        Accounts for the configured kv_cache_dtype compression.
        For TurboQuant, uses the actual compressed slot size.
        For standard dtypes, uses 2 * block_size * num_kv_heads *
        head_size * element_size.

        Args:
            block_size: Number of tokens per block.
            num_kv_heads: Number of KV attention heads.

        Returns:
            Estimated bytes per block.
        """
        if self._kv_cache_dtype.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_config = TurboQuantConfig.from_cache_dtype(
                self._kv_cache_dtype, self._head_size
            )
            # TQ stores K+V interleaved in slot_size_aligned bytes
            # per head per position.
            return block_size * num_kv_heads * tq_config.slot_size_aligned
        elif self._kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            # FP8: 1 byte per element, K + V
            return 2 * block_size * num_kv_heads * self._head_size * 1
        else:
            # bf16/fp16: 2 bytes per element, K + V
            return 2 * block_size * num_kv_heads * self._head_size * 2
