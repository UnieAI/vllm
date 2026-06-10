# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics and periodic logging for Adaptive Speculative Serving.

Exposes four Gauge metrics for monitoring adaptive serving performance:
- vllm:prefix_cache_hit_rate — percentage of requests with prefix cache hits
- vllm:prefix_warmup_entries_count — number of actively warmed prefix entries
- vllm:self_speculation_skip_rate — percentage of decode steps skipped
- vllm:confidence_tracker_mean_threshold — average confidence threshold

Also provides AdaptiveServingLogging for periodic INFO-level log output.
"""

from __future__ import annotations

import time

from prometheus_client import Counter, Gauge

from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


class AdaptiveServingProm:
    """Register and update Prometheus metrics for adaptive serving.

    Follows the same pattern as SpecDecodingProm and KVConnectorProm:
    metrics are registered once at init and updated via observe().
    """

    _gauge_cls = Gauge
    _counter_cls = Counter

    def __init__(
        self,
        enabled: bool,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.enabled = enabled
        if not enabled:
            return

        gauge_prefix_cache_hit_rate = self._gauge_cls(
            name="vllm:prefix_cache_hit_rate",
            documentation=(
                "Percentage of requests with prefix cache hits (adaptive serving)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_prefix_cache_hit_rate = create_metric_per_engine(
            gauge_prefix_cache_hit_rate, per_engine_labelvalues
        )

        gauge_prefix_warmup_entries_count = self._gauge_cls(
            name="vllm:prefix_warmup_entries_count",
            documentation=(
                "Number of actively warmed prefix cache entries (adaptive serving)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_prefix_warmup_entries_count = create_metric_per_engine(
            gauge_prefix_warmup_entries_count, per_engine_labelvalues
        )

        gauge_self_speculation_skip_rate = self._gauge_cls(
            name="vllm:self_speculation_skip_rate",
            documentation=(
                "Percentage of decode steps skipped due to successful self-speculation."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_self_speculation_skip_rate = create_metric_per_engine(
            gauge_self_speculation_skip_rate, per_engine_labelvalues
        )

        gauge_confidence_tracker_mean_threshold = self._gauge_cls(
            name="vllm:confidence_tracker_mean_threshold",
            documentation=(
                "Average confidence threshold across all tracked "
                "patterns (adaptive serving)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_confidence_tracker_mean_threshold = create_metric_per_engine(
            gauge_confidence_tracker_mean_threshold, per_engine_labelvalues
        )

        counter_warmup_prefills_executed = self._counter_cls(
            name="vllm:warmup_prefills_executed",
            documentation=(
                "Total number of successful GPU warmup prefill "
                "operations since startup (adaptive serving)."
            ),
            labelnames=labelnames,
        )
        self.counter_warmup_prefills_executed = create_metric_per_engine(
            counter_warmup_prefills_executed, per_engine_labelvalues
        )

        counter_warmup_prefills_skipped_no_tokens = self._counter_cls(
            name="vllm:warmup_prefills_skipped_no_tokens",
            documentation=(
                "Total warmup candidates skipped due to missing "
                "token registry entries (adaptive serving)."
            ),
            labelnames=labelnames,
        )
        self.counter_warmup_prefills_skipped_no_tokens = create_metric_per_engine(
            counter_warmup_prefills_skipped_no_tokens,
            per_engine_labelvalues,
        )

    def observe(
        self,
        prefix_cache_hit_rate: float,
        prefix_warmup_entries_count: int,
        self_speculation_skip_rate: float,
        confidence_tracker_mean_threshold: float,
        engine_idx: int = 0,
    ) -> None:
        """Update all adaptive serving metrics.

        Args:
            prefix_cache_hit_rate: Percentage (0-100) of requests
                with prefix cache hits.
            prefix_warmup_entries_count: Number of actively warmed
                prefix cache entries.
            self_speculation_skip_rate: Percentage (0-100) of decode
                steps skipped due to successful speculation.
            confidence_tracker_mean_threshold: Average confidence
                threshold across all tracked patterns.
            engine_idx: Engine index for labeled metrics.
        """
        if not self.enabled:
            return

        self.gauge_prefix_cache_hit_rate[engine_idx].set(prefix_cache_hit_rate)
        self.gauge_prefix_warmup_entries_count[engine_idx].set(
            prefix_warmup_entries_count
        )
        self.gauge_self_speculation_skip_rate[engine_idx].set(
            self_speculation_skip_rate
        )
        self.gauge_confidence_tracker_mean_threshold[engine_idx].set(
            confidence_tracker_mean_threshold
        )

    def observe_warmup_counters(
        self,
        prefills_executed: int,
        prefills_skipped_no_tokens: int,
        engine_idx: int = 0,
    ) -> None:
        """Increment warmup counter metrics.

        Args:
            prefills_executed: Number of new successful warmup
                prefills since last call.
            prefills_skipped_no_tokens: Number of new warmup
                candidates skipped due to missing tokens since
                last call.
            engine_idx: Engine index for labeled metrics.
        """
        if not self.enabled:
            return

        if prefills_executed > 0:
            self.counter_warmup_prefills_executed[engine_idx].inc(prefills_executed)
        if prefills_skipped_no_tokens > 0:
            self.counter_warmup_prefills_skipped_no_tokens[engine_idx].inc(
                prefills_skipped_no_tokens
            )


class AdaptiveServingLogging:
    """Periodic logging for adaptive serving warmup progress.

    Logs every 60 seconds with:
    - Number of entries warmed
    - EMA score distribution (min, mean, max)
    - Estimated hit rate improvement
    """

    LOG_INTERVAL_SECONDS = 60.0

    def __init__(self) -> None:
        self._last_log_time: float = time.monotonic()
        self._last_entries_warmed: int = 0

    def maybe_log(
        self,
        total_entries_warmed: int,
        frequency_tracker_scores: dict[int, float],
        prefix_cache_hit_rate: float,
    ) -> None:
        """Log warmup progress if the 60s interval has elapsed.

        Args:
            total_entries_warmed: Total entries warmed across all
                idle windows.
            frequency_tracker_scores: Current EMA scores from the
                frequency tracker (hash -> score mapping).
            prefix_cache_hit_rate: Current prefix cache hit rate
                percentage (0-100).
        """
        now = time.monotonic()
        elapsed = now - self._last_log_time
        if elapsed < self.LOG_INTERVAL_SECONDS:
            return

        # Calculate entries warmed since last log
        entries_this_interval = total_entries_warmed - self._last_entries_warmed

        # Compute EMA score distribution
        if frequency_tracker_scores:
            scores = list(frequency_tracker_scores.values())
            score_min = min(scores)
            score_mean = sum(scores) / len(scores)
            score_max = max(scores)
            score_dist_str = (
                f"min={score_min:.4f}, mean={score_mean:.4f}, max={score_max:.4f}"
            )
        else:
            score_dist_str = "no entries"

        # Estimate hit rate improvement (difference from a
        # hypothetical baseline of 5% if we have warmed entries)
        baseline_hit_rate = 5.0
        estimated_improvement = max(0.0, prefix_cache_hit_rate - baseline_hit_rate)

        logger.info(
            "Adaptive warmup progress: "
            "entries warmed (interval/total): %d/%d, "
            "EMA score distribution: [%s], "
            "estimated hit rate improvement: +%.1f%%",
            entries_this_interval,
            total_entries_warmed,
            score_dist_str,
            estimated_improvement,
        )

        # Update tracking state
        self._last_log_time = now
        self._last_entries_warmed = total_entries_warmed
