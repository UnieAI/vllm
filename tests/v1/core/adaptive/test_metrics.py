# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AdaptiveServingProm and AdaptiveServingLogging.

Verifies all 4 metrics are exposed and correctly updated, and that
periodic logging respects the 60s interval.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.core.adaptive.metrics import (
    AdaptiveServingLogging,
    AdaptiveServingProm,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# AdaptiveServingProm tests
# ---------------------------------------------------------------------------


class FakeGauge:
    """Fake Gauge that records constructor args and creates labeled children."""

    def __init__(self, name, documentation, multiprocess_mode, labelnames):
        self.name = name
        self.documentation = documentation
        self.multiprocess_mode = multiprocess_mode
        self.labelnames = labelnames
        self._children: dict[tuple, MagicMock] = {}

    def labels(self, *args):
        key = args
        if key not in self._children:
            child = MagicMock()
            child._name = self.name
            self._children[key] = child
        return self._children[key]


class FakeCounter:
    """Fake Counter that records constructor args and creates children."""

    def __init__(self, name, documentation, labelnames):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._children: dict[tuple, MagicMock] = {}

    def labels(self, *args):
        key = args
        if key not in self._children:
            child = MagicMock()
            child._name = self.name
            self._children[key] = child
        return self._children[key]


@pytest.fixture
def single_engine_labels():
    """Standard single-engine label setup."""
    return {
        "labelnames": ["engine_index"],
        "per_engine_labelvalues": {0: ["0"]},
    }


@pytest.fixture
def multi_engine_labels():
    """Multi-engine label setup."""
    return {
        "labelnames": ["engine_index"],
        "per_engine_labelvalues": {0: ["0"], 1: ["1"]},
    }


class TestAdaptiveServingPromEnabled:
    """Tests for AdaptiveServingProm when enabled=True."""

    def test_instantiation_enabled(self, single_engine_labels):
        """Test that AdaptiveServingProm can be instantiated when enabled."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            assert prom.enabled is True
            # All 4 gauge metric dicts should exist
            assert hasattr(prom, "gauge_prefix_cache_hit_rate")
            assert hasattr(prom, "gauge_prefix_warmup_entries_count")
            assert hasattr(prom, "gauge_self_speculation_skip_rate")
            assert hasattr(prom, "gauge_confidence_tracker_mean_threshold")
            # Counter metrics should exist
            assert hasattr(prom, "counter_warmup_prefills_executed")
            assert hasattr(prom, "counter_warmup_prefills_skipped_no_tokens")

    def test_observe_updates_prefix_cache_hit_rate(self, single_engine_labels):
        """Test that observe() updates prefix_cache_hit_rate gauge."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe(
                prefix_cache_hit_rate=75.5,
                prefix_warmup_entries_count=10,
                self_speculation_skip_rate=25.0,
                confidence_tracker_mean_threshold=0.85,
                engine_idx=0,
            )
            prom.gauge_prefix_cache_hit_rate[0].set.assert_called_once_with(75.5)

    def test_observe_updates_prefix_warmup_entries_count(self, single_engine_labels):
        """Test that observe() updates prefix_warmup_entries_count gauge."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe(
                prefix_cache_hit_rate=50.0,
                prefix_warmup_entries_count=42,
                self_speculation_skip_rate=10.0,
                confidence_tracker_mean_threshold=0.7,
                engine_idx=0,
            )
            prom.gauge_prefix_warmup_entries_count[0].set.assert_called_once_with(42)

    def test_observe_updates_self_speculation_skip_rate(self, single_engine_labels):
        """Test that observe() updates self_speculation_skip_rate gauge."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe(
                prefix_cache_hit_rate=50.0,
                prefix_warmup_entries_count=10,
                self_speculation_skip_rate=33.3,
                confidence_tracker_mean_threshold=0.7,
                engine_idx=0,
            )
            prom.gauge_self_speculation_skip_rate[0].set.assert_called_once_with(33.3)

    def test_observe_updates_confidence_tracker_mean_threshold(
        self, single_engine_labels
    ):
        """Test that observe() updates confidence_tracker_mean_threshold."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe(
                prefix_cache_hit_rate=50.0,
                prefix_warmup_entries_count=10,
                self_speculation_skip_rate=10.0,
                confidence_tracker_mean_threshold=0.92,
                engine_idx=0,
            )
            prom.gauge_confidence_tracker_mean_threshold[0].set.assert_called_once_with(
                0.92
            )

    def test_metric_names_correct(self, single_engine_labels):
        """Test that all 4 gauge metrics have correct vllm: prefixed names."""
        created_gauges: list[FakeGauge] = []

        class TrackingGauge(FakeGauge):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_gauges.append(self)

        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", TrackingGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            AdaptiveServingProm(enabled=True, **single_engine_labels)

        names = {g.name for g in created_gauges}
        assert "vllm:prefix_cache_hit_rate" in names
        assert "vllm:prefix_warmup_entries_count" in names
        assert "vllm:self_speculation_skip_rate" in names
        assert "vllm:confidence_tracker_mean_threshold" in names
        assert len(created_gauges) == 4


class TestAdaptiveServingPromDisabled:
    """Tests for AdaptiveServingProm when enabled=False."""

    def test_instantiation_disabled(self, single_engine_labels):
        """Test that AdaptiveServingProm does nothing when enabled=False."""
        prom = AdaptiveServingProm(enabled=False, **single_engine_labels)
        assert prom.enabled is False
        # No metric attributes should be created
        assert not hasattr(prom, "gauge_prefix_cache_hit_rate")
        assert not hasattr(prom, "gauge_prefix_warmup_entries_count")
        assert not hasattr(prom, "gauge_self_speculation_skip_rate")
        assert not hasattr(prom, "gauge_confidence_tracker_mean_threshold")

    def test_observe_noop_when_disabled(self, single_engine_labels):
        """Test that observe() does not raise when disabled."""
        prom = AdaptiveServingProm(enabled=False, **single_engine_labels)
        # Should not raise
        prom.observe(
            prefix_cache_hit_rate=50.0,
            prefix_warmup_entries_count=10,
            self_speculation_skip_rate=10.0,
            confidence_tracker_mean_threshold=0.7,
            engine_idx=0,
        )


# ---------------------------------------------------------------------------
# AdaptiveServingLogging tests
# ---------------------------------------------------------------------------


class TestAdaptiveServingPromWarmupCounters:
    """Tests for warmup counter metrics in AdaptiveServingProm.

    Requirements: 7.3, 7.4
    """

    def test_observe_warmup_counters_increments_executed(self, single_engine_labels):
        """Test that observe_warmup_counters increments the
        warmup_prefills_executed counter."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe_warmup_counters(
                prefills_executed=5,
                prefills_skipped_no_tokens=0,
            )
            prom.counter_warmup_prefills_executed[0].inc.assert_called_once_with(5)
            prom.counter_warmup_prefills_skipped_no_tokens[0].inc.assert_not_called()

    def test_observe_warmup_counters_increments_skipped(self, single_engine_labels):
        """Test that observe_warmup_counters increments the
        warmup_prefills_skipped_no_tokens counter."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe_warmup_counters(
                prefills_executed=0,
                prefills_skipped_no_tokens=3,
            )
            prom.counter_warmup_prefills_executed[0].inc.assert_not_called()
            prom.counter_warmup_prefills_skipped_no_tokens[
                0
            ].inc.assert_called_once_with(3)

    def test_observe_warmup_counters_both(self, single_engine_labels):
        """Test that both counters increment when both values are
        positive."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe_warmup_counters(
                prefills_executed=5,
                prefills_skipped_no_tokens=3,
            )
            prom.counter_warmup_prefills_executed[0].inc.assert_called_once_with(5)
            prom.counter_warmup_prefills_skipped_no_tokens[
                0
            ].inc.assert_called_once_with(3)

    def test_observe_warmup_counters_noop_when_disabled(self, single_engine_labels):
        """Test that observe_warmup_counters is a no-op when
        disabled."""
        prom = AdaptiveServingProm(enabled=False, **single_engine_labels)
        # Should not raise even though counter attrs don't exist
        prom.observe_warmup_counters(
            prefills_executed=5,
            prefills_skipped_no_tokens=3,
        )

    def test_observe_warmup_counters_zero_values_no_inc(self, single_engine_labels):
        """Test that counters are NOT incremented when values are
        zero."""
        with (
            patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge),
            patch.object(AdaptiveServingProm, "_counter_cls", FakeCounter),
        ):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            prom.observe_warmup_counters(
                prefills_executed=0,
                prefills_skipped_no_tokens=0,
            )
            prom.counter_warmup_prefills_executed[0].inc.assert_not_called()
            prom.counter_warmup_prefills_skipped_no_tokens[0].inc.assert_not_called()


class TestWarmupWorkerMetricsCounters:
    """Tests for PrefixWarmupWorker's internal metrics counters.

    Requirements: 7.3, 7.4
    Tests that warmup_prefills_executed increments on successful
    prefill and warmup_prefills_skipped_no_tokens increments on
    token registry miss.
    """

    def _make_worker_with_registry(self, block_size: int = 4):
        """Create a PrefixWarmupWorker with a token registry."""
        from vllm.v1.core.adaptive.prefix_frequency_tracker import (
            PrefixFrequencyTracker,
        )
        from vllm.v1.core.adaptive.prefix_warmup_worker import (
            PrefixWarmupWorker,
        )
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = MagicMock()
        block_pool.get_cached_block_hashes.return_value = set()
        block_pool.get_num_free_blocks.return_value = 80
        kv_manager = MagicMock()
        kv_manager.usage = 0.2
        kv_manager.block_pool = block_pool
        executor = MagicMock()
        executor.execute_warmup_prefill.return_value = [0, 1]
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
        return worker, tracker, registry, executor

    def test_prefills_executed_increments_on_success(self):
        """warmup_prefills_executed increments when prefill
        succeeds."""
        worker, tracker, registry, executor = self._make_worker_with_registry(
            block_size=4
        )

        # Register tokens for a candidate
        registry.register(10, [1, 2, 3, 4])

        # Build up EMA score above threshold
        for _ in range(20):
            tracker.update(10)

        assert worker.warmup_prefills_executed == 0

        # Use a tight VRAM budget so the loop stops after 1 prefill
        # floor(80 * ratio) — set budget to exactly 2 blocks
        worker._config.warmup_vram_budget_ratio = 0.025

        engine_core = MagicMock()
        worker.on_idle(engine_core)

        # Executor returns [0, 1] (2 blocks), budget is
        # floor(80*0.025)=2, so loop executes once then stops
        assert worker.warmup_prefills_executed == 1

    def test_skipped_no_tokens_increments_on_registry_miss(self):
        """warmup_prefills_skipped_no_tokens increments when token
        registry returns None for a candidate."""
        from vllm.v1.core.adaptive.prefix_frequency_tracker import (
            PrefixFrequencyTracker,
        )
        from vllm.v1.core.adaptive.prefix_warmup_worker import (
            PrefixWarmupWorker,
        )
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        config = AdaptiveServingConfig(
            adaptive_profile="dev",
            warmup_min_hit_count=0.3,
            warmup_budget_ms=5.0,  # very short budget
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        block_pool = MagicMock()
        block_pool.get_cached_block_hashes.return_value = set()
        block_pool.get_num_free_blocks.return_value = 80
        kv_manager = MagicMock()
        kv_manager.usage = 0.2
        kv_manager.block_pool = block_pool
        executor = MagicMock()
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

        # Hash 99 has NO tokens in registry — will cause a skip
        for _ in range(30):
            tracker.update(99)

        assert worker.warmup_prefills_skipped_no_tokens == 0

        engine_core = MagicMock()
        worker.on_idle(engine_core)

        # Hash 99 has no tokens in registry -> skipped
        assert worker.warmup_prefills_skipped_no_tokens >= 1
        # Executor should NOT have been called (no valid tokens)
        executor.execute_warmup_prefill.assert_not_called()


# ---------------------------------------------------------------------------
# AdaptiveServingLogging tests
# ---------------------------------------------------------------------------


class TestAdaptiveServingLogging:
    """Tests for periodic logging behavior."""

    def test_logs_after_60s_interval(self):
        """Test that AdaptiveServingLogging logs after 60s interval."""
        logging_inst = AdaptiveServingLogging()
        # Move the last log time back by 61 seconds
        logging_inst._last_log_time = time.monotonic() - 61.0

        with patch("vllm.v1.core.adaptive.metrics.logger") as mock_logger:
            logging_inst.maybe_log(
                total_entries_warmed=5,
                frequency_tracker_scores={1: 0.8, 2: 0.6, 3: 0.9},
                prefix_cache_hit_rate=45.0,
            )
            mock_logger.info.assert_called_once()

    def test_does_not_log_before_60s_interval(self):
        """Test that AdaptiveServingLogging does NOT log before 60s."""
        logging_inst = AdaptiveServingLogging()
        # Last log time is now (just initialized)

        with patch("vllm.v1.core.adaptive.metrics.logger") as mock_logger:
            logging_inst.maybe_log(
                total_entries_warmed=5,
                frequency_tracker_scores={1: 0.8},
                prefix_cache_hit_rate=45.0,
            )
            mock_logger.info.assert_not_called()

    def test_log_updates_tracking_state(self):
        """Test that logging updates _last_log_time and _last_entries_warmed."""
        logging_inst = AdaptiveServingLogging()
        logging_inst._last_log_time = time.monotonic() - 61.0
        logging_inst._last_entries_warmed = 3

        with patch("vllm.v1.core.adaptive.metrics.logger"):
            before_time = time.monotonic()
            logging_inst.maybe_log(
                total_entries_warmed=10,
                frequency_tracker_scores={1: 0.5},
                prefix_cache_hit_rate=30.0,
            )
            after_time = time.monotonic()

        assert logging_inst._last_entries_warmed == 10
        assert before_time <= logging_inst._last_log_time <= after_time
