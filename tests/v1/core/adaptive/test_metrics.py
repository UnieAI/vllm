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
        with patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge):
            prom = AdaptiveServingProm(enabled=True, **single_engine_labels)
            assert prom.enabled is True
            # All 4 metric dicts should exist
            assert hasattr(prom, "gauge_prefix_cache_hit_rate")
            assert hasattr(prom, "gauge_prefix_warmup_entries_count")
            assert hasattr(prom, "gauge_self_speculation_skip_rate")
            assert hasattr(prom, "gauge_confidence_tracker_mean_threshold")

    def test_observe_updates_prefix_cache_hit_rate(self, single_engine_labels):
        """Test that observe() updates prefix_cache_hit_rate gauge."""
        with patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge):
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
        with patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge):
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
        with patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge):
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
        with patch.object(AdaptiveServingProm, "_gauge_cls", FakeGauge):
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
        """Test that all 4 metrics have correct vllm: prefixed names."""
        created_gauges: list[FakeGauge] = []

        class TrackingGauge(FakeGauge):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_gauges.append(self)

        with patch.object(AdaptiveServingProm, "_gauge_cls", TrackingGauge):
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
