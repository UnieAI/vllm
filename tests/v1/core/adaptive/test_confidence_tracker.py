# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ConfidenceTracker."""

import pytest

from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker


class TestConfidenceTrackerInit:
    """Tests for ConfidenceTracker initialization."""

    def test_default_construction(self):
        tracker = ConfidenceTracker(
            ema_decay=0.95,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        assert tracker._ema_decay == 0.95
        assert tracker._min_hit_rate == 0.5
        assert tracker._activation_hit_rate == 0.7
        assert tracker._hit_rates == {}
        assert tracker._enabled == {}

    def test_mean_threshold_empty(self):
        tracker = ConfidenceTracker(
            ema_decay=0.95,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        assert tracker.mean_threshold() == 0.0


class TestConfidenceTrackerUpdate:
    """Tests for the update method and EMA computation."""

    def test_first_hit_initializes_rate_to_one(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        tracker.update(42, hit=True)
        assert tracker._hit_rates[42] == 1.0

    def test_first_miss_initializes_rate_to_zero(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        tracker.update(42, hit=False)
        assert tracker._hit_rates[42] == 0.0

    def test_ema_formula_applied_correctly(self):
        tracker = ConfidenceTracker(
            ema_decay=0.8,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        # First update: rate = 1.0 (hit)
        tracker.update(1, hit=True)
        assert tracker._hit_rates[1] == 1.0

        # Second update: rate = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        tracker.update(1, hit=False)
        assert tracker._hit_rates[1] == pytest.approx(0.8)

        # Third update: rate = 0.8 * 0.8 + 0.2 * 1.0 = 0.84
        tracker.update(1, hit=True)
        assert tracker._hit_rates[1] == pytest.approx(0.84)


class TestConfidenceTrackerHysteresis:
    """Tests for hysteresis enable/disable logic."""

    def test_new_pattern_starts_disabled(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        tracker.update(10, hit=False)
        assert tracker.should_speculate(10) is False

    def test_unknown_pattern_returns_false(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        assert tracker.should_speculate(999) is False

    def test_enables_at_activation_threshold(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        # First hit: rate = 1.0, which is >= 0.7 -> enabled
        tracker.update(1, hit=True)
        assert tracker.should_speculate(1) is True

    def test_disables_below_min_hit_rate(self):
        tracker = ConfidenceTracker(
            ema_decay=0.5,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        # First hit: rate = 1.0 -> enabled
        tracker.update(1, hit=True)
        assert tracker.should_speculate(1) is True

        # rate = 0.5 * 1.0 + 0.5 * 0.0 = 0.5 (still >= 0.5)
        tracker.update(1, hit=False)
        # 0.5 is NOT < 0.5, so it stays in the hysteresis band
        # Since it was enabled, it should stay enabled
        assert tracker.should_speculate(1) is True

        # rate = 0.5 * 0.5 + 0.5 * 0.0 = 0.25 (< 0.5)
        tracker.update(1, hit=False)
        assert tracker.should_speculate(1) is False

    def test_hysteresis_band_retains_state(self):
        """Between min and activation, state doesn't change."""
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.3,
            activation_hit_rate=0.8,
        )
        # First hit: rate = 1.0 -> enabled (>= 0.8)
        tracker.update(1, hit=True)
        assert tracker.should_speculate(1) is True

        # rate = 0.9 * 1.0 + 0.1 * 0.0 = 0.9 -> still enabled
        tracker.update(1, hit=False)
        assert tracker.should_speculate(1) is True

        # rate = 0.9 * 0.9 + 0.1 * 0.0 = 0.81 -> still enabled
        tracker.update(1, hit=False)
        assert tracker.should_speculate(1) is True

        # rate = 0.9 * 0.81 + 0.1 * 0.0 = 0.729 -> in band
        tracker.update(1, hit=False)
        # 0.729 is >= 0.3 and < 0.8, hysteresis retains enabled
        assert tracker.should_speculate(1) is True

    def test_hysteresis_band_retains_disabled_state(self):
        """Between min and activation, disabled stays disabled."""
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.3,
            activation_hit_rate=0.8,
        )
        # First miss: rate = 0.0 -> disabled (< 0.3)
        tracker.update(1, hit=False)
        assert tracker.should_speculate(1) is False

        # rate = 0.9 * 0.0 + 0.1 * 1.0 = 0.1 -> disabled
        tracker.update(1, hit=True)
        assert tracker.should_speculate(1) is False

        # rate = 0.9 * 0.1 + 0.1 * 1.0 = 0.19 -> disabled
        tracker.update(1, hit=True)
        assert tracker.should_speculate(1) is False

        # rate = 0.9 * 0.19 + 0.1 * 1.0 = 0.271 -> disabled
        tracker.update(1, hit=True)
        assert tracker.should_speculate(1) is False

        # rate = 0.9 * 0.271 + 0.1 * 1.0 = 0.3439 -> band
        tracker.update(1, hit=True)
        # In band, retains disabled
        assert tracker.should_speculate(1) is False


class TestConfidenceTrackerMeanThreshold:
    """Tests for mean_threshold method."""

    def test_single_pattern(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        tracker.update(1, hit=True)
        assert tracker.mean_threshold() == 1.0

    def test_multiple_patterns(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        tracker.update(1, hit=True)  # rate = 1.0
        tracker.update(2, hit=False)  # rate = 0.0
        assert tracker.mean_threshold() == pytest.approx(0.5)


class TestConfidenceTrackerPersistence:
    """Tests for to_dict and from_dict."""

    def test_round_trip(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.4,
            activation_hit_rate=0.8,
        )
        tracker.update(100, hit=True)
        tracker.update(200, hit=False)
        tracker.update(100, hit=True)

        data = tracker.to_dict()
        restored = ConfidenceTracker.from_dict(data)

        assert restored._ema_decay == tracker._ema_decay
        assert restored._min_hit_rate == tracker._min_hit_rate
        assert restored._activation_hit_rate == tracker._activation_hit_rate
        assert restored._hit_rates == pytest.approx(tracker._hit_rates)
        assert restored._enabled == tracker._enabled

    def test_from_dict_with_overrides(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.4,
            activation_hit_rate=0.8,
        )
        tracker.update(100, hit=True)

        data = tracker.to_dict()
        restored = ConfidenceTracker.from_dict(
            data,
            ema_decay=0.95,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )

        # Config overridden
        assert restored._ema_decay == 0.95
        assert restored._min_hit_rate == 0.5
        assert restored._activation_hit_rate == 0.7
        # Learned state preserved
        assert restored._hit_rates[100] == tracker._hit_rates[100]
        assert restored._enabled[100] == tracker._enabled[100]

    def test_from_dict_empty_entries(self):
        data = {
            "ema_decay": 0.9,
            "min_hit_rate": 0.5,
            "activation_hit_rate": 0.7,
            "entries": {},
        }
        tracker = ConfidenceTracker.from_dict(data)
        assert tracker._hit_rates == {}
        assert tracker._enabled == {}
        assert tracker.mean_threshold() == 0.0

    def test_to_dict_structure(self):
        tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        tracker.update(42, hit=True)

        data = tracker.to_dict()
        assert data["ema_decay"] == 0.9
        assert data["min_hit_rate"] == 0.5
        assert data["activation_hit_rate"] == 0.7
        assert "42" in data["entries"]
        assert data["entries"]["42"]["hit_rate_ema"] == 1.0
        assert data["entries"]["42"]["speculation_enabled"] is True
