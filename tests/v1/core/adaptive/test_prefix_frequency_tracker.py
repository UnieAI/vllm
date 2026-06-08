# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PrefixFrequencyTracker."""

import pytest

from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)


class TestPrefixFrequencyTrackerInit:
    """Tests for constructor validation."""

    def test_valid_construction(self):
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        assert tracker.max_entries == 100
        assert tracker.ema_decay == 0.8
        assert len(tracker) == 0

    def test_invalid_max_entries_zero(self):
        with pytest.raises(ValueError, match="max_entries"):
            PrefixFrequencyTracker(max_entries=0, ema_decay=0.8)

    def test_invalid_max_entries_negative(self):
        with pytest.raises(ValueError, match="max_entries"):
            PrefixFrequencyTracker(max_entries=-1, ema_decay=0.8)

    def test_invalid_ema_decay_zero(self):
        with pytest.raises(ValueError, match="ema_decay"):
            PrefixFrequencyTracker(max_entries=10, ema_decay=0.0)

    def test_invalid_ema_decay_one(self):
        with pytest.raises(ValueError, match="ema_decay"):
            PrefixFrequencyTracker(max_entries=10, ema_decay=1.0)

    def test_invalid_ema_decay_negative(self):
        with pytest.raises(ValueError, match="ema_decay"):
            PrefixFrequencyTracker(max_entries=10, ema_decay=-0.5)


class TestUpdate:
    """Tests for the update method."""

    def test_first_observation(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        tracker.update(42)
        assert len(tracker) == 1
        # First observation: score = 1 - decay = 0.2
        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
        assert len(candidates) == 1
        assert candidates[0] == (42, pytest.approx(0.2))

    def test_second_observation_ema(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        tracker.update(42)
        tracker.update(42)
        # After second update: score = 0.8 * 0.2 + 0.2 = 0.36
        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
        assert candidates[0][1] == pytest.approx(0.36)

    def test_multiple_updates_ema_formula(self):
        decay = 0.9
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=decay)
        # Apply 5 updates
        expected = 0.0
        for _ in range(5):
            tracker.update(1)
            expected = decay * expected + (1.0 - decay)

        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
        assert candidates[0][1] == pytest.approx(expected)

    def test_eviction_at_capacity(self):
        tracker = PrefixFrequencyTracker(max_entries=3, ema_decay=0.8)
        # Insert 3 entries with different scores
        tracker.update(1)  # score = 0.2
        tracker.update(2)  # score = 0.2
        tracker.update(2)  # score = 0.36
        tracker.update(3)  # score = 0.2
        tracker.update(3)  # score = 0.36
        tracker.update(3)  # score = 0.488

        assert len(tracker) == 3

        # Now insert a 4th - should evict the lowest (hash=1, 0.2)
        tracker.update(4)  # score = 0.2
        assert len(tracker) == 3
        # Hash 1 should be evicted (it had the lowest score 0.2)
        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
        hashes = {h for h, _ in candidates}
        assert 1 not in hashes
        # But 2, 3, 4 should remain
        assert {2, 3, 4} == hashes


class TestGetWarmupCandidates:
    """Tests for warmup candidate selection."""

    def test_excludes_cached_hashes(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        tracker.update(1)
        tracker.update(2)
        tracker.update(3)
        candidates = tracker.get_warmup_candidates(cached_hashes={1, 3}, min_score=0.0)
        assert len(candidates) == 1
        assert candidates[0][0] == 2

    def test_filters_by_min_score(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        tracker.update(1)  # score = 0.2
        tracker.update(2)
        tracker.update(2)  # score = 0.36
        tracker.update(3)
        tracker.update(3)
        tracker.update(3)  # score = 0.488

        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.35)
        # Only entries with score >= 0.35 (hashes 2 and 3)
        assert len(candidates) == 2
        assert candidates[0][0] == 3  # highest score first
        assert candidates[1][0] == 2

    def test_sorted_descending_by_score(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        # Create entries with varying scores
        for _ in range(5):
            tracker.update(100)
        for _ in range(3):
            tracker.update(200)
        for _ in range(1):
            tracker.update(300)

        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
        scores = [s for _, s in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_empty_tracker_returns_empty(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
        assert candidates == []


class TestSerialization:
    """Tests for to_dict / from_dict round-trip."""

    def test_round_trip(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        tracker.update(1)
        tracker.update(2)
        tracker.update(2)
        tracker.update(3)

        data = tracker.to_dict()
        restored = PrefixFrequencyTracker.from_dict(data, max_entries=10, ema_decay=0.8)

        # Verify same scores
        orig_candidates = tracker.get_warmup_candidates(
            cached_hashes=set(), min_score=0.0
        )
        restored_candidates = restored.get_warmup_candidates(
            cached_hashes=set(), min_score=0.0
        )
        assert len(orig_candidates) == len(restored_candidates)
        for (h1, s1), (h2, s2) in zip(orig_candidates, restored_candidates):
            assert h1 == h2
            assert s1 == pytest.approx(s2)

    def test_from_dict_empty_data(self):
        tracker = PrefixFrequencyTracker.from_dict({}, max_entries=10, ema_decay=0.8)
        assert len(tracker) == 0

    def test_from_dict_trims_to_max_entries(self):
        # Create data with more entries than new max
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        for i in range(50):
            tracker.update(i)

        data = tracker.to_dict()
        # Restore with smaller max_entries
        restored = PrefixFrequencyTracker.from_dict(data, max_entries=10, ema_decay=0.8)
        assert len(restored) == 10

    def test_to_dict_format(self):
        tracker = PrefixFrequencyTracker(max_entries=10, ema_decay=0.8)
        tracker.update(42)
        data = tracker.to_dict()
        assert "scores" in data
        assert "42" in data["scores"]
        assert isinstance(data["scores"]["42"], float)
