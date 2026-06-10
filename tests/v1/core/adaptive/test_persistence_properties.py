# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for StatePersister.

# Feature: adaptive-speculative-serving
"""

import json
import math
import os
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.adaptive.state_persister import StatePersister

# --- Strategies ---

# EMA decay in the open interval (0, 1)
ema_decay_st = st.floats(min_value=0.01, max_value=0.99)

# Max entries for frequency tracker (reasonable bounds)
max_entries_st = st.integers(min_value=1, max_value=200)

# Prefix hashes: bounded range to get collisions sometimes
prefix_hash_st = st.integers(min_value=0, max_value=10000)

# Hit rate thresholds: min < activation, both in (0, 1)
threshold_pair_st = st.tuples(
    st.floats(min_value=0.1, max_value=0.89),
    st.floats(min_value=0.1, max_value=0.89),
).filter(lambda t: t[0] < t[1])

# Context pattern hashes
context_pattern_st = st.integers(min_value=0, max_value=5000)


class TestPersistenceRoundTrip:
    """Property 11: Persistence Round-Trip.

    For any valid PrefixFrequencyTracker state and ConfidenceTracker
    state, serializing via to_dict() and then deserializing via
    from_dict() SHALL produce a state equivalent to the original
    (within floating-point tolerance for EMA scores).

    **Validates: Requirements 10.3**
    """

    @settings(max_examples=100)
    @given(
        max_entries=max_entries_st,
        ema_decay=ema_decay_st,
        updates=st.lists(prefix_hash_st, min_size=0, max_size=50),
    )
    def test_prefix_frequency_tracker_round_trip(
        self,
        max_entries: int,
        ema_decay: float,
        updates: list[int],
    ):
        """Serialize then deserialize a PrefixFrequencyTracker.

        Verifies equivalent state within floating-point tolerance.

        # Feature: adaptive-speculative-serving, Property 11: Persistence Round-Trip
        """  # noqa: E501
        # Build tracker state by applying random updates
        tracker = PrefixFrequencyTracker(max_entries=max_entries, ema_decay=ema_decay)
        for h in updates:
            tracker.update(h)

        # Serialize
        data = tracker.to_dict()

        # Deserialize
        restored = PrefixFrequencyTracker.from_dict(
            data, max_entries=max_entries, ema_decay=ema_decay
        )

        # Verify same set of tracked hashes
        assert set(tracker._scores.keys()) == set(restored._scores.keys())

        # Verify EMA scores are approximately equal
        for h in tracker._scores:
            assert math.isclose(
                tracker._scores[h],
                restored._scores[h],
                rel_tol=1e-9,
                abs_tol=1e-12,
            ), (
                f"Score mismatch for hash {h}: "
                f"{tracker._scores[h]} vs {restored._scores[h]}"
            )

    @settings(max_examples=100)
    @given(
        ema_decay=ema_decay_st,
        thresholds=threshold_pair_st,
        observations=st.lists(
            st.tuples(context_pattern_st, st.booleans()),
            min_size=0,
            max_size=50,
        ),
    )
    def test_confidence_tracker_round_trip(
        self,
        ema_decay: float,
        thresholds: tuple[float, float],
        observations: list[tuple[int, bool]],
    ):
        """Serialize then deserialize a ConfidenceTracker.

        Verifies equivalent state within floating-point tolerance.

        # Feature: adaptive-speculative-serving, Property 11: Persistence Round-Trip
        """  # noqa: E501
        min_hit_rate, activation_hit_rate = thresholds

        # Build tracker state by applying random observations
        tracker = ConfidenceTracker(
            ema_decay=ema_decay,
            min_hit_rate=min_hit_rate,
            activation_hit_rate=activation_hit_rate,
        )
        for pattern, hit in observations:
            tracker.update(pattern, hit)

        # Serialize
        data = tracker.to_dict()

        # Deserialize
        restored = ConfidenceTracker.from_dict(
            data,
            ema_decay=ema_decay,
            min_hit_rate=min_hit_rate,
            activation_hit_rate=activation_hit_rate,
        )

        # Verify same set of tracked patterns
        assert set(tracker._hit_rates.keys()) == set(restored._hit_rates.keys())

        # Verify EMA hit rates are approximately equal
        for pattern in tracker._hit_rates:
            assert math.isclose(
                tracker._hit_rates[pattern],
                restored._hit_rates[pattern],
                rel_tol=1e-9,
                abs_tol=1e-12,
            ), (
                f"Hit rate mismatch for pattern {pattern}: "
                f"{tracker._hit_rates[pattern]} vs "
                f"{restored._hit_rates[pattern]}"
            )

        # Verify same enabled/disabled states
        assert tracker._enabled == restored._enabled

    @settings(max_examples=100)
    @given(
        max_entries=max_entries_st,
        freq_decay=ema_decay_st,
        conf_decay=ema_decay_st,
        thresholds=threshold_pair_st,
        freq_updates=st.lists(prefix_hash_st, min_size=0, max_size=30),
        conf_observations=st.lists(
            st.tuples(context_pattern_st, st.booleans()),
            min_size=0,
            max_size=30,
        ),
    )
    def test_full_persistence_round_trip_via_state_persister(
        self,
        max_entries: int,
        freq_decay: float,
        conf_decay: float,
        thresholds: tuple[float, float],
        freq_updates: list[int],
        conf_observations: list[tuple[int, bool]],
    ):
        """End-to-end: save via StatePersister, load, reconstruct.

        # Feature: adaptive-speculative-serving, Property 11: Persistence Round-Trip
        """  # noqa: E501
        min_hit_rate, activation_hit_rate = thresholds

        # Build frequency tracker state
        freq_tracker = PrefixFrequencyTracker(
            max_entries=max_entries, ema_decay=freq_decay
        )
        for h in freq_updates:
            freq_tracker.update(h)

        # Build confidence tracker state
        conf_tracker = ConfidenceTracker(
            ema_decay=conf_decay,
            min_hit_rate=min_hit_rate,
            activation_hit_rate=activation_hit_rate,
        )
        for pattern, hit in conf_observations:
            conf_tracker.update(pattern, hit)

        # Persist via StatePersister
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            persister = StatePersister(path=path, interval_seconds=60.0)
            persister.save(freq_tracker, conf_tracker)

            # Load back
            freq_data, conf_data, _registry_data = persister.load()

        # Verify data was loaded successfully
        assert freq_data is not None
        assert conf_data is not None

        # Reconstruct trackers from loaded data
        restored_freq = PrefixFrequencyTracker.from_dict(
            freq_data,
            max_entries=max_entries,
            ema_decay=freq_decay,
        )
        restored_conf = ConfidenceTracker.from_dict(
            conf_data,
            ema_decay=conf_decay,
            min_hit_rate=min_hit_rate,
            activation_hit_rate=activation_hit_rate,
        )

        # Verify frequency tracker equivalence
        assert set(freq_tracker._scores.keys()) == set(restored_freq._scores.keys())
        for h in freq_tracker._scores:
            assert math.isclose(
                freq_tracker._scores[h],
                restored_freq._scores[h],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )

        # Verify confidence tracker equivalence
        assert set(conf_tracker._hit_rates.keys()) == set(
            restored_conf._hit_rates.keys()
        )
        for pattern in conf_tracker._hit_rates:
            assert math.isclose(
                conf_tracker._hit_rates[pattern],
                restored_conf._hit_rates[pattern],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        assert conf_tracker._enabled == restored_conf._enabled


# --- Strategies for Property 12 ---


@st.composite
def truncated_json_strategy(draw):
    """Generate valid JSON truncated at a random position."""
    keys = draw(
        st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=5,
        )
    )
    values = draw(
        st.lists(
            st.one_of(
                st.integers(),
                st.floats(allow_nan=False),
                st.text(),
            ),
            min_size=len(keys),
            max_size=len(keys),
        )
    )
    obj = dict(zip(keys, values))
    full_json = json.dumps(obj)
    # Truncate at a random position (at least 1 char, < full)
    if len(full_json) <= 1:
        return full_json[:0]
    cut = draw(st.integers(min_value=1, max_value=len(full_json) - 1))
    return full_json[:cut]


@st.composite
def wrong_schema_version_strategy(draw):
    """Generate valid JSON with a schema_version != 1."""
    version = draw(st.integers().filter(lambda v: v != StatePersister.SCHEMA_VERSION))
    obj = {
        "schema_version": version,
        "timestamp": "2024-01-15T10:30:00+00:00",
        "prefix_frequencies": draw(
            st.dictionaries(st.text(min_size=1, max_size=5), st.floats())
        ),
        "confidence_thresholds": draw(
            st.dictionaries(st.text(min_size=1, max_size=5), st.floats())
        ),
    }
    return json.dumps(obj)


@st.composite
def missing_required_fields_strategy(draw):
    """Generate valid JSON with schema_version=1 but missing fields."""
    obj: dict = {"schema_version": StatePersister.SCHEMA_VERSION}
    if draw(st.booleans()):
        obj["timestamp"] = "2024-01-15T10:30:00+00:00"
    # Include at most one of the two required fields
    include_freq = draw(st.booleans())
    include_conf = draw(st.booleans())
    # Ensure at least one is missing
    if include_freq and include_conf:
        if draw(st.booleans()):
            include_freq = False
        else:
            include_conf = False
    if include_freq:
        obj["prefix_frequencies"] = {"h": {"ema_score": 0.5}}
    if include_conf:
        obj["confidence_thresholds"] = {"p": {"hit_rate_ema": 0.7}}
    return json.dumps(obj)


class TestCorruptedDataGracefulFallback:
    """Property 12: Corrupted Data Graceful Fallback.

    For any byte sequence that is not valid persisted state
    (truncated data, invalid JSON, missing required fields,
    wrong schema version), StatePersister.load() SHALL return
    fresh/empty state without raising an exception.

    **Validates: Requirements 10.4**
    """

    @given(data=st.binary(min_size=1, max_size=1024))
    @settings(max_examples=100)
    def test_random_bytes_returns_none_none(self, data: bytes):
        """Random byte sequences never cause load() to raise.

        # Feature: adaptive-speculative-serving, Property 12: Corrupted Data Graceful Fallback
        """  # noqa: E501
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "wb") as f:
                f.write(data)

            persister = StatePersister(path=path, interval_seconds=60.0)
            result = persister.load()
            assert result == (None, None, None)

    @given(truncated=truncated_json_strategy())
    @settings(max_examples=100)
    def test_truncated_json_returns_none_none(self, truncated: str):
        """Truncated JSON never causes load() to raise.

        # Feature: adaptive-speculative-serving, Property 12: Corrupted Data Graceful Fallback
        """  # noqa: E501
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "w") as f:
                f.write(truncated)

            persister = StatePersister(path=path, interval_seconds=60.0)
            result = persister.load()
            assert result == (None, None, None)

    @given(content=wrong_schema_version_strategy())
    @settings(max_examples=100)
    def test_wrong_schema_version_returns_none_none(self, content: str):
        """Wrong schema version never causes load() to raise.

        # Feature: adaptive-speculative-serving, Property 12: Corrupted Data Graceful Fallback
        """  # noqa: E501
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "w") as f:
                f.write(content)

            persister = StatePersister(path=path, interval_seconds=60.0)
            result = persister.load()
            assert result == (None, None, None)

    @given(content=missing_required_fields_strategy())
    @settings(max_examples=100)
    def test_missing_required_fields_returns_none_none(self, content: str):
        """Missing required fields never cause load() to raise.

        # Feature: adaptive-speculative-serving, Property 12: Corrupted Data Graceful Fallback
        """  # noqa: E501
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "w") as f:
                f.write(content)

            persister = StatePersister(path=path, interval_seconds=60.0)
            result = persister.load()
            assert result == (None, None, None)
