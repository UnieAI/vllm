# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for StatePersister."""

import json
import os
import tempfile

import pytest

from vllm.v1.core.adaptive.state_persister import StatePersister


class FakeTracker:
    """A fake tracker that implements the _Serializable protocol."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


@pytest.fixture
def tmp_path_file():
    """Create a temp file path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "state.json")


class TestStatePersisterSave:
    def test_save_creates_file(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({"hash1": {"ema_score": 0.9}})
        conf = FakeTracker({"pattern1": {"hit_rate_ema": 0.8}})

        persister.save(freq, conf)

        assert os.path.exists(tmp_path_file)

    def test_save_includes_schema_version(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({"hash1": {"ema_score": 0.9}})
        conf = FakeTracker({"pattern1": {"hit_rate_ema": 0.8}})

        persister.save(freq, conf)

        with open(tmp_path_file) as f:
            data = json.load(f)
        assert data["schema_version"] == 1

    def test_save_includes_timestamp(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({})
        conf = FakeTracker({})

        persister.save(freq, conf)

        with open(tmp_path_file) as f:
            data = json.load(f)
        assert "timestamp" in data
        # Verify it's a valid ISO format timestamp
        assert "T" in data["timestamp"]

    def test_save_includes_tracker_data(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq_data = {"hash_abc": {"ema_score": 0.85}}
        conf_data = {"pattern_xyz": {"hit_rate_ema": 0.72}}
        freq = FakeTracker(freq_data)
        conf = FakeTracker(conf_data)

        persister.save(freq, conf)

        with open(tmp_path_file) as f:
            data = json.load(f)
        assert data["prefix_frequencies"] == freq_data
        assert data["confidence_thresholds"] == conf_data

    def test_save_atomic_no_partial_file(self, tmp_path_file):
        """Verify that the temp file is cleaned up after save."""
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({"a": 1})
        conf = FakeTracker({"b": 2})

        persister.save(freq, conf)

        # No .tmp file should remain
        parent = os.path.dirname(tmp_path_file)
        files = os.listdir(parent)
        tmp_files = [f for f in files if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_save_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir", "state.json")
            persister = StatePersister(path=nested, interval_seconds=60.0)
            freq = FakeTracker({"x": 1})
            conf = FakeTracker({"y": 2})

            persister.save(freq, conf)

            assert os.path.exists(nested)


class TestStatePersisterLoad:
    def test_load_returns_none_when_file_missing(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)

    def test_load_returns_data_on_valid_file(self, tmp_path_file):
        freq_data = {"hash1": {"ema_score": 0.9}}
        conf_data = {"pattern1": {"hit_rate_ema": 0.75}}
        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": freq_data,
            "confidence_thresholds": conf_data,
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (freq_data, conf_data)

    def test_load_returns_none_on_corrupted_json(self, tmp_path_file):
        with open(tmp_path_file, "w") as f:
            f.write("not valid json {{{")

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)

    def test_load_returns_none_on_wrong_schema_version(self, tmp_path_file):
        state = {
            "schema_version": 999,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {},
            "confidence_thresholds": {},
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)

    def test_load_returns_none_on_missing_fields(self, tmp_path_file):
        # Missing confidence_thresholds
        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {"a": 1},
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)

    def test_load_returns_none_on_invalid_field_types(self, tmp_path_file):
        # prefix_frequencies should be a dict, not a list
        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": ["not", "a", "dict"],
            "confidence_thresholds": {"ok": True},
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)

    def test_load_returns_none_on_non_dict_root(self, tmp_path_file):
        # Top-level should be a dict, not a list
        with open(tmp_path_file, "w") as f:
            json.dump([1, 2, 3], f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)

    def test_load_returns_none_on_empty_file(self, tmp_path_file):
        with open(tmp_path_file, "w") as f:
            f.write("")

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None)


class TestStatePersisterRoundTrip:
    def test_save_then_load_round_trip(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=300.0)
        freq_data = {
            "hash_a": {"ema_score": 0.95, "last_update_step": 100},
            "hash_b": {"ema_score": 0.42, "last_update_step": 50},
        }
        conf_data = {
            "pattern_1": {
                "hit_rate_ema": 0.78,
                "speculation_enabled": True,
                "total_observations": 500,
            },
        }
        freq = FakeTracker(freq_data)
        conf = FakeTracker(conf_data)

        persister.save(freq, conf)
        loaded_freq, loaded_conf = persister.load()

        assert loaded_freq == freq_data
        assert loaded_conf == conf_data

    def test_save_overwrites_previous_state(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)

        # First save
        persister.save(FakeTracker({"old": 1}), FakeTracker({"old": 2}))

        # Second save
        new_freq = {"new": 99}
        new_conf = {"new": 88}
        persister.save(FakeTracker(new_freq), FakeTracker(new_conf))

        loaded_freq, loaded_conf = persister.load()
        assert loaded_freq == new_freq
        assert loaded_conf == new_conf
