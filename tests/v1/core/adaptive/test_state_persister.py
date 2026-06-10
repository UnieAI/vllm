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
        assert result == (None, None, None)

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
        assert result == (freq_data, conf_data, None)

    def test_load_returns_none_on_corrupted_json(self, tmp_path_file):
        with open(tmp_path_file, "w") as f:
            f.write("not valid json {{{")

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None, None)

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
        assert result == (None, None, None)

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
        assert result == (None, None, None)

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
        assert result == (None, None, None)

    def test_load_returns_none_on_non_dict_root(self, tmp_path_file):
        # Top-level should be a dict, not a list
        with open(tmp_path_file, "w") as f:
            json.dump([1, 2, 3], f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None, None)

    def test_load_returns_none_on_empty_file(self, tmp_path_file):
        with open(tmp_path_file, "w") as f:
            f.write("")

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        result = persister.load()
        assert result == (None, None, None)


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
        loaded_freq, loaded_conf, loaded_registry = persister.load()

        assert loaded_freq == freq_data
        assert loaded_conf == conf_data
        assert loaded_registry is None

    def test_save_overwrites_previous_state(self, tmp_path_file):
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)

        # First save
        persister.save(FakeTracker({"old": 1}), FakeTracker({"old": 2}))

        # Second save
        new_freq = {"new": 99}
        new_conf = {"new": 88}
        persister.save(FakeTracker(new_freq), FakeTracker(new_conf))

        loaded_freq, loaded_conf, _loaded_registry = persister.load()
        assert loaded_freq == new_freq
        assert loaded_conf == new_conf


class TestTokenRegistryPersistence:
    """Tests for TokenRegistry save/load integration with StatePersister.

    Validates: Requirements 2.6
    """

    def test_save_includes_token_registry_key(self, tmp_path_file):
        """Verify save() includes token_registry key in JSON when provided."""
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({"h1": {"ema_score": 0.5}})
        conf = FakeTracker({"p1": {"hit_rate_ema": 0.6}})
        registry = FakeTracker({"entries": {"123": [1, 2, 3, 4]}, "block_size": 4})

        persister.save(freq, conf, token_registry=registry)

        with open(tmp_path_file) as f:
            data = json.load(f)
        assert "token_registry" in data
        assert data["token_registry"]["block_size"] == 4
        assert data["token_registry"]["entries"] == {"123": [1, 2, 3, 4]}

    def test_save_without_token_registry_omits_key(self, tmp_path_file):
        """Verify save() omits token_registry key when not provided."""
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({"h1": {"ema_score": 0.5}})
        conf = FakeTracker({"p1": {"hit_rate_ema": 0.6}})

        persister.save(freq, conf)

        with open(tmp_path_file) as f:
            data = json.load(f)
        assert "token_registry" not in data

    def test_round_trip_with_token_registry(self, tmp_path_file):
        """Save TokenRegistry data, load back, verify identical content."""
        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq_data = {"hash_a": {"ema_score": 0.9}}
        conf_data = {"pat_1": {"hit_rate_ema": 0.7}}
        registry_data = {
            "entries": {
                "111": [10, 20, 30, 40],
                "222": [50, 60, 70, 80],
                "333": [90, 100, 110, 120],
            },
            "block_size": 4,
        }
        freq = FakeTracker(freq_data)
        conf = FakeTracker(conf_data)
        registry = FakeTracker(registry_data)

        persister.save(freq, conf, token_registry=registry)
        loaded_freq, loaded_conf, loaded_registry = persister.load()

        assert loaded_freq == freq_data
        assert loaded_conf == conf_data
        assert loaded_registry == registry_data

    def test_round_trip_reconstruct_token_registry(self, tmp_path_file):
        """Full round-trip: create TokenRegistry, save, load, reconstruct."""
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        # Create a registry with entries
        original = TokenRegistry(max_entries=10, block_size=4)
        original.register(100, [1, 2, 3, 4])
        original.register(200, [5, 6, 7, 8])
        original.register(300, [9, 10, 11, 12])

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq = FakeTracker({"f": 1})
        conf = FakeTracker({"c": 2})

        persister.save(freq, conf, token_registry=original)
        _, _, loaded_registry_data = persister.load()

        assert loaded_registry_data is not None

        # Reconstruct from loaded data
        restored = TokenRegistry.from_dict(
            loaded_registry_data, max_entries=10, block_size=4
        )

        # Verify all original entries are present
        assert restored.get_tokens(100) == [1, 2, 3, 4]
        assert restored.get_tokens(200) == [5, 6, 7, 8]
        assert restored.get_tokens(300) == [9, 10, 11, 12]
        assert len(restored) == 3

    def test_load_returns_none_registry_when_key_missing(self, tmp_path_file):
        """Backward compat: missing token_registry key returns None."""
        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {"h": {"ema": 0.5}},
            "confidence_thresholds": {"p": {"hr": 0.6}},
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq, conf, registry_data = persister.load()

        assert freq is not None
        assert conf is not None
        assert registry_data is None

    def test_load_returns_none_registry_on_invalid_type(self, tmp_path_file):
        """Invalid token_registry type (not dict) returns None for it."""
        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {"h": {"ema": 0.5}},
            "confidence_thresholds": {"p": {"hr": 0.6}},
            "token_registry": "not a dict",
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq, conf, registry_data = persister.load()

        assert freq is not None
        assert conf is not None
        assert registry_data is None

    def test_load_returns_none_registry_on_list_type(self, tmp_path_file):
        """Invalid token_registry type (list) returns None for it."""
        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {"h": {"ema": 0.5}},
            "confidence_thresholds": {"p": {"hr": 0.6}},
            "token_registry": [1, 2, 3],
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        freq, conf, registry_data = persister.load()

        assert freq is not None
        assert conf is not None
        assert registry_data is None

    def test_load_corrupted_registry_entries_not_dict(self, tmp_path_file):
        """token_registry with entries as non-dict is handled by
        TokenRegistry.from_dict gracefully."""
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {"h": {"ema": 0.5}},
            "confidence_thresholds": {"p": {"hr": 0.6}},
            "token_registry": {"entries": "not_a_dict", "block_size": 4},
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        _, _, registry_data = persister.load()

        # StatePersister returns the dict (it's a valid dict shape)
        assert registry_data is not None
        # But TokenRegistry.from_dict handles malformed entries gracefully
        restored = TokenRegistry.from_dict(registry_data, max_entries=10, block_size=4)
        assert len(restored) == 0

    def test_load_corrupted_registry_malformed_entry_values(self, tmp_path_file):
        """token_registry with malformed entry values is handled
        gracefully by TokenRegistry.from_dict."""
        from vllm.v1.core.adaptive.token_registry import TokenRegistry

        state = {
            "schema_version": 1,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "prefix_frequencies": {"h": {"ema": 0.5}},
            "confidence_thresholds": {"p": {"hr": 0.6}},
            "token_registry": {
                "entries": {
                    "abc": [1, 2, 3, 4],  # key not parseable as int
                    "999": "not_a_list",  # value not a list
                },
                "block_size": 4,
            },
        }
        with open(tmp_path_file, "w") as f:
            json.dump(state, f)

        persister = StatePersister(path=tmp_path_file, interval_seconds=60.0)
        _, _, registry_data = persister.load()

        assert registry_data is not None
        # from_dict should skip malformed entries without crashing
        restored = TokenRegistry.from_dict(registry_data, max_entries=10, block_size=4)
        # "abc" can't be converted to int -> skipped (actually it can't)
        # "999" has "not_a_list" -> TypeError during iteration -> skipped
        # But wait, "abc" CAN'T be parsed as int, so it's skipped.
        assert len(restored) == 0
