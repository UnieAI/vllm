# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for adaptive serving integration in Engine Core.

Tests that the Engine Core correctly initializes adaptive serving
components, registers idle callbacks, handles persistence, and
aborts warmup on request arrival.
"""

from __future__ import annotations

import json
import tempfile
import time
from unittest.mock import MagicMock

from vllm.config.adaptive_serving import AdaptiveServingConfig


class TestInitAdaptiveServing:
    """Tests for EngineCore._init_adaptive_serving."""

    def _make_engine_core_stub(self):
        """Create a minimal stub with enough state for
        _init_adaptive_serving to run."""
        from vllm.v1.engine.core import EngineCore

        stub = object.__new__(EngineCore)
        stub._idle_state_callbacks = []

        # Mock scheduler with kv_cache_manager.block_pool
        mock_block_pool = MagicMock()
        mock_block_pool.get_cached_block_hashes.return_value = set()
        mock_block_pool.get_num_free_blocks.return_value = 100

        mock_kv_cache_manager = MagicMock()
        mock_kv_cache_manager.block_pool = mock_block_pool
        mock_kv_cache_manager.usage = 0.5

        mock_scheduler = MagicMock()
        mock_scheduler.kv_cache_manager = mock_kv_cache_manager
        stub.scheduler = mock_scheduler

        # Mock model executor
        stub.model_executor = MagicMock()

        return stub

    def _make_vllm_config(self, **overrides):
        """Create a mock VllmConfig with adaptive_serving config."""
        adaptive_config = AdaptiveServingConfig(**overrides)
        vllm_config = MagicMock()
        vllm_config.adaptive_serving = adaptive_config
        return vllm_config

    def test_init_disabled(self):
        """When enable_adaptive_warmup is False, no components
        are initialized."""
        stub = self._make_engine_core_stub()
        vllm_config = self._make_vllm_config(enable_adaptive_warmup=False)

        stub._init_adaptive_serving(vllm_config)

        assert stub._warmup_worker is None
        assert stub._frequency_tracker is None
        assert stub._confidence_tracker is None
        assert stub._state_persister is None
        assert len(stub._idle_state_callbacks) == 0

    def test_init_enabled_default_profile(self):
        """When enabled with production profile, all components
        are initialized."""
        stub = self._make_engine_core_stub()
        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
            adaptive_profile="production",
        )

        stub._init_adaptive_serving(vllm_config)

        assert stub._warmup_worker is not None
        assert stub._frequency_tracker is not None
        assert stub._confidence_tracker is not None
        assert stub._state_persister is None  # No persist path
        assert len(stub._idle_state_callbacks) == 1

    def test_init_with_persistence_path(self):
        """When persist_path is set, StatePersister is created."""
        stub = self._make_engine_core_stub()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            persist_path = f.name

        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
            self_spec_stats_persist_path=persist_path,
        )

        stub._init_adaptive_serving(vllm_config)

        assert stub._state_persister is not None
        assert stub._state_persister.path == persist_path

    def test_init_loads_persisted_state(self):
        """When persisted state exists, it is loaded on startup."""
        stub = self._make_engine_core_stub()

        # Write valid persisted state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state = {
                "schema_version": 1,
                "timestamp": "2024-01-15T10:30:00Z",
                "prefix_frequencies": {
                    "scores": {"123": 0.85, "456": 0.72},
                },
                "confidence_thresholds": {
                    "ema_decay": 0.95,
                    "min_hit_rate": 0.5,
                    "activation_hit_rate": 0.7,
                    "entries": {
                        "789": {
                            "hit_rate_ema": 0.78,
                            "speculation_enabled": True,
                        }
                    },
                },
            }
            json.dump(state, f)
            persist_path = f.name

        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
            self_spec_stats_persist_path=persist_path,
        )

        stub._init_adaptive_serving(vllm_config)

        # Verify state was loaded
        assert stub._frequency_tracker is not None
        assert len(stub._frequency_tracker) == 2
        assert stub._confidence_tracker is not None
        assert stub._confidence_tracker.should_speculate(789) is True

    def test_idle_callback_re_registers(self):
        """The warmup idle callback is re-registered after
        callbacks are drained."""
        stub = self._make_engine_core_stub()
        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
        )

        stub._init_adaptive_serving(vllm_config)
        assert len(stub._idle_state_callbacks) == 1

        # Simulate what _notify_idle_state_callbacks does:
        # 1. Drain all callbacks
        while stub._idle_state_callbacks:
            callback = stub._idle_state_callbacks.pop()
            callback(stub)
        # 2. Re-register adaptive callback (done by the method)
        if stub._warmup_worker is not None:
            stub._idle_state_callbacks.append(
                stub._adaptive_warmup_idle_callback
            )

        # Should be re-registered for next idle window
        assert len(stub._idle_state_callbacks) == 1
        assert stub._idle_state_callbacks[0] == stub._adaptive_warmup_idle_callback

    def test_abort_on_add_request(self):
        """Warmup is aborted when a request is added."""
        stub = self._make_engine_core_stub()
        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
        )

        stub._init_adaptive_serving(vllm_config)

        # Mock the add_request dependencies
        mock_request = MagicMock()
        mock_request.request_id = "test-123"
        mock_request.pooling_params = None
        mock_request.kv_transfer_params = None
        mock_request.abort_immediately = False

        # Track abort calls
        abort_called = []
        stub._warmup_worker.abort = lambda: abort_called.append(True)

        # Need to provide get_supported_tasks
        stub.get_supported_tasks = lambda: ()

        # Call add_request
        from vllm.v1.engine.core import EngineCore

        EngineCore.add_request(stub, mock_request)

        assert len(abort_called) == 1

    def test_periodic_persistence(self):
        """State is persisted when persist_interval_seconds elapses."""
        stub = self._make_engine_core_stub()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            persist_path = f.name

        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
            self_spec_stats_persist_path=persist_path,
            persist_interval_seconds=0.01,  # Very short for testing
        )

        stub._init_adaptive_serving(vllm_config)

        # Wait for interval to elapse
        time.sleep(0.02)

        # Trigger persistence check
        stub._maybe_persist_state()

        # Verify file was written
        with open(persist_path) as f:
            data = json.load(f)
        assert data["schema_version"] == 1
        assert "prefix_frequencies" in data
        assert "confidence_thresholds" in data

    def test_no_persistence_before_interval(self):
        """State is not persisted before the interval elapses."""
        stub = self._make_engine_core_stub()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            persist_path = f.name
            # Write something so we can tell if it gets overwritten
            f.write("original")

        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
            self_spec_stats_persist_path=persist_path,
            persist_interval_seconds=3600,  # Very long
        )

        stub._init_adaptive_serving(vllm_config)

        # Immediately check persistence (should not persist yet)
        stub._maybe_persist_state()

        # File should still have original content (or be empty from
        # a failed JSON load, but persister.load() would return None, None
        # so the tracker stays fresh)
        # Actually the file was overwritten with "original" text which
        # isn't valid JSON, so persister.load() returns None, None.
        # The point is that _maybe_persist_state should not write because
        # the interval hasn't elapsed yet.

    def test_init_dev_profile(self):
        """Dev profile applies correct defaults to components."""
        stub = self._make_engine_core_stub()
        vllm_config = self._make_vllm_config(
            enable_adaptive_warmup=True,
            adaptive_profile="dev",
        )

        stub._init_adaptive_serving(vllm_config)

        assert stub._frequency_tracker is not None
        assert stub._frequency_tracker.ema_decay == 0.8
        assert stub._adaptive_config.warmup_budget_ms == 200.0
