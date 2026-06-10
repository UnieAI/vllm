# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for execute_warmup_prefill error handling.

Tests that UniProcExecutor.execute_warmup_prefill returns None when
the worker raises an exception, and that a WARNING log is emitted.

Requirements: 3.5
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm.v1.executor.uniproc_executor import UniProcExecutor


class TestExecuteWarmupPrefillErrorHandling:
    """Tests for UniProcExecutor.execute_warmup_prefill error handling."""

    def _make_executor_stub(self):
        """Create a minimal UniProcExecutor stub for testing."""
        stub = object.__new__(UniProcExecutor)
        stub.driver_worker = MagicMock()
        return stub

    def test_returns_none_on_runtime_error(self):
        """execute_warmup_prefill returns None when collective_rpc
        raises RuntimeError."""
        stub = self._make_executor_stub()
        stub.driver_worker.execute_method = MagicMock(
            side_effect=RuntimeError("CUDA error")
        )

        with patch.object(
            UniProcExecutor,
            "collective_rpc",
            side_effect=RuntimeError("CUDA error"),
        ):
            result = UniProcExecutor.execute_warmup_prefill(stub, [1, 2, 3, 4])

        assert result is None

    def test_returns_none_on_value_error(self):
        """execute_warmup_prefill returns None when collective_rpc
        raises ValueError."""
        stub = self._make_executor_stub()

        with patch.object(
            UniProcExecutor,
            "collective_rpc",
            side_effect=ValueError("invalid input"),
        ):
            result = UniProcExecutor.execute_warmup_prefill(stub, [10, 20, 30])

        assert result is None

    def test_returns_none_on_generic_exception(self):
        """execute_warmup_prefill returns None when collective_rpc
        raises a generic Exception."""
        stub = self._make_executor_stub()

        with patch.object(
            UniProcExecutor,
            "collective_rpc",
            side_effect=Exception("unexpected failure"),
        ):
            result = UniProcExecutor.execute_warmup_prefill(stub, [5, 6, 7, 8])

        assert result is None

    def test_returns_block_ids_on_success(self):
        """execute_warmup_prefill returns block IDs when
        collective_rpc succeeds."""
        stub = self._make_executor_stub()

        with patch.object(
            UniProcExecutor,
            "collective_rpc",
            return_value=[0, 1, 2],
        ):
            result = UniProcExecutor.execute_warmup_prefill(stub, [1, 2, 3, 4])

        assert result == [0, 1, 2]

    def test_warning_logged_on_runtime_error(self):
        """A WARNING log is emitted when collective_rpc raises
        RuntimeError."""
        stub = self._make_executor_stub()

        with (
            patch.object(
                UniProcExecutor,
                "collective_rpc",
                side_effect=RuntimeError("CUDA error"),
            ),
            patch("vllm.v1.executor.uniproc_executor.logger") as mock_logger,
        ):
            UniProcExecutor.execute_warmup_prefill(stub, [1, 2, 3, 4])

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        log_msg = call_args[0][0] % call_args[0][1:]
        assert "Warmup prefill failed" in log_msg
        assert "CUDA error" in log_msg

    def test_warning_logged_on_generic_exception(self):
        """A WARNING log is emitted when collective_rpc raises
        a generic Exception."""
        stub = self._make_executor_stub()

        with (
            patch.object(
                UniProcExecutor,
                "collective_rpc",
                side_effect=Exception("worker crashed"),
            ),
            patch("vllm.v1.executor.uniproc_executor.logger") as mock_logger,
        ):
            UniProcExecutor.execute_warmup_prefill(stub, [1, 2])

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        log_msg = call_args[0][0] % call_args[0][1:]
        assert "Warmup prefill failed" in log_msg
        assert "worker crashed" in log_msg

    def test_warning_not_logged_on_success(self):
        """No WARNING is emitted when collective_rpc succeeds."""
        stub = self._make_executor_stub()

        with (
            patch.object(
                UniProcExecutor,
                "collective_rpc",
                return_value=[0, 1],
            ),
            patch("vllm.v1.executor.uniproc_executor.logger") as mock_logger,
        ):
            UniProcExecutor.execute_warmup_prefill(stub, [1, 2, 3])

        mock_logger.warning.assert_not_called()
