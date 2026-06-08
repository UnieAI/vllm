# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomically persists and loads adaptive serving state."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)


class _Serializable(Protocol):
    """Protocol for objects that can be serialized to a dict."""

    def to_dict(self) -> dict: ...


class StatePersister:
    """Atomically persists and loads adaptive serving state.

    Uses a write-to-temp-then-rename strategy to ensure atomic writes
    on POSIX systems. On load, validates schema version and required
    fields, falling back to fresh state on any error.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str, interval_seconds: float):
        self.path = path
        self.interval_seconds = interval_seconds

    def save(
        self,
        frequency_tracker: _Serializable,
        confidence_tracker: _Serializable,
    ) -> None:
        """Atomically write state to disk (write tmp -> rename).

        Serializes tracker state to JSON with a schema_version field,
        writes to a temporary file, then atomically replaces the
        target file using os.replace().
        """
        state: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prefix_frequencies": frequency_tracker.to_dict(),
            "confidence_thresholds": confidence_tracker.to_dict(),
        }

        tmp_path = f"{self.path}.tmp"
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(self.path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Write to temp file first
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=parent_dir or ".",
                prefix=".adaptive_state_",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(state, f)

            # Atomic rename (POSIX guarantees atomicity)
            os.replace(tmp_path, self.path)
        except OSError:
            logger.warning(
                "Failed to persist adaptive serving state to %s. "
                "Will retry next cycle.",
                self.path,
            )
            # Clean up temp file if it still exists
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    def load(self) -> tuple[dict | None, dict | None]:
        """Load and validate persisted state.

        Returns:
            A tuple of (freq_data, conf_data). Returns (None, None)
            if the file doesn't exist, is corrupted, has a wrong
            schema version, or is missing required fields.
        """
        if not os.path.exists(self.path):
            return (None, None)

        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.info(
                "Could not read persisted state from %s (%s). "
                "Starting with fresh state.",
                self.path,
                e,
            )
            return (None, None)

        # Validate top-level structure
        if not isinstance(data, dict):
            logger.info(
                "Persisted state at %s has invalid format. Starting with fresh state.",
                self.path,
            )
            return (None, None)

        # Validate schema version
        schema_version = data.get("schema_version")
        if schema_version != self.SCHEMA_VERSION:
            logger.info(
                "Persisted state at %s has schema version %s "
                "(expected %d). Starting with fresh state.",
                self.path,
                schema_version,
                self.SCHEMA_VERSION,
            )
            return (None, None)

        # Validate required fields
        freq_data = data.get("prefix_frequencies")
        conf_data = data.get("confidence_thresholds")

        if freq_data is None or conf_data is None:
            logger.info(
                "Persisted state at %s is missing required fields. "
                "Starting with fresh state.",
                self.path,
            )
            return (None, None)

        if not isinstance(freq_data, dict) or not isinstance(conf_data, dict):
            logger.info(
                "Persisted state at %s has invalid field types. "
                "Starting with fresh state.",
                self.path,
            )
            return (None, None)

        return (freq_data, conf_data)
