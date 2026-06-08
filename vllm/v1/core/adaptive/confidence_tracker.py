# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence Tracker for adaptive self-speculation.

Tracks per-context-pattern hit rates using exponential moving averages
and implements hysteresis logic to enable/disable speculation per
pattern.
"""

from __future__ import annotations

from typing import Self


class ConfidenceTracker:
    """Tracks per-context-pattern hit rates with adaptive thresholds.

    Uses EMA-based hit rate tracking with hysteresis to avoid rapid
    toggling of speculation on/off. A pattern starts with speculation
    disabled, becomes enabled when EMA hit rate >= activation_hit_rate,
    and is disabled again only when EMA hit rate < min_hit_rate.
    """

    def __init__(
        self,
        ema_decay: float,
        min_hit_rate: float,
        activation_hit_rate: float,
    ) -> None:
        self._ema_decay = ema_decay
        self._min_hit_rate = min_hit_rate
        self._activation_hit_rate = activation_hit_rate

        # Maps context_pattern -> EMA hit rate
        self._hit_rates: dict[int, float] = {}
        # Maps context_pattern -> speculation enabled flag
        self._enabled: dict[int, bool] = {}

    def update(self, context_pattern: int, hit: bool) -> None:
        """Update hit rate EMA for a context pattern.

        For new patterns, initializes the hit rate to 1.0 if hit else
        0.0. For existing patterns, applies EMA:
            hit_rate = decay * old + (1 - decay) * observation
        Then applies hysteresis logic to enable/disable speculation.
        """
        observation = 1.0 if hit else 0.0

        if context_pattern in self._hit_rates:
            old_rate = self._hit_rates[context_pattern]
            new_rate = (
                self._ema_decay * old_rate + (1.0 - self._ema_decay) * observation
            )
        else:
            new_rate = observation
            # New patterns start with speculation disabled
            self._enabled[context_pattern] = False

        self._hit_rates[context_pattern] = new_rate

        # Apply hysteresis logic
        currently_enabled = self._enabled[context_pattern]
        if new_rate >= self._activation_hit_rate:
            self._enabled[context_pattern] = True
        elif new_rate < self._min_hit_rate:
            self._enabled[context_pattern] = False
        else:
            # In the hysteresis band: retain current state
            self._enabled[context_pattern] = currently_enabled

    def should_speculate(self, context_pattern: int) -> bool:
        """Whether speculation is enabled for a given context pattern.

        Returns False for unknown patterns.
        """
        return self._enabled.get(context_pattern, False)

    def mean_threshold(self) -> float:
        """Mean hit rate across all tracked patterns (for metrics).

        Returns 0.0 if no patterns are tracked.
        """
        if not self._hit_rates:
            return 0.0
        return sum(self._hit_rates.values()) / len(self._hit_rates)

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        entries: dict[str, dict] = {}
        for pattern, hit_rate in self._hit_rates.items():
            entries[str(pattern)] = {
                "hit_rate_ema": hit_rate,
                "speculation_enabled": self._enabled[pattern],
            }
        return {
            "ema_decay": self._ema_decay,
            "min_hit_rate": self._min_hit_rate,
            "activation_hit_rate": self._activation_hit_rate,
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> Self:
        """Deserialize from persisted state.

        Keyword arguments override persisted config values, allowing
        config changes to take effect on restart while preserving
        learned hit rates.
        """
        ema_decay = kwargs.get("ema_decay", data.get("ema_decay", 0.95))
        min_hit_rate = kwargs.get("min_hit_rate", data.get("min_hit_rate", 0.5))
        activation_hit_rate = kwargs.get(
            "activation_hit_rate",
            data.get("activation_hit_rate", 0.7),
        )

        tracker = cls(
            ema_decay=ema_decay,
            min_hit_rate=min_hit_rate,
            activation_hit_rate=activation_hit_rate,
        )

        entries = data.get("entries", {})
        for pattern_str, entry in entries.items():
            pattern = int(pattern_str)
            hit_rate = float(entry["hit_rate_ema"])
            enabled = bool(entry["speculation_enabled"])
            tracker._hit_rates[pattern] = hit_rate
            tracker._enabled[pattern] = enabled

        return tracker
