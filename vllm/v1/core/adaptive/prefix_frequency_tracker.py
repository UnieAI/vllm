# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EMA-based prefix frequency tracker for adaptive warmup."""

from __future__ import annotations

import heapq


class PrefixFrequencyTracker:
    """Tracks prefix pattern frequencies using exponential moving
    averages (EMA).

    Each observed prefix hash gets an EMA score that increases on
    repeated observations and decays towards zero otherwise.  The
    tracker maintains a bounded-size map and evicts the lowest-score
    entry when at capacity.
    """

    def __init__(self, max_entries: int, ema_decay: float) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must be in the open interval (0, 1)")
        self._max_entries = max_entries
        self._ema_decay = ema_decay
        # prefix_hash -> EMA score
        self._scores: dict[int, float] = {}

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ema_decay(self) -> float:
        return self._ema_decay

    def __len__(self) -> int:
        return len(self._scores)

    def update(self, prefix_hash: int) -> None:
        """Update the EMA score for *prefix_hash*.

        If the hash already exists:
            score = decay * old_score + (1 - decay)
        If the hash is new:
            score = (1 - decay)   (first observation)

        When the map is at capacity and a new hash is inserted, the
        entry with the minimum EMA score is evicted.
        """
        if prefix_hash in self._scores:
            old_score = self._scores[prefix_hash]
            self._scores[prefix_hash] = self._ema_decay * old_score + (
                1.0 - self._ema_decay
            )
        else:
            # Need to make room if at capacity
            if len(self._scores) >= self._max_entries:
                self._evict_min()
            self._scores[prefix_hash] = 1.0 - self._ema_decay

    def get_warmup_candidates(
        self,
        cached_hashes: set[int],
        min_score: float,
    ) -> list[tuple[int, float]]:
        """Return uncached prefixes with score >= *min_score*,
        sorted in descending score order.

        Args:
            cached_hashes: Set of prefix hashes already present in
                the block pool cache.
            min_score: Minimum EMA score threshold for candidacy.

        Returns:
            List of (prefix_hash, score) tuples sorted by score
            descending.
        """
        candidates: list[tuple[int, float]] = [
            (h, score)
            for h, score in self._scores.items()
            if h not in cached_hashes and score >= min_score
        ]
        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def to_dict(self) -> dict:
        """Serialize tracker state for persistence."""
        return {
            "scores": {str(k): v for k, v in self._scores.items()},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        max_entries: int,
        ema_decay: float,
    ) -> PrefixFrequencyTracker:
        """Restore tracker state from persisted data.

        Args:
            data: Dictionary previously returned by ``to_dict()``.
            max_entries: Maximum capacity for the restored tracker.
            ema_decay: EMA decay parameter for the restored tracker.

        Returns:
            A new PrefixFrequencyTracker initialized with the
            persisted scores.
        """
        tracker = cls(max_entries=max_entries, ema_decay=ema_decay)
        raw_scores: dict[str, float] = data.get("scores", {})
        for k, v in raw_scores.items():
            tracker._scores[int(k)] = float(v)

        # If persisted data exceeds new max_entries, trim to fit
        while len(tracker._scores) > tracker._max_entries:
            tracker._evict_min()

        return tracker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_min(self) -> None:
        """Evict the entry with the lowest EMA score using a
        min-heap selection."""
        if not self._scores:
            return
        # Build list of (score, hash) and find the minimum
        # Use heapq.nsmallest for O(n) single-min extraction
        min_score, min_hash = heapq.nsmallest(
            1, ((score, h) for h, score in self._scores.items())
        )[0]
        del self._scores[min_hash]
