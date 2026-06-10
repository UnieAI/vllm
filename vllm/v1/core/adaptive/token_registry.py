# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block hash to token ID sequence registry for adaptive warmup."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TokenRegistry:
    """Bounded mapping from block hash to token ID sequence.

    Stores token sequences at block granularity, where each entry
    maps a single block hash to a fixed-size token sequence of
    length equal to the configured block size.  Used by
    PrefixWarmupWorker to resolve block hashes back to token IDs
    for executing warmup prefill forward passes.

    Eviction strategy:
        1. Evict stale entries (not in active_hashes) first.
        2. Fallback to LRU (oldest registered) when all entries
           are active.
    """

    def __init__(self, max_entries: int, block_size: int) -> None:
        """
        Args:
            max_entries: Maximum number of entries, matching
                warmup_max_prefixes.
            block_size: Number of tokens per block.
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._max_entries = max_entries
        self._block_size = block_size
        # block_hash -> token_ids; insertion-ordered (Python 3.7+)
        self._entries: dict[int, list[int]] = {}

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def block_size(self) -> int:
        return self._block_size

    def __len__(self) -> int:
        return len(self._entries)

    def register(self, block_hash: int, token_ids: list[int]) -> None:
        """Register a block hash to token ID sequence mapping.

        If the registry is at capacity, stale entries are evicted
        first.  If all entries are active, the oldest entry is
        evicted (LRU fallback).

        Registrations where ``len(token_ids) != block_size`` are
        silently ignored with a DEBUG log.
        """
        if len(token_ids) != self._block_size:
            logger.debug(
                "TokenRegistry: ignoring registration for hash %d "
                "with %d tokens (expected %d)",
                block_hash,
                len(token_ids),
                self._block_size,
            )
            return

        # If already present, move to end (refresh insertion order)
        if block_hash in self._entries:
            del self._entries[block_hash]
            self._entries[block_hash] = list(token_ids)
            return

        # Evict if at capacity
        if len(self._entries) >= self._max_entries:
            self._evict_one()

        self._entries[block_hash] = list(token_ids)

    def get_tokens(self, block_hash: int) -> list[int] | None:
        """Retrieve the token ID sequence for a block hash.

        Returns:
            List of token IDs, or None if the mapping does not
            exist.
        """
        tokens = self._entries.get(block_hash)
        if tokens is not None:
            return list(tokens)
        return None

    def evict_stale(self, active_hashes: set[int]) -> None:
        """Remove all entries whose block hash is not in
        *active_hashes*."""
        stale_keys = [k for k in self._entries if k not in active_hashes]
        for k in stale_keys:
            del self._entries[k]

    def to_dict(self) -> dict:
        """Serialize registry state for persistence.

        Returns:
            Dictionary with ``entries`` mapping (str keys) and
            ``block_size``.
        """
        return {
            "entries": {str(k): v for k, v in self._entries.items()},
            "block_size": self._block_size,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        max_entries: int,
        block_size: int,
    ) -> TokenRegistry:
        """Restore registry from persisted data.

        Args:
            data: Dictionary previously returned by ``to_dict()``.
            max_entries: Maximum capacity for the restored registry.
            block_size: Expected block size.

        Returns:
            A new TokenRegistry initialized with persisted
            mappings.  If the data is malformed, logs a WARNING and
            returns an empty registry.
        """
        registry = cls(max_entries=max_entries, block_size=block_size)

        if not isinstance(data, dict):
            logger.warning(
                "TokenRegistry: invalid persisted data type, creating empty registry"
            )
            return registry

        raw_entries: dict = data.get("entries", {})
        if not isinstance(raw_entries, dict):
            logger.warning(
                "TokenRegistry: invalid 'entries' field, creating empty registry"
            )
            return registry

        for k, v in raw_entries.items():
            try:
                block_hash = int(k)
                token_ids = [int(t) for t in v]
            except (ValueError, TypeError):
                logger.warning(
                    "TokenRegistry: skipping malformed entry key=%s",
                    k,
                )
                continue

            if len(token_ids) != block_size:
                continue

            # Respect max_entries during restore
            if len(registry._entries) >= max_entries:
                break

            registry._entries[block_hash] = token_ids

        return registry

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Evict one entry to make room.

        Strategy: evict the oldest entry (first in insertion order),
        which approximates LRU since re-registration refreshes
        position.
        """
        if not self._entries:
            return
        # dict preserves insertion order; pop the first key
        oldest_key = next(iter(self._entries))
        del self._entries[oldest_key]
