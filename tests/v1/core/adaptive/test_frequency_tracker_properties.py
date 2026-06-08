# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for PrefixFrequencyTracker.

# Feature: adaptive-speculative-serving, Property 2: Bounded Frequency
# Map with Min-Score Eviction
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)

# --- Strategies ---

# Max entries between 1 and 50 to keep tests fast while covering
# edge cases (capacity=1 is the tightest bound).
max_entries_st = st.integers(min_value=1, max_value=50)

# EMA decay in the open interval (0, 1).
ema_decay_st = st.floats(
    min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
)

# Prefix hashes: use a moderate range so collisions can occur
# naturally, exercising both "existing key" and "new key" paths.
prefix_hash_st = st.integers(min_value=0, max_value=200)

# Sequences of prefix hashes to feed into the tracker.
hash_sequence_st = st.lists(prefix_hash_st, min_size=1, max_size=300)


class TestBoundedMapProperty:
    """Property 2: Bounded Frequency Map with Min-Score Eviction.

    **Validates: Requirements 1.2, 1.3**

    For any sequence of prefix hash insertions into the frequency
    tracker with a configured maximum capacity:
    - The map size SHALL never exceed the maximum.
    - Whenever an eviction occurs, the evicted entry SHALL have the
      minimum EMA score among all entries prior to insertion.
    """

    @settings(max_examples=100)
    @given(
        max_entries=max_entries_st,
        ema_decay=ema_decay_st,
        hashes=hash_sequence_st,
    )
    def test_map_size_never_exceeds_max(
        self,
        max_entries: int,
        ema_decay: float,
        hashes: list[int],
    ):
        """After any sequence of updates, len(tracker) <= max_entries.

        **Validates: Requirements 1.2**
        """
        tracker = PrefixFrequencyTracker(max_entries=max_entries, ema_decay=ema_decay)
        for h in hashes:
            tracker.update(h)
            assert len(tracker) <= max_entries

    @settings(max_examples=100)
    @given(
        max_entries=max_entries_st,
        ema_decay=ema_decay_st,
        hashes=hash_sequence_st,
    )
    def test_evicted_entry_has_minimum_score(
        self,
        max_entries: int,
        ema_decay: float,
        hashes: list[int],
    ):
        """When eviction occurs, the evicted entry had the minimum
        EMA score among all entries prior to insertion.

        **Validates: Requirements 1.3**

        Strategy: maintain a shadow model of the tracker's scores.
        Before each insertion that triggers eviction, verify the
        entry that disappears had the minimum score.
        """
        tracker = PrefixFrequencyTracker(max_entries=max_entries, ema_decay=ema_decay)
        # Shadow score map mirrors internal state.
        shadow: dict[int, float] = {}

        for h in hashes:
            if h in shadow:
                # Existing key — just update EMA, no eviction.
                shadow[h] = ema_decay * shadow[h] + (1.0 - ema_decay)
                tracker.update(h)
            else:
                # New key — eviction may happen.
                if len(shadow) >= max_entries:
                    # Find the minimum score BEFORE insertion.
                    min_score = min(shadow.values())

                    # Perform the update (triggers eviction internally).
                    tracker.update(h)

                    # Determine which key was evicted by comparing
                    # shadow keys to tracker's remaining candidates.
                    remaining = {
                        k
                        for k, _ in tracker.get_warmup_candidates(
                            cached_hashes=set(), min_score=0.0
                        )
                    }

                    # The evicted key is in shadow but not remaining.
                    # (The new key h is in remaining.)
                    evicted_keys = set(shadow.keys()) - remaining
                    # Exactly one key should be evicted.
                    assert len(evicted_keys) == 1
                    evicted_key = evicted_keys.pop()

                    # The evicted entry must have had the minimum
                    # score prior to insertion.
                    assert shadow[evicted_key] <= min_score + 1e-9

                    # Update shadow: remove evicted, add new.
                    del shadow[evicted_key]
                    shadow[h] = 1.0 - ema_decay
                else:
                    # No eviction needed — just insert.
                    tracker.update(h)
                    shadow[h] = 1.0 - ema_decay
