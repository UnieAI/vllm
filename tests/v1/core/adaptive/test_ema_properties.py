# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for EMA computation correctness.

# Feature: adaptive-speculative-serving, Property 1: EMA Computation Correctness
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)


@settings(max_examples=100)
@given(
    decay=st.floats(min_value=0.01, max_value=0.99),
    num_updates=st.integers(min_value=1, max_value=200),
)
def test_ema_iterative_computation_matches_formula(
    decay: float,
    num_updates: int,
) -> None:
    """Property 1: EMA Computation Correctness

    For any decay in (0,1) and any number of updates N applied to
    the same prefix hash, the tracker's final score SHALL equal the
    value computed by iteratively applying:
        score = decay * score + (1 - decay)
    starting from score = 0 (first observation uses
    score = (1 - decay), which is equivalent to decay * 0 + (1 - decay)).

    **Validates: Requirements 1.1, 4.1**
    """
    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=decay)

    # Apply N updates to the same hash
    prefix_hash = 12345
    for _ in range(num_updates):
        tracker.update(prefix_hash)

    # Compute expected score iteratively
    expected_score = 0.0
    for _ in range(num_updates):
        expected_score = decay * expected_score + (1.0 - decay)

    # Retrieve actual score from tracker
    candidates = tracker.get_warmup_candidates(cached_hashes=set(), min_score=0.0)
    assert len(candidates) == 1
    actual_score = candidates[0][1]

    assert math.isclose(
        actual_score,
        expected_score,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ), (
        f"EMA mismatch: actual={actual_score}, "
        f"expected={expected_score}, "
        f"decay={decay}, updates={num_updates}"
    )
