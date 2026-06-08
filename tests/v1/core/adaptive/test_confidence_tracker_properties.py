# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for Confidence Tracker hysteresis.

# Feature: adaptive-speculative-serving, Property 7: Adaptive Threshold Hysteresis
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker


@settings(max_examples=100)
@given(
    ema_decay=st.floats(min_value=0.01, max_value=0.99),
    min_hit_rate=st.floats(min_value=0.05, max_value=0.45),
    activation_offset=st.floats(min_value=0.1, max_value=0.5),
    observations=st.lists(st.booleans(), min_size=1, max_size=200),
)
def test_hysteresis_invariant(
    ema_decay: float,
    min_hit_rate: float,
    activation_offset: float,
    observations: list[bool],
) -> None:
    """Property 7: Adaptive Threshold Hysteresis

    For any sequence of hit/miss observations for a context pattern,
    the confidence tracker's speculation-enabled state SHALL follow
    hysteresis: enabled when EMA hit rate >= activation_hit_rate,
    disabled when EMA hit rate < min_hit_rate, and unchanged
    otherwise.

    **Validates: Requirements 4.2, 4.3**
    """
    activation_hit_rate = min(min_hit_rate + activation_offset, 0.99)

    tracker = ConfidenceTracker(
        ema_decay=ema_decay,
        min_hit_rate=min_hit_rate,
        activation_hit_rate=activation_hit_rate,
    )

    context_pattern = 1

    # Track expected state manually
    expected_enabled = False
    expected_hit_rate: float | None = None

    for obs in observations:
        observation_val = 1.0 if obs else 0.0

        if expected_hit_rate is None:
            # First observation: initialized directly
            expected_hit_rate = observation_val
            # New patterns start disabled
            expected_enabled = False
        else:
            expected_hit_rate = (
                ema_decay * expected_hit_rate + (1.0 - ema_decay) * observation_val
            )

        # Apply hysteresis rules
        if expected_hit_rate >= activation_hit_rate:
            expected_enabled = True
        elif expected_hit_rate < min_hit_rate:
            expected_enabled = False
        # else: retain current state (hysteresis band)

        tracker.update(context_pattern, hit=obs)

        actual_enabled = tracker.should_speculate(context_pattern)
        assert actual_enabled == expected_enabled, (
            f"Hysteresis violation: "
            f"hit_rate={expected_hit_rate:.6f}, "
            f"min={min_hit_rate:.4f}, "
            f"activation={activation_hit_rate:.4f}, "
            f"expected_enabled={expected_enabled}, "
            f"actual_enabled={actual_enabled}, "
            f"obs={obs}"
        )
