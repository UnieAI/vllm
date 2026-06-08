# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for Self-Speculation decision correctness.

# Feature: adaptive-speculative-serving, Property 6: Speculation Decision Correctness
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker
from vllm.v1.spec_decode.self_speculation import SelfSpeculationProposer


def _make_proposer(
    confidence_threshold: float,
) -> SelfSpeculationProposer:
    """Create a SelfSpeculationProposer with a mock VllmConfig."""
    mock_config = MagicMock()
    mock_config.adaptive_serving.self_spec_confidence_threshold = confidence_threshold
    mock_config.adaptive_serving.adaptive_profile = "dev"
    mock_config.adaptive_serving.self_spec_min_hit_rate = 0.5
    mock_config.adaptive_serving.self_spec_activation_hit_rate = 0.7
    return SelfSpeculationProposer(mock_config)


def _pretrain_tracker_with_hits(
    tracker: ConfidenceTracker, pattern: int, n: int = 20
) -> None:
    """Pre-train a tracker pattern with hits to enable speculation."""
    for _ in range(n):
        tracker.update(pattern, hit=True)


@settings(max_examples=100)
@given(
    vocab_size=st.integers(min_value=2, max_value=1000),
    threshold=st.floats(min_value=0.01, max_value=0.99),
    logit_scale=st.floats(min_value=0.1, max_value=100.0),
    hot_index=st.integers(min_value=0, max_value=999),
)
def test_speculation_decision_matches_softmax_threshold(
    vocab_size: int,
    threshold: float,
    logit_scale: float,
    hot_index: int,
) -> None:
    """Property 6: Speculation Decision Correctness (part a)

    For any logit distribution and configured confidence threshold,
    the SelfSpeculationProposer SHALL perform speculation if and
    only if the softmax top-1 probability >= threshold.

    The proposed token (when speculation occurs) SHALL be the argmax
    of the softmax distribution.

    **Validates: Requirements 3.2**
    """
    # Constrain hot_index to valid range for this vocab_size
    hot_index = hot_index % vocab_size

    proposer = _make_proposer(threshold)

    # Generate logits: a base random tensor with one "hot" logit
    # scaled to create a controlled confidence scenario
    logits = torch.zeros(1, vocab_size)
    logits[0, hot_index] = logit_scale

    # Compute expected decision
    probs = torch.softmax(logits[0].float(), dim=-1)
    top_prob, top_idx = probs.max(dim=-1)
    expected_speculates = top_prob.item() >= threshold
    expected_token = int(top_idx.item())

    # Set up proposer with logits
    proposer.set_last_logits(logits)

    # Create minimal context: need at least 1 sampled token,
    # valid num_tokens and token_ids
    sampled_token_ids = [[0]]
    num_tokens_no_spec = np.array([4], dtype=np.int32)
    token_ids_cpu = np.array([[1, 2, 3, 4]], dtype=np.int32)

    # Pre-train tracker so it approves speculation for the
    # context pattern derived from token_ids_cpu
    context_pattern = proposer._compute_context_pattern(
        token_ids_cpu[0], int(num_tokens_no_spec[0])
    )
    _pretrain_tracker_with_hits(proposer._confidence_tracker, context_pattern)

    # Call propose
    proposals = proposer.propose(
        sampled_token_ids=sampled_token_ids,
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )

    if expected_speculates:
        assert len(proposals[0]) == 1, (
            f"Expected speculation (top_prob={top_prob.item():.6f} "
            f">= threshold={threshold:.6f}) but got empty proposal"
        )
        assert proposals[0][0] == expected_token, (
            f"Expected proposed token={expected_token}, got {proposals[0][0]}"
        )
    else:
        assert len(proposals[0]) == 0, (
            f"Expected no speculation "
            f"(top_prob={top_prob.item():.6f} "
            f"< threshold={threshold:.6f}) but got proposal: "
            f"{proposals[0]}"
        )


@settings(max_examples=100)
@given(
    speculated_token=st.integers(min_value=0, max_value=50000),
    actual_token=st.integers(min_value=0, max_value=50000),
    request_id=st.text(min_size=1, max_size=20),
)
def test_hit_miss_update_correctness(
    speculated_token: int,
    actual_token: int,
    request_id: str,
) -> None:
    """Property 6: Speculation Decision Correctness (parts b, c)

    For any speculated and actual token pair:
    - Skip (hit): speculated == actual -> tracker updated with
      hit=True
    - Discard (miss): speculated != actual -> tracker updated
      with hit=False

    **Validates: Requirements 3.3, 3.4**
    """
    proposer = _make_proposer(confidence_threshold=0.9)

    # Patch the tracker's update method to capture calls
    with patch.object(
        proposer._confidence_tracker, "update", wraps=None
    ) as mock_update:
        proposer.update_speculation_result(
            request_id=request_id,
            speculated_token=speculated_token,
            actual_token=actual_token,
        )

        # Verify update was called exactly once
        mock_update.assert_called_once()
        call_args = mock_update.call_args

        # Extract the hit argument
        _, kwargs = call_args
        hit_value = kwargs.get("hit", call_args[0][1]) if kwargs else call_args[0][1]

        expected_hit = speculated_token == actual_token

        assert hit_value == expected_hit, (
            f"speculated={speculated_token}, "
            f"actual={actual_token}: "
            f"expected hit={expected_hit}, got hit={hit_value}"
        )
