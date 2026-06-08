# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SelfSpeculationProposer interface and error fallback.

Requirements: 3.5, 9.3
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.spec_decode.self_speculation import SelfSpeculationProposer


def _make_vllm_config(
    confidence_threshold: float = 0.9,
    profile: str = "dev",
) -> MagicMock:
    """Create a minimal mock VllmConfig with AdaptiveServingConfig."""
    adaptive_cfg = AdaptiveServingConfig(
        adaptive_profile=profile,
        self_spec_confidence_threshold=confidence_threshold,
    )
    vllm_config = MagicMock()
    vllm_config.adaptive_serving = adaptive_cfg
    return vllm_config


def _make_proposer(
    confidence_threshold: float = 0.9,
    profile: str = "dev",
) -> SelfSpeculationProposer:
    """Create a SelfSpeculationProposer with test configuration."""
    vllm_config = _make_vllm_config(
        confidence_threshold=confidence_threshold,
        profile=profile,
    )
    return SelfSpeculationProposer(vllm_config)


class TestInterfaceCompliance:
    """Verify SelfSpeculationProposer has the expected proposer
    interface methods.

    Requirements: 3.5
    """

    def test_has_propose_method(self):
        proposer = _make_proposer()
        assert hasattr(proposer, "propose")
        assert callable(proposer.propose)

    def test_has_load_model_method(self):
        proposer = _make_proposer()
        assert hasattr(proposer, "load_model")
        assert callable(proposer.load_model)

    def test_has_set_last_logits_method(self):
        proposer = _make_proposer()
        assert hasattr(proposer, "set_last_logits")
        assert callable(proposer.set_last_logits)

    def test_has_update_speculation_result_method(self):
        proposer = _make_proposer()
        assert hasattr(proposer, "update_speculation_result")
        assert callable(proposer.update_speculation_result)


class TestNumSpeculativeTokens:
    """Verify class attribute num_speculative_tokens == 1."""

    def test_class_attribute_is_one(self):
        assert SelfSpeculationProposer.num_speculative_tokens == 1

    def test_instance_attribute_is_one(self):
        proposer = _make_proposer()
        assert proposer.num_speculative_tokens == 1


class TestNoLogitsReturnsEmpty:
    """When _last_logits is None, propose() returns all-empty lists."""

    def test_returns_empty_for_single_request(self):
        proposer = _make_proposer()
        # No logits set
        result = proposer.propose(
            sampled_token_ids=[[5]],
            num_tokens_no_spec=np.array([10]),
            token_ids_cpu=np.zeros((1, 20), dtype=np.int32),
        )
        assert result == [[]]

    def test_returns_empty_for_batch(self):
        proposer = _make_proposer()
        batch_size = 4
        result = proposer.propose(
            sampled_token_ids=[[1], [2], [3], [4]],
            num_tokens_no_spec=np.array([5, 5, 5, 5]),
            token_ids_cpu=np.zeros((batch_size, 20), dtype=np.int32),
        )
        assert result == [[], [], [], []]


class TestBelowThresholdReturnsEmpty:
    """When top-1 confidence < threshold, no token proposed."""

    def test_low_confidence_returns_empty(self):
        proposer = _make_proposer(confidence_threshold=0.9)
        # Create logits where no token has high probability
        # Uniform distribution: confidence = 1/vocab_size << 0.9
        vocab_size = 100
        logits = torch.zeros(1, vocab_size)
        proposer.set_last_logits(logits)

        result = proposer.propose(
            sampled_token_ids=[[0]],
            num_tokens_no_spec=np.array([5]),
            token_ids_cpu=np.ones((1, 20), dtype=np.int32),
        )
        assert result == [[]]

    def test_just_below_threshold_returns_empty(self):
        proposer = _make_proposer(confidence_threshold=0.95)
        # Create logits where top-1 prob is just below 0.95
        # Use a distribution that gives ~0.9 top-1 probability
        vocab_size = 10
        logits = torch.zeros(1, vocab_size)
        # Set token 0 to have a high logit but not high enough
        logits[0, 0] = 2.0  # softmax will give < 0.95
        proposer.set_last_logits(logits)

        result = proposer.propose(
            sampled_token_ids=[[0]],
            num_tokens_no_spec=np.array([5]),
            token_ids_cpu=np.ones((1, 20), dtype=np.int32),
        )
        assert result == [[]]


class TestAboveThresholdProposesToken:
    """When confidence >= threshold and tracker approves,
    proposes top-1 token."""

    def test_high_confidence_with_tracker_approval(self):
        proposer = _make_proposer(confidence_threshold=0.5)
        # Create logits where token 7 dominates
        vocab_size = 100
        logits = torch.full((1, vocab_size), -10.0)
        logits[0, 7] = 10.0  # Very high, softmax prob ~1.0

        proposer.set_last_logits(logits)

        # Force tracker to approve speculation
        with patch.object(
            proposer._confidence_tracker,
            "should_speculate",
            return_value=True,
        ):
            result = proposer.propose(
                sampled_token_ids=[[5]],
                num_tokens_no_spec=np.array([5]),
                token_ids_cpu=np.ones((1, 20), dtype=np.int32),
            )
        assert result == [[7]]

    def test_proposes_correct_top1_token(self):
        proposer = _make_proposer(confidence_threshold=0.5)
        vocab_size = 50
        logits = torch.full((1, vocab_size), -10.0)
        logits[0, 42] = 20.0  # Token 42 is the top-1

        proposer.set_last_logits(logits)

        with patch.object(
            proposer._confidence_tracker,
            "should_speculate",
            return_value=True,
        ):
            result = proposer.propose(
                sampled_token_ids=[[1]],
                num_tokens_no_spec=np.array([5]),
                token_ids_cpu=np.ones((1, 20), dtype=np.int32),
            )
        assert result == [[42]]


class TestTrackerDisapprovalReturnsEmpty:
    """When confidence is high but tracker says don't speculate,
    returns empty."""

    def test_tracker_disapproval_blocks_proposal(self):
        proposer = _make_proposer(confidence_threshold=0.5)
        vocab_size = 100
        logits = torch.full((1, vocab_size), -10.0)
        logits[0, 7] = 10.0  # High confidence

        proposer.set_last_logits(logits)

        # Force tracker to disapprove speculation
        with patch.object(
            proposer._confidence_tracker,
            "should_speculate",
            return_value=False,
        ):
            result = proposer.propose(
                sampled_token_ids=[[5]],
                num_tokens_no_spec=np.array([5]),
                token_ids_cpu=np.ones((1, 20), dtype=np.int32),
            )
        assert result == [[]]


class TestErrorFallback:
    """When logits have unexpected shape or are corrupted,
    proposer returns empty proposals (graceful degradation).

    Requirements: 9.3
    """

    def test_logits_wrong_shape_returns_empty(self):
        """Logits with fewer rows than batch size should not
        crash, and out-of-bounds indices return empty."""
        proposer = _make_proposer(confidence_threshold=0.5)
        # Only 1 row of logits but batch_size = 3
        logits = torch.full((1, 50), -10.0)
        logits[0, 5] = 10.0
        proposer.set_last_logits(logits)

        with patch.object(
            proposer._confidence_tracker,
            "should_speculate",
            return_value=True,
        ):
            result = proposer.propose(
                sampled_token_ids=[[1], [2], [3]],
                num_tokens_no_spec=np.array([5, 5, 5]),
                token_ids_cpu=np.ones((3, 20), dtype=np.int32),
            )
        # First request may get proposal, but requests 2 and 3
        # should get empty (index out of bounds handled)
        assert result[1] == []
        assert result[2] == []

    def test_empty_sampled_tokens_returns_empty(self):
        """When sampled_token_ids has empty lists, proposer
        handles gracefully."""
        proposer = _make_proposer(confidence_threshold=0.5)
        logits = torch.full((2, 50), -10.0)
        logits[0, 5] = 10.0
        logits[1, 5] = 10.0
        proposer.set_last_logits(logits)

        result = proposer.propose(
            sampled_token_ids=[[], [1]],
            num_tokens_no_spec=np.array([5, 5]),
            token_ids_cpu=np.ones((2, 20), dtype=np.int32),
        )
        # First request has no sampled tokens — should return empty
        assert result[0] == []

    def test_zero_dim_logits_returns_empty(self):
        """If logits tensor has zero dimension, returns empty."""
        proposer = _make_proposer(confidence_threshold=0.5)
        # Empty tensor (0 rows)
        logits = torch.zeros(0, 50)
        proposer.set_last_logits(logits)

        result = proposer.propose(
            sampled_token_ids=[[1]],
            num_tokens_no_spec=np.array([5]),
            token_ids_cpu=np.ones((1, 20), dtype=np.int32),
        )
        assert result == [[]]


class TestLoadModelIsNoOp:
    """Calling load_model with any args doesn't raise."""

    def test_load_model_no_args(self):
        proposer = _make_proposer()
        # Should not raise
        proposer.load_model()

    def test_load_model_with_args(self):
        proposer = _make_proposer()
        # Should not raise with arbitrary arguments
        proposer.load_model("arg1", "arg2", key="value")

    def test_load_model_returns_none(self):
        proposer = _make_proposer()
        result = proposer.load_model()
        assert result is None


class TestConfidenceTrackerIntegration:
    """Verify ConfidenceTracker is correctly connected to the proposer
    for per-pattern gating.

    Requirements: 3.6, 4.2, 4.3
    """

    def test_external_confidence_tracker_injection(self):
        """When an external ConfidenceTracker is provided, the proposer
        uses it instead of creating its own."""
        from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker

        external_tracker = ConfidenceTracker(
            ema_decay=0.9,
            min_hit_rate=0.4,
            activation_hit_rate=0.6,
        )
        vllm_config = _make_vllm_config()
        proposer = SelfSpeculationProposer(
            vllm_config, confidence_tracker=external_tracker
        )

        assert proposer.confidence_tracker is external_tracker

    def test_default_creates_internal_tracker(self):
        """When no external tracker is provided, the proposer
        creates its own ConfidenceTracker."""
        from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker

        proposer = _make_proposer()
        assert isinstance(proposer.confidence_tracker, ConfidenceTracker)

    def test_external_tracker_used_for_gating(self):
        """The external tracker's should_speculate() is used
        during proposal gating."""
        from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker

        external_tracker = ConfidenceTracker(
            ema_decay=0.8,
            min_hit_rate=0.5,
            activation_hit_rate=0.7,
        )
        vllm_config = _make_vllm_config(confidence_threshold=0.5)
        proposer = SelfSpeculationProposer(
            vllm_config, confidence_tracker=external_tracker
        )

        # Set high-confidence logits
        vocab_size = 50
        logits = torch.full((1, vocab_size), -10.0)
        logits[0, 7] = 20.0
        proposer.set_last_logits(logits)

        # External tracker hasn't been trained — should_speculate
        # returns False for unknown patterns
        result = proposer.propose(
            sampled_token_ids=[[1]],
            num_tokens_no_spec=np.array([5]),
            token_ids_cpu=np.ones((1, 20), dtype=np.int32),
        )
        # Unknown patterns return False by default
        assert result == [[]]

        # Now train the external tracker to approve a pattern
        # (multiple hits to exceed activation_hit_rate=0.7)
        context_pattern = proposer._compute_context_pattern(
            np.ones(20, dtype=np.int32), 5
        )
        for _ in range(10):
            external_tracker.update(context_pattern, True)

        # Now the same propose call should succeed
        result = proposer.propose(
            sampled_token_ids=[[1]],
            num_tokens_no_spec=np.array([5]),
            token_ids_cpu=np.ones((1, 20), dtype=np.int32),
        )
        assert result == [[7]]


class TestSpeculativeConfigRegistration:
    """Verify that 'self_speculation' is registered as a valid method
    in SpeculativeConfig.

    Requirements: 3.5, 3.6
    """

    def test_self_speculation_in_speculative_method_literal(self):
        """self_speculation is a valid SpeculativeMethod value."""
        from typing import get_args

        from vllm.config.speculative import SpeculativeMethod

        # Flatten nested Literal types
        all_methods = set()
        for arg in get_args(SpeculativeMethod):
            if hasattr(arg, "__args__"):
                all_methods.update(get_args(arg))
            else:
                all_methods.add(arg)

        assert "self_speculation" in all_methods
