# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-Speculation Proposer for adaptive speculative serving.

Proposes speculative tokens using the same model's top-1 predictions.
Uses softmax confidence gating and per-pattern confidence tracking
to decide when speculation is beneficial.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.core.adaptive.confidence_tracker import ConfidenceTracker

logger = init_logger(__name__)


class SelfSpeculationProposer:
    """Proposes at most 1 speculative token per decode step.

    The proposer checks two conditions before proposing:
    1. The softmax probability of the top-1 token must exceed
       the configured confidence threshold.
    2. The ConfidenceTracker must indicate that speculation is
       worthwhile for the given context pattern.

    If both conditions pass, the top-1 token is proposed as a
    speculative candidate. Otherwise, an empty proposal is
    returned (no speculation for that request).

    The proposer is KV cache dtype-aware: speculative KV
    computation produces results in the configured compressed
    format (e.g., TurboQuant) when applicable, ensuring seamless
    integration with the KV cache subsystem.
    """

    # At most 1 speculative token per decode step
    num_speculative_tokens: int = 1

    def __init__(
        self,
        vllm_config: VllmConfig,
        confidence_tracker: ConfidenceTracker | None = None,
    ) -> None:
        adaptive_cfg = vllm_config.adaptive_serving
        self._confidence_threshold = adaptive_cfg.self_spec_confidence_threshold

        # KV cache dtype for speculative KV computation.
        # When TurboQuant or FP8 is configured, speculative KV blocks
        # are computed and stored in the same compressed format as
        # regular KV cache blocks, ensuring compatibility.
        cache_dtype = getattr(
            getattr(vllm_config, "cache_config", None), "cache_dtype", "auto"
        )
        # Ensure we have a valid string (handles MagicMock in tests)
        if not isinstance(cache_dtype, str):
            cache_dtype = "auto"
        self._kv_cache_dtype: CacheDType = cache_dtype  # type: ignore[assignment]

        if isinstance(self._kv_cache_dtype, str) and self._kv_cache_dtype.startswith(
            "turboquant_"
        ):
            logger.info(
                "SelfSpeculationProposer will compute speculative KV "
                "in compressed dtype=%s",
                self._kv_cache_dtype,
            )

        # Determine EMA decay from profile
        profile = adaptive_cfg.adaptive_profile
        ema_decay = 0.8 if profile == "dev" else 0.95

        if confidence_tracker is not None:
            # Use an externally-provided tracker (e.g., from Engine Core)
            # to allow shared state across components and persistence.
            self._confidence_tracker = confidence_tracker
        else:
            self._confidence_tracker = ConfidenceTracker(
                ema_decay=ema_decay,
                min_hit_rate=adaptive_cfg.self_spec_min_hit_rate,
                activation_hit_rate=adaptive_cfg.self_spec_activation_hit_rate,
            )

        # Storage for last logits (set externally before propose)
        self._last_logits: torch.Tensor | None = None

    @property
    def kv_cache_dtype(self) -> CacheDType:
        """The KV cache dtype used for speculative KV computation."""
        return self._kv_cache_dtype

    @property
    def confidence_tracker(self) -> ConfidenceTracker:
        """Expose tracker for persistence and metrics."""
        return self._confidence_tracker

    def set_last_logits(self, logits: torch.Tensor) -> None:
        """Store logits from the current decode step.

        Args:
            logits: Tensor of shape (batch_size, vocab_size) containing
                the logits output from the most recent forward pass.
        """
        self._last_logits = logits

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: (
            dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
        ) = None,
    ) -> list[list[int]]:
        """Propose speculative tokens based on confidence.

        For each request in the batch, checks:
        1. Softmax top-1 probability >= confidence threshold
        2. ConfidenceTracker says speculation is OK for the
           context pattern (derived from recent token hash)

        Returns a list of proposed token lists (each with 0 or 1
        token).
        """
        batch_size = len(sampled_token_ids)
        proposals: list[list[int]] = []

        if self._last_logits is None:
            # No logits available — return empty proposals
            return [[] for _ in range(batch_size)]

        for i in range(batch_size):
            if not sampled_token_ids[i]:
                # No sampled tokens for this request (skip)
                proposals.append([])
                continue

            # Get logits for this request
            if i >= self._last_logits.shape[0]:
                proposals.append([])
                continue

            logits_i = self._last_logits[i]

            # Compute softmax and get top-1 confidence
            probs = torch.softmax(logits_i.float(), dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            confidence = top_prob.item()

            # Check 1: confidence threshold
            if confidence < self._confidence_threshold:
                proposals.append([])
                continue

            # Derive context pattern from recent tokens
            num_tokens = int(num_tokens_no_spec[i])
            context_pattern = self._compute_context_pattern(
                token_ids_cpu[i], num_tokens
            )

            # Check 2: confidence tracker approval
            if not self._confidence_tracker.should_speculate(context_pattern):
                proposals.append([])
                continue

            # Both checks pass: propose top-1 token
            proposals.append([int(top_idx.item())])

        return proposals

    def update_speculation_result(
        self,
        request_id: str,
        speculated_token: int,
        actual_token: int,
    ) -> None:
        """Update the confidence tracker with speculation outcome.

        Args:
            request_id: The request identifier (used to derive
                context pattern; currently hashed directly).
            speculated_token: The token that was speculatively
                proposed.
            actual_token: The token that was actually sampled.
        """
        hit = speculated_token == actual_token
        # Use request_id hash as context pattern
        context_pattern = hash(request_id) & 0x7FFFFFFF
        self._confidence_tracker.update(context_pattern, hit)

    def load_model(self, *args, **kwargs):
        """No separate model needed — uses the target model."""
        pass

    @staticmethod
    def _compute_context_pattern(token_ids: np.ndarray, num_tokens: int) -> int:
        """Compute a context pattern hash from recent tokens.

        Uses the last few tokens as a fingerprint for the context.
        This allows the confidence tracker to learn per-pattern
        hit rates.
        """
        # Use last 4 tokens (or fewer if not available) as pattern
        window = min(4, num_tokens)
        if window <= 0:
            return 0
        recent = token_ids[num_tokens - window : num_tokens]
        # Simple hash combining recent token IDs
        h = 0
        for tok in recent:
            h = ((h * 31) + int(tok)) & 0x7FFFFFFF
        return h
