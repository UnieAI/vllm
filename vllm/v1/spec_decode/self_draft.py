# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Self-Speculative Decoding: use the first K layers of the target model
as a draft model, with the full model verifying.

Zero extra GPU memory — shares all weights with the target model by
temporarily patching the model's layer iteration bounds during draft
generation.

Usage:
    vllm serve <model> \\
        --speculative-config '{"method": "self_draft", \\
            "num_speculative_tokens": 3, \\
            "self_draft_depth": 8}'

Or programmatically:
    from vllm.v1.spec_decode.self_draft import SelfDraftProposer
    proposer = SelfDraftProposer(model, draft_depth=8, k=3, device="cuda")
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


class SelfDraftProposer:
    """Draft proposer that reuses the target model's first K layers.

    During draft generation:
    1. Patch model.end_layer → draft_depth (skip deeper layers)
    2. Run forward pass (embedding → K layers → norm → LM head)
    3. Restore model.end_layer → original value

    During verification: the full model runs normally (no patching).

    This is a standalone class that does NOT inherit from
    SpecDecodeBaseProposer to avoid coupling with the existing spec
    decode pipeline.  It can be integrated later if validated.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_depth: int,
        k: int,
        device: torch.device,
        vllm_config: "VllmConfig | None" = None,
    ):
        """
        Args:
            target_model: The full model (e.g., LlamaForCausalLM).
            draft_depth: Number of layers to use for drafting (e.g., 8
                         out of 32 for Llama-7B).
            k: Number of draft tokens to propose per step.
            device: CUDA device.
            vllm_config: Optional vllm config for metadata.
        """
        self.model = target_model
        self.draft_depth = draft_depth
        self.k = k
        self.device = device
        self.vllm_config = vllm_config

        # Find the inner model (e.g., LlamaModel inside LlamaForCausalLM).
        self._inner_model = self._find_inner_model(target_model)
        self._full_end_layer = self._inner_model.end_layer
        self._full_start_layer = self._inner_model.start_layer

        total_layers = self._full_end_layer - self._full_start_layer
        if draft_depth > total_layers:
            raise ValueError(
                f"draft_depth ({draft_depth}) > total layers ({total_layers})"
            )
        if draft_depth < 1:
            raise ValueError(f"draft_depth must be >= 1, got {draft_depth}")

        logger.info(
            "SelfDraftProposer: using %d/%d layers for drafting, k=%d",
            draft_depth, total_layers, k,
        )

    @staticmethod
    def _find_inner_model(model: nn.Module) -> nn.Module:
        """Find the inner transformer model that has start_layer/end_layer."""
        # Most HuggingFace models: model.model (e.g., LlamaForCausalLM.model)
        if hasattr(model, "model") and hasattr(model.model, "end_layer"):
            return model.model
        # Direct model (already the inner one)
        if hasattr(model, "end_layer"):
            return model
        raise ValueError(
            f"Cannot find inner model with start_layer/end_layer in "
            f"{type(model).__name__}. Supported architectures: "
            f"Llama, Qwen2, Mistral, and other HuggingFace-style models."
        )

    @contextmanager
    def _shallow_forward_mode(self):
        """Temporarily patch the model to only run first K layers."""
        original_end = self._inner_model.end_layer
        self._inner_model.end_layer = (
            self._full_start_layer + self.draft_depth
        )
        try:
            yield
        finally:
            self._inner_model.end_layer = original_end

    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Generate draft tokens using the first K layers.

        This is a simplified proposer for prototyping.  For production
        integration, wrap this in a SpecDecodeBaseProposer subclass.

        Args:
            input_ids: [batch_size] — the last accepted token per request.
            positions: [batch_size] — position IDs for each token.
            target_hidden_states: unused (for API compat).

        Returns:
            List of draft token ID lists, one per request.
        """
        batch_size = input_ids.shape[0]
        all_drafts: list[list[int]] = [[] for _ in range(batch_size)]

        # Current token IDs and positions for iterative draft generation.
        cur_ids = input_ids.clone()
        cur_pos = positions.clone()

        for step in range(self.k):
            # Run forward pass with only the first K layers.
            with self._shallow_forward_mode():
                # The model's forward() will:
                #   embed → layers[0:draft_depth] → norm → lm_head
                logits = self.model(
                    input_ids=cur_ids.unsqueeze(1)
                    if cur_ids.dim() == 1
                    else cur_ids,
                    positions=cur_pos.unsqueeze(1)
                    if cur_pos.dim() == 1
                    else cur_pos,
                    intermediate_tensors=None,
                )

            # Handle different return types.
            if isinstance(logits, tuple):
                logits = logits[0]

            # Greedy sample.
            if logits.dim() == 3:
                # [batch, seq_len, vocab] → take last token
                next_token_logits = logits[:, -1, :]
            elif logits.dim() == 2:
                # [batch, vocab]
                next_token_logits = logits
            else:
                break

            next_tokens = next_token_logits.argmax(dim=-1)  # [batch]

            # Append to drafts.
            next_tokens_cpu = next_tokens.cpu().tolist()
            for i in range(batch_size):
                all_drafts[i].append(next_tokens_cpu[i])

            # Prepare next step.
            cur_ids = next_tokens
            cur_pos = cur_pos + 1

        return all_drafts

    def get_acceptance_stats(self) -> dict:
        """Return metadata about the draft configuration."""
        total = self._full_end_layer - self._full_start_layer
        return {
            "draft_depth": self.draft_depth,
            "total_layers": total,
            "layer_ratio": self.draft_depth / total,
            "k": self.k,
        }
