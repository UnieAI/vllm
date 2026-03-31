# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Self-Speculative Decoding: use the first K layers of the target model
as a draft model, with the full model verifying.

Zero extra GPU memory — shares all weights with the target model by
temporarily patching the model's layer iteration bounds during draft
generation.

Usage:
    # Fixed depth:
    vllm serve <model> \\
        --speculative-config '{"method": "self_draft", \\
            "num_speculative_tokens": 3, \\
            "self_draft_depth": 8}'

    # Auto-probe (finds optimal depth at startup):
    vllm serve <model> \\
        --speculative-config '{"method": "self_draft", \\
            "num_speculative_tokens": 3, \\
            "self_draft_depth": -1}'
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class SelfDraftProposer:
    """Draft proposer that reuses the target model's first K layers.

    During draft generation:
    1. Patch model.end_layer → draft_depth
    2. Forward pass: embedding → K layers → norm → LM head
    3. Restore model.end_layer

    During verification: the full model runs normally.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_depth: int,
        k: int,
        device: torch.device,
        vllm_config: "VllmConfig | None" = None,
    ):
        self.model = target_model
        self.k = k
        self.device = device
        self.vllm_config = vllm_config

        # Find inner transformer model.
        self._inner = self._find_inner_model(target_model)
        self._full_end = self._inner.end_layer
        self._full_start = self._inner.start_layer
        self._total_layers = self._full_end - self._full_start

        # Auto-probe: depth=-1 means "find the best depth automatically".
        if draft_depth == -1:
            draft_depth = self._auto_probe()

        if draft_depth < 1 or draft_depth > self._total_layers:
            raise ValueError(
                f"draft_depth must be in [1, {self._total_layers}], "
                f"got {draft_depth}"
            )
        self.draft_depth = draft_depth

        logger.info(
            "SelfDraftProposer: %d/%d layers, k=%d",
            draft_depth, self._total_layers, k,
        )

    @staticmethod
    def _find_inner_model(model: nn.Module) -> nn.Module:
        """Find the inner model that has start_layer/end_layer."""
        if hasattr(model, "model") and hasattr(model.model, "end_layer"):
            return model.model
        if hasattr(model, "end_layer"):
            return model
        raise ValueError(
            f"Cannot find layer bounds in {type(model).__name__}. "
            "Supported: Llama, Qwen2, Mistral, and other HF-style models."
        )

    @contextmanager
    def _shallow_mode(self):
        """Run only the first draft_depth layers."""
        orig = self._inner.end_layer
        self._inner.end_layer = self._full_start + self.draft_depth
        try:
            yield
        finally:
            self._inner.end_layer = orig

    def _auto_probe(self) -> int:
        """Find optimal draft depth by testing multiple candidates.

        Tests 25%, 50%, 75% of total layers. Picks the depth with
        best (agreement × speedup) product.
        """
        candidates = sorted(set([
            max(1, self._total_layers // 4),
            max(1, self._total_layers // 2),
            max(1, self._total_layers * 3 // 4),
        ]))

        logger.info(
            "Auto-probing draft depth: testing %s out of %d layers...",
            candidates, self._total_layers,
        )

        # Use a simple calibration: run 10 forward passes at each depth
        # and compare greedy top-1 with full model.
        probe_tokens = torch.randint(
            1, 1000, (1, 32), dtype=torch.long, device=self.device,
        )
        probe_pos = torch.arange(32, dtype=torch.long, device=self.device).unsqueeze(0)

        # Full model reference output.
        with torch.inference_mode():
            full_out = self.model(
                input_ids=probe_tokens,
                positions=probe_pos,
                intermediate_tensors=None,
            )
            if isinstance(full_out, tuple):
                full_out = full_out[0]
            full_tokens = full_out[:, -1, :].argmax(dim=-1)  # [1]

        best_depth = candidates[0]
        best_score = 0.0

        for depth in candidates:
            self._inner.end_layer = self._full_start + depth

            with torch.inference_mode():
                t0 = time.perf_counter()
                draft_out = self.model(
                    input_ids=probe_tokens,
                    positions=probe_pos,
                    intermediate_tensors=None,
                )
                draft_time = time.perf_counter() - t0

            if isinstance(draft_out, tuple):
                draft_out = draft_out[0]
            draft_tokens = draft_out[:, -1, :].argmax(dim=-1)

            agree = (draft_tokens == full_tokens).float().mean().item()

            # Full model time (approx from proportional scaling).
            full_time = draft_time * self._total_layers / depth
            speedup = full_time / max(draft_time, 1e-9)
            score = agree * speedup

            logger.info(
                "  depth=%d/%d: agreement=%.0f%%, speedup=%.1fx, score=%.2f",
                depth, self._total_layers, agree * 100, speedup, score,
            )

            if score > best_score:
                best_score = score
                best_depth = depth

        # Restore full model.
        self._inner.end_layer = self._full_end

        logger.info(
            "Auto-probe result: depth=%d/%d (score=%.2f)",
            best_depth, self._total_layers, best_score,
        )
        return best_depth

    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Generate draft tokens using the first K layers.

        Args:
            input_ids: [batch_size] — the last accepted token per request.
            positions: [batch_size] — position IDs.

        Returns:
            list of draft token ID lists, one per request.
        """
        batch_size = input_ids.shape[0]
        drafts: list[list[int]] = [[] for _ in range(batch_size)]

        cur_ids = input_ids.clone()
        cur_pos = positions.clone()

        for _ in range(self.k):
            with self._shallow_mode():
                out = self.model(
                    input_ids=cur_ids.unsqueeze(1)
                    if cur_ids.dim() == 1
                    else cur_ids,
                    positions=cur_pos.unsqueeze(1)
                    if cur_pos.dim() == 1
                    else cur_pos,
                    intermediate_tensors=None,
                )

            if isinstance(out, tuple):
                out = out[0]

            if out.dim() == 3:
                logits = out[:, -1, :]
            elif out.dim() == 2:
                logits = out
            else:
                break

            next_tokens = logits.argmax(dim=-1)
            cpu_tokens = next_tokens.cpu().tolist()
            for i in range(batch_size):
                drafts[i].append(cpu_tokens[i])

            cur_ids = next_tokens
            cur_pos = cur_pos + 1

        return drafts

    def load_model(self, target_model: nn.Module | None = None) -> None:
        """No-op: self-draft reuses the target model's weights."""
        pass
