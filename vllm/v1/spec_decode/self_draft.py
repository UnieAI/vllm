# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Self-Speculative Decoding: use a subset of the target model's layers
as a draft model, with the full model verifying.

Zero extra GPU memory — shares all weights with the target model by
temporarily patching layer behaviour during draft generation.

Three skip patterns are supported:

  prefix  — Use first K layers only (default)
  even    — Skip every other layer (run 0,2,4,...) — better agreement
  custom  — Specify exact layer indices to keep

Usage:
    # Prefix (first 8 layers):
    vllm serve <model> --speculative-config \\
        '{"method": "self_draft", "num_speculative_tokens": 3,
          "self_draft_depth": 8}'

    # Even skip (half the layers, evenly distributed):
    vllm serve <model> --speculative-config \\
        '{"method": "self_draft", "num_speculative_tokens": 3,
          "self_draft_depth": 16, "self_draft_skip_pattern": "even"}'

    # Auto-probe (finds optimal depth and pattern):
    vllm serve <model> --speculative-config \\
        '{"method": "self_draft", "num_speculative_tokens": 3,
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

# Sentinel for identity-replaced layers.
_ORIG_FORWARD_ATTR = "_self_draft_orig_forward"


class SelfDraftProposer:
    """Draft proposer that reuses a subset of the target model's layers.

    Supports three skip patterns:
      - ``prefix``: Run layers [0, draft_depth). Simple end_layer patch.
      - ``even``: Run every other layer. Same compute budget as prefix
        at half the depth, but preserves features from all depths.
      - ``custom``: Run only the specified ``layer_indices``.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_depth: int,
        k: int,
        device: torch.device,
        skip_pattern: str = "prefix",
        layer_indices: list[int] | None = None,
        vllm_config: "VllmConfig | None" = None,
    ):
        self.model = target_model
        self.k = k
        self.device = device
        self.skip_pattern = skip_pattern
        self.vllm_config = vllm_config

        # Find inner transformer model.
        self._inner = self._find_inner_model(target_model)
        self._full_end = self._inner.end_layer
        self._full_start = self._inner.start_layer
        self._total_layers = self._full_end - self._full_start

        # Auto-probe: depth=-1.
        if draft_depth == -1:
            draft_depth, skip_pattern = self._auto_probe()
            self.skip_pattern = skip_pattern

        if draft_depth < 1 or draft_depth > self._total_layers:
            raise ValueError(
                f"draft_depth must be in [1, {self._total_layers}], "
                f"got {draft_depth}"
            )
        self.draft_depth = draft_depth

        # Compute which layers to SKIP based on pattern.
        if skip_pattern == "prefix":
            # Skip layers [draft_depth, total).
            self._skip_indices: list[int] = []  # handled by end_layer patch
        elif skip_pattern == "even":
            # Keep every other layer: 0, 2, 4, ...
            keep = set(range(self._full_start, self._full_end, 2))
            self._skip_indices = [
                i for i in range(self._full_start, self._full_end)
                if i not in keep
            ]
        elif skip_pattern == "custom":
            if layer_indices is None:
                raise ValueError("layer_indices required for custom pattern")
            keep = set(layer_indices)
            self._skip_indices = [
                i for i in range(self._full_start, self._full_end)
                if i not in keep
            ]
        else:
            raise ValueError(f"Unknown skip_pattern: {skip_pattern}")

        n_active = self._total_layers - len(self._skip_indices)
        if skip_pattern == "prefix":
            n_active = draft_depth

        logger.info(
            "SelfDraftProposer: pattern=%s, %d/%d active layers, k=%d",
            skip_pattern, n_active, self._total_layers, k,
        )

    @staticmethod
    def _find_inner_model(model: nn.Module) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "end_layer"):
            return model.model
        if hasattr(model, "end_layer"):
            return model
        raise ValueError(
            f"Cannot find layer bounds in {type(model).__name__}. "
            "Supported: Llama, Qwen2, Mistral, and other HF-style models."
        )

    # ── Skip mode context managers ──────────────────────────────────

    @contextmanager
    def _draft_mode(self):
        """Enter draft mode with the configured skip pattern."""
        if self.skip_pattern == "prefix":
            with self._prefix_mode():
                yield
        else:
            with self._layer_skip_mode():
                yield

    @contextmanager
    def _prefix_mode(self):
        """Run only the first draft_depth layers (end_layer patch)."""
        orig = self._inner.end_layer
        self._inner.end_layer = self._full_start + self.draft_depth
        try:
            yield
        finally:
            self._inner.end_layer = orig

    @contextmanager
    def _layer_skip_mode(self):
        """Skip specific layers by replacing their forward with identity.

        This approach does NOT modify the model's layer iteration or
        the forward() method — it patches individual layer.forward
        callables to pass through (hidden_states, residual) unchanged.
        """
        patched: list[tuple[int, object]] = []
        for idx in self._skip_indices:
            layer = self._inner.layers[idx]
            orig = layer.forward
            # Store original for restoration.
            setattr(layer, _ORIG_FORWARD_ATTR, orig)
            # Replace with identity: return (hidden_states, residual) as-is.
            layer.forward = _identity_layer_forward
            patched.append((idx, orig))
        try:
            yield
        finally:
            for idx, orig in patched:
                layer = self._inner.layers[idx]
                layer.forward = orig
                if hasattr(layer, _ORIG_FORWARD_ATTR):
                    delattr(layer, _ORIG_FORWARD_ATTR)

    # ── Auto-probe ──────────────────────────────────────────────────

    def _auto_probe(self) -> tuple[int, str]:
        """Find optimal (depth, pattern) by testing candidates.

        Tests prefix and even patterns at multiple depths.
        Returns (best_depth, best_pattern).
        """
        candidates = []
        for depth in sorted(set([
            max(1, self._total_layers // 4),
            max(1, self._total_layers // 2),
            max(1, self._total_layers * 3 // 4),
        ])):
            candidates.append((depth, "prefix"))
        # Even skip: test at 50% of layers (every other layer).
        candidates.append((self._total_layers, "even"))

        logger.info(
            "Auto-probing: testing %d configurations...", len(candidates),
        )

        probe_tokens = torch.randint(
            1, 1000, (1, 32), dtype=torch.long, device=self.device,
        )
        probe_pos = torch.arange(
            32, dtype=torch.long, device=self.device,
        ).unsqueeze(0)

        # Full model reference.
        with torch.inference_mode():
            full_out = self.model(
                input_ids=probe_tokens,
                positions=probe_pos,
                intermediate_tensors=None,
            )
            if isinstance(full_out, tuple):
                full_out = full_out[0]
            full_tokens = full_out[:, -1, :].argmax(dim=-1)

        best = (candidates[0][0], candidates[0][1], 0.0)

        for depth, pattern in candidates:
            # Temporarily configure skip indices.
            if pattern == "prefix":
                skip = []
                n_active = depth
            elif pattern == "even":
                keep = set(range(self._full_start, self._full_end, 2))
                skip = [i for i in range(self._full_start, self._full_end)
                        if i not in keep]
                n_active = self._total_layers - len(skip)
            else:
                continue

            # Patch model.
            old_skip = self._skip_indices
            old_pattern = self.skip_pattern
            old_depth = getattr(self, "draft_depth", depth)
            self._skip_indices = skip
            self.skip_pattern = pattern
            self.draft_depth = depth

            with torch.inference_mode():
                t0 = time.perf_counter()
                with self._draft_mode():
                    draft_out = self.model(
                        input_ids=probe_tokens,
                        positions=probe_pos,
                        intermediate_tensors=None,
                    )
                draft_time = time.perf_counter() - t0

            # Restore.
            self._skip_indices = old_skip
            self.skip_pattern = old_pattern
            self.draft_depth = old_depth

            if isinstance(draft_out, tuple):
                draft_out = draft_out[0]
            draft_tokens = draft_out[:, -1, :].argmax(dim=-1)
            agree = (draft_tokens == full_tokens).float().mean().item()
            speedup = self._total_layers / max(n_active, 1)
            score = agree * speedup

            logger.info(
                "  %s depth=%d: %d/%d active, agreement=%.0f%%, "
                "speedup=%.1fx, score=%.2f",
                pattern, depth, n_active, self._total_layers,
                agree * 100, speedup, score,
            )

            if score > best[2]:
                best = (depth, pattern, score)

        logger.info(
            "Auto-probe result: pattern=%s, depth=%d (score=%.2f)",
            best[1], best[0], best[2],
        )
        return best[0], best[1]

    # ── Propose ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Generate draft tokens using the configured layer subset."""
        batch_size = input_ids.shape[0]
        drafts: list[list[int]] = [[] for _ in range(batch_size)]

        cur_ids = input_ids.clone()
        cur_pos = positions.clone()

        for _ in range(self.k):
            with self._draft_mode():
                out = self.model(
                    input_ids=(cur_ids.unsqueeze(1) if cur_ids.dim() == 1
                               else cur_ids),
                    positions=(cur_pos.unsqueeze(1) if cur_pos.dim() == 1
                               else cur_pos),
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


def _identity_layer_forward(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Identity replacement for skipped layers.

    Returns (hidden_states, residual) unchanged, matching the
    signature of LlamaDecoderLayer.forward().
    """
    return hidden_states, residual
