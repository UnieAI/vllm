# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


@dataclass
class JacobiDraftTokenIds:
    tokens: torch.Tensor
    lengths: list[int]


class JacobiProposer:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.method == "jacobi"

        self.device = device
        self.max_model_len = vllm_config.model_config.max_model_len
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        self.num_blocks = max(self.speculative_config.jacobi_num_blocks, 1)
        self.block_size = self.num_speculative_tokens // self.num_blocks
        self.steps_per_yield = max(self.speculative_config.jacobi_steps_per_yield, 1)
        self.ngram_pool_size = max(self.speculative_config.jacobi_ngram_pool_size, 0)
        self.prefill_random = self.speculative_config.jacobi_prefill_random

        # DSC parameters
        self.max_batch_size = self.speculative_config.jacobi_max_batch_size
        self.accept_rate_low = self.speculative_config.jacobi_accept_rate_low
        self.accept_rate_high = self.speculative_config.jacobi_accept_rate_high
        self.ema_decay = self.speculative_config.jacobi_accept_rate_ema_decay
        self.warmup_steps = self.speculative_config.jacobi_accept_rate_warmup
        self.probe_interval = self.speculative_config.jacobi_accept_rate_probe_interval

        # DSC state
        self.current_ema = 1.0
        self.step_counter = 0
        self.gate_open = True
        self.probe_counter = 0

        self._offsets_buf: torch.Tensor | None = None
        self._idx_buf: torch.Tensor | None = None
        self._mask_buf: torch.Tensor | None = None

    def propose(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
        logits: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
        return_device: torch.device | None = None,
    ) -> JacobiDraftTokenIds:
        num_reqs = input_batch.num_reqs
        if not self._use_jacobi_for_decode(input_batch, request_states):
            return JacobiDraftTokenIds(
                tokens=torch.empty(
                    (num_reqs, 0),
                    device=return_device or "cpu",
                    dtype=torch.int32,
                ),
                lengths=[0] * num_reqs,
            )

        if spec_decode_metadata is None:
            return self._init_drafts(
                input_batch, request_states, return_device=return_device
            )

        return self._update_drafts(
            input_batch,
            request_states,
            logits,
            spec_decode_metadata,
            return_device=return_device,
        )

    def _use_jacobi_for_decode(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
    ) -> bool:
        if self.max_batch_size is not None and input_batch.num_reqs > self.max_batch_size:
            return False

        if not self._is_greedy_sampling(input_batch, request_states):
            return False

        if self.accept_rate_low is None:
            return True

        self.step_counter += 1
        if self.step_counter < self.warmup_steps:
            return True

        if self.gate_open:
            self.probe_counter = 0
            if self.current_ema < self.accept_rate_low:
                logger.info(
                    "[DSC] Closing Jacobi gate. EMA: %.3f < %.3f",
                    self.current_ema,
                    self.accept_rate_low,
                )
                self.gate_open = False
        else:
            if self.accept_rate_high is not None and self.current_ema > self.accept_rate_high:
                logger.info(
                    "[DSC] Re-opening Jacobi gate. EMA: %.3f > %.3f",
                    self.current_ema,
                    self.accept_rate_high,
                )
                self.gate_open = True

            if not self.gate_open and self.probe_interval > 0:
                self.probe_counter += 1
                if self.probe_counter >= self.probe_interval:
                    self.probe_counter = 0
                    return True

        return self.gate_open

    def refine_drafts(
        self,
        input_batch: InputBatch,
        logits: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> tuple[torch.Tensor, bool]:
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return torch.empty((0,), device=logits.device, dtype=torch.int32), True

        max_spec_len = spec_decode_metadata.max_spec_len
        if max_spec_len == 0:
            return torch.empty((0,), device=logits.device, dtype=torch.int32), True

        target_logits = logits[spec_decode_metadata.target_logits_indices]
        greedy_flat = target_logits.argmax(dim=-1)

        lengths = torch.tensor(
            spec_decode_metadata.num_draft_tokens,
            device=logits.device,
            dtype=torch.int64,
        )
        start_indices = torch.zeros_like(lengths)
        cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens.to(
            device=logits.device, dtype=lengths.dtype
        )
        if num_reqs > 1:
            start_indices[1:] = cu_num_draft_tokens[:-1]

        offsets, idx, mask = self._get_index_buffers(
            num_reqs, max_spec_len, logits.device
        )
        idx.copy_(start_indices.unsqueeze(1).expand_as(idx))
        idx.add_(offsets)
        torch.lt(offsets, lengths.unsqueeze(1), out=mask)
        idx.masked_fill_(~mask, 0)

        draft_padded = spec_decode_metadata.draft_token_ids[idx]
        greedy_padded = greedy_flat[idx]
        mismatch = (draft_padded != greedy_padded) & mask
        all_converged = not mismatch.any().item()

        return greedy_flat.to(torch.int32), all_converged

    def _is_greedy_sampling(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
    ) -> bool:
        for req_id in input_batch.req_ids:
            params = request_states[req_id].sampling_params
            if params is None:
                continue
            if params.temperature > _SAMPLING_EPS:
                return False
            if params.top_p < 1.0:
                return False
            if params.top_k not in (0, 1, -1):
                return False
            if params.min_p > _SAMPLING_EPS:
                return False
            if params.frequency_penalty != 0.0:
                return False
            if params.presence_penalty != 0.0:
                return False
            if params.repetition_penalty != 1.0:
                return False
            if params.logit_bias:
                return False
            if params.n != 1:
                return False
        return True

    def _init_drafts(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
        return_device: torch.device | None = None,
    ) -> JacobiDraftTokenIds:
        num_reqs = input_batch.num_reqs
        lengths = [0] * num_reqs
        max_spec_len = 0
        use_gpu = return_device is not None and return_device.type == "cuda"
        drafts = [None] * num_reqs
        for i, req_id in enumerate(input_batch.req_ids):
            req_state = request_states[req_id]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens <= input_batch.num_prompt_tokens[i]:
                continue
            if num_tokens <= 0:
                continue
            if num_tokens >= self.max_model_len:
                continue

            draft_len = min(
                self.num_speculative_tokens,
                max(self.max_model_len - num_tokens, 0),
            )
            if draft_len <= 0:
                continue

            if self.prefill_random and self.vocab_size > 0:
                draft = torch.randint(
                    0, self.vocab_size, (draft_len,), device=self.device
                )
            else:
                last_token = int(input_batch.token_ids_cpu[i, num_tokens - 1])
                draft = torch.full(
                    (draft_len,),
                    last_token,
                    dtype=torch.long,
                    device=self.device,
                )

            req_state.jacobi_needs_bootstrap = True
            if self.ngram_pool_size > 0:
                req_state.jacobi_ngram_pool = deque(
                    maxlen=self.ngram_pool_size
                )
            else:
                req_state.jacobi_ngram_pool.clear()
            if use_gpu:
                drafts[i] = draft.to(torch.int32)
            else:
                drafts[i] = draft.to(torch.int32).cpu()
            lengths[i] = draft_len
            if draft_len > max_spec_len:
                max_spec_len = draft_len

        if max_spec_len == 0:
            return JacobiDraftTokenIds(
                tokens=torch.empty(
                    (num_reqs, 0),
                    device=return_device or "cpu",
                    dtype=torch.int32,
                ),
                lengths=lengths,
            )

        padded = torch.zeros(
            (num_reqs, max_spec_len),
            dtype=torch.int32,
            device=return_device if use_gpu else "cpu",
        )
        for i, draft in enumerate(drafts):
            if draft is None:
                continue
            padded[i, : lengths[i]] = draft

        return JacobiDraftTokenIds(tokens=padded, lengths=lengths)

    def _update_drafts(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
        logits: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
        return_device: torch.device | None = None,
    ) -> JacobiDraftTokenIds:
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return JacobiDraftTokenIds(
                tokens=torch.empty(
                    (0, 0), device=return_device or "cpu", dtype=torch.int32
                ),
                lengths=[],
            )

        target_logits = logits[spec_decode_metadata.target_logits_indices]
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        draft_token_ids = spec_decode_metadata.draft_token_ids

        num_draft_tokens = spec_decode_metadata.num_draft_tokens
        max_spec_len = spec_decode_metadata.max_spec_len
        if max_spec_len == 0:
            return JacobiDraftTokenIds(
                tokens=torch.empty(
                    (num_reqs, 0),
                    device=return_device or "cpu",
                    dtype=torch.int32,
                ),
                lengths=num_draft_tokens,
            )

        lengths = torch.tensor(
            num_draft_tokens, device=logits.device, dtype=torch.int64
        )
        start_indices = torch.zeros_like(lengths)
        cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens.to(
            device=logits.device, dtype=lengths.dtype
        )
        if num_reqs > 1:
            start_indices[1:] = cu_num_draft_tokens[:-1]

        offsets, idx, mask = self._get_index_buffers(
            num_reqs, max_spec_len, logits.device
        )
        idx.copy_(start_indices.unsqueeze(1).expand_as(idx))
        idx.add_(offsets)
        torch.lt(offsets, lengths.unsqueeze(1), out=mask)
        idx.masked_fill_(~mask, 0)

        greedy_flat = target_logits.argmax(dim=-1)
        draft_padded = draft_token_ids[idx]
        greedy_padded = greedy_flat[idx]

        mismatch = (draft_padded != greedy_padded) & mask
        has_mismatch = mismatch.any(dim=1)
        first_mismatch = mismatch.float().argmax(dim=1).to(lengths.dtype)
        accepted_lens = torch.where(has_mismatch, first_mismatch, lengths)
        bonus_tokens = bonus_logits.argmax(dim=-1).to(torch.int64)

        bootstrap_mask = torch.tensor(
            [
                request_states[req_id].jacobi_needs_bootstrap
                for req_id in input_batch.req_ids
            ],
            device=logits.device,
            dtype=torch.bool,
        )

        safe_lengths = torch.maximum(lengths, torch.ones_like(lengths))
        shift_indices = accepted_lens.unsqueeze(1) + offsets.unsqueeze(0)
        clamp_shift = torch.minimum(
            shift_indices, (safe_lengths - 1).unsqueeze(1)
        ).to(torch.int64)
        greedy_shifted = torch.gather(greedy_padded, 1, clamp_shift)

        accept_all_mask = accepted_lens == lengths
        base_drafts = torch.where(
            accept_all_mask.unsqueeze(1),
            bonus_tokens.unsqueeze(1),
            greedy_shifted,
        )
        base_drafts = torch.where(
            bootstrap_mask.unsqueeze(1), greedy_padded, base_drafts
        )
        padded_drafts = torch.where(
            mask, base_drafts, torch.zeros_like(base_drafts)
        ).to(torch.int32)

        combined = torch.cat(
            [accepted_lens.to(torch.int32).unsqueeze(1), padded_drafts], dim=1
        ).cpu()
        accepted_lens_cpu = combined[:, 0].tolist()
        padded_tensor = combined[:, 1:].to(torch.int32)

        total_accepted_tokens = 0
        total_drafted_tokens = 0
        for i, req_id in enumerate(input_batch.req_ids):
            draft_len = num_draft_tokens[i]
            if draft_len <= 0:
                continue

            req_state = request_states[req_id]
            if req_state.jacobi_needs_bootstrap:
                req_state.jacobi_needs_bootstrap = False
                continue

            accepted_len = accepted_lens_cpu[i]
            total_accepted_tokens += accepted_len
            total_drafted_tokens += draft_len

            if self.ngram_pool_size > 0:
                if accepted_len < draft_len:
                    pool_tail = self._select_ngram_tail(req_state, draft_len - 1)
                    if pool_tail is not None:
                        padded_tensor[i, 1:draft_len] = torch.tensor(
                            pool_tail, dtype=padded_tensor.dtype
                        )
                self._append_ngram_pool(
                    req_state, padded_tensor[i, :draft_len].tolist()
                )

        if self.accept_rate_low is not None:
            self._update_acceptance_rate(total_accepted_tokens, total_drafted_tokens)

        if return_device is not None and return_device.type == "cuda":
            return JacobiDraftTokenIds(
                tokens=padded_drafts, lengths=num_draft_tokens
            )
        return JacobiDraftTokenIds(tokens=padded_tensor, lengths=num_draft_tokens)

    def _get_index_buffers(
        self, num_reqs: int, max_spec_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self._offsets_buf is None or self._offsets_buf.device != device
                or self._offsets_buf.numel() < max_spec_len):
            self._offsets_buf = torch.arange(
                max_spec_len, device=device, dtype=torch.int64
            )
        offsets = self._offsets_buf[:max_spec_len]

        if (self._idx_buf is None or self._idx_buf.device != device
                or self._idx_buf.shape[0] < num_reqs
                or self._idx_buf.shape[1] < max_spec_len):
            self._idx_buf = torch.empty(
                (num_reqs, max_spec_len), device=device, dtype=torch.int64
            )
        idx = self._idx_buf[:num_reqs, :max_spec_len]

        if (self._mask_buf is None or self._mask_buf.device != device
                or self._mask_buf.shape[0] < num_reqs
                or self._mask_buf.shape[1] < max_spec_len):
            self._mask_buf = torch.empty(
                (num_reqs, max_spec_len), device=device, dtype=torch.bool
            )
        mask = self._mask_buf[:num_reqs, :max_spec_len]

        return offsets, idx, mask

    def _pad_or_truncate(self, tokens: torch.Tensor, draft_len: int) -> torch.Tensor:
        if tokens.numel() < draft_len:
            pad_len = draft_len - tokens.numel()
            pad_token = tokens[-1]
            pad_tokens = pad_token.repeat(pad_len)
            return torch.cat([tokens, pad_tokens], dim=0)
        if tokens.numel() > draft_len:
            return tokens[:draft_len]
        return tokens

    def _update_acceptance_rate(self, num_accepted: int, total_proposed: int) -> None:
        if total_proposed == 0:
            return
        current_rate = num_accepted / total_proposed
        self.current_ema = (
            self.ema_decay * self.current_ema + (1.0 - self.ema_decay) * current_rate
        )

    def _append_ngram_pool(self, req_state: CachedRequestState, tokens: list[int]):
        if self.ngram_pool_size <= 0:
            return
        pool = req_state.jacobi_ngram_pool
        if pool.maxlen != self.ngram_pool_size:
            pool = deque(pool, maxlen=self.ngram_pool_size)
            req_state.jacobi_ngram_pool = pool
        pool.append(tokens)

    def _select_ngram_tail(
        self, req_state: CachedRequestState, length: int
    ) -> list[int] | None:
        if self.ngram_pool_size <= 0 or length <= 0:
            return None
        pool = req_state.jacobi_ngram_pool
        if not pool:
            return None
        candidate = pool[-1]
        if not candidate:
            return None
        if len(candidate) >= length:
            return candidate[:length]
        pad = [candidate[-1]] * (length - len(candidate))
        return candidate + pad
