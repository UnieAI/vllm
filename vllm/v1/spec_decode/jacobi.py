# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


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

        if self.steps_per_yield > 1:
            logger.warning(
                "Jacobi steps_per_yield > 1 is not supported in vLLM v1; "
                "using a single step per iteration."
            )

    def propose(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
        logits: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> list[list[int]]:
        num_reqs = input_batch.num_reqs
        if not self._use_jacobi_for_decode(input_batch, request_states):
            return [[] for _ in range(num_reqs)]

        if spec_decode_metadata is None:
            return self._init_drafts(input_batch, request_states)

        return self._update_drafts(
            input_batch, request_states, logits, spec_decode_metadata
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
    ) -> list[list[int]]:
        draft_token_ids: list[list[int]] = []
        for i, req_id in enumerate(input_batch.req_ids):
            req_state = request_states[req_id]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens <= input_batch.num_prompt_tokens[i]:
                draft_token_ids.append([])
                continue
            if num_tokens <= 0:
                draft_token_ids.append([])
                continue
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            draft_len = min(
                self.num_speculative_tokens,
                max(self.max_model_len - num_tokens, 0),
            )
            if draft_len <= 0:
                draft_token_ids.append([])
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
            req_state.jacobi_ngram_pool.clear()
            draft_token_ids.append(draft.tolist())

        return draft_token_ids

    def _update_drafts(
        self,
        input_batch: InputBatch,
        request_states: dict[str, CachedRequestState],
        logits: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> list[list[int]]:
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return []

        target_logits = logits[spec_decode_metadata.target_logits_indices]
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        draft_token_ids = spec_decode_metadata.draft_token_ids

        num_draft_tokens = spec_decode_metadata.num_draft_tokens
        max_spec_len = spec_decode_metadata.max_spec_len
        if max_spec_len == 0:
            return [[] for _ in range(num_reqs)]

        lengths = torch.tensor(
            num_draft_tokens, device=logits.device, dtype=torch.int32
        )
        start_indices = torch.zeros_like(lengths)
        cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens.to(
            device=logits.device
        )
        if num_reqs > 1:
            start_indices[1:] = cu_num_draft_tokens[:-1]

        offsets = torch.arange(max_spec_len, device=logits.device, dtype=torch.int32)
        idx = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        mask = offsets.unsqueeze(0) < lengths.unsqueeze(1)
        idx_safe = torch.where(mask, idx, torch.zeros_like(idx))

        greedy_flat = target_logits.argmax(dim=-1)
        draft_padded = draft_token_ids[idx_safe]
        greedy_padded = greedy_flat[idx_safe]

        mismatch = (draft_padded != greedy_padded) & mask
        has_mismatch = mismatch.any(dim=1)
        first_mismatch = mismatch.float().argmax(dim=1)
        accepted_lens = torch.where(has_mismatch, first_mismatch, lengths)
        accepted_lens_cpu = accepted_lens.cpu().tolist()
        bonus_tokens = bonus_logits.argmax(dim=-1)

        total_accepted_tokens = 0
        total_drafted_tokens = 0
        padded_drafts = torch.zeros(
            (num_reqs, max_spec_len),
            dtype=torch.int32,
            device=logits.device,
        )
        append_pool = [False] * num_reqs

        for i, req_id in enumerate(input_batch.req_ids):
            req_state = request_states[req_id]
            draft_len = num_draft_tokens[i]
            if draft_len <= 0:
                continue

            cur_draft = draft_padded[i, :draft_len]
            greedy_tokens = greedy_padded[i, :draft_len]

            if req_state.jacobi_needs_bootstrap:
                new_draft = greedy_tokens.to(torch.int32)
                req_state.jacobi_needs_bootstrap = False
                padded_drafts[i, :draft_len] = new_draft
                continue

            accepted_len = accepted_lens_cpu[i]

            had_rejection = accepted_len < draft_len
            total_accepted_tokens += accepted_len
            total_drafted_tokens += draft_len

            if accepted_len < draft_len:
                next_token = greedy_tokens[accepted_len : accepted_len + 1]
                tail_tokens = greedy_tokens[accepted_len + 1 : draft_len]
                new_draft = (
                    torch.cat([next_token, tail_tokens], dim=0)
                    if tail_tokens.numel() > 0
                    else next_token
                )
            else:
                new_draft = bonus_tokens[i].view(1)

            if had_rejection and self.ngram_pool_size > 0:
                pool_tail = self._select_ngram_tail(req_state, draft_len - 1)
                if pool_tail is not None:
                    pool_tail_tensor = torch.tensor(
                        pool_tail, dtype=new_draft.dtype, device=self.device
                    )
                    new_draft = torch.cat([new_draft[:1], pool_tail_tensor], dim=0)

            new_draft = self._pad_or_truncate(new_draft, draft_len).to(torch.int32)

            padded_drafts[i, :draft_len] = new_draft
            if self.ngram_pool_size > 0:
                append_pool[i] = True

        if self.accept_rate_low is not None:
            self._update_acceptance_rate(total_accepted_tokens, total_drafted_tokens)

        padded_drafts_cpu = padded_drafts.cpu().tolist()
        if self.ngram_pool_size > 0:
            for i, req_id in enumerate(input_batch.req_ids):
                if not append_pool[i]:
                    continue
                draft_len = num_draft_tokens[i]
                if draft_len <= 0:
                    continue
                self._append_ngram_pool(
                    request_states[req_id], padded_drafts_cpu[i][:draft_len]
                )
        return [
            padded_drafts_cpu[i][: num_draft_tokens[i]] for i in range(num_reqs)
        ]

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
        pool.append(tokens)
        if len(pool) > self.ngram_pool_size:
            del pool[0]

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
