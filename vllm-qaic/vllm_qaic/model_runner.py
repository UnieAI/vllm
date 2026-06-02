"""QaicModelRunner: vLLM 0.21+ port of the v1_ngram fork's V1 QAIC runner.

PROVENANCE (read docs/UnieAI_Quic_integrated.md):
  * The base QaicModelRunner is Qualcomm's (marked "Confidential and
    Proprietary" in the fork). It runs the QPC on the AIC and keeps vLLM's
    scheduling/sampling on the host CPU.
  * The ngram speculative-decoding pieces in this file — the seven
    `_qaic_rejection_sample*` helpers, the 2D decode packing, and the ngram
    gating in __init__ — are UnieAI's original contribution. They are ported
    here VERBATIM from the fork (only SpecDecodeMetadata field access may need
    touch-ups, flagged inline).

The old fork carried QAIC-specific input packing inside execute_model. vLLM
0.21 moved most of that state into InputBatch/CpuGpuBuffer, so this runner lets
the upstream input-prep path populate the new buffers, then repacks the host
arrays into the QPC decode/prefill inputs expected by Qualcomm's loader.
"""

import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.logger import init_logger
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

# Ported QAIC model loader (see vllm_qaic/model_loader.py / port_from_fork.sh).
from vllm_qaic.model_loader import load_qaic_model

logger = init_logger(__name__)

# Resolve the sentinels from the SAME module the V1 RejectionSampler uses, so
# our CPU rejection sampler agrees with RejectionSampler.parse_output().
# Import (do NOT silently default): a mismatched PLACEHOLDER_TOKEN_ID would
# silently corrupt accepted/rejected speculative tokens, and the greedy
# sentinel changed across releases (v0.10.1 used -1.0; 0.21 uses
# GREEDY_TEMPERATURE == 0). gpu_model_runner already imports this module, so
# this adds no new dependency. If the import path moves, fail loudly here.
# (Verified against vllm/v1/sample/rejection_sampler.py on 0.21:
#  PLACEHOLDER_TOKEN_ID == -1, GREEDY_TEMPERATURE == 0.)
from vllm.v1.sample.rejection_sampler import (
    GREEDY_TEMPERATURE,
    PLACEHOLDER_TOKEN_ID,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import ModelRunnerOutput


class QaicModelRunner(GPUModelRunner):

    # -------------------------------------------------------------------------
    # __init__
    #   OLD: def __init__(self, vllm_config, device, speculative_model_type=None)
    #        -> super().__init__(vllm_config, device)
    #   NEW: GPUModelRunner.__init__ is (vllm_config, device) only AND its class
    #        now mixes in LoRA/KVConnector/ECConnector. There is no
    #        speculative_model_type parameter anymore — derive it internally.
    # -------------------------------------------------------------------------
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        # Check device first so a wrong device fails with a clear message.
        # NOTE: super().__init__ below is the 0.21 GPUModelRunner init (7000-line
        # class) and is the FIRST concrete on-machine failure point to test — it
        # may make CUDA/device assumptions that need re-fitting for a CPU-only
        # QAIC host (allocating device buffers, querying torch.cuda, cudagraphs).
        assert device == torch.device("cpu"), "QAIC keeps host tensors on CPU."
        super().__init__(vllm_config, device)

        # --- UnieAI ngram gating (ported verbatim from the fork) ------------
        self.speculative_model_type: Optional[str] = None
        if self.speculative_config is not None:
            if self.speculative_config.method != "ngram":
                raise ValueError(
                    "Only ngram speculative decoding is supported on the QAIC "
                    "backend when using the vLLM V1 engine.")
            self.speculative_model_type = "target"
        # --------------------------------------------------------------------

        assert not self.uses_mrope, "mrope is not supported on QAIC."
        assert not self.supports_mm_inputs, "multimodal is not supported on QAIC."

        self.max_seq_len = self.model_config.max_seq_len_to_capture
        self.num_kv_heads = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.head_size = self.model_config.get_head_size()

        # TODO(port): _postprocess_tensors() cast int32->int64 the fork's host
        # arrays on self.input_batch / self.input_ids_cpu. In 0.21 those live in
        # InputBatch + CpuGpuBuffer (self.input_ids). Re-fit against the new
        # InputBatch field names before calling it.
        # self._postprocess_tensors()

    # -------------------------------------------------------------------------
    # execute_model  ***THE BIG ARCHITECTURAL BREAK***
    #   OLD: a SINGLE method did input-prep + QPC forward + sampling and
    #        returned ModelRunnerOutput.
    #   NEW: 0.21 splits this into execute_model() (may return None and stash
    #        self.execute_model_state) followed by sample_tokens(grammar_output).
    #
    #   Decision for QAIC: the QPC forward is synchronous and host-driven, so
    #   the simplest faithful port is to do the full QAIC path inside
    #   execute_model and return ModelRunnerOutput directly (i.e. NOT use the
    #   deferred sample_tokens machinery). Confirm the base class allows a
    #   subclass to fully override execute_model and short-circuit
    #   sample_tokens on your target build.
    #
    #   Input-prep dependency note: the fork built decode/prefill batches from
    #   self.num_decodes / self.cu_num_tokens / self.positions_np / self.input_ids_cpu.
    #   ALL REMOVED in 0.21. Rebuild from scheduler_output + self.input_batch:
    #     - num_scheduled_tokens per req: scheduler_output.num_scheduled_tokens
    #     - token history:               self.input_batch.token_ids_cpu
    #     - computed tokens:             self.input_batch.num_computed_tokens_cpu
    #     - block ids:                   self.input_batch.block_table[...]
    #   and recompute the decode/prefill split locally (the fork relied on
    #   reorder_batch_to_split_decodes_and_prefills + self.num_decodes).
    # -------------------------------------------------------------------------
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ) -> "ModelRunnerOutput":
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called after "
                "execute_model() returns None.")

        deferred_state_corrections_fn = self._update_states(scheduler_output)
        if self.num_prompt_logprobs:
            raise NotImplementedError(
                "prompt_logprobs is not supported by V1 QAICModelRunner.")
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        num_scheduled_tokens_np = np.array(
            [scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids],
            dtype=np.int32,
        )
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        _, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens_np,
        )
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens_np,
            self.query_pos.np,
            cumsum_dtype=np.int32,
        )

        input_ids = (
            self.input_ids.cpu[:total_num_scheduled_tokens]
            .to(torch.int64)
            .numpy()
        )
        positions_np = (
            self.positions[:total_num_scheduled_tokens]
            .to(torch.int64)
            .cpu()
            .numpy()
        )

        is_decode = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
            >= self.input_batch.num_prompt_tokens[:num_reqs]
        )
        decode_req_indices = np.nonzero(is_decode)[0]
        prefill_req_indices = np.nonzero(~is_decode)[0]
        num_decodes = int(decode_req_indices.size)

        block_table = self.input_batch.block_table[0].get_numpy_array()
        block_ids = block_table[:num_reqs, 0].astype(np.int64)
        if np.any(block_ids < 1):
            raise RuntimeError(
                "QAIC expects allocated vLLM KV block ids to be >= 1; "
                f"got {block_ids.tolist()}.")
        batch_indices = block_ids - 1

        token_starts = np.concatenate((
            np.array([0], dtype=cu_num_tokens.dtype),
            cu_num_tokens[:-1],
        ))
        token_ends = cu_num_tokens
        decode_token_counts = num_scheduled_tokens_np[decode_req_indices]
        prefill_token_counts = num_scheduled_tokens_np[prefill_req_indices]

        decode_token_indices = self._qaic_flatten_req_token_indices(
            decode_req_indices, token_starts, token_ends)
        prefill_token_indices = self._qaic_flatten_req_token_indices(
            prefill_req_indices, token_starts, token_ends)
        decode_cu_num_tokens = np.cumsum(decode_token_counts, dtype=np.int32)
        prefill_cum_sum = np.cumsum(prefill_token_counts, dtype=np.int32)

        decode_block_ids = batch_indices[decode_req_indices]
        decode_lora_ids: Optional[np.ndarray] = None
        prefill_lora_ids: Optional[np.ndarray] = None
        if self.lora_config:
            request_lora_mapping = self.input_batch.request_lora_mapping
            decode_lora_ids = request_lora_mapping[decode_req_indices]
            prefill_lora_ids = request_lora_mapping[prefill_req_indices]

        (
            decode_input_ids,
            decode_positions,
            decode_lengths,
            _,
        ) = self._pack_decode_batch(
            input_ids[decode_token_indices],
            positions_np[decode_token_indices],
            num_decodes,
            decode_cu_num_tokens,
        )

        prefill_input_ids = input_ids[prefill_token_indices]
        prefill_positions = positions_np[prefill_token_indices]
        prefill_block_ids = batch_indices[prefill_req_indices]

        hidden_states_decode = (
            self.model(
                input_ids=decode_input_ids,
                positions=decode_positions,
                batch_indices=decode_block_ids,
                is_prompt=False,
                lora_ids=decode_lora_ids,
                decode_lengths=decode_lengths,
            )
            if decode_input_ids.size > 0
            else None
        )
        hidden_states_prefill = (
            self.model(
                input_ids=prefill_input_ids,
                positions=prefill_positions,
                batch_indices=prefill_block_ids,
                is_prompt=True,
                bypass_model_exec=False,
                kv_caches=[],
                logits_mem_buffs=None,
                prefill_is_partial=False,
                lora_ids=prefill_lora_ids,
                prefill_cum_sum=prefill_cum_sum,
            )
            if prefill_input_ids.size > 0
            else None
        )

        hidden_states = self._qaic_merge_model_outputs(
            hidden_states_decode,
            hidden_states_prefill,
            decode_req_indices,
            prefill_req_indices,
            decode_token_counts,
        )

        logits = self.model.compute_logits(
            hidden_states,
            self.input_batch.sampling_metadata,
        )
        if scheduler_output.grammar_bitmask is not None:
            raise NotImplementedError(
                "Grammar bitmask is not supported by V1 QAICModelRunner.")

        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            assert self.speculative_config is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            sampler_output.sampled_token_ids = self._qaic_rejection_sample(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            _,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            total_num_scheduled_tokens,
        )

        if deferred_state_corrections_fn is not None:
            deferred_state_corrections_fn()

        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
            routed_experts=None,
        )

    @staticmethod
    def _qaic_flatten_req_token_indices(
        req_indices: np.ndarray,
        token_starts: np.ndarray,
        token_ends: np.ndarray,
    ) -> np.ndarray:
        if req_indices.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([
            np.arange(token_starts[i], token_ends[i], dtype=np.int64)
            for i in req_indices
        ])

    @staticmethod
    def _qaic_merge_model_outputs(
        hidden_states_decode: Optional[torch.Tensor],
        hidden_states_prefill: Optional[torch.Tensor],
        decode_req_indices: np.ndarray,
        prefill_req_indices: np.ndarray,
        decode_token_counts: np.ndarray,
    ) -> torch.Tensor:
        """Restore QAIC QPC outputs to the row order vLLM sampling expects.

        QAIC must run decode and prefill requests through separate QPC shapes.
        Unlike the fork, the 0.21 port does not reorder InputBatch into
        decode-first order. Build QPC inputs by mask, then scatter the returned
        logits back to request/logits order before sampling.
        """
        pieces: list[torch.Tensor] = []
        if hidden_states_decode is not None:
            decode_rows: list[torch.Tensor] = []
            cursor = 0
            for count in decode_token_counts:
                next_cursor = cursor + int(count)
                decode_rows.append(hidden_states_decode[cursor:next_cursor])
                cursor = next_cursor
            if cursor != hidden_states_decode.shape[0]:
                raise RuntimeError(
                    "QAIC decode output row count does not match scheduled "
                    "decode tokens: got "
                    f"{hidden_states_decode.shape[0]}, expected {cursor}.")
        else:
            decode_rows = []
            if decode_req_indices.size:
                raise RuntimeError(
                    "QAIC decode output is missing for scheduled decode "
                    "requests.")

        if hidden_states_prefill is not None:
            prefill_rows = list(hidden_states_prefill)
            if len(prefill_rows) != prefill_req_indices.size:
                raise RuntimeError(
                    "QAIC prefill output row count does not match scheduled "
                    f"prefills: got {len(prefill_rows)}, expected "
                    f"{prefill_req_indices.size}.")
        else:
            prefill_rows = []
            if prefill_req_indices.size:
                raise RuntimeError(
                    "QAIC prefill output is missing for scheduled prefill "
                    "requests.")

        decode_by_req = {
            int(req_idx): row
            for req_idx, row in zip(decode_req_indices, decode_rows)
        }
        prefill_by_req = {
            int(req_idx): row.unsqueeze(0)
            for req_idx, row in zip(prefill_req_indices, prefill_rows)
        }
        for req_idx in sorted((*decode_by_req.keys(), *prefill_by_req.keys())):
            if req_idx in decode_by_req:
                pieces.append(decode_by_req[req_idx])
            else:
                pieces.append(prefill_by_req[req_idx])

        if not pieces:
            raise RuntimeError("QAIC model execution produced no logits.")

        return torch.cat(pieces, dim=0)

    # -------------------------------------------------------------------------
    # 2D decode packing — UnieAI's ngram change to execute_model, factored out.
    # OLD context: when speculative_config is set, decode requests are packed as
    # [num_decodes, num_speculative_tokens+1] so the target QPC verifies N+1
    # positions per request in one pass. Ported here as a helper; wire it into
    # your rebuilt execute_model. Depends on the decode arrays you reconstruct.
    # -------------------------------------------------------------------------
    def _pack_decode_batch(
        self,
        input_ids: np.ndarray,
        positions_np: np.ndarray,
        num_decodes: int,
        cu_num_tokens: np.ndarray,
    ):
        """Returns (decode_input_ids, decode_positions, decode_lengths,
        decode_token_count) — verbatim logic from the fork's execute_model."""
        decode_lengths: Optional[np.ndarray] = None
        decode_token_count = num_decodes
        if self.speculative_config is not None and num_decodes > 0:
            max_decode_tokens = self.speculative_config.num_speculative_tokens + 1
            decode_lengths = np.diff(
                np.concatenate((
                    np.array([0], dtype=cu_num_tokens.dtype),
                    cu_num_tokens[:num_decodes],
                ))).astype(np.int32)
            decode_input_ids = np.full(
                (num_decodes, max_decode_tokens), -1, dtype=input_ids.dtype)
            decode_positions = np.full(
                (num_decodes, max_decode_tokens), -1, dtype=positions_np.dtype)
            cursor = 0
            for i, num_tokens in enumerate(decode_lengths):
                assert num_tokens <= max_decode_tokens
                nxt = cursor + int(num_tokens)
                decode_input_ids[i, :num_tokens] = input_ids[cursor:nxt]
                decode_positions[i, :num_tokens] = positions_np[cursor:nxt]
                cursor = nxt
            decode_token_count = cursor
        else:
            decode_input_ids = input_ids[:num_decodes]
            decode_positions = positions_np[:num_decodes]
        return decode_input_ids, decode_positions, decode_lengths, decode_token_count

    # =========================================================================
    # UnieAI ngram CPU rejection sampler — PORTED VERBATIM from the fork.
    # These depend ONLY on `sampling_metadata`, `SpecDecodeMetadata` and torch,
    # so they survive the 0.21 port intact. The ONLY thing to verify is that the
    # SpecDecodeMetadata fields used here (num_draft_tokens, max_spec_len,
    # cu_num_draft_tokens, draft_token_ids) still exist with these names on the
    # target (they do on 0.21, though cu_num_draft_tokens is now a tensor).
    # =========================================================================
    def _qaic_rejection_sample(
        self,
        metadata: SpecDecodeMetadata,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        sampling_metadata: Any,
    ) -> torch.Tensor:
        """CPU fallback for ngram speculative decoding on QAIC.

        V1's default rejection sampler launches Triton kernels. QAIC runs this
        path on CPU and can have no active Triton driver, so use a small
        PyTorch/CPU implementation for ngram, where draft_probs is None.
        """
        batch_size = len(metadata.num_draft_tokens)
        output_token_ids = torch.full(
            (batch_size, metadata.max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=target_logits.device,
        )
        target_argmax = target_logits.argmax(dim=-1)

        for req_idx, num_draft_tokens in enumerate(metadata.num_draft_tokens):
            start_idx = 0 if req_idx == 0 else int(
                metadata.cu_num_draft_tokens[req_idx - 1].item())
            end_idx = int(metadata.cu_num_draft_tokens[req_idx].item())
            assert end_idx - start_idx == num_draft_tokens

            if self._qaic_is_greedy_request(sampling_metadata, req_idx):
                self._qaic_rejection_sample_greedy_req(
                    output_token_ids, metadata.draft_token_ids, target_argmax,
                    bonus_token_ids, req_idx, start_idx, num_draft_tokens)
            else:
                self._qaic_rejection_sample_random_req(
                    output_token_ids, metadata.draft_token_ids, target_logits,
                    bonus_token_ids, sampling_metadata, req_idx, start_idx,
                    num_draft_tokens)

        return output_token_ids

    @staticmethod
    def _qaic_is_greedy_request(sampling_metadata: Any, req_idx: int) -> bool:
        if sampling_metadata.all_greedy:
            return True
        if sampling_metadata.all_random:
            return False
        assert sampling_metadata.temperature is not None
        return (
            float(sampling_metadata.temperature[req_idx].item())
            == float(GREEDY_TEMPERATURE)
        )

    @staticmethod
    def _qaic_rejection_sample_greedy_req(
        output_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        req_idx: int,
        start_idx: int,
        num_draft_tokens: int,
    ) -> None:
        rejected = False
        for pos in range(num_draft_tokens):
            if rejected:
                break
            token_idx = start_idx + pos
            target_token_id = int(target_argmax[token_idx].item())
            output_token_ids[req_idx, pos] = target_token_id
            if int(draft_token_ids[token_idx].item()) != target_token_id:
                rejected = True

        if not rejected:
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[
                req_idx, 0]

    def _qaic_rejection_sample_random_req(
        self,
        output_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        sampling_metadata: Any,
        req_idx: int,
        start_idx: int,
        num_draft_tokens: int,
    ) -> None:
        rejected = False
        generator = sampling_metadata.generators.get(req_idx)
        for pos in range(num_draft_tokens):
            if rejected:
                break

            token_idx = start_idx + pos
            draft_token_id = int(draft_token_ids[token_idx].item())
            target_probs = self._qaic_target_probs_for_req(
                target_logits[token_idx], sampling_metadata, req_idx)
            uniform = torch.rand((), device=target_logits.device,
                                 generator=generator)

            if float(target_probs[draft_token_id].item()) >= float(
                    uniform.item()):
                output_token_ids[req_idx, pos] = draft_token_id
            else:
                rejected = True
                recovered_probs = target_probs.clone()
                recovered_probs[draft_token_id] = 0
                if recovered_probs.sum() <= 0:
                    recovered_token_id = int(target_probs.argmax().item())
                else:
                    recovered_token_id = self._qaic_sample_from_probs(
                        recovered_probs, generator)
                output_token_ids[req_idx, pos] = recovered_token_id

        if not rejected:
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[
                req_idx, 0]

    def _qaic_target_probs_for_req(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        req_idx: int,
    ) -> torch.Tensor:
        logits = logits.float().clone()
        if sampling_metadata.temperature is not None:
            temperature = float(sampling_metadata.temperature[req_idx].item())
            if temperature != float(GREEDY_TEMPERATURE):
                logits.div_(temperature)

        self._qaic_apply_top_k_top_p(logits, sampling_metadata, req_idx)
        return logits.softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def _qaic_apply_top_k_top_p(
        logits: torch.Tensor,
        sampling_metadata: Any,
        req_idx: int,
    ) -> None:
        if sampling_metadata.top_k is not None:
            top_k = int(sampling_metadata.top_k[req_idx].item())
            if 0 < top_k < logits.numel():
                cutoff = logits.topk(top_k).values[-1]
                logits.masked_fill_(logits < cutoff, -float("inf"))

        if sampling_metadata.top_p is not None:
            top_p = float(sampling_metadata.top_p[req_idx].item())
            if top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(descending=True)
                sorted_probs = sorted_logits.softmax(dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                logits[sorted_indices[remove_mask]] = -float("inf")

    @staticmethod
    def _qaic_sample_from_probs(
        probs: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> int:
        q = torch.empty_like(probs)
        if generator is None:
            q.exponential_()
        else:
            q.exponential_(generator=generator)
        return int(probs.div(q).argmax().item())

    # =========================================================================
    # Methods coupled to GPUModelRunner — NEW 0.21 signatures + OLD->NEW notes.
    # =========================================================================

    def propose_draft_token_ids(self, *args, **kwargs):
        """Use vLLM 0.21's ngram proposer.

        The v1_ngram fork implemented this against the old V1 signature. The
        current upstream implementation already handles the new NgramProposer
        contract, and this QAIC runner rejects non-ngram speculative methods in
        __init__, so delegating here keeps the scheduler-facing API aligned.
        """
        return super().propose_draft_token_ids(*args, **kwargs)

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        """Ported from the fork (low risk). Builds one FullAttentionSpec per
        layer. OLD==NEW shape; verify FullAttentionSpec fields on your target
        (block_size, num_kv_heads, head_size, dtype, use_mla).
        """
        block_size = self.cache_config.block_size
        kv_cache_spec: dict[str, "KVCacheSpec"] = {}
        n_layers = self.model_config.get_num_layers(self.parallel_config)
        for i in range(n_layers):
            layer_name = f"layer_{i}"
            kv_cache_spec[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_dtype,
                use_mla=False,
            )
        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: "KVCacheConfig",
                            is_profiling: bool = False) -> None:
        """Ported from the fork. On QAIC the KV cache is owned by the QPC on the
        card, so we only stash the config; no host tensors are allocated.
        OLD: initialize_kv_cache(kv_cache_config); NEW adds is_profiling.
        """
        self.kv_cache_config = kv_cache_config
        # NOTE (fork): KV-transfer registration intentionally left out (only
        # needed for disaggregated serving). Re-add if you port qaic_connector.

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Ported from the fork. Loads/compiles the QPC via load_qaic_model.
        OLD: load_model(*args, **kwargs); NEW: (load_dummy_weights=False).
        """
        logger.info("Starting to load model %s...", self.model_config.model)
        t0 = time.perf_counter()
        with set_current_vllm_config(self.vllm_config):
            self.model = load_qaic_model(
                self.vllm_config,
                speculative_model_type=self.speculative_model_type,
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.model_config, self.scheduler_config,
                    self.lora_config, self.device)
        logger.info("Model loading took %.6f seconds", time.perf_counter() - t0)

    def get_model(self) -> nn.Module:
        return self.model

    def _qaic_dummy_run(self) -> None:
        """Warmup hook used by QaicWorker.

        Mirrors the fork's warmup. Besides shape validation, this primes the
        QPC prefill/decode phase buffers before the first real request.
        """
        if self.model.disagg_serving_en or self.model.disagg_producer_en:
            return

        prefill_bsz = self.model.prefill_bsz
        prefill_input_ids = self.input_batch.token_ids_cpu[
            :prefill_bsz, :self.max_seq_len].flatten()
        prefill_positions = self.arange_np[:self.max_seq_len].repeat(prefill_bsz)
        prefill_block_ids = np.arange(prefill_bsz)
        prefill_cum_sum = np.array(
            [self.max_seq_len] * prefill_bsz, dtype=np.int64).cumsum()
        prefill_lora_ids = None
        if self.lora_config:
            prefill_lora_ids = np.arange(prefill_bsz, dtype=np.int64)

        self.model(
            input_ids=prefill_input_ids,
            positions=prefill_positions,
            batch_indices=prefill_block_ids,
            lora_ids=prefill_lora_ids,
            is_prompt=True,
            prefill_cum_sum=prefill_cum_sum,
        )

        decode_bsz = self.model.decode_bsz
        decode_input_ids = np.zeros(decode_bsz, dtype=np.int64)
        decode_positions = np.zeros(decode_bsz, dtype=np.int64)
        decode_block_ids = np.arange(decode_bsz, dtype=np.int64)
        decode_lora_ids = None
        if self.lora_config:
            decode_lora_ids = np.arange(decode_bsz, dtype=np.int64)

        self.model(
            input_ids=decode_input_ids,
            positions=decode_positions,
            batch_indices=decode_block_ids,
            lora_ids=decode_lora_ids,
            is_prompt=False,
            bypass_model_exec=False,
        )
