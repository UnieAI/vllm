"""QaicModelRunner — vLLM 0.21+ port of the v1_ngram fork's QaicModelRunner.

PROVENANCE (read docs/UnieAI_Quic_integrated.md):
  * The base QaicModelRunner is Qualcomm's (marked "Confidential and
    Proprietary" in the fork). It runs the QPC on the AIC and keeps vLLM's
    scheduling/sampling on the host CPU.
  * The ngram speculative-decoding pieces in this file — the seven
    `_qaic_rejection_sample*` helpers, the 2D decode packing, and the ngram
    gating in __init__ — are UnieAI's original contribution. They are ported
    here VERBATIM from the fork (only SpecDecodeMetadata field access may need
    touch-ups, flagged inline).

STATUS: scaffold. The self-contained ngram helpers are complete and portable.
The methods that touch GPUModelRunner internals carry the NEW 0.21 signatures
plus inline "OLD(v0.10.1) -> NEW(0.21)" notes; their bodies that depend on the
host input-prep arrays (self.positions_np / self.cu_num_tokens / self.num_decodes,
all REMOVED in 0.21) are marked TODO. See docs/MIGRATION_GPUModelRunner_old_vs_new.md.
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

# TODO(verify import path on target): PLACEHOLDER_TOKEN_ID moved across releases.
# In the fork it came from the spec-decode utils. On 0.21 check:
#   from vllm.v1.spec_decode.utils import PLACEHOLDER_TOKEN_ID   (or)
#   from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
try:  # be tolerant until pinned
    from vllm.v1.spec_decode.utils import PLACEHOLDER_TOKEN_ID
except Exception:  # pragma: no cover
    PLACEHOLDER_TOKEN_ID = -1

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
        super().__init__(vllm_config, device)

        assert device == torch.device("cpu"), "QAIC keeps host tensors on CPU."

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
        raise NotImplementedError(
            "Port execute_model from the fork and re-fit input-prep against the "
            "0.21 InputBatch/CpuGpuBuffer model. See the OLD->NEW note above and "
            "docs/MIGRATION_GPUModelRunner_old_vs_new.md. The 2D decode packing "
            "(_pack_decode_batch below) and the ngram rejection sampling "
            "(_qaic_rejection_sample) are ready to call once inputs exist.")

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
        return float(sampling_metadata.temperature[req_idx].item()) == -1.0

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
            if temperature != -1.0:
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
        """OLD: propose_draft_token_ids(scheduler_output, sampled_token_ids)
        -> thin wrapper over self.propose_ngram_draft_token_ids(...).
        NEW: base signature expanded massively (hidden_states,
        sample_hidden_states, aux_hidden_states, spec_decode_metadata,
        common_attn_metadata, slot_mappings) AND NgramProposer.propose now takes
        (sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings).
        For ngram, prefer delegating to super().propose_draft_token_ids(...).
        """
        raise NotImplementedError(
            "Re-fit against the new propose_draft_token_ids / NgramProposer.propose "
            "signatures, or delegate to super(). See migration doc.")

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        """OLD == NEW shape: dict[layer_name -> FullAttentionSpec]. Low risk.
        Port the fork body; verify FullAttentionSpec field names (block_size,
        num_kv_heads, head_size, dtype, use_mla) on the target.
        """
        raise NotImplementedError("Port get_kv_cache_spec from the fork (low risk).")

    def initialize_kv_cache(self, kv_cache_config: "KVCacheConfig",
                            is_profiling: bool = False) -> None:
        """OLD: initialize_kv_cache(kv_cache_config).
        NEW: added is_profiling param. Fork body just stored the config.
        """
        raise NotImplementedError("Port initialize_kv_cache; mind the new is_profiling arg.")

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """OLD: load_model(*args, **kwargs) -> load_qaic_model(vllm_config, spec_type).
        NEW: load_model(load_dummy_weights=False). Call into
        vllm_qaic.model_loader.load_qaic_model and pass self.speculative_model_type.
        """
        raise NotImplementedError("Port load_model -> vllm_qaic.model_loader.load_qaic_model.")

    def get_model(self) -> nn.Module:
        return self.model
