# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import time
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.qaic_v1 import load_qaic_model
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

class QaicModelRunner(GPUModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        speculative_model_type: Optional[str] = None,
    ):
        super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        if self.speculative_config is not None:
            if self.speculative_config.method != "ngram":
                raise ValueError(
                    "Only ngram speculative decoding is supported on qaic "
                    "backend when using vllm v1.")
            if speculative_model_type in (None, "default"):
                speculative_model_type = "target"
        assert not self.uses_mrope, "mrope is not supported."
        assert not self.supports_mm_inputs, "multimodal inputs are not suported ."


        # Extract configuration params
        self.speculative_model_type = speculative_model_type
        self.max_seq_len = self.model_config.max_seq_len_to_capture
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        # Post-process tensors
        self._postprocess_tensors()

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        _, num_decodes = reorder_batch_to_split_decodes_and_prefills(
            self.input_batch, scheduler_output
        )
        self.num_decodes = num_decodes

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def _postprocess_tensors(self) -> None:
        # Cast below tensors from `int32` -> `int64`
        self.input_batch.request_lora_mapping = self.input_batch.request_lora_mapping.astype(np.int64)
        self.input_ids_cpu = self.input_ids_cpu.to(torch.int64)
        self.input_batch.token_ids_cpu_tensor = self.input_batch.token_ids_cpu_tensor.to(torch.int64)
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy()
        self.input_batch.block_table[0].block_table_cpu = self.input_batch.block_table[0].block_table_cpu.to(torch.int64)
        self.input_batch.block_table[0].block_table_np = self.input_batch.block_table[0].block_table_cpu.numpy()

    def _prepare_qaic_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> SpecDecodeMetadata:
        """
        :return: SpecDecodeMetadata
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        # num_scheduled_tokens, [2, 5, 3]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Qaic: cache to split inputs into decode and prefill requests during `execute_model`
        self.cu_num_tokens = cu_num_tokens

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Qaic: cache block_ids associated with each request (`-1` is to account for null block)
        self.batch_indices = self.input_batch.block_table[0].get_numpy_array()[:num_reqs, 0] - 1

        # Prepare spec decode metadata
        spec_decode_metadata = None
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if use_spec_decode:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)

        return spec_decode_metadata

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids_cpu[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=None,
        )
        return metadata

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        # Prepare inputs
        spec_decode_metadata = self._prepare_qaic_inputs(scheduler_output)

        # Split positions and inputs into decode and prefill

        # decode arrays
        input_ids = self.input_ids_cpu.numpy()
        decode_block_ids: np.ndarray = self.batch_indices[:self.num_decodes]
        decode_lora_ids: Optional[np.ndarray] = None
        decode_lengths: Optional[np.ndarray] = None
        decode_token_count = self.num_decodes

        if self.speculative_config is not None and self.num_decodes > 0:
            max_decode_tokens = self.speculative_config.num_speculative_tokens + 1
            decode_lengths = np.diff(
                np.concatenate((
                    np.array([0], dtype=self.cu_num_tokens.dtype),
                    self.cu_num_tokens[:self.num_decodes],
                ))).astype(np.int32)

            decode_input_ids = np.full(
                (self.num_decodes, max_decode_tokens), -1, dtype=input_ids.dtype)
            decode_positions = np.full(
                (self.num_decodes, max_decode_tokens), -1,
                dtype=self.positions_np.dtype)

            cursor = 0
            for i, num_tokens in enumerate(decode_lengths):
                assert num_tokens <= max_decode_tokens
                next_cursor = cursor + int(num_tokens)
                decode_input_ids[i, :num_tokens] = input_ids[cursor:next_cursor]
                decode_positions[i, :num_tokens] = self.positions_np[
                    cursor:next_cursor]
                cursor = next_cursor
            decode_token_count = cursor
        else:
            decode_input_ids = input_ids[:self.num_decodes]
            decode_positions = self.positions_np[:self.num_decodes]

        # prefill arrays
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        prefill_input_ids: np.ndarray = input_ids[
            decode_token_count:total_num_scheduled_tokens]
        prefill_positions: np.ndarray = self.positions_np[
            decode_token_count:total_num_scheduled_tokens]
        prefill_block_ids: np.ndarray = self.batch_indices[self.num_decodes:total_num_scheduled_tokens]
        prefill_cum_sum = self.cu_num_tokens[
            self.num_decodes:] - decode_token_count
        prefill_lora_ids: Optional[np.ndarray] = None

        if self.lora_config:
            decode_lora_ids: np.ndarray = self.input_batch.request_lora_mapping[:self.num_decodes]
            prefill_lora_ids: np.ndarray = self.input_batch.request_lora_mapping[self.num_decodes:self.input_batch.num_reqs]


        # Run Decode & Prefill requests separately
        hidden_states_decode = (
            self.model(
                input_ids=decode_input_ids,
                positions=decode_positions,
                batch_indices=decode_block_ids,
                is_prompt=False,
                lora_ids=decode_lora_ids,
                decode_lengths=decode_lengths,
            )
            if decode_input_ids.size>0
            else None
        )

        kv_caches = []
        bypass_model_exec = False
        hidden_states = None
        prefill_is_partial = False

        hidden_states_prefill = (
            self.model(
                input_ids=prefill_input_ids,
                positions=prefill_positions,
                batch_indices=prefill_block_ids,
                is_prompt=True,
                bypass_model_exec=bypass_model_exec,
                kv_caches=kv_caches,
                logits_mem_buffs=hidden_states,
                prefill_is_partial=prefill_is_partial,
                lora_ids=prefill_lora_ids,
                prefill_cum_sum=prefill_cum_sum,
            )
            if prefill_input_ids.size>0
            else None
        )

        # Concat hidden states
        if hidden_states_prefill is not None and hidden_states_decode is not None:
            hidden_states = torch.cat(
                (hidden_states_decode, hidden_states_prefill), dim=0
            )
        else:
            hidden_states = (
                hidden_states_prefill
                if hidden_states_prefill is not None
                else hidden_states_decode
            )
        # TODO: Add pooling
        logits = self.model.compute_logits(hidden_states, None)

        # TODO: Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            raise NotImplementedError(
                "Grammar bitmask is not supported by V1 QAICModelRunner."
            )

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self._qaic_rejection_sample(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # Identify partial prefill requests to discard their sampled tokens
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids[self.num_decodes:], start=self.num_decodes):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                discard_sampled_tokens_req_indices.append(i)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states,
            scheduler_output,
        )

        # --- Batch Writeback Optimization ---
        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        
        # Mask out partial prefill sampled tokens
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Efficiently write back to input_batch and requests
        # (This replaces the slow per-request loop with slightly more optimized logic)
        req_ids = self.input_batch.req_ids
        token_ids_cpu = self.input_batch.token_ids_cpu
        num_tokens_no_spec = self.input_batch.num_tokens_no_spec
        num_tokens = self.input_batch.num_tokens

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = num_tokens_no_spec[req_idx]
            n_new = len(sampled_ids)
            end_idx = start_idx + n_new
            
            # Direct assignment to numpy views is fast
            token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            num_tokens_no_spec[req_idx] = end_idx
            num_tokens[req_idx] = end_idx
            
            # Update request state
            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Propose next set of draft tokens if speculative decoding is enabled
        if self.speculative_config is None:
            spec_token_ids = None
        else:
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
            )

        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)
        else:
            num_nans_in_logits = {}

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=None, # Logprobs not supported in optimized path for now
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=num_nans_in_logits,
        )

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
        if num_draft_tokens == 0:
            output_token_ids[req_idx, 0] = bonus_token_ids[req_idx, 0]
            return

        # Vectorized greedy rejection
        req_draft = draft_token_ids[start_idx : start_idx + num_draft_tokens]
        req_target = target_argmax[start_idx : start_idx + num_draft_tokens]
        
        matches = (req_draft == req_target)
        rejection_indices = (~matches).nonzero()
        
        if rejection_indices.numel() > 0:
            first_rejection = rejection_indices[0].item()
            output_token_ids[req_idx, :first_rejection + 1] = req_target[:first_rejection + 1]
        else:
            # All accepted, add bonus
            output_token_ids[req_idx, :num_draft_tokens] = req_target
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[req_idx, 0]

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


    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        valid_sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        assert isinstance(self.drafter, NgramProposer)
        
        # Determine requests eligible for N-gram proposal
        # (Must have at least one valid sampled token and not at max model len)
        num_reqs = self.input_batch.num_reqs
        num_tokens_no_spec = self.input_batch.num_tokens_no_spec
        
        valid_ngram_requests = []
        for i in range(num_reqs):
            if not valid_sampled_token_ids[i]:
                continue
            if num_tokens_no_spec[i] >= self.max_model_len:
                continue
            valid_ngram_requests.append(i)
            
        if not valid_ngram_requests:
            return [[] for _ in range(num_reqs)]
            
        valid_ngram_requests = np.array(valid_ngram_requests, dtype=np.int32)
        
        # Use optimized batch_propose
        spec_token_ids = self.drafter.batch_propose(
            num_reqs,
            valid_ngram_requests,
            num_tokens_no_spec,
            self.input_batch.token_ids_cpu,
        )
        return spec_token_ids

    def load_model(self, *args, **kwargs) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        time_before_load = time.perf_counter()
        with set_current_vllm_config(self.vllm_config):
            self.model = load_qaic_model(
                self.vllm_config, speculative_model_type=self.speculative_model_type
            )
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)

        time_after_load = time.perf_counter()
        logger.info(
            "Model loading took %.6f seconds", time_after_load - time_before_load
        )

    def _qaic_dummy_run(self) -> None:
        if self.model.disagg_serving_en or self.model.disagg_producer_en:
            # it's either prefill only or decode only
            # dummy run would cause device error, so skip it for now
            # TODO: add disagg_dummy_run once it's stable
            return

        prefill_bsz = self.model.prefill_bsz
        prefill_input_ids = self.input_batch.token_ids_cpu[:prefill_bsz,:self.max_seq_len].flatten()
        prefill_positions = self.arange_np[:self.max_seq_len].repeat(prefill_bsz)
        prefill_block_ids = np.arange(prefill_bsz)
        prefill_cum_sum = np.array([self.max_seq_len]*prefill_bsz, dtype=np.int64).cumsum()
        prefill_lora_ids = None
        if self.lora_config:
            prefill_lora_ids = np.arange(prefill_bsz, dtype=np.int64)

        self.model(
            input_ids=prefill_input_ids,
            positions=prefill_positions,
            batch_indices=prefill_block_ids,
            lora_ids=prefill_lora_ids,
            is_prompt=True,
            prefill_cum_sum=prefill_cum_sum
        )

        decode_bsz = self.model.decode_bsz
        decode_input_ids = np.array([0]*decode_bsz, dtype=np.int64)
        decode_positions = np.array([0]*decode_bsz, dtype=np.int64)
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

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config

        # TODO: Setup KV transfer
        # kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # if has_kv_transfer_group():
        #     get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        block_size = self.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
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

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return True

def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
) -> tuple[bool, int]:
    """
    Modified from the original method in utils to customize it for qaic

    Reorders the batch to split into prefill and decode requests; places all
    prefill requests at the front of the batch.

    Returns:
        True if the batch was modified, False otherwise.
    """
    # We now want to reorder the batch so that the "decode" requests are at
    # the front and the "prefill" requests are at the back using the least
    # amount of swaps possible. (NOTE for now we loosely use "decode" to mean
    # requests where attention is likely memory-bound and "prefill" to mean
    # requests where attention is likely compute-bound, TODO(lucas): figure out
    # a better naming here)
    decodes = []
    prefills = []
    num_decode_tokens = 0
    num_prefill_tokens = 0

    for i, req_id in enumerate(input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        req_index = input_batch.req_id_to_index.get(req_id)
        if (
            input_batch.num_computed_tokens_cpu[req_index]
            < input_batch.num_prompt_tokens[req_index]
        ):
            prefills.append(i)
            num_prefill_tokens += num_tokens
        else:
            decodes.append(i)
            num_decode_tokens += num_tokens

    # We hope that this is fairly minimal since decodes
    # should be around for a number of iterations so hopefully they are
    # relatively stationary (and new request are generally appended to the
    # persistent batch so already should be at the back)
    # To achieve this we loop over the decodes in descending order and
    # the prefills in ascending order. We swap decodes from the  "back"
    # i.e. past where the last decode should be in the reodorered with
    # prefills from the front of the batch.
    # `decodes` and `prefills` are already in ascending order just based on
    # the above loop
    num_decodes = len(decodes)
    num_prefills = len(prefills)
    modified_batch = False

    for i in range(1, min(num_decodes, num_prefills) + 1):
        # If the decode is at the "back" of the batch, i, we can swap it
        # with the prefill closest to the front of the batch
        decode_idx = decodes[num_decodes - i]
        if decode_idx < num_decodes:
            break

        input_batch.swap_states(prefills[i - 1], decode_idx)
        modified_batch = True

    return modified_batch, num_decodes
