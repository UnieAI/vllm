# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Utilities for selecting and loading qaic models."""
import json
import math
import os
import signal
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from QEfficient.generation.cloud_infer import QAICInferenceSession
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.model_loader.qaic import (
    QAIC_DEVICE_CONFIG, _get_qaic_compile_config, check_qpc_exists,
    get_hf_model, search_adapters_in_cache,
    verify_adaptername_to_id_consistency)
from vllm.model_executor.model_loader.qaic_session_np import (
    DisAgg_QAICInferenceSession, aic_to_np_dtype_mapping)
from vllm.model_executor.sampling_metadata import SamplingMetadata

logger = init_logger(__name__)

chunk_id = 0
lock = threading.Lock()

class QaicCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config

        # TODO: Add new variables for pooling and turbo

        self.config = config
        self.vocab_size = config.get_text_config().vocab_size
        self.seq_len = model_config.max_seq_len_to_capture
        self.ctx_len = model_config.max_model_len
        self.decode_bsz = vllm_config.scheduler_config.max_num_seqs
        self.full_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_seq_len = model_config.max_seq_len_to_capture
        self.prefill_bsz = 1
        self.lora_mode = bool(vllm_config.lora_config)
        self.last_decode = False

        self.num_logits_to_keep = None
        self.encode_num_logits_buffer = None
        self.is_spec_decode_target_model = False

        self.sampler = get_sampler()
        self.pad = np.full(self.max_seq_len, fill_value=-1, dtype=np.int64)
        self.logits_processor = LogitsProcessor(self.vocab_size, logits_as_input=True)

        self.decode_batch_inputs = {
            "input_ids": np.full((self.decode_bsz,1),-1, dtype=np.int64),
            "position_ids": np.full((self.decode_bsz,1),-1, dtype=np.int64),
            "batch_index": np.full((self.decode_bsz,1),-1, dtype=np.int64),
        }
        self.list_of_comp_ctx_lengths = None
        if self.lora_mode:
            self.decode_batch_inputs["lora_ids"] = np.full((self.decode_bsz,1),-1, dtype=np.int64)

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        is_prompt: bool,
        lora_ids: Optional[np.ndarray] = None,
        sampling_params: Optional[Dict[str, Union[List[float], List[int]]]] = None,
        bypass_model_exec: Optional[bool] = False,
        kv_caches: Optional[List[List[np.ndarray]]] = None,
        logits_mem_buffs: Optional[List[np.ndarray]] = None,
        callback: Optional[Callable] = None,
        multi_modal_kwargs_list: Optional[List[dict]] = None,
        prefill_is_partial: Optional[List[bool]] = None,
        prefill_cum_sum: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if is_prompt and (self.disagg_producer_en or self.disagg_serving_en):
            assert kv_caches is not None and logits_mem_buffs is not None
            if self.disagg_producer_en:
                assert prefill_is_partial is not None
                logits = self._run_pipeline_prefill(
                    input_ids,
                    positions,
                    batch_indices,
                    kv_caches,
                    logits_mem_buffs,
                    prefill_is_partial,
                    lora_ids,
                )
            else:
                assert bypass_model_exec == True
                logits = self._kv_handoff(batch_indices, kv_caches, logits_mem_buffs)
        else:
            with (
                lock
            ):  # TODO: Re-evaluate if it's needed for for MultiProcExecutor on a single machine
                if is_prompt:
                    logits = self._run_prefill(
                        input_ids, positions, batch_indices, prefill_cum_sum, lora_ids
                    )
                else:
                    logits = self._run_decode(
                        input_ids, positions, batch_indices, lora_ids
                    )
                    # logits is a non-writable array. pytorch needs to have a
                    # writable array to work properyly (else, behavior is undefined)
                    # https://python-code.dev/articles/413443632
                    logits = np.copy(logits)

        return torch.from_numpy(logits)

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_comp_ctx_lengths(self):

        comp_ctx_lengths_prefill, comp_ctx_lengths_decode = [], []
        if "comp_ctx_lengths" not in self.session.binding_index_map:
            return None, None
        input_idx = self.session.binding_index_map["input_ids"]
        ccl_idx = self.session.binding_index_map["comp_ctx_lengths"]
        for i in range(len(self.session.allowed_shapes)):
            if self.session.allowed_shapes[i][input_idx][1][1]==self.seq_len:
                comp_ctx_lengths_prefill.append(self.session.allowed_shapes[i][ccl_idx][1][0])
            elif self.session.allowed_shapes[i][input_idx][1][1]==1 or self.num_logits_to_keep:
                comp_ctx_lengths_decode.append(self.session.allowed_shapes[i][ccl_idx][1][0])
            else:
                raise ValueError("QPC not compiled for required seq_len")

        comp_ctx_lengths_prefill.sort()
        comp_ctx_lengths_decode.sort()
        if comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
            self.list_of_comp_ctx_lengths = {comp_ctx_len:np.empty(comp_ctx_len, dtype=np.int8) for comp_ctx_len in comp_ctx_lengths_prefill+comp_ctx_lengths_decode}
        return comp_ctx_lengths_prefill, comp_ctx_lengths_decode

    def load_model(
        self,
        qpc_path: str,
        device_id: list,
        num_logits_to_keep: Optional[int] = None,
        stages: Optional[int] = 1,
        kv_transfer_role: Optional[str] = None,
    ) -> None:
        self.qpc_path: str = qpc_path
        self.device_id: list = device_id
        self.num_logits_to_keep = num_logits_to_keep
        self.stages = stages
        self.disagg_serving_en = kv_transfer_role != None
        self.disagg_producer_en = kv_transfer_role == "kv_producer"

        logger.info(f"Loading QPC...")
        logger.info(
            f"This may take some time, please don't press CTRL-C during this phase..."
        )
        s = time.perf_counter()
        if not self.disagg_serving_en:
            self.session = QAICInferenceSession(qpc_path, device_ids=device_id)

            self.session.skip_buffers(
                [x for x in self.session.input_names if x.startswith("past_")]
            )
            self.session.skip_buffers(
                [x for x in self.session.output_names if x.endswith("_RetainedState")]
            )
        else:
            stages = stages if stages else 1
            self.session = DisAgg_QAICInferenceSession(
                qpc_path,
                device_ids=device_id,
                stages=stages,
                cluster_id="Prefill" if self.disagg_producer_en else "decode",
            )
            for y in range(stages + 1):
                self.session.skip_buffers(
                    [x for x in self.session.input_names if x.startswith("past_")],
                    y,
                )
                self.session.skip_buffers(
                    [
                        x
                        for x in self.session.output_names
                        if x.endswith("_RetainedState")
                    ],
                    y,
                )
        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = self.get_comp_ctx_lengths()
        e = time.perf_counter() - s
        logger.info(f"Successfully loaded QPC in {e} secs")

        self.prefill_num_logits_buffer = None
        self.prefill_logits = dict(
            logits=np.random.randn(self.prefill_bsz, 1, self.vocab_size).astype(
                np.float32
            )
        )
        self.session.set_buffers(self.prefill_logits)
        self.batch_prefill_logits = np.empty(
            (self.decode_bsz, self.vocab_size), dtype=np.float32
        )
        self.decode_num_logits_buffer = None
        if self.num_logits_to_keep is not None:
            self.is_spec_decode_target_model = True
            self.decode_logits = dict(
                logits=np.random.randn(
                    self.decode_bsz, self.num_logits_to_keep, self.vocab_size
                ).astype(np.float32)
            )
            self.prefill_num_logits_buffer = dict(
                num_logits_to_keep=np.zeros((1, 1), np.int64)
            )
            self.decode_num_logits_buffer = dict(
                num_logits_to_keep=np.zeros((self.num_logits_to_keep, 1), np.int64)
            )
            self.session.set_buffers(self.prefill_num_logits_buffer)
        else:
            self.decode_logits = dict(
                logits=np.random.randn(self.decode_bsz, 1, self.vocab_size).astype(
                    np.float32
                )
            )
        if "batch_index" in self.session.input_names:
            self.ignore_batch_index = False
        else:
            self.ignore_batch_index = True
        if self.disagg_producer_en:
            self.decode_bsz = 0

    def kv_cache_info(self):
        no_buff_l = []
        for name in self.session.input_names:
            if name.startswith("past_"):
                no_buff_l.append(name)
        return [self.get_input_shape_and_dtype(no_buff_l[0]), len(no_buff_l)]

    def get_input_shape_and_dtype(
        self, input_name: str
    ) -> Optional[Tuple[list, np.dtype, int]]:
        if input_name not in self.session.input_names:
            return None
        binding = self.session.bindings[self.session.binding_index_map[input_name]]
        logger.info(type(binding.size))
        return list(binding.dims), aic_to_np_dtype_mapping[binding.type], binding.size

    def _kv_handoff(
        self,
        batch_indices: List[int],
        kv_caches: List[List[np.ndarray]],
        logits_mem_buffs: List[np.ndarray],
    ):
        logits_list = []
        for bidx, index in enumerate(batch_indices):
            logits_list.append(logits_mem_buffs[index].squeeze(1))
            # Update kv cache setDataWith
            _ = self.session.set_data_for_kv_handoff(
                kv_caches[index],
                [("batch_index", bidx), ("ctx_start", 0)],
                0,
                self.session.decode_buff_map,
            )
        return np.concatenate(logits_list)

    def _run_pipeline_prefill(
        self,
        input_ids: List[np.ndarray],
        positions: List[np.ndarray],
        batch_indices: List[int],
        kv_caches: List[List[np.ndarray]],
        logits_mem_buffs: List[np.ndarray],
        prefill_is_partial: List[bool],
        lora_ids: Optional[List[int]] = None,
    ):
        # TODO: Pull in latest code when adding disagg support in v1
        # set qpc prefill state
        if self.last_decode:
            self.last_decode = False

        running_chunk_ids = deque()
        logits = np.empty(
            (len(batch_indices), 1, self.vocab_size), dtype=logits_mem_buffs[0].dtype
        )
        i = 0
        with lock:  # for max_concurrent_batches > 1 on a single machine
            global chunk_id
            for index, (iids, pids, bidx) in enumerate(
                zip(input_ids, positions, batch_indices)
            ):
                n_prompt_tokens = iids.shape[-1]
                n_chunks: int = math.ceil(n_prompt_tokens / self.seq_len)
                assert n_chunks > 0

                shape = self.session.kv_shape
                shape[2] = n_prompt_tokens
                for chunk in range(n_chunks):
                    last_chunk = 0
                    if chunk + 1 == n_chunks:
                        lower_idx = -self.seq_len
                        upper_idx = n_prompt_tokens
                        if not prefill_is_partial[index]:
                            last_chunk = 1
                    else:
                        lower_idx = int(chunk * self.seq_len)
                        upper_idx = int((chunk + 1) * self.seq_len)
                    chunk_inputs = {
                        "input_ids": iids[:, lower_idx:upper_idx],
                        "position_ids": pids[:, lower_idx:upper_idx],
                        "batch_index": np.array([[bidx]]),
                        "logits": np.zeros((1, 1, self.vocab_size), dtype=np.float32),
                    }

                    while self.session.execObj_available < 1:
                        chunk_out, batch_id = running_chunk_ids.popleft()
                        self.session.complete_inf(chunk_out)
                        if batch_id != None:
                            logits[batch_id] = logits_mem_buffs[batch_id]

                    if last_chunk:
                        running_chunk_ids.append((chunk_id, bidx))
                        _ = self.session.set_data_for_kv_handoff(
                            kv_caches[index],
                            [("batch_index", bidx), ("ctx_start", 0)],
                            chunk_id,
                            self.session.prefill_buff_map[:-1],
                        )
                        logits[i] = logits_mem_buffs[index]
                        chunk_inputs["logits"] = logits_mem_buffs[index]
                        i += 1
                    else:
                        running_chunk_ids.append((chunk_id, None))
                    # Submit Chunk to LRT Queue
                    self.session.np_run_pipeline(
                        inputs=chunk_inputs, index=chunk_id, last_chunk=last_chunk
                    )
                    # time.sleep(0.01)
                    chunk_id += 1
                    chunk_id %= self.stages + 1

        # wait for all chunks to finish
        while running_chunk_ids:
            chunk_out, batch_id = running_chunk_ids.popleft()
            self.session.complete_inf(chunk_out)
            if batch_id != None:
                logits[batch_id] = logits_mem_buffs[batch_id]

        return logits.squeeze(1)

    def _run_prefill(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        prefill_cum_sum: np.ndarray,
        lora_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # set qpc prefill state
        if self.last_decode:
            self.session.set_buffers(self.prefill_logits)
            if self.is_spec_decode_target_model:
                self.session.set_buffers(self.prefill_num_logits_buffer)
            self.last_decode = False

        # perform prefill (only prefill_bsz=1 is supported)
        logits_list = []
        idx_start = 0
        for i,idx_end in enumerate(prefill_cum_sum):
            # extract indices of specific request
            iids = input_ids[idx_start:idx_end]
            pids = positions[idx_start:idx_end]
            idx_start = idx_end
            # concatenate to multiple of `self.seq_len`
            # to avoid reading/writing KV$ on `num_pads` tokens
            n_prompt_tokens = iids.shape[-1]
            if (remainder := n_prompt_tokens%self.seq_len) > 0:
                num_pads = self.seq_len - remainder
                pad = self.pad[:num_pads]
                iids = np.concatenate([pad, iids], dtype=np.int64)
                pids = np.concatenate([pad, pids], dtype=np.int64)
            # create chunk inputs
            chunk_inputs = dict()
            if not self.ignore_batch_index:
                batch_index = batch_indices[i:i+1].reshape(1,1)
                chunk_inputs["batch_index"] = batch_index
            if lora_ids is not None:
                lora_index = lora_ids[i:i+1].reshape(1,1)
                chunk_inputs["lora_ids"] = lora_index
            # chunk the request
            n_chunks: int = iids.shape[-1] // self.seq_len
            prefill_ccl_id = 0
            for chunk in range(n_chunks):
                lower_idx = int(chunk * self.seq_len)
                upper_idx = int((chunk + 1) * self.seq_len)
                chunk_inputs["input_ids"] = iids[lower_idx:upper_idx].reshape(1,self.seq_len)
                chunk_inputs["position_ids"] = pids[lower_idx:upper_idx].reshape(1,self.seq_len)
                if self.comp_ctx_lengths_prefill is not None:
                    prefill_ccl = self.comp_ctx_lengths_prefill[0]
                    for j in range(prefill_ccl_id, len(self.comp_ctx_lengths_prefill)):
                        if max(chunk_inputs['position_ids'][0]) < self.comp_ctx_lengths_prefill[j]:
                            prefill_ccl_id, prefill_ccl = j, self.comp_ctx_lengths_prefill[j]
                            break
                    chunk_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[prefill_ccl]
                outputs: dict = self.session.run(chunk_inputs)
            logits = outputs["logits"][:, -1] # shape: [1, vocab_size]
            logits_list.append(logits)
        return np.concatenate(logits_list, axis=0)

    def _run_decode(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        lora_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        num_decodes = input_ids.shape[-1]
        self.decode_batch_inputs["input_ids"][:num_decodes,0] = input_ids
        self.decode_batch_inputs["position_ids"][:num_decodes,0] = positions
        if num_decodes < self.decode_bsz:
            self.decode_batch_inputs["input_ids"][num_decodes:] = -1
            self.decode_batch_inputs["position_ids"][num_decodes:] = -1


        if not self.ignore_batch_index:

            self.decode_batch_inputs["batch_index"][:num_decodes,0] = batch_indices
            if num_decodes < self.decode_bsz:
                self.decode_batch_inputs["batch_index"][num_decodes:] = -1

        if lora_ids is not None:
            self.decode_batch_inputs["lora_ids"][:num_decodes,0] = lora_ids
            if num_decodes < self.decode_bsz:
                self.decode_batch_inputs["lora_ids"][num_decodes:] = -1

        if not self.last_decode:
            # set qpc sesstion state to decode phase
            self.session.set_buffers(self.decode_logits)
            if self.is_spec_decode_target_model:
                self.session.set_buffers(self.decode_num_logits_buffer)
            self.last_decode = True

        if self.comp_ctx_lengths_decode is not None:
            max_position_id = positions.max().item()
            self.decode_batch_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[max(self.list_of_comp_ctx_lengths.keys())]
            for comp_ctx_len in self.comp_ctx_lengths_decode:
                if max_position_id < comp_ctx_len:
                    self.decode_batch_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[comp_ctx_len]
                    break

        outputs = self.session.run(self.decode_batch_inputs)
        logits: np.ndarray = outputs["logits"]
        return logits[:num_decodes].squeeze(1)

    def run_encode(
        self, qpc_inputs: dict, encode_num_logits_buffer: Optional[dict] = None
    ) -> np.ndarray:
        """run qpc_inputs

        Args:
            qpc_inputs (dict): qpc inputs of incoming requests to process
        Returns:
            np.ndarray: fixed slot generated tokens
        """
        if encode_num_logits_buffer:
            if (
                self.encode_num_logits_buffer is None
                or encode_num_logits_buffer["output"].shape
                != self.encode_num_logits_buffer["output"].shape
            ):
                self.session.set_buffers(encode_num_logits_buffer)
                self.encode_num_logits_buffer = encode_num_logits_buffer
        outputs: dict = self.session.run(qpc_inputs)

        return outputs


def load_qaic_model(
    vllm_config: VllmConfig, speculative_model_type: Optional[str] = None
) -> nn.Module:
    # Create a model instance.
    model = QaicCausalLM(vllm_config)

    if speculative_model_type is None:
        speculative_model_type = "default"
        model.sampler.include_gpu_probs_tensor = False
    else:
        speculative_model_type = speculative_model_type.lower()
        model.sampler.include_gpu_probs_tensor = True

    if speculative_model_type not in QAIC_DEVICE_CONFIG.keys():
        raise ValueError(
            f"Unable to find default profile for model type {speculative_model_type}!!\n"
        )

    qaic_compile_config = _get_qaic_compile_config(vllm_config, speculative_model_type)
    qpc_path = qaic_compile_config.qpc_path

    # set lora max adapters
    if vllm_config.lora_config:
        qaic_max_adapters = int(os.environ.get("VLLM_QAIC_LORA_MAX_ID_SUPPORTED", 128))

    # if provided qpc is valid
    if qpc_path and not check_qpc_exists(qpc_path):
        raise ValueError(
            f"Environment variable VLLM_QAIC_QPC_PATH is set!\n"
            f"QAIC qpc path {qpc_path} doesn't exist or didn't have compiled binary!\n"
            "Unset VLLM_QAIC_QPC_PATH, if you don't want to provide compiled qpc.\n"
        )

    # set adaptername_to_id from previous dump file if qpc_path exist
    adaptername_to_id = {}
    if vllm_config.lora_config and (qpc_path and check_qpc_exists(qpc_path)):
        # check if json file exist
        if os.path.exists(f"{qpc_path}/adaptername_to_id.json"):
            with open(f"{qpc_path}/adaptername_to_id.json", "r") as file:
                adaptername_to_id = json.load(file)
        else:
            raise FileNotFoundError(
                f"The file at {qpc_path}/adaptername_to_id.json was not found. Please provide a correct VLLM_QAIC_QPC_PATH."
            )

        # check if json file content is correct
        if not verify_adaptername_to_id_consistency(
            adaptername_to_id, vllm_config.lora_config.lora_modules
        ):
            raise ValueError(
                f"Inconsistent file content in {qpc_path}/adaptername_to_id.json and input lora modules."
            )

    # Generate qpc using QEfficient transformer
    if not qpc_path:
        quant_cfg = vllm_config.model_config._parse_quant_hf_config()
        quant_method = None
        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

        if (
            vllm_config.model_config.quantization is not None
            and vllm_config.model_config.quantization in ["awq", "gptq"]
            and quant_method != vllm_config.model_config.quantization
        ):
            raise ValueError(
                f"Currently qaic backend only supports pre-quantized AWQ | GPTQ models"
                " via vllm!"
            )

        try:
            qeff_model = get_hf_model(
                vllm_config.model_config,
                qaic_compile_config.qaic_config,
                vllm_config.lora_config,
                qaic_compile_config.kv_offload,
            )
            from QEfficient import QEFFAutoModelForCausalLM

            if not isinstance(qeff_model, QEFFAutoModelForCausalLM):
                # Only QEFFAutoModelForCausalLM supports the prefill-only option
                qaic_compile_config.cfg.pop("prefill_only", None)
            if vllm_config.lora_config:
                logger.info(
                    "Transforming and compiling lora model using QEfficient library"
                )

                # search adapter in cache
                if not vllm_config.lora_config.lora_modules:

                    # search adapter in cache
                    filtered_cached_lora_module_paths = search_adapters_in_cache(
                        vllm_config.model_config.model
                    )

                    # error out if cache is empty
                    if len(filtered_cached_lora_module_paths) == 0:
                        raise ValueError(
                            "No adapter in cache, please either download some into HF_HOME or provide lora_modules list."
                        )
                    # set lora_modules
                    vllm_config.lora_config.lora_modules = (
                        filtered_cached_lora_module_paths
                    )

                # error out if reach adapter limit
                assert (
                    len(vllm_config.lora_config.lora_modules) <= qaic_max_adapters
                ), f"Number of cached adapters exceed limitation of {qaic_max_adapters}. Please either delete adapters from HF_HOME or specify adapters in lora_modules."

                # load adapters to model
                for lora_module_path in vllm_config.lora_config.lora_modules:
                    model_dir = lora_module_path.path.split("/")[-3]
                    adapter_model_id = (
                        f"{model_dir.split('--')[1]}/{model_dir.split('--')[2]}"
                    )
                    qeff_model.load_adapter(
                        adapter_model_id, lora_module_path.name
                    )  # adapters with inconsistent target_modules or ranks will not be added here (TODO: otherwise should add another check here)

                # get adaptername_to_id
                adaptername_to_id = qeff_model.active_adapter_to_id

            else:
                logger.info(
                    f"Transforming and compiling model[{speculative_model_type}] using QEfficient library"
                )
            qeff_model.compile(**qaic_compile_config.cfg)
            qpc_path = qeff_model.qpc_path
            if isinstance(qpc_path, list):
                qpc_path = qpc_path[qaic_compile_config.qpc_idx]
        except Exception as e:
            logger.error("Failed to transform and compile the model! {e}")
            raise e

    # dump adaptername_to_id to folder for the first compilation
    if vllm_config.lora_config and not os.path.exists(
        f"{qpc_path}/adaptername_to_id.json"
    ):
        with open(f"{qpc_path}/adaptername_to_id.json", "w") as file:
            json.dump(adaptername_to_id, file)
            logger.info(
                f"Dump adaptername_to_id mapping to {qpc_path}/adaptername_to_id.json"
            )

    if speculative_model_type != "default":
        logger.info(
            f"Spec model type {speculative_model_type}_{qaic_compile_config.num_logits_to_keep}"
        )

    logger.info(f"Using qpc:-{qpc_path}")

    if qaic_compile_config.compile_only:
        # Hack for Model-IP execution flow
        # TODO: remove this in future
        # This will create error in parent process if exited,
        # need better solution in future
        logger.info("Compilation completed, exiting...")
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(10)
        exit(0)
    # Load the weights from the cached or downloaded files.
    # model_config.qpc in None
    model.load_model(
        qpc_path=qpc_path,
        device_id=qaic_compile_config.device_group,
        num_logits_to_keep=qaic_compile_config.num_logits_to_keep,
        stages=qaic_compile_config.stages,
        kv_transfer_role=(
            vllm_config.kv_transfer_config.kv_role
            if vllm_config.kv_transfer_config
            else None
        ),
    )

    return model.eval()
