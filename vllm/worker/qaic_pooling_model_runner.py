# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import dataclasses
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.pooling_params import PoolingParams
from vllm.sequence import (
    IntermediateTensors,
    PoolerOutput,
    PoolingSequenceGroupOutput,
    SequenceData,
    SequenceGroupMetadata,
)
from vllm.worker.qaic_model_runner import QaicModelRunner, ModelInputForQaic
from vllm.config import VllmConfig
import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class QaicPoolingModelRunner(QaicModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        speculative_model_type: Optional[str] = None):
        super().__init__(vllm_config, speculative_model_type)
        self.hidden_dimension = self.model_config.get_hidden_size()
        self.is_pooler = False
        self.is_qaic_pooler = False
        self.normalize = False
        self.softmax = False
        self.bsz = vllm_config.scheduler_config.max_num_seqs
        if self.vllm_config.model_config.override_qaic_config:
            if "pooling_device" in self.vllm_config.model_config.override_qaic_config:
                self.is_pooler = True
                if self.vllm_config.model_config.override_qaic_config["pooling_device"] == 'qaic':
                    self.is_qaic_pooler = True
                    normalize = self.vllm_config.model_config.override_qaic_config.get("normalize", False)
                    if (isinstance(normalize, str) and normalize.lower() in ('true', '1')) or normalize is True:
                        self.normalize = True
                    softmax = self.vllm_config.model_config.override_qaic_config.get("softmax", False)
                    if (isinstance(softmax, str) and softmax.lower() in ('true', '1')) or softmax is True:
                        self.softmax = True

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForQaic:
        lora_ids: List[int] = []
        (
            input_tokens,
            input_positions,
            input_block_ids,
            seq_lens,
            lora_ids,
            multi_modal_kwargs_list,
            _,
            _,
        ) = self._prepare_prompt(seq_group_metadata_list)

        # Prepare PoolingMetadata.
        assert seq_lens is not None
        pooling_metadata = self._prepare_pooling(seq_group_metadata_list, seq_lens)
        self.encode_num_logits_buffer = None
        if not self.is_qaic_pooler:
            self.encode_num_logits_buffer = dict(output=np.empty((self.bsz, input_tokens[0].shape[1], self.hidden_dimension), np.float32))
        return ModelInputForQaic(
            input_tokens=input_tokens,
            input_positions=input_positions,
            input_block_ids=input_block_ids,
            pooling_metadata=pooling_metadata,
            lora_ids=lora_ids if self.vllm_config.lora_config else None,
            multi_modal_kwargs_list=multi_modal_kwargs_list,
        )

    def _prepare_pooling(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> PoolingMetadata:
        """Prepare PoolingMetadata for the sequence group metadata list."""
        seq_groups: List[Tuple[List[int], PoolingParams]] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            pooling_params = seq_group_metadata.pooling_params
            seq_groups.append((seq_ids, pooling_params))

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        pooling_metadata = PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
        )

        return pooling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForQaic,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[PoolerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "QaicPoolingModelRunner does not support multi-step execution."
            )
        if len(model_input.multi_modal_kwargs_list) != 0:   ## multimodal
            outputs = []
            for qpc_inputs in model_input.multi_modal_kwargs_list:
                output = self.model.model.run_mm_encode(qpc_inputs)

                if len(output) == 1:
                    embeds = list(output.values())[0]
                    # TODO: it should be done by QEfficient
                    # Embeddings are in the shape of (num_images, image_feature_size, hidden_size)
                    if embeds.shape[0] != 1: # only support 1 image
                        embeds = embeds.reshape(1, np.prod(embeds.shape[:-1]), embeds.shape[-1])
                    outputs.append(torch.from_numpy(np.copy(embeds)))
                else:
                    # Temporary solution for mllama
                    # Expected sorted keys:
                    # dict_keys(['past_key.3', 'past_value.3', 'past_key.8', 'past_value.8', 'past_key.13', 'past_value.13', 'past_key.18', 'past_value.18', 'past_key.23', 'past_value.23', 'past_key.28', 'past_value.28', 'past_key.33', 'past_value.33', 'past_key.38', 'past_value.38'])
                    get_idx = lambda item: int(item[0].split(".")[1])
                    sorted_output = np.concatenate([v for _, v in sorted(output.items(), key=get_idx)])
                    outputs.append(torch.from_numpy(sorted_output).unsqueeze(0))

            pooling_output = [PoolingSequenceGroupOutput(data) for data in outputs]
            return [PoolerOutput(outputs=pooling_output)]


        else:    ## embedding
            seq_len = model_input.input_tokens[0].shape[-1]

            merged_inputs = {
                "input_ids": np.empty((self.bsz,seq_len),dtype=model_input.input_tokens[0].dtype),
                "attention_mask": np.empty((self.bsz,seq_len),dtype=model_input.input_tokens[0].dtype)
            }
            valid_ids = []
            i = 0
            for input_id, posn in zip(model_input.input_tokens, model_input.input_positions):
                merged_inputs["input_ids"][i] = input_id
                merged_inputs["attention_mask"][i] = np.where(posn == -1, 0, 1)
                valid_ids.append(np.max(posn)+1)
                i+=1

            output = self.model.model.run_encode(merged_inputs, self.encode_num_logits_buffer)
            output_array = output["output"][:i]

            if self.is_pooler and not self.is_qaic_pooler:
                output_array = np.concatenate([output_array[i, :valid_ids[i]] for i in range(output_array.shape[0])])

            outputs = torch.tensor(output_array)
            if self.is_pooler:
                if self.is_qaic_pooler:
                    if self.normalize:
                        outputs = F.normalize(outputs, p=2, dim=1)
                    if self.softmax:
                        outputs = F.softmax(outputs, dim=1)
                else:
                    outputs = self.model.pooler(hidden_states=outputs, pooling_metadata=model_input.pooling_metadata)

            return [outputs]