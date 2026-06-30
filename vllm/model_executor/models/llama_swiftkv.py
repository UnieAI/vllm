# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from typing import Optional, Union, List
import torch
from torch import nn
from .interfaces import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput

# Wrapper class to bypass Model registery checks
# Model is implemented in QEFFicient transformer library for qaic devices
class LlamaSwiftKVForCausalLM(nn.Module, SupportsPP):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        pass

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        pass

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        pass