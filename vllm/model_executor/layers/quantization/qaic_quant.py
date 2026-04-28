# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Not a contribution.
# Original file did not have any copyright or license markings.
#
# Qualcomm Technologies, Inc. Copyright added in regards to Qualcomm Technologies, Inc.
# modifications only.
# ---------------------------------------------------------------------------------------
import torch
from typing import Any, Dict, List, Optional
from torch.nn import Module
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.platforms import current_platform

QAICQuantList: List[str]= ["awq", "gptq", "mxfp6", "fp8", "compressed-tensors"]

class QaicQuantConfig(QuantizationConfig):
    """MxFP6 Quantization Config class for QAIC Backend."""

    def __init__(
        self,
        quantize_method: str = "None",
    ) -> None:
        self.quantize_method = quantize_method

    def get_name(self) -> str:
        return "qaic_quant"

    @classmethod
    def get_supported_act_dtypes(self) -> List[str]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "This function should not be called with QAIC Backend")

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QaicQuantConfig":
        try:
            quantize_method = cls.get_from_keys(config, ["quant_method"])
        except:
            quantize_method = cls.get_from_keys(config, ["quantize_method"])
        return cls(quantize_method=quantize_method)

    def get_quant_method(self, layer: Module, prefix: str) -> Optional[Any]:
        raise NotImplementedError(
            "This function should not be called with QAIC Backend")

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        if current_platform.is_qaic():
            if quant_method in QAICQuantList and user_quant == "mxfp6":
                return user_quant
        return None


    def get_scaled_act_names(self) -> List[str]:
        return []
