"""mxfp6 quantization registration (out-of-tree, no core patch).

Ported from the fork's vllm/model_executor/layers/quantization/qaic_quant.py
(marked "Confidential and Proprietary — Qualcomm Technologies" in the fork).
The fork wired "mxfp6" -> QaicQuantConfig by editing the core registry; here we
register from outside via register_quantization_config().

Only adaptation vs the fork: `current_platform.is_qaic()` does not exist on the
0.21 OOT platform, so we check `device_type == "qaic"` instead.
"""

from typing import Any, List, Optional

import torch
from torch.nn import Module

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

QAICQuantList: List[str] = ["awq", "gptq", "mxfp6", "fp8", "compressed-tensors"]

_REGISTERED = False


def register_qaic_quant() -> None:
    """Idempotent (general_plugins may run per-process)."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_quantization_config("mxfp6")(QaicQuantConfig)
    _REGISTERED = True


class QaicQuantConfig(QuantizationConfig):
    """MxFP6 Quantization Config class for QAIC Backend."""

    def __init__(self, quantize_method: str = "None") -> None:
        self.quantize_method = quantize_method

    def get_name(self) -> str:
        return "qaic_quant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("Not applicable on the QAIC backend.")

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QaicQuantConfig":
        try:
            quantize_method = cls.get_from_keys(config, ["quant_method"])
        except Exception:
            quantize_method = cls.get_from_keys(config, ["quantize_method"])
        return cls(quantize_method=quantize_method)

    def get_quant_method(self, layer: Module, prefix: str) -> Optional[Any]:
        raise NotImplementedError("Not applicable on the QAIC backend.")

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        # OLD (fork): current_platform.is_qaic()
        # NEW (0.21 OOT): the platform has no is_qaic(); check device_type.
        if getattr(current_platform, "device_type", None) == "qaic":
            if quant_method in QAICQuantList and user_quant == "mxfp6":
                return user_quant
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
