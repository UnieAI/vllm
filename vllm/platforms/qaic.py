# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Optional
import torch
import os
import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class QaicPlatform(Platform):
    _enum = PlatformEnum.QAIC
    device_name: str = "qaic"
    device_type: str = "qaic"
    supported_quantization: list[str] = ["mxfp6", "awq", "gptq", "fp8", "compressed-tensors", "mxfp4"]

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "qaic"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm.v1.worker.qaic_worker.QaicWorker"
            elif vllm_config.speculative_config:
                parallel_config.worker_cls = \
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "vllm.worker.qaic_worker.QaicWorker"
            else:
                parallel_config.worker_cls = \
                "vllm.worker.qaic_worker.QaicWorker"

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        assert not (vllm_config.lora_config and vllm_config.speculative_config), (
            "LORA with SPD is not yet supported for QAIC backend")
        assert not (vllm_config.lora_config and model_config.is_multimodal_model), (
            "LORA with Multi-modality is not yet supported for QAIC backend")
        assert not (vllm_config.speculative_config and model_config.is_multimodal_model), (
            "SPD with Multi-modality is not yet supported for QAIC backend")

        cache_config = vllm_config.cache_config
        if cache_config:
            # Set block size
            if envs.VLLM_USE_V1 and cache_config.enable_prefix_caching:
                cache_config.enable_prefix_caching = False
                logger.warning_once(
                    "Prefix caching is not yet supported on v1 Engine. "
                    "Will automatically disable it."
                )
            if not cache_config.enable_prefix_caching:
                if model_config.is_multimodal_model and model_config.is_encoder_decoder:
                    # This is a hack for time-being as
                    # multi-modality support is only for single batch currrently
                    cache_config.block_size = 100000
                else:
                    cache_config.block_size = vllm_config.model_config.max_model_len # ctx_len
            else:
                cache_config.block_size = model_config.max_seq_len_to_capture

        on_device_sampling_en = vllm_config.model_config.override_qaic_config and vllm_config.model_config.override_qaic_config.get('aic_include_sampler', False)
        if isinstance(on_device_sampling_en, str):
            on_device_sampling_en = vllm_config.model_config.override_qaic_config['aic_include_sampler'].lower() in ['true', '1']

        assert not (on_device_sampling_en and vllm_config.speculative_config), (
            "SPD with On-device sampling is not yet supported for QAIC backend")

        # Disaggregated prefill/decode is supported standalone for now
        if vllm_config.kv_transfer_config:
            assert not vllm_config.lora_config, (
            "LORA with Disaggregated serving not yet supported for QAIC backend")
            assert not vllm_config.speculative_config or vllm_config.speculative_config.method in ["ngram", "draft_model"], (
            "PLD and DLM based SPD Types are supported with Disaggregated serving, other SPD types such as Turbo is not yet supported with Disaggregated serving for QAIC backend")
            assert not vllm_config.model_config.is_multimodal_model, (
            "Multi-modality with Disaggregated serving not yet supported for QAIC backend")
            assert not (vllm_config.kv_transfer_config.kv_role != "kv_producer" and
                        vllm_config.cache_config.enable_prefix_caching), (
            "Prefix caching with KV-role \'kv_consumer\' or \'kv_both\' not yet supported for QAIC backend")
            if on_device_sampling_en and vllm_config.kv_transfer_config.kv_role == "kv_producer":
                logger.warning_once(
                    "On-device sampling with Disaggregated serving is only supported in "
                    "Decode cluster with no support for repetition penalty"
                )
        if vllm_config.model_config.override_qaic_config is not None \
            and "disable_multimodal" in vllm_config.model_config.override_qaic_config:
            val = vllm_config.model_config.override_qaic_config["disable_multimodal"]
            if (isinstance(val,str) and val.lower() in {"true","1"}) or bool(val):
                vllm_config.model_config.multimodal_config = None

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Qaic.")
        return False

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def default_v1(cls, model_config: "ModelConfig") -> bool:
        """
        Returns whether the current platform supports v1 by default.
        """
        return False

    def __getattr__(self, key: str):
        if key == "Event":
            return True
        device = getattr(torch, self.device_name, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            logger.warning("Current platform %s does not have '%s'" \
            " attribute.", self.device_name, key)
            return None

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str) -> bool:
        return kv_cache_dtype in ["fp8", "mxint8"]
