"""QaicPlatform: the out-of-tree (OOT) platform class vLLM loads for QAIC.

Ported and trimmed from the fork's vllm/platforms/qaic.py, with two changes:
  1. ``_enum = PlatformEnum.OOT`` (upstream has no QAIC enum value).
  2. QAIC knobs are read from ``vllm_config.additional_config`` instead of the
     fork's custom ``ModelConfig.override_qaic_config`` field, so we never have
     to patch arg_utils.py / config.

IMPORTANT: method signatures on vllm.platforms.interface.Platform drift between
releases. After ``pip install vllm==<target>``, open the installed
``vllm/platforms/interface.py`` and confirm the signatures below still match
(the README PART 1 tells you exactly how). Where unsure, a TODO marks it.
"""

import os
import shlex
from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


class QaicPlatform(Platform):
    _enum = PlatformEnum.OOT
    # vLLM still uses `device_name` in a few places to construct torch.device
    # strings (for example distributed GroupCoordinator). QAIC is not a torch
    # device type, so use CPU for host-side coordination and expose the hardware
    # name through get_device_name().
    device_name: str = "cpu"
    device_type: str = "qaic"
    # Host-side tensors live on CPU; real compute runs on the AIC via qaicrt.
    dispatch_key: str = "CPU"
    # Mirrors the fork's list (incl. mxfp4). Only mxfp6 is wired through
    # register_quantization_config today; the rest are declared for parity so we
    # don't silently reject models the fork accepted.
    supported_quantization: list[str] = [
        "mxfp6", "mxfp4", "awq", "gptq", "fp8", "compressed-tensors",
    ]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "qaic"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # QAIC execution is owned by qaicrt; vLLM host-side tensors remain CPU.
        return None

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.manual_seed(seed)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    def uses_host_device_handling(self) -> bool:
        # QAIC is not a torch device type. vLLM host tensors stay on CPU while
        # the compiled QPC executes on the AIC through qaicrt, so DeviceConfig
        # must not call torch.device("qaic").
        return True

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def is_kv_cache_dtype_supported(
        cls, kv_cache_dtype: str, model_config: "ModelConfig" = None
    ) -> bool:
        # KNOWN CORE GAP: upstream CacheDType Literal (vllm/config/cache.py)
        # does NOT include "mxint8" in vLLM 0.21+. So mxint8 KV cache fails
        # *validation* before it ever reaches us. Until that is resolved
        # (small core patch or upstream PR — see README "Known core gaps"),
        # only fp8 passes here.
        return kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        # This plugin supports the V1 engine ONLY. The fork's V0 path
        # (and ~half of the Qualcomm core patch) is intentionally dropped.
        if parallel_config.worker_cls == "auto":
            assert getattr(envs, "VLLM_USE_V1", True), (
                "vllm-qaic supports the V1 engine only. Set VLLM_USE_V1=1."
            )
            parallel_config.worker_cls = "vllm_qaic.worker.QaicWorker"

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        if scheduler_config is not None:
            scheduler_config.async_scheduling = False

            # Decode-priority scheduling: QAIC runs decode and prefill as separate
            # sequential QPC graphs, so vLLM's default prefill/decode mixing
            # (chunked prefill) adds the prefill QPC latency to every decode token's
            # TPOT at high concurrency. Install a scheduler that keeps decode steps
            # pure unless a user opts out or set their own scheduler_cls.
            if (os.environ.get("QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER") != "1"
                    and scheduler_config.scheduler_cls in (None, "")):
                scheduler_config.scheduler_cls = (
                    "vllm_qaic.scheduler.QaicDecodePriorityScheduler")

        # QAIC knobs come from --additional-config '{"num_cores":16, ...}'
        # (replaces the fork's --override-qaic-config / --device-group).
        qaic_cfg = cls._normalize_qaic_config(
            dict(vllm_config.additional_config or {}))

        # block_size for QAIC: one logical block == full context (ctx_len).
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        max_seq_len_to_capture = qaic_cfg.get("max_seq_len_to_capture")
        if max_seq_len_to_capture is not None:
            model_config.max_seq_len_to_capture = int(max_seq_len_to_capture)
        if (
            model_config is not None
            and not hasattr(model_config, "max_seq_len_to_capture")
        ):
            model_config.max_seq_len_to_capture = model_config.max_model_len
        # Paged (block-table) KV mode. Requires a QPC compiled with paged KV
        # (QEfficient export(paged_kv=True)). When on: one logical block == one
        # page (block_size = page_size), and prefix caching is allowed (paging
        # supports shared/reused blocks); the runner feeds the full block_table
        # to the QPC. When off: the legacy QAIC layout (one block == full ctx,
        # prefix caching disabled).
        paged_kv = bool(qaic_cfg.get("paged_kv", False))
        if cache_config is not None:
            if paged_kv:
                page_size = int(qaic_cfg.get("page_size", 128))
                cache_config.block_size = page_size
                logger.warning_once(
                    "QAIC paged KV enabled: block_size(page_size)=%d, "
                    "prefix_caching=%s", page_size, cache_config.enable_prefix_caching)
            else:
                if (
                    getattr(envs, "VLLM_USE_V1", True)
                    and cache_config.enable_prefix_caching
                ):
                    cache_config.enable_prefix_caching = False
                    logger.warning_once(
                        "Prefix caching is not supported on QAIC V1 (non-paged); "
                        "disabling. Set additional_config paged_kv=true to enable.")
                cache_config.block_size = model_config.max_model_len

        # Pass QAIC knobs through verbatim. NORMALIZATION IS NOT DONE HERE:
        # the fork normalizes keys (num_cores/aic_num_cores, mxfp6->mxfp6_matmul,
        # mxint8_kv_cache, prefill_seq_len, aic_enable_depth_first/dfs,
        # device_group/device_ids, ...) inside compile-config assembly via
        # _clean_config(). That lives in compile_config.py (ported from the
        # fork's model_loader/qaic.py by port_from_fork.sh). CONTRACT: the ported
        # compile_config MUST read these knobs from `vllm_config.additional_config`
        # and run `_clean_config()` on them before calling QEfficient.compile(),
        # otherwise the compile keys won't match QEfficient's expectations.
        vllm_config.additional_config = qaic_cfg

    @staticmethod
    def _normalize_qaic_config(config: dict) -> dict:
        legacy_override = config.pop("override_qaic_config", None)
        if legacy_override:
            if isinstance(legacy_override, str):
                for item in shlex.split(legacy_override):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        config[key] = value
                    else:
                        config[item] = True
            elif isinstance(legacy_override, dict):
                config.update(legacy_override)
            else:
                raise TypeError(
                    "override_qaic_config must be a string or dict when "
                    "passed through --additional-config.")

        device_group = config.get("device_group")
        if isinstance(device_group, str):
            config["device_group"] = [
                int(x) for x in device_group.replace(" ", "").split(",") if x
            ]
        elif isinstance(device_group, int):
            config["device_group"] = [device_group]

        return config
