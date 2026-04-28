# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""A QAIC worker class."""
import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import get_uniform_page_size
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.qaic_model_runner import QaicModelRunner
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler_output import SchedulerOutput


class QaicWorker(WorkerBase):
    """A worker class that executes the model on a group of qaic devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
        speculative_model_type: str = "default",
    ):

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.speculative_model_type = speculative_model_type

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        self.profiler = None

        if self.speculative_config:
            assert self.model_config.hf_config.model_type not in [
                "mlp_speculator",
                "medusa",
                "eagle",
            ], "qaic backend currently doesn't support mlp_speculator or medusa or eagle models"

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on QAIC, ignore it.")
        pass

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        logger.warning("sleep mode is not supported on QAIC, ignore it.")
        pass

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        # disable sliding window
        self.cache_config.sliding_window = None

        assert num_cpu_blocks == 0
        if not self.cache_config.enable_prefix_caching:
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            if (
                self.model_config.is_multimodal_model
                and self.model_config.is_encoder_decoder
            ):
                # This is a hack for time-being as
                # multi-modality support is only for single batch currrently
                self.cache_config.num_gpu_blocks *= 2
            # Sanity check
            assert num_gpu_blocks == self.scheduler_config.max_num_seqs + 1
            return
        else:
            raise NotImplementedError("prefix caching is not supported on QAIC in V1")

    def init_device(self) -> None:
        """Initialize qaic device"""
        self.device = self.device_config.device

        init_qaic_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Device ID check
        from qaicrt import Util as qaic_util

        _devices_available = qaic_util().getDeviceIds()[1]
        if self.device_config.device_group is not None:
            for device_id in self.device_config.device_group:
                if device_id not in _devices_available:
                    logger.error(f"Device id {device_id} not available!!")

        self._configure_thread_parallelism()

        # Construct the model runner
        self.model_runner: QaicModelRunner = QaicModelRunner(
            self.vllm_config, self.device, self.speculative_model_type
        )

    def _configure_thread_parallelism(self):
        # Configure thread parallelism
        #
        # Avoid oversubscription of CPU threads, during multi-instance execution.
        # By default there is no limit, if user set an environment variable
        # VLLM_QAIC_MAX_CPU_THREADS, then number of cpu thread running pytorch
        # sampling on cpu is limited, to avoid over-subscription.
        # The contention is amplified when running in a container where CPU limits
        # can cause throttling.
        default_limit = min(torch.get_num_threads(), 8)
        thread_limit = os.environ.get("VLLM_QAIC_MAX_CPU_THREADS", default_limit)
        if thread_limit:
            logger.warning(
                f"Reducing Torch parallelism from {torch.get_num_threads()} threads to {thread_limit}"
                " to avoid unnecessary CPU contention. Set VLLM_QAIC_MAX_CPU_THREADS to tune this value as needed."
            )
            torch.set_num_threads(int(thread_limit))
            if "OMP_NUM_THREADS" not in os.environ:
                os.environ["OMP_NUM_THREADS"] = str(thread_limit)

    def load_model(self):
        """Load model from QEfficient Transformer library"""
        self.model_runner.load_model()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        logger.warning("reloading weights is not supported on QAIC, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        # QAIC does not support paged attention,
        # so we set available memory based on desired num_gpu_blocks.
        # The number of QAIC KV blocks should match the maximum number of
        # sequences that can be processed in a single batch.

        # Since the v1 block pool creates a null_block using self.free_block_queue.popleft(),
        # the actual number of usable blocks is num_gpu_blocks - 1.
        num_gpu_blocks = (
            self.cache_config.num_gpu_blocks_override
            if self.cache_config.num_gpu_blocks_override
            else self.scheduler_config.max_num_seqs
        ) + 1
        if (
            self.model_config.is_multimodal_model
            and self.model_config.is_encoder_decoder
        ):
            # This is a hack for time-being as
            # multi-modality support is only for single batch currrently
            num_gpu_blocks *= 2
        page_size = get_uniform_page_size(self.get_kv_cache_spec())
        return (
            num_gpu_blocks
            * page_size
            * self.model_config.get_num_layers(self.parallel_config)
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        self.model_runner._qaic_dummy_run()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        assert isinstance(output, ModelRunnerOutput)
        return output

    def profile(self, is_start: bool = True):
        logger.warning("profile is not supported on QAIC, ignore it.")
        pass

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_qaic_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Qaic uses tensor parallelism using device-group argument

    vLLM still needs the environment inited when TP/PP > 1
    """
    init_distributed_environment(
        world_size=1,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="gloo",
    )
    ensure_model_parallel_initialized(
        1,
        1,
    )
    ensure_kv_transfer_initialized(vllm_config)
