# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""A QAIC worker class."""
from typing import List, Optional, Tuple, Type, Set

import os
import torch
import torch.distributed

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import SequenceGroupMetadata, ExecuteModelRequest
from vllm.worker.qaic_model_runner import QaicModelRunner
from vllm.worker.qaic_pooling_model_runner import QaicPoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.spec_decode.target_model_runner import TargetModelRunner

from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

logger = init_logger(__name__)


class QaicWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes the model on a group of qaic devices.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
        speculative_model_type:str = "default",
        model_runner_cls: Optional[Type[ModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.speculative_config = vllm_config.speculative_config
        self.device = self.device_config.device
        self.speculative_model_type = speculative_model_type

        if (model_runner_cls == TargetModelRunner):
            if vllm_config.speculative_config.method == "turbo":
                self.speculative_model_type = "turbo"
            else:
                self.speculative_model_type = "target"

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        ModelRunnerClass: Type[ModelRunnerBase] = QaicModelRunner
        if self.model_config.runner_type == "pooling":
            ModelRunnerClass = QaicPoolingModelRunner

        if self.speculative_config:
            if "draft" in self.speculative_model_type:
                self.vllm_config.kv_transfer_config = None

        self.model_runner = ModelRunnerClass(vllm_config,
                                            self.speculative_model_type)

        if self.speculative_config:
            assert self.model_config.hf_config.model_type not in ["mlp_speculator", "medusa",
                "eagle"], "qaic backend currently doesn't support mlp_speculator or medusa or eagle models"

        self.is_driver_worker = is_driver_worker
        self.num_phy_kv_cache_blks = self.scheduler_config.max_num_seqs
        self.vllm_config = vllm_config
        self._configure_thread_parallelism()
        # Set Stack size to 16MB
        import resource
        _curr_stack_limit = resource.getrlimit(resource.RLIMIT_STACK)[0]
        _set_stack_limit = int(os.environ.get("VLLM_QAIC_RLIMIT_STACK", 16777216))
        if _curr_stack_limit != _set_stack_limit:
            logger.info(f"Setting RLIMIT_STACK to {_set_stack_limit/(1024*1024)}MB.")
            resource.setrlimit(resource.RLIMIT_STACK,(_set_stack_limit, -1))

    def _configure_thread_parallelism(self):
        # Configure thread parallelism
        #
        # Avoid oversubscription of CPU threads, during multi-instance execution.
        # By default there is no limit, if user set an environment variable
        # VLLM_QAIC_MAX_CPU_THREADS, then number of cpu thread running pytorch
        # sampling on cpu is limited, to avoid over-subscription.
        # The contention is amplified when running in a container where CPU limits
        # can cause throttling.
        default_limit  = min(torch.get_num_threads(),8)
        thread_limit = os.environ.get("VLLM_QAIC_MAX_CPU_THREADS", default_limit)
        if thread_limit:
            logger.warning(
                f"Reducing Torch parallelism from {torch.get_num_threads()} threads to {thread_limit}"
                " to avoid unnecessary CPU contention. Set VLLM_QAIC_MAX_CPU_THREADS to tune this value as needed.")
            torch.set_num_threads(int(thread_limit))
            if "OMP_NUM_THREADS" not in os.environ:
                os.environ["OMP_NUM_THREADS"] = str(thread_limit)

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    def init_device(self) -> None:
        """Initialize qaic device
        """
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Device ID check
        from qaicrt import Util as qaic_util
        _devices_available = qaic_util().getDeviceIds()[1]
        if self.device_config.device_group is not None:
            for device_id in self.device_config.device_group:
                if device_id not in _devices_available:
                    logger.error(f"Device id {device_id} not available!!")

        # VLLM core implemented of Chuncked prefill not supported
        if self.scheduler_config.chunked_prefill_enabled:
            logger.warning(
                "vLLM chunked prefill is not supported"
                " it is disabled automatically..."
                " qaic backend supports its own internal"
                " chunking that is enabled by default.")
            self.scheduler_config.chunked_prefill_enabled = False

    def load_model(self):
        """Load model from QEfficient Transformer library
        """
        self.model_runner.init_model()
        if self.model_config.runner_type == "pooling":
            # for pooling models initialize_cache won't be called
            self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This determines how many KV blocks can fit into the configured QAIC
        KV cache space.

        Note that since swapping is not yet supported, so this return num_cpu_blocks as 0.

        """
        # Set the number of QAIC KV blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with Qaic backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        # disable sliding window
        self.cache_config.sliding_window = None
        # Set max num batched token
        self.scheduler_config.max_num_batched_tokens = \
                                 self.scheduler_config.max_model_len * \
                                 self.scheduler_config.max_num_seqs

        assert num_cpu_blocks == 0
        if not self.cache_config.enable_prefix_caching:
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            if self.model_config.is_multimodal_model and self.model_config.is_encoder_decoder:
                # This is a hack for time-being as
                # multi-modality support is only for single batch currrently
                self.cache_config.block_size = 100000
                self.cache_config.num_gpu_blocks *=2
            if self.speculative_config is not None:
                self.cache_config.block_size = self.model_config.max_model_len + self.speculative_config.num_speculative_tokens
            assert num_gpu_blocks >= self.scheduler_config.max_num_seqs
            if self.model_config.runner_type != "pooling":
                self.model_runner.load_model()
            return

        #If prefix caching enabled
        self.cache_config.block_size = self.model_config.max_seq_len_to_capture
        self.num_phy_kv_cache_blks = num_gpu_blocks
        assert (num_gpu_blocks >= self.scheduler_config.max_num_seqs), (
                "num-gpu-blocks should be greater than equal to max-num-seqs")
        self.cache_config.num_gpu_blocks = self.num_phy_kv_cache_blks * cdiv(self.scheduler_config.max_model_len,
                                                                 self.cache_config.block_size)
        #For qaic num_cpu_blk is always 0, which is hardcoded in blk manager
        #reusing num_cpu_blocks variable for saving KV cache size
        self.cache_config.num_cpu_blocks = self.num_phy_kv_cache_blks
        if self.model_config.runner_type != "pooling":
            self.model_runner.load_model()

    @property
    def do_metadata_broadcast(self) -> bool:
        return False

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(num_seq_groups=len(
            execute_model_req.seq_group_metadata_list), )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass


    # def execute_model(
    #     self,
    #     seq_group_metadata_list: List[SequenceGroupMetadata],
    # ) -> List[SamplerOutput]:
    #     """Execute model on qaic devices.
    #     """
    #     #breakpoint()
    #     # Avoid executing models, if input group is empty..
    #     if len(seq_group_metadata_list) == 0:
    #         return []

    #     # execute model on qaic
    #     output = self.model_runner.execute_model(seq_group_metadata_list)

    #     # Qaic worker only supports single-step output.
    #     return [output]

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.

        speculative decoding is not yet supported in qaic devices.
        """
        # Set of checks to qualify
        # QAIC doesn't support disabling speculative decoding
        # on-the-fly
        if self.speculative_config is not None:
            if self.speculative_config.disable_by_batch_size is not None:
                raise ValueError("Qaic backend doesn't support disabling "
                                 "speculative decoding for new incoming requests when the number "
                                 "of enqueue requests is larger than prameter speculative_disable_by_batch_size"
                                 "re-run with speculative_disable_by_batch_size=None")

        # In case of qaic, each block is size of context length
        # Hence num_gpu_block should be same as batch size
        # In case of speculative decoding the batch size is
        # same for both draft and target model, thus num_gpu_blok
        # should be same for both the networks.
        if self.speculative_model_type != "draft":
            return 1
        return 0

    def init_distributed_environment(self):
        """Qaic uses tensor parallelism using device-group argument

        vLLM still needs the environment inited when TP/PP > 1
        """
        # Spawning a large number of workers may lead to race condition in using the same port in distributed_init_method
        # Retry logic is similar to https://github.com/vllm-project/vllm/pull/20151
        max_retries = 5
        distributed_init_method = self.distributed_init_method
        for i in range(max_retries):
            try:
                init_distributed_environment(
                    world_size=1,
                    rank=self.rank,
                    local_rank=self.local_rank,
                    distributed_init_method=distributed_init_method,
                    backend="gloo",
                )
                self.distributed_init_method = distributed_init_method
                break
            except torch.distributed.DistNetworkError as e:
                if "EADDRINUSE" in str(e):
                    if i < max_retries - 1:
                        logger.warning(f"Address {distributed_init_method} already in use. Retrying with a new port.")
                        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
                        continue
                    else:
                        logger.error(f"Failed to initialize distributed environment after {i+1} attempts: {e}")
                raise e
        ensure_model_parallel_initialized(
            1,
            1,
        )
        ensure_kv_transfer_initialized(self.vllm_config)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()