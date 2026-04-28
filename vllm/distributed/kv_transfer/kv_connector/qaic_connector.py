# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------

"""
QaicCache KV Cache Connector for Disaggregated Serving

The QaicConnector can transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using separate process
for book keeping, to support xPyD via shared memory access between processes with
KV caching support.
"""

from typing import TYPE_CHECKING, List, Tuple, Union, Optional

import torch
import queue
import signal
import atexit
import copy
import os
import sys
import threading
import time
import uuid
import numpy as np
import copy
import gc
import random
from multiprocessing import resource_tracker
from logging import DEBUG
from vllm.config import VllmConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from qaic_disagg.kv_handoff.protocol import QaicBufferType, QaicKvHandOffReqType, QaicKvHandOffGetResp, QaicKvHandOffGetReq, QaicKvHandOffPutReq
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
import zmq
from torch.distributed import ProcessGroup
from zmq import IPV6  # type: ignore
from zmq import SUB, SUBSCRIBE, XPUB, XPUB_VERBOSE, Context  # type: ignore
import msgspec
from vllm.utils import is_valid_ipv6_address, resolve_obj_by_qualname, zmq_socket_ctx, make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from unittest.mock import patch
from dataclasses import replace
if TYPE_CHECKING:
    from vllm.worker.qaic_model_runner import ModelInputForQaic

logger = init_logger(__name__)
KV_LOOKUP_RETRIES= 5
KV_LOOKUP_RETRIES_INTERVAL = 0.005
FORCE_CLEAN_UP_MULTIPLIER = 2
MAX_UID = 1000000

class ShmBuffer:

    def __init__(self,
                 shape: List[int],
                 dtype: np.dtype,
                 total_bytes_of_buffer : Optional[int] = None,
                 num_buffs: int = 1,
                 name: Optional[str] = None,
                 create: bool = True):
        """
        A shared memory buffer implementation, wrapper over shared_memory.SharedMemory
        During creation, `name` is None and the buffer is created. We can pass the
        created object to other processes by pickling it. The other processes will
        get the name of the shared memory and open it, so that they can access the
        same shared memory buffer.
        """# noqa
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.list_of_np_buff = []
        self.cleanup_done = False
        if not isinstance(shape, list) and not isinstance(shape, tuple):
            if isinstance(shape, int):
                shape = [shape]
            else:
                raise ValueError("shape must be a list or tuple or int")

        # Calculate number of bytes in the buffer
        if not total_bytes_of_buffer:
            self.total_bytes_of_buffer = 1
            for dim in shape:
                self.total_bytes_of_buffer *= dim
            if self.dtype  == np.float16:
                self.total_bytes_of_buffer *= 2
            elif self.dtype in [np.float32, np.int32]:
                self.total_bytes_of_buffer *= 4
            elif self.dtype in [np.float64, np.int64]:
                self.total_bytes_of_buffer *= 8
        else:
            self.total_bytes_of_buffer = total_bytes_of_buffer

        self.buff_size = self.total_bytes_of_buffer
        self.total_bytes_of_buffer *= num_buffs
        self.num_buffs = num_buffs
        self.is_creator = create
        #print ("Total bytes of buffer: ", self.total_bytes_of_buffer)
        if self.is_creator is True:
            try:
                self.shared_memory = shared_memory.SharedMemory(name=name,
                    create=True, size=self.total_bytes_of_buffer)
                self.name = self.shared_memory.name
            except Exception as e:
                raise ValueError("Exception occured during creation of shared memory!") from e
        else:
            # we are opening an existing buffer
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            # with patch("multiprocessing.resource_tracker.register",
            #            lambda *args, **kwargs: None):
            try:
                self.shared_memory = shared_memory.SharedMemory(name=name)
                # See https://docs.python.org/3/library/multiprocessing.shared_memory.html # noqa
                # Some platforms allocate memory based on page size,
                # so the shared memory block size may be larger or equal
                # to the requested size. The size parameter is ignored
                # when attaching to an existing block.
                assert (self.shared_memory.size
                        >= self.total_bytes_of_buffer)
            except FileNotFoundError:
                raise ValueError("Shared memory buffer not found")
            except Exception as e:
                raise ValueError("Shared memory buffer not found") from e

        # create list of numpy arrays
        for i in range(self.num_buffs):
            with self.get_data(i) as buff:
                    self.list_of_np_buff.append(np.ndarray(self.shape, dtype=self.dtype, buffer=buff))

    def cleanup(self):
        if not self.cleanup_done:
            self.cleanup_done = True
            self.shared_memory.close()
            if not self.is_creator:
                self.shared_memory.unlink()
            del self.list_of_np_buff
            del self.shared_memory

    def __del__(self):
        self.cleanup()

    @contextmanager
    def get_data(self, current_idx: int = 0):
        start = current_idx * self.buff_size
        end = start +  self.buff_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

class QaicKVCacheBank():

    def __init__(self):
        self.MemBank = dict()
        self.uid = 0
        self.pid = hex(os.getpid())[2:]

    def get_Storage(self, shape:Tuple, dtype:np.dtype, num_buffs:int, name:Optional[str]=None):
        if name is None:
            self.uid  = (self.uid + 1) % MAX_UID
            name = "psm_" + hex(self.uid)[2:] + f"_" + self.pid
            create = True
        else:
            create = False
        shm = ShmBuffer(shape = shape,
                    dtype = dtype,
                    num_buffs = num_buffs,
                    name = name,
                    create = create)
        self.MemBank[shm.name] = shm
        return shm.name, shm.list_of_np_buff

    def release_Storage(self, name:Optional[str]=None):
        assert name in self.MemBank, f"Storage {name} not found in MemBank"
        shm =  self.MemBank[name]
        if shm.is_creator:
            #print("Unregistering tracking for ",shm.shared_memory._name)
            resource_tracker.unregister(shm.shared_memory._name, "shared_memory")
        shm.cleanup()
        del self.MemBank[name]

    def cleanup(self):
        buff_name = list(self.MemBank.keys())
        for _name in buff_name:
            self.release_Storage(_name)

class QaicConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.transfer_config: KVTransferConfig = config.kv_transfer_config
        self.kv_role: Optional[str] = config.kv_transfer_config.kv_role
        self.vllm_config : VllmConfig = config
        self.identity = str(uuid.uuid4()).encode('utf-8')
        self.kv_rank = config.kv_transfer_config.kv_rank
        self.kv_port = config.kv_transfer_config.kv_port
        self.is_producer = self.kv_role == "kv_producer"
        self.mem_bank = QaicKVCacheBank()
        self.shm_tracker:List[str] = []
        self.force_clean_up_threshold = FORCE_CLEAN_UP_MULTIPLIER * config.scheduler_config.max_num_seqs

        logger.info("Initializing QaicConnector under kv_transfer_config %s",
                     self.transfer_config)
        print(self.transfer_config, flush=True)

        #KV_both mode is not supported yet
        assert self.kv_role != "kv_both", "KV_BOTH mode is not supported yet"

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        kv_ip = self.transfer_config.kv_ip
        if is_valid_ipv6_address(kv_ip):
            kv_ip = '[' + kv_ip + ']'
            self.ctx.setsockopt(zmq.IPV6,1)

        ipc_path = f"tcp://{kv_ip}:{self.transfer_config.kv_port}"

        self.socket = make_zmq_socket(self.ctx,
                            ipc_path,
                            zmq.constants.DEALER,
                            bind=False,
                            identity=self.identity)
        self.decoder = MsgpackDecoder(QaicKvHandOffGetResp)
        self.encoder = MsgpackEncoder()
        self.encoder_send = MsgpackEncoder(QaicKvHandOffPutReq)

        #Invoke Threads
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.close)

    def signal_handler(self, signum, frame):
        logger.info("Received signal %s, exiting...", signum)
        self.close()

    def send_kv_cache_to_store(self, pkt: QaicKvHandOffPutReq):
        """Send KV cache to external connector"""
        put_req = self.encoder_send.encode(pkt)
        try:
            resp = QaicKvHandOffReqType.RESP_BUFFER_FULL
            retries_cnt = 0
            while resp == QaicKvHandOffReqType.RESP_BUFFER_FULL:
                self.socket.send_multipart((QaicKvHandOffReqType.PUT.value,put_req[0]), copy=False)
                (resp, _ ) = self.socket.recv_multipart(copy=False)
                resp = QaicKvHandOffReqType(bytes(resp.buffer))
                if resp == QaicKvHandOffReqType.RESP_BUFFER_FULL:
                    time.sleep(random.randint(1,10)/1000)
                    retries_cnt +=1
                if retries_cnt > KV_LOOKUP_RETRIES:
                    retries_cnt = 0
                    logger.warning(f"KV Handoff Storage Full...")
                    # Trigger garbage collection
                    gc.collect()
                    time.sleep(random.randint(20,100)/1000)
        except Exception as e:
            raise ValueError(f"Unable to access KV store due to an exception: {e}")

    def get_kvcache_from_store(self, prompt_hash) -> Optional[QaicKvHandOffGetResp]:
        """Get kv cache from kv_store."""
        result = None
        # Get kv cache from kv_store
        req_pkt = QaicKvHandOffGetReq(
                        buff_type = 0,
                        timestamp = time.perf_counter(),
                        key_hash = prompt_hash,
                        rank= self.kv_rank,
                    )
        encode_req_pkt = self.encoder.encode(req_pkt)[0]
        max_retries = KV_LOOKUP_RETRIES
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Send request to KV store
                self.socket.send_multipart((QaicKvHandOffReqType.GET.value, encode_req_pkt))
                (resp, resp_payload) = self.socket.recv_multipart(copy=False)
                resp = QaicKvHandOffReqType(bytes(resp.buffer))
                if QaicKvHandOffReqType.RESP_OK == resp:
                    result = self.decoder.decode(resp_payload)
                    break
                if QaicKvHandOffReqType.RESP_NOT_FOUND == resp:
                    retry_count +=1
                    time.sleep(KV_LOOKUP_RETRIES_INTERVAL)
                    print(f"KV Resp Not Found Received for {req_pkt}")
                    logger.debug(f"KV Resp Not Found Received for {req_pkt}")
                    continue
                elif QaicKvHandOffReqType.RESP_ERROR == resp:
                    logger.debug(f"KV Resp Error Received for {req_pkt}")
                    retry_count = max_retries
                else:
                    raise ValueError(f"Invalid response from KV store: {resp}")
            except Exception as e:
                raise ValueError(f"Unable to access KV store due to an exception: {e}")
        if self.kv_role == "kv_consumer":
            if retry_count >= max_retries or result is None:
                raise ValueError(
                    f"Unable to find prompt hash {req_pkt.key_hash} in KV store!")
        return result

    def cleanup_callback(self, clean_all=False):
        """Cleanup callback for kv_store."""
        if len(self.shm_tracker) > 0:
            for buff in (self.shm_tracker[0:2] if not clean_all else self.shm_tracker):
                self.mem_bank.release_Storage(buff)
            self.shm_tracker = self.shm_tracker[2:] if not clean_all else []

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForQaic",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForQaic"]:
        """Receive kv_caches and hidden states from kv_store."""
        if not model_input.is_prompt:
            model_input = replace(model_input,
                            cleanup_callback = self.cleanup_callback)
            return [], False, model_input
        # Current Assumption is that model_input will be for prefill or decode
        # will never be together as it is a case in chunked prefill with splitfuse
        # Need new implementation for V1 architecture
        kv_info = model_executable.kv_cache_info()
        kv_shape = kv_info[0][0]
        bypass_model_exec = False
        hidden_or_intermediate_states = []
        if model_input.kv_handoff_metadata is not None:
            kv_handoff_metadata = model_input.kv_handoff_metadata
        else:
            kv_handoff_metadata = []
        kv_shm_buff, logit_shm_buff_name  = None, None
        skip_sampling = False
        for pl,prompt_hash  in zip(model_input.seq_lens, model_input.prompt_hashes):
            kv_shape[2] = pl
            kv_shape[0] = 1
            if not self.is_producer:
                # Get kv cache from kv_store
                resp = self.get_kvcache_from_store(prompt_hash)
                assert resp.buff_type == 0, "Raw np.ndarray KV exchange not supported yet"
                kv_shm_buff, logit_shm_buff_name  = resp.payload

            kv_storage_shm_name, kv_buff = self.mem_bank.get_Storage(shape=kv_shape,
                                      dtype = kv_info[0][1],
                                      num_buffs = kv_info[1],
                                      name = kv_shm_buff)
            logits_shm_name, logit_buff = self.mem_bank.get_Storage(
                    shape=(1,1,model_executable.vocab_size),
                            dtype = np.float32,
                            num_buffs = 1,
                            name = logit_shm_buff_name)
            kv_handoff_metadata.append([kv_storage_shm_name,logits_shm_name])
            kv_caches.append(kv_buff)
            hidden_or_intermediate_states.append(logit_buff[0])

        if self.is_producer:
            skip_sampling = True
        else:
            bypass_model_exec = True

        model_input = replace(model_input,
                              kv_handoff_metadata = kv_handoff_metadata,
                              skip_sampling = skip_sampling,
                              cleanup_callback = self.cleanup_callback)
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForQaic",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        if not self.is_producer:
            #shm unlink logic
            if model_input.is_prompt:
                # Add shm to local clean-up tracker
                for buff_list in model_input.kv_handoff_metadata:
                    for buff_name in buff_list:
                        self.shm_tracker.append(buff_name)
            else:
                #Force cleanup if average GL is less than batch size
                if len(self.shm_tracker) > self.force_clean_up_threshold:
                    logger.info(f"Forcing clean_up of {len(self.shm_tracker)} shm buffers, as average GL is less than batch size")
                    self.cleanup_callback(clean_all=True)
            return

        # if kv_producer
        assert model_input.is_prompt, "kv_producer should only be called for prefills"
        assert model_input.kv_handoff_metadata, "kv_handoff_metadata should be not be None or empty list"
        for buff_info,prompt_hash  in zip(model_input.kv_handoff_metadata, model_input.prompt_hashes):
            for buff in buff_info:
                self.shm_tracker.append(buff)
            req_pk  = QaicKvHandOffPutReq(
                    buff_type = QaicBufferType.SHM,
                    timestamp = time.perf_counter(),
                    key_hash = prompt_hash,
                    rank = self.kv_rank,
                    payload = [buff for buff in buff_info],
                    num_buff = 2,
                    )
            self.send_kv_cache_to_store(req_pk)

    def close(self):
        self.mem_bank.cleanup()
        if hasattr(self, 'ctx'):
            if self.ctx:
                self.ctx.destroy(linger=10)
