# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------

import numpy as np
import time
import warnings
from collections import defaultdict, deque
from multiprocessing import shared_memory
from dataclasses import dataclass
from typing import List, Union, Optional
from qaic_disagg.kv_handoff.protocol import QaicKvHandOffGetResp, QaicBufferType, QaicKvHandOffPutReq

@dataclass(slots=True)
class KVCacheEntry:
    kv_cache_payload: Union[List[np.ndarray], List[str]]
    buff_type: QaicBufferType
    #prompt_shape: Tuple[int,int]
    prefill_rank: Optional[int] = None
    prefill_kvhandoff_time: Optional[float] = None
    timestamp: float = time.perf_counter()

    @classmethod
    def create_obj(cls, header: QaicKvHandOffPutReq):
        # Validate KV Cache payload
        if len(header.payload) != header.num_buff:
            raise ValueError(f"KV Cache payload length {len(header.payload)} does not match header num_kv_cache_entries {header.num_buffs}")
        # Convert KV Cache payload to numpy arrays
        return cls(
                    header.payload,
                    header.buff_type,
                    #header.prompt_shape,
                    header.rank,
                    time.perf_counter() - header.timestamp,
                    time.perf_counter()
                    )
        
    def get_pkt_to_send(self):
        """ Get packet to send to Qaic VLLM connector"""
        return QaicKvHandOffGetResp(
            self.buff_type,
            time.perf_counter(),
            self.prefill_rank,
            #self.prefill_shape,
            len(self.kv_cache_payload),
            self.kv_cache_payload)

class QaicKVStore:
    """ Key value store"""
    def __init__(self):
        self._store = defaultdict(deque)

    def add(self, key, value: KVCacheEntry):
        self._store[key].append(value)

    def get(self, key):
        try:
            result =  self._store[key].popleft()
            if len(self._store[key]) == 0:
                del self._store[key]
            return result
        except (KeyError, IndexError):
            raise KeyError(f"No entries found for prompt hash: {key}")

    def peek(self, key):
        try:
            return self._store[key][0]
        except (KeyError, IndexError):
            raise KeyError(f"No entries found for prompt hash: {key}")

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        return sum(map(len, self._store.values()))

    def __del__(self):
        """ Clear Key value store"""
        count = 0
        if hasattr(self, '_store'):
            for key in list(self._store):
                for _ in list(range(len(self._store[key]))):
                    pkt = self.get(key)
                    if pkt and pkt.buff_type != QaicBufferType.SHM:
                        del pkt
                        continue
                    #shm store found
                    count+=1
                    for name in pkt.kv_cache_payload:
                        try:
                            shm = shared_memory.SharedMemory(name=name)
                            print(f"Cleaning shared memory buffers {name}")
                            shm.close()
                            shm.unlink()
                        except FileNotFoundError:
                            continue
                        except Exception as e:
                            warnings.warn(f"Failed to unlink shared memory '{name}': {e}")
                    del pkt
            if count != 0:
                print(f"Cleaned shared memory buffers for {count} entries.")
            del self._store