# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import msgspec
import numpy as np

from enum import Enum, IntEnum
from typing import Optional, Tuple, List, Union
PrefixHash = int

class QaicKvHandOffReqType(Enum):
    """The type of kv_handoff message."""
    GET = b'\x00'
    PUT = b'\x01'
    RESP_OK =  b'\x02'
    RESP_ERROR =  b'\x03'
    RESP_BUFFER_FULL=  b'\x04'
    RESP_NOT_FOUND =  b'\x05'
    RESP_INVALID_CMD =  b'\x06'


class QaicBufferType(IntEnum):
    """The type of kv_handoff message."""
    SHM = 0
    NP = 1

class QaicKvHandOffGetResp(msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]):
    """
    Header for kv_handoff messages.
    """
    buff_type : QaicBufferType
    timestamp : float
    rank : int
    num_buff : int
    payload: Optional[List[str]] = None

class QaicKvHandOffPutReq(msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]):
    """
    Header for kv_handoff messages.
    """
    buff_type : QaicBufferType
    timestamp : float
    key_hash : PrefixHash
    rank : int
    num_buff : int
    payload: List[str]

class QaicKvHandOffGetReq(msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]):
    """
    Header for kv_handoff messages.
    """
    buff_type : QaicBufferType
    timestamp : float
    key_hash : PrefixHash
    rank : int


ReqType = Union[QaicKvHandOffPutReq, QaicKvHandOffGetReq]