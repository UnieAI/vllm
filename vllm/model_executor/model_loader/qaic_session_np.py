# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Technologies, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
# Adapted from https://github.com/quic/efficient-transformers/blob/main/QEfficient/generation/cloud_infer.py
# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------

from typing import Dict, List, Optional, Any, Tuple
from warnings import warn
import numpy as np
import sys
import json
from time import perf_counter
from multiprocessing import Event, Queue
import copy
# import queue
try:
    import qaicrt
except ImportError:
    import platform
    import sys

    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    import qaicrt
try:
    import QAicApi_pb2 as aicapi
except ImportError:
    import sys
    sys.path.append("/opt/qti-aic/dev/python")
import QAicApi_pb2 as aicapi

aic_to_np_dtype_mapping = {
    aicapi.FLOAT_TYPE: np.dtype(np.float32),
    aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
    aicapi.INT8_Q_TYPE: np.dtype(np.int8),
    aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
    aicapi.INT16_Q_TYPE: np.dtype(np.int16),
    aicapi.INT32_Q_TYPE: np.dtype(np.int32),
    aicapi.INT32_I_TYPE: np.dtype(np.int32),
    aicapi.INT64_I_TYPE: np.dtype(np.int64),
    aicapi.INT8_TYPE: np.dtype(np.int8),
}

from threading import Event
output_qbuffers = {}

class DisAgg_QAICInferenceSession:
    def __init__(
        self,
        qpc_path: str,
        device_ids: Optional[List[int]] = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
        stages: int = 1, 
        cluster_id=None
    ):
        self.queue_len = stages+1
        self.execObj = [qaicrt.ExecObj] * (self.queue_len)
        self.execObj_available = self.queue_len
        self.execObj_ids_dict = {}
        self.event_pp_full = Event()
        self.read_output = Event()
        self.output = Queue()
        # Load QPC
        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)  # Async API
        if enable_debug_logs:
            assert (
                self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG)
                == qaicrt.QStatus.QS_SUCCESS
            ), "Failed to setLogLevel"
        qpc = qaicrt.Qpc(str(qpc_path))
        # Load IO Descriptor
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to getIoDescriptor"
        iodesc.ParseFromString(bytes(iodesc_data))
        self.allowed_shapes = [
            [
                (aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims))
                for x in allowed_shape.shapes
            ]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {
            binding.name: binding.index for binding in self.bindings
        }

        # Create and load Program
        prog_properties = qaicrt.QAicProgramProperties()
        prog_properties.SubmitRetryTimeoutMs = 60_000
        queueProperties = qaicrt.QAicQueueProperties()
        queueProperties.numThreadsPerQueue = 1
        self.queue.initProperties(queueProperties)

        if device_ids and len(device_ids) > 1:
            prog_properties.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, None, qpc, prog_properties)
        assert (
            self.program.load() == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to load program"
        self.activate_done = False
        if activate:
            self.activate()
        # Create input qbuffers and buf_dims
        self.qbuffers = [[
            qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings
        ]] * (self.queue_len)
        self.buf_dims = [qaicrt.BufferDimensionsVecRef(
            [
                (
                    aic_to_np_dtype_mapping[binding.type].itemsize,
                    list(binding.dims),
                )
                for binding in self.bindings
            ]
        )] * (self.queue_len)

        self.kv_slicing_spec_handle = None
        self.repetition_penalty_spec_handle = None
        if "past_key.0_RetainedState" in self.binding_index_map:
            self.kv_shape=self.bindings[self.binding_index_map["past_key.0_RetainedState"]].dims
            self.kv_shape[0] = 1
            self.kv_size=aic_to_np_dtype_mapping[self.bindings[self.binding_index_map["past_key.0_RetainedState"]].type]
            buffer_spec_json = self.get_json_for_kv_cache_slicing()
            self.kv_slicing_spec_handle = self.get_slicing_spec_handle(buffer_spec_json=buffer_spec_json)
        past_repetition_penalty_buffer = "past_repetition_penalty_buffer"
        if past_repetition_penalty_buffer in self.input_names:
            self.repetition_penalty_map = [
                (
                    past_repetition_penalty_buffer,
                    self.binding_index_map[past_repetition_penalty_buffer],
                )
            ]
            buffer_spec_json = self.get_json_for_repetition_penalty_slicing()
            self.repetition_penalty_spec_handle = self.get_slicing_spec_handle(
                buffer_spec_json=buffer_spec_json
            )

        self.timers = [0] * self.queue_len
        self.cluster_id = cluster_id

        self.decode_buff_map = []
        for name in self.input_names:
            if name.startswith("past_key") or name.startswith("past_value"):
                self.decode_buff_map.append((name, self.binding_index_map[name]))
        self.decode_buff_map.sort(key=lambda x: x[0])

        self.prefill_buff_map = []
        for name in self.output_names:
            if name.endswith("RetainedState"):
                self.prefill_buff_map.append((name.replace('_RetainedState',''), self.binding_index_map[name]))
        self.prefill_buff_map.sort(key=lambda x: x[0])
        for name in self.output_names:
            if name.startswith("log"):
                self.prefill_buff_map.append((name, self.binding_index_map[name]))

    def get_json_for_kv_cache_slicing(self):
        size = aic_to_np_dtype_mapping[self.bindings[self.binding_index_map["past_key.0"]].type]
        json_spec = {
            "BufferSpecs": [
                {
                    "Name": "past_key.*",
                    "ElemSize": size.itemsize,
                    "DimSpecs": [
                        {"start": "batch_index"},
                        {"start": 0},
                        {"start": "ctx_start"},
                        {"start": 0},
                    ],
                },
                {
                    "Name": "past_value.*",
                    "ElemSize": size.itemsize,
                    "DimSpecs": [
                        {"start": "batch_index"},
                        {"start": 0},
                        {"start": "ctx_start"},
                        {"start": 0},
                    ],
                }
            ]
        }
        return json.dumps(json_spec)

    def get_json_for_repetition_penalty_slicing(self):
        size = aic_to_np_dtype_mapping[
            self.bindings[self.binding_index_map["past_repetition_penalty_buffer"]].type
        ]
        json_spec = {
        "BufferSpecs": [
                {
                "Name": "past_repetition_penalty_buffer",
                "ElemSize": size.itemsize,
                "DimSpecs": [
                    {"start": "batch_index"},
                    {"start": 0},
                    ],
                }
            ]
        }
        return json.dumps(json_spec)

    def debugprint(self, *str_in):
        print(f"[{self.cluster_id}]", end="")
        print(str_in)

    def get_slicing_spec_handle(self, buffer_spec_json):
        status, slicingSpecHandle = self.program.createSlicingSpecHandle(buffer_spec_json)
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to create SlicingSpecHandle"
        return slicingSpecHandle

    @property
    def input_names(self) -> List[str]:
        return [
            binding.name
            for binding in self.bindings
            if binding.dir == aicapi.BUFFER_IO_TYPE_INPUT
        ]

    @property
    def output_names(self) -> List[str]:
        return [
            binding.name
            for binding in self.bindings
            if binding.dir == aicapi.BUFFER_IO_TYPE_OUTPUT
        ]

    def get_bindings(self, binding_names: List[str]) -> List[aicapi.IoBinding]:
        bindings: List[aicapi.IoBinding] = [
            binding
            for binding in self.bindings
            if binding.name in binding_names
        ]
        return bindings

    def get_bindings_shapes(self, binding_name: List[str]) -> Dict[str, List[List[int]]]:
        """Function returns all possible shapes of requested buffers

        Args:
            binding_name (List[str]): List of I/O buffer names

        Returns:
            Dict[str, List[List[int]]]: All possible shapes of requested buffers
        """
        result = {}
        for name in binding_name:
            try:
                result[name] = []
                idx = self.binding_index_map[name]
            except Exception as e:
                warn(
                        f"Unable to find binding: {name}"
                    )
                continue
            for allowed_shaped in self.allowed_shapes:
                result[name].append(allowed_shaped[idx][1])
        return result

    def activate(self):
        self.activate_done = True
        self.program.activate()
        for i in range (self.queue_len):
            self.execObj[i] = qaicrt.ExecObj(self.context, self.program)
            self.execObj_ids_dict[self.execObj[i].getId()] = i

    def deactivate(self):
        print("Deactivating qpc..")
        if self.activate_done:
            self.program.deactivate()
            self.activate_done = False

    def set_buffers(self, buffers: Dict[str, np.ndarray], index: int=0):
        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index: int = self.binding_index_map[buffer_name]
            buffer = np.ascontiguousarray(buffer)
            self.qbuffers[index][buffer_index] = qaicrt.QBuffer(buffer)
            self.buf_dims[index][buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def unskip_buffers(self, skipped_buffer_names: List[str], index: int=0) -> None:
        if not skipped_buffer_names:
            return
        bindings: List[aicapi.IoBinding] = self.get_bindings(
            skipped_buffer_names
        )
        buffers: Dict[str, np.ndarray] = dict()
        for binding in bindings:
            aic_dtype: int = binding.type
            np_dtype: np.dtype = aic_to_np_dtype_mapping[aic_dtype]
            dims: List[int] = binding.dims
            arr = np.zeros(dims, dtype=np_dtype)
            buffers[binding.name] = arr
        self.set_buffers(buffers, index)

    def skip_buffers(self, skipped_buffer_names: List[str], index: int = 0):
        self.set_buffers({k: np.array([]) for k in skipped_buffer_names}, index)

    def get_tuple_list_from_dict(self, dict_in):
        # Convert the buffer_dict to a list of tuples
        buffer_idx_to_buffer = []
        for buffer_name, buffer in dict_in.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index: int = self.binding_index_map[buffer_name]
            if buffer is None:
                continue
            buffer = np.ascontiguousarray(buffer)
            buffer_idx_to_buffer.append((buffer_index, buffer))
        return buffer_idx_to_buffer

    def extract_outputs(self, input_dict):
        output_dict = dict()
        for bufname in self.output_names:
            if bufname in input_dict.keys():
                output_dict[bufname] = input_dict[bufname]
        return output_dict

    def create_numpy_buffers(self, input_dict, direction, shape, size):
        bufnames = []
        if direction == "in":
            bufnames = [
                n for n in self.input_names if n.startswith("past_key") or n.startswith("past_value")
            ]
        elif direction == "out":
            bufnames = [
                n
                for n in self.output_names
                if (n.startswith("past_key") or n.startswith("past_value"))
                and n.endswith("_RetainedState")
            ]
        else:
            raise ValueError("invalid buffer direction to create_numpy_buffers")
        for bufname in bufnames:
            if len(shape) == 0:
                input_dict[bufname] = np.array([])
            else:
                input_dict[bufname] = np.zeros(shape=shape, dtype=size)

    def create_numpy_penalty_buffers(self, input_dict, direction, shape, dtype):
        bufnames = []
        if direction == "in":
            bufnames = [
                "past_repetition_penalty_buffer",
                "past_presence_penalty_buffer",
            ]
        elif direction == "out":
            bufnames = [
                "past_repetition_penalty_buffer_RetainedState",
                "past_presence_penalty_buffer_RetainedState",
            ]
        else:
            raise ValueError("invalid buffer direction to create_numpy_penalty_buffers")
        for bufname in bufnames:
            if len(shape) == 0:
                input_dict[bufname] = np.array([])
            else:
                input_dict[bufname] = np.zeros(shape=shape, dtype=dtype)

    def create_output_buffers(self, input_dict, shape, size, buffer_name="logits"):
        if buffer_name not in self.binding_index_map:
            warn(f'Buffer: "{buffer_name}" not found')
            return
        input_dict[buffer_name] = np.empty(shape=shape, dtype=size)

    def set_data_for_kv_handoff(
        self,
        kv_cache_buffers,
        slicing_parameters,
        index=0,
        buff_map: Optional[List[Tuple[str, int]]] = None,
    ):
        return self._set_data_with_slices(
            kv_cache_buffers,
            slicing_parameters,
            self.kv_slicing_spec_handle,
            index,
            buff_map,
        )

    def set_data_for_repetition_penalty(
        self, repetition_penalty_buffers, slicing_parameters, index=0
    ):
        return self._set_data_with_slices(
            repetition_penalty_buffers,
            slicing_parameters,
            self.repetition_penalty_spec_handle,
            index,
            self.repetition_penalty_map,
        )

    def _set_data_with_slices(
        self,
        buffers,
        slicing_parameters,
        slicing_spec_handle,
        index=0,
        buff_map: Optional[List[Tuple[str, int]]] = None,
    ):
        if isinstance(buffers, list):
            assert buff_map is not None
            assert len(buffers) == len(buff_map) or len(buffers) + 1 == len(
                buff_map
            ), "buffers must be a list of numpy arrays or a dictionary of numpy arrays"
            slices_as_tuple_list = [
                (name[1], buff) for name, buff in zip(buff_map, buffers)
            ]
        else:
            slices_as_tuple_list = self.get_tuple_list_from_dict(buffers)
        status, slicingHandle = self.execObj[index].setDataWithSlices(
            slices_as_tuple_list, slicing_spec_handle, slicing_parameters
        )
        assert status == qaicrt.QStatus.QS_SUCCESS, f"Failed to setDataWithSlices"
        return buffers

    def np_run(self, inputs: Dict[str,Any], slicing_parameters: List[Tuple[str, int]]=None, index=0):
        #setdata with slices for each instance of the sliceddata
        slices_as_tuple_list = self.get_tuple_list_from_dict(inputs)
        if slicing_parameters is None:
            status = self.execObj[index].setData(slices_as_tuple_list)
            assert (
            status == qaicrt.QStatus.QS_SUCCESS
                ), f"Failed to setDataWithSlices"
        else:
            status, slicingHandle = self.execObj[index].setDataWithSlices(slices_as_tuple_list, self.kv_slicing_spec_handle, slicing_parameters)
            assert (
            status == qaicrt.QStatus.QS_SUCCESS
                ), f"Failed to setDataWithSlices"
        assert (
            self.queue.enqueue(self.execObj[index]) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to enqueue"

    def run(self, inputs: Dict[str, np.ndarray], index: int = 0) -> Dict[str, np.ndarray]:
        # Set inputs
        self.set_buffers(inputs, index)
        assert (
            self.execObj[index].setData(self.qbuffers[index], self.buf_dims[index]) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to setData"
        # # Run with sync API
        # if self.execObj.run(self.qbuffers) != qaicrt.QStatus.QS_SUCCESS:
        # Run with async API
        assert (
            self.queue.enqueue(self.execObj[index]) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to enqueue"
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
            # Print additional error messages for unmatched dimension error
            if self.allowed_shapes:
                error_message += "\n\n"
                error_message += '(Only if "No matching dimension found" error is present above)'
                error_message += "\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings, allowed_shape, self.buf_dims[index]
                    ):
                        if passed_shape[0] == 0:
                            if not binding.is_partial_buf_allowed:
                                warn(
                                    f"Partial buffer not allowed for: {binding.name}"
                                )
                            continue
                        error_message += (
                            f"{binding.name}:\t{elemsize}\t{shape}\n"
                        )
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(
                    self.bindings, self.buf_dims[index]
                ):
                    if shape[0] == 0:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)
        # Get output buffers
        status, output_qbuffers = self.execObj[index].getData()
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to getData"
        # Build output
        outputs = {}
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            if self.buf_dims[index][buffer_index][1][0] == 0:
                continue
            outputs[output_name] = np.frombuffer(
                output_qbuffers[buffer_index],
                aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            ).reshape(self.buf_dims[index][buffer_index][1])
        return outputs
    
    def run_pipeline_stage(self, inputs: Dict[str,Any], slicing_parameters: List[Tuple[str, int]]=None, index=0):
        self.timers[index] = perf_counter()
        slices_as_tuple_list = self.get_tuple_list_from_dict(inputs)
        if slicing_parameters is None:
            status = self.execObj[index].setData(slices_as_tuple_list)
            assert (
                status == qaicrt.QStatus.QS_SUCCESS
            ), f"Failed to setData"
        else:
            slicing_spec_handle = self.kv_slicing_spec_handle
            status, slicingHandle = self.execObj[index].setDataWithSlices(slices_as_tuple_list, slicing_spec_handle, slicing_parameters)
            assert (
                status == qaicrt.QStatus.QS_SUCCESS
            ), f"Failed to setDataWithSlices"
        assert (
            self.queue.enqueue(self.execObj[index]) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to enqueue"

    def run_pipeline(self, inputs: Dict[str, np.ndarray] , index: int = 0, last_chunk: int = 0) -> Dict[str, np.ndarray]:
        skip = False
        #check if index is more than execobjs
        if index >= self.queue_len:
            index = index % self.queue_len
        if self.execObj_ids_dict[self.execObj[index].getId()] != -1:
            skip = True
        self.execObj_ids_dict[self.execObj[index].getId()] = index
        # Set callback for last chunk
        if last_chunk:
            self.unskip_buffers(set([x for x in self.output_names if x.endswith("_RetainedState")]), index)
            assert (
                    self.execObj[index].setCallback(self.cb_get_data) == qaicrt.QStatus.QS_SUCCESS
                ), "Failed to setCallBack"
        else:
            if skip:
                self.skip_buffers(set([x for x in self.output_names if x.endswith("_RetainedState")]), index)
            assert (
                    self.execObj[index].setCallback(self.cb_get_data) == qaicrt.QStatus.QS_SUCCESS
                ), "Failed to setCallBack"
        # Set inputs
        self.set_buffers(inputs, index)
        assert (
            self.execObj[index].setData(self.qbuffers[index], self.buf_dims[index]) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to setData"
        # # Run with sync API
        # if self.execObj.run(self.qbuffers) != qaicrt.QStatus.QS_SUCCESS:
        # Run with async API
        assert (
            self.queue.enqueue(self.execObj[index]) == qaicrt.QStatus.QS_SUCCESS
            # self.execObj.run(self.qbuffers) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to enqueue"
        self.execObj_available -= 1

    def np_run_pipeline(self, inputs: Dict[str, np.ndarray] , slicing_parameters: List[Tuple[str, int]]=None, index: int = 0, last_chunk: int = 0) -> Dict[str, np.ndarray]:
        skip = False
        #check if index is more than execobjs
        if index >= self.queue_len:
            index = index % self.queue_len
        if self.execObj_ids_dict[self.execObj[index].getId()] != -1:
            skip = True
        self.execObj_ids_dict[self.execObj[index].getId()] = index
        slices_as_tuple_list = self.get_tuple_list_from_dict(inputs)
 
        if slicing_parameters is None:
            status = self.execObj[index].setData(slices_as_tuple_list)
            assert (
                status == qaicrt.QStatus.QS_SUCCESS
            ), f"Failed to setData"
        else:
            slicing_spec_handle = self.kv_slicing_spec_handle
            status, slicingHandle = self.execObj[index].setDataWithSlices(slices_as_tuple_list, slicing_spec_handle, slicing_parameters)
            assert (
                status == qaicrt.QStatus.QS_SUCCESS
            ), f"Failed to setDataWithSlices"
        # Run with async API
        assert (
            self.queue.enqueue(self.execObj[index]) == qaicrt.QStatus.QS_SUCCESS
            # self.execObj.run(self.qbuffers) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to enqueue"
        self.execObj_available -= 1

    def get_output(self , out_buffers, index : int = 0):
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
            # Print additional error messages for unmatched dimension error
            if self.allowed_shapes:
                error_message += "\n\n"
                error_message += '(Only if "No matching dimension found" error is present above)'
                error_message += "\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings, allowed_shape, self.buf_dims[index]
                    ):
                        if passed_shape[0] == 0:
                            if not binding.is_partial_buf_allowed:
                                warn(
                                    f"Partial buffer not allowed for: {binding.name}"
                                )
                            continue
                        error_message += (
                            f"{binding.name}:\t{elemsize}\t{shape}\n"
                        )
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(
                    self.bindings, self.buf_dims[index]
                ):
                    if shape[0] == 0:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)
        # Build output
        # outputs = dict()
        # outputs["logits"] = self.logits["logits"]
        # for output_name in self.output_names:
        #     if out_buffers[output_name].size == 0:
        #         continue
        #     outputs[output_name] = out_buffers[output_name]
        self.execObj_ids_dict[self.execObj[index].getId()] = -1
        self.execObj_available += 1
        return out_buffers
    
    def complete_inf(self, index: int = 0):
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
            # Print additional error messages for unmatched dimension error
            if self.allowed_shapes:
                error_message += "\n\n"
                error_message += '(Only if "No matching dimension found" error is present above)'
                error_message += "\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings, allowed_shape, self.buf_dims[index]
                    ):
                        if passed_shape[0] == 0:
                            if not binding.is_partial_buf_allowed:
                                warn(
                                    f"Partial buffer not allowed for: {binding.name}"
                                )
                            continue
                        error_message += (
                            f"{binding.name}:\t{elemsize}\t{shape}\n"
                        )
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(
                    self.bindings, self.buf_dims[index]
                ):
                    if shape[0] == 0:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)
        self.execObj_ids_dict[self.execObj[index].getId()] = -1
        self.execObj_available += 1