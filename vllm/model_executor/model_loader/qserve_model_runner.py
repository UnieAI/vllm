# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import numpy as np
import math
import time
from typing import Optional, List, Dict, Union, Set, Tuple, Callable
from QEfficient.generation.cloud_infer import QAICInferenceSession
from vllm.model_executor.model_loader.qaic_session_np import DisAgg_QAICInferenceSession, aic_to_np_dtype_mapping
from vllm.logger import init_logger
import atexit

from multiprocessing import Queue

logger = init_logger(__name__)

class QServeModelRunner:
    """model runner tasked with generating tokens from a given input"""

    def __init__(
        self,
        qpc_path: str,
        vocab_size: int,
        device_id: list,
        seq_len: Optional[int] = None,
        ctx_len: Optional[int] = None,
        decode_bsz: Optional[int] = None,
        prefill_bsz: int = 1,
        num_logits_to_keep: Optional[int] = None,
        speculative_model_type: Optional[str] = None,
        lora_mode: bool = False,
        include_sampler: Optional[bool] = None,
        include_guided_decoding: Optional[bool] = None,
        return_pdfs: Optional[bool] = None,
        max_top_k_ids: Optional[int] = None,
        is_multimodal_model: Optional[bool] = False,
        stages: Optional[int] = 1,
        queue: Optional[Queue] = None,
        kv_transfer_role: Optional[str] = None,
        image_token_index: Optional[int] = None,
    ) -> None:
        """initialize ModelRunner

        Args:
            qpc_path (str): path to qpc
            vocab_size (int): vocab size
            device_id (str, int): path to mdp json file or int QID
            seq_len (int): prompt length
            ctx_len (int): context length
            decode_bsz (int): decode batch size
            prefill_bsz (int): prefill batch size
        """
        self.qpc_path: str = qpc_path
        self.vocab_size: int = vocab_size
        self.device_id: list = device_id
        self.seq_len: Optional[int] = seq_len
        self.ctx_len: Optional[int] = ctx_len
        self.decode_bsz: Optional[int] = decode_bsz
        self.full_batch_size: Optional[int] = decode_bsz
        self.prefill_bsz: Optional[int] = prefill_bsz

        self.disagg_serving_en = kv_transfer_role != None
        self.disagg_producer_en = kv_transfer_role == "kv_producer"

        print("Stages: ", stages, flush=True)
        # atexit.register(self.cleanup)
        logger.info(f"Loading QPC...")
        logger.info(f"This may take some time, please don't press CTRL-C during this phase...")
        s = time.perf_counter()
        if not self.disagg_serving_en:
            if not include_sampler:
                self.session = QAICInferenceSession(qpc_path, device_ids=device_id)
            else:
                self.session = DisAgg_QAICInferenceSession(qpc_path, device_ids=device_id)
            self.session.skip_buffers(
            set([x for x in self.session.input_names if x.startswith("past_")])
                )
            self.session.skip_buffers(
            set([x for x in self.session.output_names if x.endswith("_RetainedState")])
                )
        else:
            self.session = DisAgg_QAICInferenceSession(qpc_path, device_ids=device_id, stages=stages, cluster_id="Prefill" if self.disagg_producer_en else "decode")
            for y in range (stages+1):
                self.session.skip_buffers(
                    set([x for x in self.session.input_names if x.startswith("past_")]), y
                    )
                self.session.skip_buffers(
                    set([x for x in self.session.output_names if x.endswith("_RetainedState")]), y
                    )
        e = time.perf_counter() - s
        logger.info(f"Successfully loaded QPC in {e} secs")
        logger.info(qpc_path)
        # self.validate_input_arguments()
        self.attention_mask = None
        self.prefill_batch_inputs = None
        self.decode_batch_inputs = None
        self.decode_single_inputs = None
        self.num_logits_to_keep = num_logits_to_keep
        self.speculative_model_type = speculative_model_type
        self.is_spec_decode_target_model = False
        self.last_decode = True
        self.prefill_num_logits_buffer = None
        self.decode_num_logits_buffer = None
        self.encode_num_logits_buffer = None
        # On Device Sampling
        self.include_sampler: Optional[bool] = include_sampler
        self.return_pdfs: Optional[bool] = return_pdfs
        self.max_top_k_ids: Optional[int] = max_top_k_ids
        self.include_guided_decoding: Optional[bool] = include_guided_decoding
        self.prefill_probs = None
        self.decode_probs = None
        self.stages = stages
        self.Q = queue
        self.past_kv_update = []
        # Validate sampler inputs
        self.sampler_inputs = []
        sampler_inputs = [
            "last_accepted_output_tokens", "repetition_penalties", "presence_penalties",
            "temperatures", "top_ks", "top_ps", "min_ps", "random_numbers"
        ]
        if self.include_guided_decoding:
            sampler_inputs.append("token_bitmasks")
        count = 0
        for session_input_name in self.session.input_names:
            if session_input_name in sampler_inputs:
                count += 1
                if count == len(sampler_inputs):
                    break
        if count == 0:
            self.include_sampler = False
        elif count < len(sampler_inputs):
            raise ValueError(
                "The provided QPC does not have the required number of inputs to run sampling "
                f"on the QAIC device (only {count}/{len(self.sampler_inputs)} inputs provided). Partial "
                "sampling support is not available. Please check the QPC and try again."
            )
        else:  # count == len(sampler_inputs)
            self.include_sampler = True
            self.sampler_inputs = sampler_inputs
        if include_sampler and not self.include_sampler:
            logger.warning_once(
                "User entered `include_sampler`=True. But the provided QPC is not compiled "
                "to run sampling on the QAIC device. Falling back to the PyTorch backend."
            )
        elif (include_sampler is None or not include_sampler) and self.include_sampler:
            raise ValueError(
                "The provided QPC is compiled to run sampling on the QAIC device. "
                "But the user did not enter `include_sampler`=True. Please make sure the input "
                "is specified correctly."
            )

        if self.num_logits_to_keep is not None:
            self.is_spec_decode_target_model = True
            prefill_logit_dims = [self.prefill_bsz, 1, self.vocab_size]
            decode_logit_dims = [self.decode_bsz, self.num_logits_to_keep, self.vocab_size]
            if self.speculative_model_type == "turbo":
                prefill_logit_dims.insert(2, self.num_logits_to_keep)
                decode_logit_dims.insert(2, self.num_logits_to_keep)
            self.prefill_logits = dict(logits=np.random.randn(*prefill_logit_dims).astype(np.float32))
            self.decode_logits = dict(logits=np.random.randn(*decode_logit_dims).astype(np.float32))
            self.prefill_num_logits_buffer = dict(num_logits_to_keep = np.zeros((1, 1), np.int64))
            self.decode_num_logits_buffer = dict(num_logits_to_keep=np.zeros((self.num_logits_to_keep, 1), np.int64))
        elif self.include_sampler:
            if self.return_pdfs:
                self.prefill_probs = dict(
                    probs=np.random.randn(self.prefill_bsz, 1, self.vocab_size).astype(np.float32)
                )
                self.decode_probs = dict(
                    probs=np.random.randn(self.decode_bsz, 1, self.vocab_size).astype(np.float32)
                )
            self.prefill_logits = dict(next_tokens=np.zeros((self.prefill_bsz, 1, 1), np.int64))
            self.decode_logits = dict(next_tokens=np.zeros((self.decode_bsz, 1, 1), np.int64))
        else:
            self.prefill_logits = dict(logits=np.random.randn(self.prefill_bsz, 1, self.vocab_size).astype(np.float32))
            self.decode_logits = dict(logits=np.random.randn(self.decode_bsz, 1, self.vocab_size).astype(np.float32))
        if "batch_index" in self.session.input_names:
            self.ignore_batch_index = False
        else:
            self.ignore_batch_index = True
        self.is_multimodal_model = is_multimodal_model
        if self.is_multimodal_model:
            if "vision_embeds" in self.session.input_names:
                self.embeds_dims, _, _ = self.get_input_shape_and_dtype("vision_embeds")
            self.need_image_idx = "image_idx" in self.session.input_names
            mm_input_names = ["pixel_values", "input_features", "vision_embeds", "cross_attention_mask"]
            self.mm_input_info = {
                input_name: self.get_input_shape_and_dtype(input_name)[0]
                for input_name in mm_input_names
                if input_name in self.session.input_names
            }

        self.image_token_index = image_token_index

        self.list_of_comp_ctx_lengths = None
        try:
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = self.get_comp_ctx_lengths()
        except Exception as e:
            logger.debug(f"Compute context lengths not configured or failed to initialize: {e}")
            self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = None, None

        logger.info(f"Warming-up with Dummy run...")
        s = time.perf_counter()
        try:
            if not self.disagg_serving_en:
                self.dummy_run(lora_mode)
            else:
                self.disagg_dummy_run(lora_mode)
        except ValueError:
            logger.info("Re-runing with different logit dimension...")
            self.prefill_logits = dict(logits=np.random.randn(self.prefill_bsz, self.vocab_size).astype(np.float32))
            self.decode_logits = dict(logits=np.random.randn(self.decode_bsz, self.vocab_size).astype(np.float32))
            if not self.disagg_serving_en:
                self.dummy_run(lora_mode)
            else:
                self.disagg_dummy_run(lora_mode)
        e = time.perf_counter() - s
        logger.info(f"Successfully finished Dummy run in {e} secs")

    def get_comp_ctx_lengths(self):

        comp_ctx_lengths_prefill, comp_ctx_lengths_decode = [], []
        if "comp_ctx_lengths" not in self.session.binding_index_map:
            return None, None
        input_idx = self.session.binding_index_map["input_ids"]
        ccl_idx = self.session.binding_index_map["comp_ctx_lengths"]

        for i in range(len(self.session.allowed_shapes)):
            if self.session.allowed_shapes[i][input_idx][1][1]==self.seq_len:
                comp_ctx_lengths_prefill.append(self.session.allowed_shapes[i][ccl_idx][1][0])
            elif self.session.allowed_shapes[i][input_idx][1][1]==1 or self.num_logits_to_keep:
                comp_ctx_lengths_decode.append(self.session.allowed_shapes[i][ccl_idx][1][0])
            else:
                raise ValueError("QPC not compiled for required seq_len")

        comp_ctx_lengths_prefill.sort()
        comp_ctx_lengths_decode.sort()
        if comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
            self.list_of_comp_ctx_lengths = {comp_ctx_len:np.empty(comp_ctx_len, dtype=np.int8) for comp_ctx_len in comp_ctx_lengths_prefill+comp_ctx_lengths_decode}
        return comp_ctx_lengths_prefill, comp_ctx_lengths_decode

    def cleanup(self):
        if hasattr(self,'session'):
            self.session.deactivate()
            del self.session

    def get_qpc_IO_dims(self):

        bindings = self.session.get_bindings_shapes(["input_ids",'past_key.0'])
        decode_bsz, seq_len = np.max(bindings['input_ids'], axis=0)
        prefill_bsz, decode_id_sz = np.min(bindings['input_ids'], axis=0)

        # Only prefill batch size 1 is supported
        if prefill_bsz!=1 or decode_id_sz!=1:
            raise ValueError(
                    self.device_id,
                    message=f"QPC not compiled for either decode or has prefill bsz>1!!",
                )
        ctx_len = bindings['past_key.0'][0][2]

        return {
            "prefill_bsz" : prefill_bsz,
            "seq_len" : seq_len,
            "decode_bsz" : decode_bsz,
            "ctx_len" : ctx_len,
        }

    def validate_input_arguments(self, class_object=None):
        if class_object is None:
            class_object = self

        arg_dims: dict = self.get_qpc_IO_dims()
        for arg, dim in arg_dims.items():
            instance_val: Optional[int] = getattr(class_object, arg)
            if instance_val is None:
                setattr(class_object, arg, dim)
            elif instance_val != dim:
                raise ValueError(
                    self.device_id,
                    message=f"arg {arg}={instance_val} does not match corresponding qpc value of {dim}",
                )

    def get_input_shape_and_dtype(self, input_name: str) -> Tuple[Tuple, np.dtype, int]:
        if input_name not in self.session.input_names:
            return None
        binding = self.session.bindings[self.session.binding_index_map[input_name]]
        return list(binding.dims), aic_to_np_dtype_mapping[binding.type], binding.size

    def dummy_run(self, lora_mode = False):
        """assert prefill and decode work by running dummy inputs

        also creates attention_mask and decode input buffers
        that will be used throughout the life of qserve
        """

        # prepare dummy run inputs
        prefill_inputs = {
            "input_ids":np.zeros((self.prefill_bsz, self.seq_len), dtype=np.int64),
            "position_ids":np.tile(np.full((self.seq_len), -1, dtype=np.int64).reshape(1,self.seq_len), (self.prefill_bsz, 1)),
            "batch_index":np.arange(self.prefill_bsz).reshape(-1, 1),
        }
        if self.list_of_comp_ctx_lengths is not None:
            prefill_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len)

        if lora_mode:
            prefill_inputs["lora_ids"] = np.arange(self.prefill_bsz).reshape(-1, 1)

        if self.include_sampler:
            prefill_inputs.update({
                "last_accepted_output_tokens": np.zeros(
                    (self.prefill_bsz, self.seq_len), dtype=np.int64,
                ),
                "repetition_penalties": np.ones((self.prefill_bsz, 1), dtype=np.float32),
                "presence_penalties": np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                # frequency_penalties: np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                "temperatures": np.ones((self.prefill_bsz, 1), dtype=np.float32),
                "top_ks": np.full((self.prefill_bsz, 1), self.vocab_size, dtype=np.int32),
                "top_ps": np.ones((self.prefill_bsz, 1), dtype=np.float32),
                "min_ps": np.ones((self.prefill_bsz, 1), dtype=np.float32),
                "random_numbers": np.zeros((self.prefill_bsz, self.max_top_k_ids), dtype=np.float32),
                "prefill_repetition_penalty_buffer": np.full((self.prefill_bsz, self.vocab_size), False, dtype=bool),
            })
            prefill_inputs.update(self.prefill_logits)
            if self.include_guided_decoding:
                prefill_inputs["token_bitmasks"] = np.full((self.prefill_bsz, self.vocab_size), True, dtype=bool)

        prefill_qpc_inputs = {0: prefill_inputs}

        # decode inputs
        decode_single_inputs = {
            "input_ids":np.array([[0]]),
            "position_ids":np.array([[0]]),
        }
        decode_batch_inputs = {
            "input_ids":np.zeros((self.decode_bsz, 1), dtype=np.int64),
            "position_ids":np.full((self.decode_bsz,1), -1, dtype=np.int64),
        }
        if self.list_of_comp_ctx_lengths is not None:
            decode_single_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len, dtype=np.int8)
            decode_batch_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len, dtype=np.int8)
        if self.is_spec_decode_target_model:
            # decode on this model has multiple tokens per batch (aka precode)
            decode_single_inputs = dict(
                input_ids=np.zeros((1,self.num_logits_to_keep), dtype=np.int64),
                position_ids=np.full((1, self.num_logits_to_keep), -1, dtype=np.int64),
            )
            decode_batch_inputs = dict(
                input_ids=np.zeros((self.decode_bsz, self.num_logits_to_keep), dtype=np.int64),
                position_ids = np.full((self.decode_bsz, self.num_logits_to_keep), -1, dtype=np.int64),
            )
            if self.list_of_comp_ctx_lengths is not None:
                decode_single_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len, dtype=np.int8)
                decode_batch_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len, dtype=np.int8)
        if input_info := self.get_input_shape_and_dtype("position_ids"):
            position_ids_shape = input_info[0]
            if len(position_ids_shape) == 3: 
                # For models that use mrope input positions
                # position_ids_shape[0] is the additional dimensional axis for mrope
                # position_ids is in the shape of (position_ids_shape[0], bsz, seq_len)
                prefill_inputs["position_ids"] = np.tile(
                    prefill_inputs["position_ids"], (position_ids_shape[0], 1, 1)
                )
                decode_single_inputs["position_ids"] = np.tile(
                    decode_single_inputs["position_ids"], (position_ids_shape[0], 1)
                )
                decode_batch_inputs["position_ids"] = np.tile(
                    decode_batch_inputs["position_ids"], (position_ids_shape[0], 1, 1)
                )
        if not self.ignore_batch_index:
            decode_single_inputs["batch_index"] = np.array([[0]])
            decode_batch_inputs["batch_index"] = np.arange(self.decode_bsz, dtype=np.int64).reshape(-1, 1)
        if input_info := self.get_input_shape_and_dtype("cross_attention_mask"):
            self.is_cross_attention = True
            (dims, dtype, _) = input_info
            self.prefill_cross_attention_mask = np.zeros((dims[0], dims[1], dims[2], dims[3]), dtype=dtype)
            decode_single_inputs["cross_attention_mask"] = np.ones((1, dims[2], dims[3]), dtype=dtype)
            decode_batch_inputs["cross_attention_mask"] = np.ones((dims[0], 1, dims[2], dims[3]), dtype=dtype)
        else:
            self.is_cross_attention = False
        if lora_mode:
            decode_single_inputs["lora_ids"] = np.array([[0]])
            decode_batch_inputs["lora_ids"] = np.arange(self.decode_bsz, dtype=np.int64).reshape(-1, 1)

        if self.include_sampler:
            decode_single_inputs.update({
                "last_accepted_output_tokens": np.array([[0]]),
                "repetition_penalties": np.array([[1.0]], dtype=np.float32),
                "presence_penalties": np.array([[0.0]], dtype=np.float32),
                # frequency_penalties: np.array([[0.0]], dtype=np.float32),
                "temperatures": np.array([[1.0]], dtype=np.float32),
                "top_ks": np.array([[self.vocab_size]], dtype=np.int32),
                "top_ps": np.array([[1.0]], dtype=np.float32),
                "min_ps": np.array([[1.0]], dtype=np.float32),
                "random_numbers": np.zeros((1, self.max_top_k_ids), dtype=np.float32),
            })
            decode_batch_inputs.update({
                "last_accepted_output_tokens": np.zeros((self.decode_bsz, 1), dtype=np.int64),
                "repetition_penalties": np.ones((self.decode_bsz, 1), dtype=np.float32),
                "presence_penalties": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                # frequency_penalties: np.zeros((self.decode_bsz, 1), dtype=np.float32),
                "temperatures": np.ones((self.decode_bsz, 1), dtype=np.float32),
                "top_ks": np.full((self.decode_bsz, 1), self.vocab_size, dtype=np.int32),
                "top_ps": np.ones((self.decode_bsz, 1), dtype=np.float32),
                "min_ps": np.ones((self.decode_bsz, 1), dtype=np.float32),
                "random_numbers": np.zeros((self.decode_bsz, self.max_top_k_ids), dtype=np.float32),
            })
            decode_batch_inputs.update(self.decode_logits)
            if self.include_guided_decoding:
                decode_single_inputs["token_bitmasks"] = np.full((1, self.vocab_size), True, dtype=bool)
                decode_batch_inputs["token_bitmasks"] = np.full((self.decode_bsz, self.vocab_size), True, dtype=bool)
            if self.return_pdfs:
                decode_batch_inputs.update(self.decode_probs)

        if self.is_multimodal_model:
            decode_single_inputs["image_idx"] = np.array([[0]])
            decode_single_inputs["image_idx_output"] = np.array([[0]])
            decode_batch_inputs["image_idx"] = np.array([[0]])
            decode_batch_inputs["image_idx_output"] = np.array([[0]])

        decode_qpc_inputs = {0: decode_single_inputs}
        self.decode_single_inputs = decode_single_inputs
        self.decode_batch_inputs = decode_batch_inputs
        self.prefill_batch_inputs = prefill_inputs.copy()

        # run dummy inputs
        if (
            "logits" in self.session.output_names or
            (self.include_sampler and "next_tokens" in self.session.output_names)
        ):
            logger.debug("starting dummy run...")
            _: Dict[int, np.ndarray] = self.run(prefill_qpc_inputs, is_prompt=True)  # prefill
            _: Dict[int, np.ndarray] = self.run(decode_qpc_inputs, is_prompt=False)  # decode
            logger.debug("finished dummy run")

    def disagg_dummy_run(self, lora_mode = False):
        """assert prefill and decode work by running dummy inputs

        also creates attention_mask and decode input buffers
        that will be used throughout the life of qserve
        """

        # Prepare dummy run inputs
        if self.session.cluster_id == "Prefill":
            prefill_inputs = {
                "input_ids":np.zeros((self.prefill_bsz, self.seq_len), dtype=np.int64),
                "position_ids":np.tile(np.full((self.seq_len), -1, dtype=np.int64).reshape(1,self.seq_len), (self.prefill_bsz, 1)),
                "batch_index":np.arange(self.prefill_bsz).reshape(-1, 1),
            }
            if input_info := self.get_input_shape_and_dtype("cross_attention_mask"):
                self.is_cross_attention = True
                (dims, dtype, _) = input_info
                self.prefill_cross_attention_mask = np.zeros((dims[0], dims[1], dims[2], dims[3]), dtype=dtype)
            else:
                self.is_cross_attention = False

            if lora_mode:
                prefill_inputs["lora_ids"] = np.arange(self.prefill_bsz).reshape(-1, 1)

            self.prefill_batch_inputs = prefill_inputs.copy()

        # Prepare decode inputs
        if self.session.cluster_id == "decode":
            # decode inputs
            decode_single_inputs = {
                "input_ids":np.array([[0]]),
                "position_ids":np.array([[0]]),
            }
            decode_batch_inputs = {
                "input_ids":np.zeros((self.decode_bsz, 1), dtype=np.int64),
                "position_ids":np.full((self.decode_bsz,1), -1, dtype=np.int64),
            }
            if self.is_spec_decode_target_model:
                # decode on this model has multiple tokens per batch (aka precode)
                decode_single_inputs = dict(
                    input_ids=np.zeros((1,self.num_logits_to_keep), dtype=np.int64),
                    position_ids=np.full((1, self.num_logits_to_keep), -1, dtype=np.int64),
                )
                decode_batch_inputs = dict(
                    input_ids=np.zeros((self.decode_bsz, self.num_logits_to_keep), dtype=np.int64),
                    position_ids = np.full((self.decode_bsz, self.num_logits_to_keep), -1, dtype=np.int64),
                )
            if "batch_index" in self.session.input_names:
                decode_single_inputs["batch_index"] = np.array([[0]])
                decode_batch_inputs["batch_index"] = np.arange(self.decode_bsz, dtype=np.int64).reshape(-1, 1)
                self.ignore_batch_index = False
            else:
                self.ignore_batch_index = True

            if lora_mode:
                decode_single_inputs["lora_ids"] = np.array([[0]])
                decode_batch_inputs["lora_ids"] = np.arange(self.decode_bsz, dtype=np.int64).reshape(-1, 1)

            if self.list_of_comp_ctx_lengths is not None:
                decode_single_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len, dtype=np.int8)
                decode_batch_inputs["comp_ctx_lengths"] = np.zeros(self.ctx_len, dtype=np.int8)

            decode_batch_inputs_temp = decode_batch_inputs.copy()

            if input_info := self.get_input_shape_and_dtype("cross_attention_mask"):
                self.is_cross_attention = True
                (dims, dtype, _) = input_info
                self.prefill_cross_attention_mask = np.zeros((dims[0], dims[1], dims[2], dims[3]), dtype=dtype)
                decode_single_inputs["cross_attention_mask"] = np.ones((1, dims[2], dims[3]), dtype=dtype)
                decode_batch_inputs["cross_attention_mask"] = np.ones((dims[0], 1, dims[2], dims[3]), dtype=dtype)
            else:
                self.is_cross_attention = False

            if self.include_sampler:
                decode_single_inputs.update({
                    "last_accepted_output_tokens": np.array([[0]]),
                    "repetition_penalties": np.array([[1.0]], dtype=np.float32),
                    "presence_penalties": np.array([[0.0]], dtype=np.float32),
                    # frequency_penalties: np.array([[0.0]], dtype=np.float32),
                    "temperatures": np.array([[1.0]], dtype=np.float32),
                    "top_ks": np.array([[self.vocab_size]], dtype=np.int32),
                    "top_ps": np.array([[1.0]], dtype=np.float32),
                    "min_ps": np.array([[1.0]], dtype=np.float32),
                    "random_numbers": np.zeros((1, self.max_top_k_ids), dtype=np.float32),
                })
                decode_batch_inputs.update({
                    "last_accepted_output_tokens": np.zeros((self.decode_bsz, 1), dtype=np.int64),
                    "repetition_penalties": np.ones((self.decode_bsz, 1), dtype=np.float32),
                    "presence_penalties": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                    # frequency_penalties: np.zeros((self.decode_bsz, 1), dtype=np.float32),
                    "temperatures": np.ones((self.decode_bsz, 1), dtype=np.float32),
                    "top_ks": np.full((self.decode_bsz, 1), self.vocab_size, dtype=np.int32),
                    "top_ps": np.ones((self.decode_bsz, 1), dtype=np.float32),
                    "min_ps": np.ones((self.decode_bsz, 1), dtype=np.float32),
                    "random_numbers": np.zeros((self.decode_bsz, self.max_top_k_ids), dtype=np.float32),
                })

            decode_qpc_inputs = {self.decode_bsz: decode_single_inputs}
            self.decode_single_inputs = decode_single_inputs
            self.decode_batch_inputs = decode_batch_inputs

        self._input_map_chg_needed = "decoder_input_ids" in self.session.input_names
        # This is a hack for mapping names to qpc input, will be changed in future releases
        embeds_name = [
            input_name
            for input_name in self.session.input_names
            if "embeds" in input_name or "image_features" in input_name
        ]
        if embeds_name:
            self.embeds_name = embeds_name[0]

        # run dummy inputs
        if (
            "logits" in self.session.output_names or
            (self.include_sampler and "next_tokens" in self.session.output_names)
        ):
            logger.debug("starting dummy run...")
            if self.session.cluster_id == "Prefill":
                logger.info(f"Running dummy prefill run for {self.stages} stages")
                bidx = 0
                while (bidx < self.stages+1):
                    prefill_inputs = {
                    "input_ids":np.zeros((self.prefill_bsz, self.seq_len), dtype=np.int64),
                    "position_ids":np.tile(np.full((self.seq_len), -1, dtype=np.int64).reshape(1,self.seq_len), (self.prefill_bsz, 1)),
                    "batch_index":np.arange(self.prefill_bsz).reshape(-1, 1),
                    "logits":np.empty(shape=(self.prefill_bsz, 1, self.vocab_size), dtype=np.float32)
                    }
                    if lora_mode:
                        prefill_inputs["lora_ids"] = np.arange(self.prefill_bsz).reshape(-1, 1)
                    s = time.perf_counter()
                    KvCache_buff = []
                    for _ in self.session.prefill_buff_map[:-1]:
                        KvCache_buff.append(np.empty(shape = self.session.kv_shape, dtype = self.session.kv_size))
                    _ = self.session.set_data_for_kv_handoff(KvCache_buff,
                                                             #[("batch_index", bidx), ("ctx_start",0)],
                                                [("batch_index", bidx%self.full_batch_size), ("ctx_start",0)],
                                                             bidx, self.session.prefill_buff_map)
                    self.session.np_run_pipeline(inputs=prefill_inputs,
                                                 slicing_parameters=None, index=bidx, last_chunk=True)
                    outputs = self.session.complete_inf(bidx)
                    bidx += 1
                logger.info("Finished dummy prefill run")
            if self.session.cluster_id == "decode":
                logger.info(f"Running dummy decode run with bsz {self.decode_bsz}")
                bidx = 0
                input_kv_buffers = {}
                shape = self.session.kv_shape
                shape[0] = self.decode_bsz
                self.session.create_numpy_buffers(
                    input_dict=input_kv_buffers,
                    direction="in",
                    shape=shape,
                    size=self.session.kv_size,
                )
                if self.include_sampler:
                    self.session.create_numpy_penalty_buffers(
                        input_dict=input_kv_buffers,
                        direction="in",
                        shape=(self.decode_bsz, self.vocab_size),
                        dtype=bool,
                    )
                    self.session.create_output_buffers(input_kv_buffers, (self.decode_bsz, 1, 1), np.int64, "next_tokens")
                else:
                    self.session.create_output_buffers(input_kv_buffers, self.decode_logits["logits"].shape, np.float32)
                # Update kv cache setDataWith
                _ = self.session.set_data_for_kv_handoff(input_kv_buffers,
                                                            [("batch_index", bidx), ("ctx_start", 0)],
                                                            0, self.session.decode_buff_map)
                bidx += 1
                self.session.np_run(self.decode_batch_inputs)
                self.session.complete_inf()
                # self.decode_batch_inputs = decode_batch_inputs_temp
                logger.info(f"Finished dummy decode run with bsz {self.decode_bsz}")
            logger.debug("finished dummy run")

    def run(self, qpc_inputs: Dict[int, dict], is_prompt: bool, callback: Optional[Callable]=None) -> np.ndarray:
        """run qpc_inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to process
            is_prompt (bool): whether this is a prefill or decode run

        Returns:
            np.ndarray: fixed slot generated tokens
        """
        if is_prompt:
            if self.disagg_producer_en:
                return self._run_pipeline_prefill(qpc_inputs, callback)
            logits_list = []

            for bidx,inputs in qpc_inputs.items():
                if self.disagg_serving_en:
                    assert inputs['ignore_prefill'] == True
                    logits_list.append(inputs['logits'].squeeze(1))
                    # Update kv cache setDataWith
                    _ = self.session.set_data_for_kv_handoff(inputs['kv_cache'],
                                                                 [("batch_index", bidx), ("ctx_start", 0)],
                                                                 0, self.session.decode_buff_map)    
                else:
                    if self.include_sampler and "next_tokens" not in inputs:
                        inputs["next_tokens"] = np.empty(
                                shape=self.prefill_logits["next_tokens"].shape, dtype=np.int64
                            )
                    logits = self._run_prefill({bidx : inputs})
                    logits_list.append(logits)
            return np.concatenate(logits_list)
        else:
            return self._run_decode(qpc_inputs, callback)

    def _run_prefill(
        self,
        qpc_inputs: Dict[int, dict],
    ) -> np.ndarray:
        """run qpc prefill inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to process

        Returns:
            np.ndarray: fixed slot generated tokens
        """

        # set qpc prefill state
        if not self.include_sampler and self.last_decode:
            self.session.set_buffers(self.prefill_logits)
            if self.is_spec_decode_target_model:
                self.session.set_buffers(self.prefill_num_logits_buffer)
            if self.include_sampler and self.return_pdfs:
                self.session.set_buffers(self.prefill_probs)
            self.last_decode = False
        # perform prefill (only prefill_bsz=1 is supported)
        bidx, inputs = next(iter(qpc_inputs.items()))
        n_prompt_tokens = inputs["input_ids"].shape[-1]  # n_prompt_tokens >= self.seq_len
        n_chunks: int = math.ceil(n_prompt_tokens / self.seq_len)
        assert n_chunks > 0

        if self.ignore_batch_index:
            del inputs["batch_index"]
        image_idx = None
        if self.is_multimodal_model:
            if not self.include_sampler:
                mm_data = dict()
                for k in self.mm_input_info:
                    if k in inputs:
                        mm_data[k] = inputs.pop(k)
                if self.is_cross_attention and "cross_attention_mask" not in mm_data:
                    # need to redo it after decode to support text-only inputs for mllama
                    mm_data["cross_attention_mask"] = self.prefill_cross_attention_mask
                self.session.set_buffers(mm_data)
            else:
                if self.is_cross_attention and "cross_attention_mask" not in inputs:
                    # need to redo it after decode to support text-only inputs for mllama
                    inputs["cross_attention_mask"] = self.prefill_cross_attention_mask
            if self.need_image_idx:
                image_idx = np.array([[0]])

        need_chunking = {"input_ids", "position_ids"}

        prefill_ccl_id = 0

        past_repetition_penalty_buffer = inputs.pop(
            "past_repetition_penalty_buffer", None
        )
        if past_repetition_penalty_buffer is not None:
            _ = self.session.set_data_for_repetition_penalty(
                {"past_repetition_penalty_buffer": past_repetition_penalty_buffer},
                [("batch_index", bidx)],
                0,
            )
        for chunk in range(n_chunks):
            if chunk+1 == n_chunks:
                lower_idx = -self.seq_len
                upper_idx = n_prompt_tokens
                if image_idx is not None:
                    if self.image_token_index is not None:
                        image_token_id = self.image_token_index
                        overlapped = inputs["input_ids"][..., n_prompt_tokens - self.seq_len:int(chunk * self.seq_len)]
                        overlapped_image_token_count = (overlapped == image_token_id).sum()
                        image_idx = np.array([image_idx[0] - overlapped_image_token_count])
                    else:
                        # This is the fallback case for new models whose image_token_index cannot be read from the HF config.
                        # Con: Prefix caching will not be supported in this case.
                        image_idx = np.array([image_idx[0] - int((chunk + 1) * self.seq_len) + n_prompt_tokens])
            else:
                lower_idx = int(chunk * self.seq_len)
                upper_idx = int((chunk + 1) * self.seq_len)
            chunk_inputs = {}
            for input_name, value in inputs.items():
                if input_name in need_chunking:
                    chunk_inputs[input_name] = inputs[input_name][..., lower_idx:upper_idx]
                else:
                    chunk_inputs[input_name] = value
            if image_idx is not None:
                chunk_inputs["image_idx"] = image_idx
                chunk_inputs["image_idx_output"] = image_idx

            if self.comp_ctx_lengths_prefill is not None:
                prefill_ccl = self.comp_ctx_lengths_prefill[0]
                for i in range(prefill_ccl_id, len(self.comp_ctx_lengths_prefill)):
                    if chunk_inputs['position_ids'].max() < self.comp_ctx_lengths_prefill[i]:
                        prefill_ccl_id, prefill_ccl = i, self.comp_ctx_lengths_prefill[i]
                        break
                chunk_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[prefill_ccl]

            if self.include_sampler:
                chunk_inputs["last_accepted_output_tokens"] = chunk_inputs["input_ids"]
                if self.return_pdfs:
                    chunk_inputs.update(self.prefill_probs)
                self.session.np_run(chunk_inputs)
                self.session.complete_inf()
                if "image_idx_output" in chunk_inputs:
                    image_idx = chunk_inputs["image_idx_output"]
                outputs = chunk_inputs
            else:
                outputs: dict = self.session.run(chunk_inputs)

            if "image_idx_output" in outputs:
                image_idx = outputs["image_idx_output"]

        if image_idx and image_idx > 0: # skip multimodal data buffers after prefill
            if self.is_cross_attention:
                self.session.skip_buffers([x for x in inputs.keys() if x.startswith("past_")])
            else:
                self.session.skip_buffers(self.mm_input_info.keys())

        if self.include_sampler:
            if self.return_pdfs:
                probs = outputs["probs"]
            next_tokens = outputs["next_tokens"]
            return next_tokens

        logits = outputs["logits"]
        return logits[:,-1]

    def _run_pipeline_prefill(
        self,
        qpc_inputs: Dict[int, dict],
        callback:Optional[Callable]=None
    ) -> np.ndarray:
        """run qpc prefill inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to process

        Returns:
            np.ndarray: fixed slot generated tokens
        """
        # set qpc prefill state
        if self.last_decode:
            self.last_decode = False
        chunk_id = 0
        counter = 0
        chunk_out = 0
        last_chunk_ids = {key: None for key in range(self.stages+1)}
        logits = np.empty((len(qpc_inputs.keys()), 1, self.vocab_size), dtype=np.float32)
        i = 0
        for bidx, inputs in qpc_inputs.items():
            n_prompt_tokens = inputs["input_ids"].shape[-1]  # n_prompt_tokens >= self.seq_len
            n_chunks: int = math.ceil(n_prompt_tokens / self.seq_len)
            assert n_chunks > 0
            for chunk in range(n_chunks):
                if chunk+1 == n_chunks:
                    lower_idx = -self.seq_len
                    upper_idx = n_prompt_tokens
                    last_chunk = 1
                else:
                    last_chunk = 0
                    lower_idx = int(chunk * self.seq_len)
                    upper_idx = int((chunk + 1) * self.seq_len)
                chunk_inputs = {
                    "input_ids":inputs["input_ids"][:, lower_idx:upper_idx],
                    "position_ids":inputs["position_ids"][:, lower_idx:upper_idx],
                    "batch_index":inputs["batch_index"],
                }
                if "lora_ids" in inputs:
                    chunk_inputs["lora_ids"] = inputs["lora_ids"]

                if self.session.execObj_available < 1 or chunk > self.stages+1:
                    batch_id = last_chunk_ids[chunk_out]
                    if callback:
                        callback()
                    self.session.complete_inf(chunk_out)
                    if (batch_id != None):
                        counter = 0
                        last_chunk_ids[chunk_out] = None
                    else:
                        counter += 1

                    chunk_out += 1
                    chunk_out %= self.stages+1
                if last_chunk:
                    last_chunk_ids[chunk_id] = bidx
                    _ = self.session.set_data_for_kv_handoff(inputs['kv_cache'],
                                                             [("batch_index", bidx), ("ctx_start",0)], 
                                                             chunk_id, self.session.prefill_buff_map[:-1])
                    chunk_inputs['logits'] = inputs['logits']
                    i +=1
                # Submit Chunk to LRT Queue
                self.session.np_run_pipeline(inputs=chunk_inputs, index=chunk_id, last_chunk=last_chunk)
                chunk_id += 1
                chunk_id %= self.stages+1
        # wait for all chunks to finish
        while self.session.execObj_available < self.stages+1:
            batch_id = last_chunk_ids[chunk_out]
            if callback:
                callback()
            self.session.complete_inf(chunk_out)
            if (batch_id != None):
                counter = 0
            else:
                counter += 1
            chunk_out += 1
            chunk_out %= self.stages+1
        return logits.squeeze(1)

    def _run_decode(self, qpc_inputs: Dict[int, dict], callback:Optional[Callable]=None) -> np.ndarray:
        """run qpc decode inputs

        Args:
            qpc_inputs (Dict[int, dict]): qpc inputs of incoming requests to process

        Returns:
            np.ndarray: fixed slot generated tokens
        """
        def _run(idx):
            if idx != self.decode_bsz - 1:
                j = idx + 1
                i=0
                while (j < self.decode_bsz):
                    if i not in qpc_inputs.keys():
                        self.decode_batch_inputs["batch_index"][j] = i
                        self.decode_batch_inputs["position_ids"][..., j, :] = -1
                        j+=1
                    i+=1
            if not self.disagg_serving_en:
                if self.include_sampler:
                    self.session.np_run(self.decode_batch_inputs)
                    self.session.complete_inf()
                else:
                    return self.session.run(self.decode_batch_inputs)
            else:
                if self.include_sampler:
                    self.session.create_output_buffers(self.decode_batch_inputs, (self.decode_bsz, 1, 1), np.int64, "next_tokens")
                else:
                    self.session.create_output_buffers(self.decode_batch_inputs, self.decode_logits["logits"].shape, np.float32)
                self.session.np_run(self.decode_batch_inputs)
                if callback:
                    callback()
                self.session.complete_inf()
            return self.decode_batch_inputs

        if not self.last_decode and not self.include_sampler:
            # set qpc session state to decode phase
            self.session.set_buffers(self.decode_logits)
            if self.is_spec_decode_target_model:
                self.session.set_buffers(self.decode_num_logits_buffer)
            if self.include_sampler and self.return_pdfs:
                self.session.set_buffers(self.decode_probs)
            self.last_decode = True
        # to handle case where we are running precode on TLM/DLM in spec-decoding
        firstinputkey = list(qpc_inputs.keys())[0]
        # Initialize decode batch inputs
        for i in range(self.decode_bsz):
            if not self.ignore_batch_index:
                self.decode_batch_inputs["batch_index"][i] = -1
            self.decode_batch_inputs["position_ids"][..., i, :] = -1

        if self.comp_ctx_lengths_decode is not None:
            max_position_id = max(qpc_inputs[bidx]["position_ids"].max().item() for bidx in qpc_inputs)
            for comp_ctx_len in self.comp_ctx_lengths_decode:
                if max_position_id < comp_ctx_len-1:
                    break
            self.decode_batch_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[comp_ctx_len]

        if not self.is_spec_decode_target_model and qpc_inputs[firstinputkey]["input_ids"].shape[1] == 2:
            # DLM precode case only
            # hacky implementation runs DLM one extra time if previous iteration had a bonus token for any batch element
            # that is currently decoding
            for i in range(2):
                for idx, bidx in enumerate(qpc_inputs.keys()):
                    self.decode_batch_inputs["input_ids"][idx] = qpc_inputs[bidx]["input_ids"][:,i:i+1]
                    self.decode_batch_inputs["position_ids"][idx] = qpc_inputs[bidx]["position_ids"][:,i:i+1]
                    self.decode_batch_inputs["batch_index"][idx] = bidx
                    if "lora_ids" in qpc_inputs[bidx]:
                        self.decode_batch_inputs["lora_ids"][idx] = qpc_inputs[bidx]["lora_ids"]
                    if self.include_sampler:
                        for op in self.sampler_inputs:
                            if op == "last_accepted_output_tokens":
                                self.decode_batch_inputs[op][idx] = qpc_inputs[bidx]["input_ids"]
                            else:
                                self.decode_batch_inputs[op][idx] = qpc_inputs[bidx][op]
                outputs: dict = _run(idx)
        else:
            for idx, bidx in enumerate(qpc_inputs.keys()):
                self.decode_batch_inputs["input_ids"][idx] = qpc_inputs[bidx]["input_ids"]
                self.decode_batch_inputs["position_ids"][..., idx, :] = qpc_inputs[bidx]["position_ids"]
                if not self.ignore_batch_index:
                    self.decode_batch_inputs["batch_index"][idx] = bidx
                if "lora_ids" in qpc_inputs[bidx]:
                    self.decode_batch_inputs["lora_ids"][idx] = qpc_inputs[bidx]["lora_ids"]
                if self.include_sampler:
                    for op in self.sampler_inputs:
                        if op == "last_accepted_output_tokens":
                            self.decode_batch_inputs[op][idx] = qpc_inputs[bidx]["input_ids"]
                        else:
                            self.decode_batch_inputs[op][idx] = qpc_inputs[bidx][op]
            outputs: dict = _run(idx)

        if self.include_sampler:
            if self.return_pdfs:
                probs = outputs["probs"]
            next_tokens = outputs["next_tokens"]
            return next_tokens

        logits: np.ndarray = outputs["logits"]
        if self.is_spec_decode_target_model:
            return logits[:idx+1]
        return logits[:idx+1].squeeze(1)

    def run_encode(self, qpc_inputs: dict, encode_num_logits_buffer: Optional[dict]= None) -> dict:
        """run qpc_inputs

        Args:
            qpc_inputs (dict): qpc inputs of incoming requests to process
        Returns:
            dict: embeddings
        """
        if encode_num_logits_buffer:
            if self.encode_num_logits_buffer is None or encode_num_logits_buffer['output'].shape != self.encode_num_logits_buffer['output'].shape:
                self.session.set_buffers(encode_num_logits_buffer)
                self.encode_num_logits_buffer = encode_num_logits_buffer
        outputs: dict = self.session.run(qpc_inputs)

        return outputs

    def run_mm_encode(self, qpc_inputs: dict) -> dict:
        """run qpc_inputs

        Args:
            qpc_inputs (dict): qpc inputs of incoming requests to process
        Returns:
            dict: extracted multimodal embeddings
        """
        # Determine number of multimodal inputs
        # TODO: This is only verified for Qwen2.5VL. Test with other VLM models.
        num_mm_input = 1
        for key, value in qpc_inputs.items():
            if key in self.mm_input_info:
                num_mm_input = value.shape[0] // self.mm_input_info[key][0]
                break
        if num_mm_input > 1:
            grouped_outputs = {}
            for i in range(num_mm_input):
                single_mm_item_qpc_input = {
                    key: value[self.mm_input_info[key][0]*i : self.mm_input_info[key][0]*(i+1)]
                    for key, value in qpc_inputs.items()
                }

                outputs = self.run_encode(single_mm_item_qpc_input)
                for k, v in outputs.items():
                    if k not in grouped_outputs:
                        grouped_outputs[k] = [v]
                    else:
                        grouped_outputs[k].append(v)

            concatenated_output = {k: np.concatenate(v, axis=1) for k, v in grouped_outputs.items()}
            return concatenated_output
        else:
            return self.run_encode(qpc_inputs)
