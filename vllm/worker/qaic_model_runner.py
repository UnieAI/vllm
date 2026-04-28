# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import math
from typing import TYPE_CHECKING, List, Callable, TypeVar, Optional, Tuple, Dict, Union, Any, Set
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

import time
from vllm.config import VllmConfig, DeviceConfig, set_current_vllm_config
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader.qaic import get_qaic_model, QaicCausalLM, update_hf_config
from vllm.model_executor.models.qaic_custom_mm_processor import register_qaic_custom_mm_processor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap)
from vllm.sequence import (
    CompletionSequenceGroupOutput,
    IntermediateTensors,
    Logprob,
    SequenceData,
    SequenceGroupMetadata,
    SequenceOutput,
)
from vllm.utils import make_tensor_with_pad, is_pin_memory_available
#from vllm.model_executor.guided_decoding.xgrammar_decoding import XGrammarLogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

def pad_to_max_length(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


@dataclass
class QaicSamplingParams:
    repetition_penalties: List[float]
    presence_penalties: List[float]
    # frequency_penalties: List[float]
    temperatures: List[float]
    top_ks: List[int]
    top_ps: List[float]
    min_ps: List[float]
    random_numbers: List[np.ndarray]
    token_bitmasks: List[Optional[np.ndarray]]
    past_repetition_penalty_buffer: List[np.ndarray]

    def to_dict(self) -> Dict[str, Union[List[float], List[int], List[np.ndarray]]]:
        return {
            "repetition_penalties": self.repetition_penalties,
            "presence_penalties": self.presence_penalties,
            # "frequency_penalties": self.frequency_penalties,
            "temperatures": self.temperatures,
            "top_ks": self.top_ks,
            "top_ps": self.top_ps,
            "min_ps": self.min_ps,
            "random_numbers": self.random_numbers,
            "token_bitmasks": self.token_bitmasks,
            "past_repetition_penalty_buffer": self.past_repetition_penalty_buffer,
        }

    def update(self, sampling_metadata: SamplingMetadata, max_top_k_ids: int):
        """
        Update random number buffers and token bitmasks
        """
        for i, seq_group in enumerate(sampling_metadata.seq_groups):
            sp = seq_group.sampling_params
            if sp.temperature != 0:
                self.random_numbers[i] = torch.rand(
                    1,
                    max_top_k_ids,
                    generator=seq_group.generator,
                    dtype=torch.float32
                ).numpy()
            if sp.logits_processors and self.token_bitmasks[i] is not None:
                logits_processor = sp.logits_processors[0]
                seq_id = seq_group.seq_ids[0]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                bitmask = logits_processor.get_next_token_bitmask(past_tokens_ids)
                if bitmask is not None:
                    self.token_bitmasks[i] = bitmask.numpy()

@dataclass(frozen=True)
class ModelInputForQaic(ModelRunnerInputBase):
    """
    Used by the QaicModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    sampling_metadata: Optional["SamplingMetadata"] = None
    is_prompt: Optional[bool] = None
    async_callback: Optional[Callable] = None
    cleanup_callback: Optional[Callable] = None
    lora_ids: Optional[torch.Tensor] = None
    multi_modal_kwargs_list: Optional[List[dict]] = None
    pooling_metadata: Optional["PoolingMetadata"] = None
    skip_sampling: bool = False
    kv_handoff_metadata: Optional[List[List[str]]] = None
    seq_lens: Optional[List[int]] = None
    prompt_hashes: Optional[List[int]] = None
    seq_groups: Optional[List[List[int]]] = None
    sampling_params: Optional[QaicSamplingParams] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        raise NotImplementedError("ModelInputForQaic cannot be broadcast.")

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForQaic":
        assert attn_backend is None
        return cls.from_broadcasted_tensor_dict(tensor_dict)

class QaicModelRunner(ModelRunnerBase[ModelInputForQaic]):
    """
    Helper class for shared methods between Qaic model runners.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        speculative_model_type: Optional[str] = None
    ):
        self._previous_batch_request_ids: List[str] = []

        update_hf_config(vllm_config.model_config)

        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        self.speculative_model_type = speculative_model_type

        if vllm_config.model_config is not None and vllm_config.model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on qaic. "
                           "The model will run without sliding window.")
        self.device_config = (vllm_config.device_config
                              if vllm_config.device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        register_qaic_custom_mm_processor()

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        #set the number of tokens required for precode inference
        if self.speculative_model_type in ("target", "turbo"):
            # "target" is to detached spd scenarios (two qpcs) while "turbo"
            # is to attached spd scenario (one qpc)
            self.num_precode_inputs = self.speculative_config.num_speculative_tokens + 1
        elif self.speculative_model_type == "draft":
            self.num_precode_inputs = 2

        # create lora table for fast lora request matching in add_lora()
        if vllm_config.lora_config:
            self.lora_table = {mod.name: mod.path for mod in self.vllm_config.lora_config.lora_modules}

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            "fp16",
            self.cache_config.block_size,
            True,
        )
        self.sampler = get_sampler()
        self.allowed_seqlens = []
        self.embed_task_en = vllm_config.model_config.runner_type == "pooling" and not vllm_config.model_config.is_multimodal_model
        self._default_random_numbers = None
        self._default_token_bitmask = None
        self._penalty_masks = None

    def load_model(self) -> None:
        get_qaic_model(self.model,
                       self.vllm_config,
                        speculative_model_type=self.speculative_model_type)
        self._on_device_sampling_enabled: Optional[bool] = self.model.model.include_sampler
        self._on_device_guided_decoding_enabled: Optional[bool] = self.model.model.include_guided_decoding
        if self._on_device_sampling_enabled:
            self._default_random_numbers = torch.rand(
                1, self.model.model.max_top_k_ids, dtype=torch.float32
            ).numpy()
            self._penalty_masks = np.zeros(
                (self.vllm_config.scheduler_config.max_num_seqs, self.vocab_size),
                dtype=bool,
            )
        if self._on_device_guided_decoding_enabled:
            self._default_token_bitmask = np.full(
                (1, self.vocab_size),
                True,
                dtype=bool
            )
        if self.embed_task_en:
            self.allowed_seqlens = self.model.get_allowed_seqlens()

    def init_model(self) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model = QaicCausalLM(self.model_config.hf_config, self.vllm_config)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[int], List[int],
        List[int], List[dict], List[int], List[int],
    ]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[np.ndarray] = []
        input_positions: List[np.ndarray] = []
        input_block_ids: List[int] = []
        hashes: List[int] = []
        seq_lens: List[int] = []
        lora_ids: List[int] = []
        multi_modal_kwargs_list: List[dict] = []
        computed_lens: List[int] = []

        max_padding = None
        max_seq_len = self.model_config.max_seq_len_to_capture
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids: List[int] = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id: int = seq_ids[0]

            seq_data: SequenceData = seq_group_metadata.seq_data[seq_id]
            prompt_tokens: List[int] = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            if self.vllm_config.kv_transfer_config:
                hashes.append(hash(tuple(prompt_tokens)))
            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]

            # For encoder-only models, the block_table is None,
            if block_table is not None:
                input_block_id = block_table[0]
                input_block_ids.append(input_block_id)

            #Prefix Caching
            computed_block_nums = seq_group_metadata.computed_block_nums
            computed_len = len(computed_block_nums) if ((computed_block_nums is not None)
                                        and (len(computed_block_nums) > 0)) else 0
            computed_len = computed_len * self.cache_config.block_size

            # Multi-modal inputs
            if seq_group_metadata.multi_modal_data:
                # Exclude vision tokens from prefix caching
                image_token_index = self.model.model.image_token_index
                if image_token_index:
                    try:
                        first_image_token = prompt_tokens.index(image_token_index)
                        computed_len = min(first_image_token, computed_len)
                    except ValueError:
                        pass
                # NOTE: mm_data only includes the subset of multi-modal items
                # that intersect with the current prefill positions.
                mm_kwargs, _ = MultiModalPlaceholderMap \
                    .from_seq_group(seq_group_metadata, range(computed_len, seq_len))
                mm_kwargs = mm_kwargs.get_data()
            else:
                mm_kwargs = None
            computed_lens.append(computed_len)

            # pad input
            prompt_positions: List[int] = list(range(computed_len, seq_len))
            if self.model_config.uses_mrope and not self.vllm_config.model_config.runner_type == "pooling":
                # special processing for mrope position deltas.
                hf_config = self.model_config.hf_config
                if mm_kwargs:
                    image_grid_thw = mm_kwargs.get("image_grid_thw", None)
                    video_grid_thw = mm_kwargs.get("video_grid_thw", None)
                    audio_feature_lengths = mm_kwargs.get("audio_feature_lengths",
                                                        None)
                    assert (
                        image_grid_thw is not None or video_grid_thw is not None
                        or audio_feature_lengths is not None), (
                            "mrope embedding type requires multi-modal input mapper "
                            "returns 'image_grid_thw' or 'video_grid_thw' or "
                            "'audio_feature_lengths'.")

                    second_per_grid_ts = mm_kwargs.get("second_per_grid_ts", None)
                    use_audio_in_video = mm_kwargs.get("use_audio_in_video", False)

                    mrope_input_positions, mrope_position_delta = \
                        MRotaryEmbedding.get_input_positions(
                            prompt_tokens,
                            hf_config=hf_config,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            second_per_grid_ts=second_per_grid_ts,
                            context_len=computed_len,
                            seq_len=seq_len,
                            audio_feature_lengths=audio_feature_lengths,
                            use_audio_in_video=use_audio_in_video,
                        )
                    stacked_prompt_positions = [prompt_positions] + mrope_input_positions
                else:
                    stacked_prompt_positions = [prompt_positions] * 4
                    mrope_position_delta = 0
                seq_data.mrope_position_delta = mrope_position_delta
                prompt_tokens = prompt_tokens[computed_len:]
                if seq_len-computed_len <= max_seq_len:
                    prompt_tokens = pad_to_max_length(prompt_tokens, max_seq_len, 2)
                    stacked_prompt_positions = [pad_to_max_length(position, max_seq_len, -1) for position in stacked_prompt_positions]
                prompt_tokens = np.asarray(prompt_tokens).reshape(1, -1)
                prompt_positions = np.asarray(stacked_prompt_positions).reshape(4, 1, -1)
            else:
                prompt_tokens = prompt_tokens[computed_len:]

                if self.embed_task_en:
                    max_seq_len = self.model_config.max_model_len
                    if self.allowed_seqlens:
                        max_padding = max(max_padding if max_padding else 0, min([sq
                                                for sq in self.allowed_seqlens if int(sq) > len(prompt_tokens)]))

                if self.vllm_config.model_config.hf_config.is_encoder_decoder:
                    # A hack for using Whisper model with OpenAI's audio.transcriptions API
                    prompt_tokens = np.full((1, max_seq_len),
                                            self.vllm_config.model_config.hf_config.decoder_start_token_id)
                    prompt_positions = np.asarray(prompt_positions[:max_seq_len]
                                            ).reshape(1, -1)
                elif seq_len-computed_len <= max_seq_len:
                    prompt_tokens = np.asarray(
                                        pad_to_max_length(prompt_tokens, max_seq_len, 2)
                                        ).reshape(1, max_seq_len)
                    prompt_positions = np.asarray(
                                        pad_to_max_length(prompt_positions, max_seq_len, -1)
                                        ).reshape(1, max_seq_len)
                else:
                    # model_runner will perform chunking
                    prompt_tokens = np.asarray(
                                            prompt_tokens
                                            ).reshape(1, -1)
                    prompt_positions = np.asarray(
                                            prompt_positions
                                            ).reshape(1, -1)

            input_tokens.append(prompt_tokens)
            input_positions.append(prompt_positions)
            if self.vllm_config.lora_config:
                lora_ids.extend(
                    [seq_group_metadata.lora_int_id] *
                    (seq_len - self.ctx_len
                    if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            # Update multi-modal inputs
            if mm_kwargs:
                if "image_embeds" in mm_kwargs:
                    image_embeds = mm_kwargs.pop("image_embeds").to(torch.float16)
                    if self.model.model.is_cross_attention:
                        # Temporary solution for mllama
                        # image_embeds is a stack of past keys and past values
                        cross_attention_layers = self.vllm_config.model_config.hf_config.text_config.cross_attention_layers
                        assert len(cross_attention_layers) == image_embeds.shape[1] // 2
                        for i, layer_idx in enumerate(cross_attention_layers):
                            mm_kwargs[f"past_key.{layer_idx}"] = image_embeds[0][i*2].unsqueeze(0)
                            mm_kwargs[f"past_value.{layer_idx}"] = image_embeds[0][i*2+1].unsqueeze(0)
                    elif self.vllm_config.model_config.multimodal_config.get_limit_per_prompt("image") > 1:
                        if image_embeds.shape[0] != 1 and self.model.model.embeds_dims[0] == 1:
                            image_embeds = image_embeds.reshape(1, math.prod(image_embeds.shape[:-1]), image_embeds.shape[-1])
                        if list(image_embeds.shape) != self.model.model.embeds_dims:
                            padded_embeds = torch.zeros(self.model.model.embeds_dims, dtype=image_embeds.dtype)
                            slices = tuple(slice(0, s) for s in image_embeds.shape)
                            padded_embeds[slices] = image_embeds
                            image_embeds = padded_embeds
                        mm_kwargs["vision_embeds"] = image_embeds
                    else:
                        mm_kwargs["vision_embeds"] = image_embeds.reshape(self.model.model.embeds_dims)
                if "pixel_values_flat" in mm_kwargs: # for internvl
                    mm_kwargs["pixel_values"] = mm_kwargs["pixel_values_flat"]
                    del mm_kwargs["pixel_values_flat"]
                for k, v in list(mm_kwargs.items()):
                    if k not in self.model.model.session.binding_index_map:
                        del mm_kwargs[k]
                        continue
                    if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                        # TODO: check if still works for multiframes for models other than qwenvl
                        v = v[0]
                        mm_kwargs[k] = v
                    if isinstance(v, torch.Tensor) and \
                        v.dtype.is_floating_point and v.dtype != torch.float16:
                        v = v.to(torch.float16)
                    v = np.asarray(v)
                    mm_kwargs[k] = v
                multi_modal_kwargs_list.append(mm_kwargs)
            elif self.model_config.is_multimodal_model:
                multi_modal_kwargs_list.append(None)

        assert max_seq_len > 0
        if max_padding and (max_seq_len != max_padding):
            input_tokens = [ip_token[:,:max_padding] for ip_token in input_tokens]
            input_positions = [ip_posn[:,:max_padding] for ip_posn in input_positions]

        return (
            input_tokens, input_positions, input_block_ids, seq_lens,
            lora_ids, multi_modal_kwargs_list, hashes, computed_lens
        )

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = [] # holds token ids of each sequence in batch
        input_positions: List[List] = [] # Either List[List[int]] for regular models, or List[List[List[int]]] for models that use mrope
        input_block_ids: List[int] = []
        lora_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids: List[int] = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data: SequenceData = seq_group_metadata.seq_data[seq_id]
                generation_token: int = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len: int = seq_data.get_len()
                position: int = seq_len - 1 # TODO: needs changing for qpc
                if seq_data.mrope_position_delta is not None:
                    next_pos = MRotaryEmbedding.get_next_input_positions(
                        seq_data.mrope_position_delta,
                        seq_data.get_num_computed_tokens(),
                        seq_len,
                    )
                    stacked_position_ids = [[position]] + next_pos
                    input_positions.append(stacked_position_ids)
                else:
                    input_positions.append([position])

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                #assert len(block_table) == 1
                input_block_id = block_table[0]
                input_block_ids.append(input_block_id)
                if self.vllm_config.lora_config:
                    lora_ids.append(seq_group_metadata.lora_int_id)

        assert len(input_tokens[0]) == 1

        input_tokens = [np.asarray(x_i).reshape(1,1) for x_i in input_tokens]
        def reshape_input_position(position):
            position = np.asarray(position)
            return position.reshape(position.size, 1)
        input_positions = [reshape_input_position(x_i) for x_i in input_positions]

        return input_tokens, input_positions, input_block_ids, lora_ids

    def _prepare_precode(self, seq_group_metadata_list: List[SequenceGroupMetadata]) -> Tuple:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = [] # holds token ids of each sequence in batch
        input_positions: List[List[int]] = []
        input_block_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            is_curr_sgm_precode = seq_group_metadata.is_precode
            #reset it to false once we know we have to run a precode, to avoid repeated precodes
            seq_group_metadata.is_precode = False
            seq_ids: List[int] = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data: SequenceData = seq_group_metadata.seq_data[seq_id]
                if is_curr_sgm_precode and len(list(seq_data.get_output_token_ids())) >= 2:
                    generation_tokens: List[int] = list(seq_data.get_output_token_ids()[-self.num_precode_inputs:])
                    input_tokens.append(generation_tokens)
                    seq_len: int = seq_data.get_len()
                    positions: List[int] = list(range(seq_len - self.num_precode_inputs, seq_len)) # TODO: needs changing for qpc
                    input_positions.append(positions)
                else:
                    #this is the case for DLM precode, when some sequences in the batch have a bonus token from last step
                    # we need to pad the other sequences in this batch to the right to match the bonus-token sequence
                    generation_tokens: List[int] = [seq_data.get_output_token_ids()[-1],0]
                    input_tokens.append(generation_tokens)

                    seq_len: int = seq_data.get_len()
                    positions: List[int] = [seq_len - 1, -1]
                    input_positions.append(positions)

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                # assert len(block_table) == 1
                input_block_id = block_table[0]
                input_block_ids.append(input_block_id)

        assert len(input_tokens[0]) == (self.num_precode_inputs)

        input_tokens = [np.asarray(x_i).reshape(1,self.num_precode_inputs) for x_i in input_tokens]
        input_positions = [np.asarray(x_i).reshape(1,self.num_precode_inputs) for x_i in input_positions]
        return input_tokens, input_positions, input_block_ids

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForQaic:
        return ModelInputForQaic.from_broadcasted_tensor_dict(tensor_dict)

    def get_penalty_mask(self, tokens: np.ndarray, i: int) -> np.ndarray:
        mask = self._penalty_masks[i:i+1, :]
        mask.fill(False)
        valid = (tokens >= 0) & (tokens < self.vocab_size)
        mask[0, tokens[valid]] = True
        return mask

    def _prepare_sample(
        self, sampling_metadata: SamplingMetadata, computed_lens: list[int]
    ) -> QaicSamplingParams:
        repetition_penalties = []
        presence_penalties = []
        # frequency_penalties = []
        temperatures = []
        top_ks = []
        top_ps = []
        min_ps = []
        random_numbers = []
        token_bitmasks = []
        past_repetition_penalty_buffer = []

        for i, seq_group in enumerate(sampling_metadata.seq_groups):
            sp = seq_group.sampling_params
            seq_ids = seq_group.seq_ids
            assert len(seq_ids) == 1, (
                "On-device sampling with guided decoding does not support speculative decoding"
            )
            seq_id = seq_ids[0]
            repetition_penalties.append(sp.repetition_penalty)
            if (
                sp.repetition_penalty != 1.0
                and len(seq_group.seq_data[seq_id].output_token_ids) == 0
                and computed_lens[i] > 0
            ):
                prompt_token_ids = seq_group.seq_data[seq_id].prompt_token_ids[
                    : computed_lens[i]
                ]
                past_repetition_penalty_buffer.append(
                    self.get_penalty_mask(np.asarray(prompt_token_ids), i)
                )
            else:
                past_repetition_penalty_buffer.append(None)

            presence_penalties.append(sp.presence_penalty)
            if sp.frequency_penalty is not None and sp.frequency_penalty != 0:
                logger.warning_once(
                    "Frequency penalties are not supported by the QAIC backend. "
                    "The Sampler will run without frequency penalties. "
                    "To use frequency penalties, please use the PyTorch backend."
                )
            # frequency_penalties.append(sp.frequency_penalty)
            temperatures.append(sp.temperature)
            if sp.top_k < 1 or sp.top_k > self.model.model.max_top_k_ids:
                logger.warning_once(
                    f"Currently, the QPC is compiled for `max_top_k_ids`={self.model.model.max_top_k_ids}. "
                    "To use a different value, please provide `max_top_k_ids` in `override_qaic_config`. "
                    "This will recompile the QPC with the updated value."
                )
                top_ks.append(self.model.model.max_top_k_ids)
            else:
                top_ks.append(sp.top_k)
            top_ps.append(sp.top_p)
            min_ps.append(sp.min_p)
            if temperatures != 0.0:
                random_numbers.append(torch.rand(1, self.model.model.max_top_k_ids, generator=seq_group.generator, dtype=torch.float32).numpy())
            else:
                random_numbers.append(self._default_random_numbers)
            token_bitmask = None
            if self._on_device_guided_decoding_enabled:
                token_bitmask = self._default_token_bitmask
                if sp.logits_processors:
                    # Guided decoding not supported in v10.1.1 v0
                    #assert len(sp.logits_processors) == 1 and isinstance(sp.logits_processors[0], XGrammarLogitsProcessor), (
                    #    "Only the XGrammar backend is supported for guided decoding with on-device sampling in QAIC."
                    #)
                    logits_processor = sp.logits_processors[0]
                    past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                    bitmask = logits_processor.get_next_token_bitmask(past_tokens_ids)
                    if bitmask is not None:
                        token_bitmask = bitmask.numpy()
            elif sp.logits_processors:
                logger.warning_once(
                        "Currently, the QPC is compiled without guided decoding support. To use guided decoding,"
                        "please provide `aic_include_guided_decoding=True` in `override_qaic_config`."
                        "Otherwise, the sampler will run without guided decoding."
                    )
            token_bitmasks.append(token_bitmask)

        sampling_params = QaicSamplingParams(
            repetition_penalties=repetition_penalties,
            presence_penalties=presence_penalties,
            # frequency_penalties=frequency_penalties,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            random_numbers=random_numbers,
            token_bitmasks=token_bitmasks,
            past_repetition_penalty_buffer=past_repetition_penalty_buffer,
        )

        return sampling_params

    def prepare_model_input(
        self,
        input_seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) ->  ModelInputForQaic:
        multi_modal_kwargs_list = []
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        seq_group_metadata_list = []
        is_prompt = input_seq_group_metadata_list[0].is_prompt
        lora_ids: List[int] = []
        computed_lens: List[int] = []
        ## with MQAScorer
        if not is_prompt and self.speculative_model_type in ("target", "turbo"):
            for sgm in input_seq_group_metadata_list:
                sgm.is_precode = True
        seq_group_metadata_list = input_seq_group_metadata_list
        ## with MQAScorer
        # check if any of the sequences require precode-type inference(multiple tokens in parallel):
        # either 1) TLM scoring inference or 2) DLM 1st token proposal when bonus token in last iteration
        is_any_sgm_precode = False
        for idx in range(len(seq_group_metadata_list)):
            if seq_group_metadata_list[idx].is_precode:
                is_any_sgm_precode = True

        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_block_ids,
             seq_lens, lora_ids, multi_modal_kwargs_list, prompt_hashes, 
             computed_lens) = self._prepare_prompt(seq_group_metadata_list)
        elif is_any_sgm_precode:
            #is_precode is True for TLM for Scoring,
            #for DLM in first speculation if last iteration had all tokens accepted
            (input_tokens, input_positions, input_block_ids
            ) = self._prepare_precode(seq_group_metadata_list)
            seq_lens, prompt_hashes = None, None
            if self.speculative_model_type in ("target", "turbo"):
                seq_lens = [(self.speculative_config.num_speculative_tokens+1) for sgm in input_seq_group_metadata_list]
        else:
            (input_tokens, input_positions, input_block_ids, decode_lora_ids
             ) = self._prepare_decode(seq_group_metadata_list)
            seq_lens, prompt_hashes = None, None
            if self.vllm_config.lora_config:
                lora_ids.extend(decode_lora_ids)
        sampling_metadata = SamplingMetadata.prepare(
            input_seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since qaic worker doesn't support chunked prefill
            # just use seq_lens instead.
            # As vllm chunking core implementation is not used
            # thus there is no need to provide query_lens
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        if self._on_device_sampling_enabled:
            # Update the entire sampling_metadata if the request IDs have changed.
            # If the request IDs remain the same,
            # only the random numbers need to be updated for requests that use random sampling.
            current_batch_request_ids = [sgm.request_id for sgm in input_seq_group_metadata_list]
            if current_batch_request_ids != self._previous_batch_request_ids:
                self.sampling_params = self._prepare_sample(
                    sampling_metadata, computed_lens
                )  # Get sampling params
            else:
                self.sampling_params.update(sampling_metadata, self.model.model.max_top_k_ids)
            self._previous_batch_request_ids = current_batch_request_ids

        return ModelInputForQaic(
            input_tokens=input_tokens,
            input_positions=input_positions,
            input_block_ids=input_block_ids,
            sampling_metadata=sampling_metadata,
            is_prompt=is_prompt,
            lora_ids=lora_ids if self.vllm_config.lora_config else None,
            multi_modal_kwargs_list=multi_modal_kwargs_list,
            seq_lens=seq_lens,
            prompt_hashes=prompt_hashes,
            seq_groups=[list(sgm.seq_data.keys()) for sgm in input_seq_group_metadata_list],
            sampling_params=self.sampling_params if self._on_device_sampling_enabled else None,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForQaic,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:

        if num_steps > 1:
            raise ValueError(
                "QaicModelRunner does not support multi-step execution.")
        kv_caches = []
        bypass_model_exec = False
        if self.vllm_config.kv_transfer_config:
            hidden_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    self.model,
                    model_input,
                    kv_caches=kv_caches, # = [dict](bsx) [160 list] [np (1, 8 , 4097, 128)]
                )
        else:
            hidden_states = None

        hidden_states: torch.Tensor = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            batch_indices=model_input.input_block_ids,
            is_prompt=model_input.is_prompt,
            lora_ids=model_input.lora_ids if (
                self.vllm_config.lora_config and model_input.lora_ids is not None) else None,
            sampling_params=model_input.sampling_params.to_dict() if \
                self._on_device_sampling_enabled else None,
            bypass_model_exec=bypass_model_exec,
            kv_caches = kv_caches,
            logits_mem_buffs=hidden_states,
            callback=model_input.cleanup_callback,
            multi_modal_kwargs_list=model_input.multi_modal_kwargs_list,
        )

        if model_input.async_callback is not None:
            model_input.async_callback()

        if model_input.skip_sampling:
            output: SamplerOutput = _make_sampler_output(
                next_token_ids=torch.zeros((len(model_input.input_block_ids),1), dtype=torch.int64),
                seq_groups=model_input.seq_groups,
            )
        else:
            if self._on_device_sampling_enabled:
                logits = hidden_states
            else:
                logits = self.model.compute_logits(hidden_states,
                                                model_input.sampling_metadata)

            if self.speculative_model_type in ("target", "turbo"):
                # Post-process the logits.
                logits = self.model.process_logits(logits, self.speculative_model_type,
                                                model_input.is_prompt,
                                                model_input.sampling_metadata)

            if not self._on_device_sampling_enabled:
                # Sample the next token.
                output: Optional[SamplerOutput] = self.sampler(
                    logits=logits,
                    sampling_metadata=model_input.sampling_metadata,
                )
            else:
                if model_input.is_prompt and self.vllm_config.kv_transfer_config:
                    output: Optional[SamplerOutput] = self.sampler(
                        logits=logits,
                        sampling_metadata=model_input.sampling_metadata,
                    )
                else:
                    output: SamplerOutput = _make_sampler_output(
                        next_token_ids=logits,
                        seq_groups=model_input.seq_groups,
                    )

            if self.speculative_model_type == "turbo":
            # Post-process sample.
                output: Optional[SamplerOutput] = self.model.process_sample(output, model_input.is_prompt)

        if self.vllm_config.kv_transfer_config:
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model is used to know which layer the current worker
                # is working on, so that we can receive KV for only those
                # layers.
                self.model,
                model_input,
                kv_caches=kv_caches,
                hidden_or_intermediate_states=hidden_states
            )

        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def get_model(self) -> nn.Module:
        return self.model.model

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # check if lora_request appears in lora_modules
        if lora_request.lora_name in self.lora_table and self.lora_table[lora_request.lora_name] == lora_request.lora_path:
            return True
        return False

    def remove_lora(self, lora_id: int) -> None:
        raise NotImplementedError("remove_lora() is not supported in QAIC backend")

    def pin_lora(self, lora_id: int) -> None:
        raise NotImplementedError("pin_lora() is not supported in QAIC backend")

    def list_loras(self) -> None:
        raise NotImplementedError("list_loras() is not supported in QAIC backend")


def _make_sampler_output(
    next_token_ids: List[int],
    seq_groups: List[List[int]],
) -> SamplerOutput:
    zero_logprob = Logprob(0.0)
    sampler_outputs = []
    batch_idx = 0
    for seq_ids in seq_groups:
        seq_outputs = []
        for seq_id in seq_ids:
            next_token_id = next_token_ids[batch_idx]
            seq_outputs.append(SequenceOutput(seq_id, next_token_id, {next_token_id: zero_logprob}))
            batch_idx += 1
        sampler_outputs.append(CompletionSequenceGroupOutput(seq_outputs, None))
    return SamplerOutput(sampler_outputs)
