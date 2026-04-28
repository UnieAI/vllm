# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""PyTorch Mllama model."""
from collections.abc import Mapping, Sequence
from typing import Union, Any, Optional

import torch
from PIL.Image import Image
from transformers import BatchFeature
from transformers.models.mllama.processing_mllama import (
    get_cross_attention_token_mask,
    convert_sparse_cross_attention_mask_to_dense,
)
import numpy as np

from vllm.model_executor.models.mllama import (
    MllamaProcessingInfo,
    MllamaDummyInputsBuilder,
    MllamaForConditionalGeneration,
)
from vllm.model_executor.models.mllama4 import (
    Mllama4ProcessingInfo,
    Mllama4MultiModalProcessor,
    Mllama4DummyInputsBuilder,
    Llama4ForConditionalGeneration,
)
from vllm.model_executor.models.gemma3_mm import (
    Gemma3ProcessingInfo,
    Gemma3MultiModalProcessor,
    Gemma3DummyInputsBuilder,
    Gemma3ForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalInputs,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargs,
    ImageItem,
    ModalityData,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    MultiModalDataDict,
    MultiModalDataItems,
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataParser,
    ModalityDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails
)


class MllamaQaicMultiModalDataParser(MultiModalDataParser):
    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, torch.Tensor):
            return ImageEmbeddingItems(data)

        return super()._parse_image_data(data)


class MllamaQaicMultiModalProcessor(BaseMultiModalProcessor[MllamaProcessingInfo]):
    def __init__(self, *args, **kwargs):
        super(MllamaQaicMultiModalProcessor, self).__init__(*args, **kwargs)
        model_config = self.info.ctx.model_config
        is_kv_offload = model_config.override_qaic_config.get("kv_offload", False)
        self.need_cross_attention_mask = not (
            is_kv_offload and model_config.runner_type == "pooling"
        )  # vision model for dual qpc doesn't need cross attention mask

    def _get_data_parser(self) -> MultiModalDataParser:
        return MllamaQaicMultiModalDataParser()

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        mm_inputs = super().apply(
            prompt, mm_data, hf_processor_mm_kwargs, tokenization_kwargs=tokenization_kwargs, return_mm_hashes=return_mm_hashes
        )

        # Check that the number of image tokens in the prompt matches
        # the number of images provided in mm_data
        image_token_index = self.info.get_hf_config().image_token_index
        prompt_token_ids = mm_inputs["prompt_token_ids"]
        num_image_tokens = prompt_token_ids.count(image_token_index)
        image_data = mm_data.get("image", [])
        num_images = 1 if isinstance(image_data, Image) else len(image_data)
        if num_image_tokens != num_images:
            raise ValueError(
                f"The number of image tokens ({num_image_tokens}) must be"
                f" the same as the number of images ({num_images})"
            )

        if mm_inputs and self.need_cross_attention_mask:
            mm_kwargs = mm_inputs["mm_kwargs"]
            if "image_embeds" in mm_kwargs:
                token_per_chunk = self.info.get_token_per_chunk_from_config()
                num_tiles = [
                    [int(mm_kwargs["image_embeds"].shape[3] / token_per_chunk)]
                ]
            else:
                if num_images == 1:
                    num_tiles = [
                        [
                            self.info.get_num_tiles_per_image(
                                image_data.height, image_data.width
                            )
                        ]
                    ]
                else:
                    num_tiles = [
                        [
                            self.info.get_num_tiles_per_image(img.height, img.width)
                            for img in image_data
                        ]
                    ]

            cross_attention_token_mask = [
                get_cross_attention_token_mask(prompt_token_ids, image_token_index)
            ]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=self.info.get_hf_config().vision_config.max_num_tiles,
                length=len(prompt_token_ids),
            )
            pad_width = (
                (0, 0),
                (
                    0,
                    self.info.ctx.model_config.max_seq_len_to_capture
                    - len(prompt_token_ids),
                ),
                (0, 0),
                (0, 0),
            )
            cross_attention_mask = np.pad(
                cross_attention_mask, pad_width, mode="constant", constant_values=0
            )
            for i in range(len(cross_attention_mask)):
                elem = MultiModalFieldElem("image", "cross_attention_mask",
                                            torch.from_numpy(cross_attention_mask[i]),
                                            MultiModalBatchedField())
                mm_kwargs._items_by_modality["image"][i]["cross_attention_mask"] = elem
                mm_kwargs._data = None
        return mm_inputs

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        if mm_data:
            processed_outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        else:
            processed_outputs = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            aspect_ratio_ids=MultiModalFieldConfig.batched("image"),
            aspect_ratio_mask=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.get_hf_config().image_token_index

        def get_replacement_mllama(item_idx):
            return [image_token_id]

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_mllama,
            )
        ]


class Mllama4QaicDummyInputsBuilder(Mllama4DummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        # Skip hf processor
        image_token = "<|image|>"
        return image_token * num_images


class Mllama4QaicMultiModalProcessor(Mllama4MultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:

        mm_fields_config = super()._get_mm_fields_config(
            hf_inputs, hf_processor_mm_kwargs
        )
        mm_fields_config["image_embeds"] = MultiModalFieldConfig.batched("image")
        return mm_fields_config

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Skip prompt update if the inputs are image embeddings
        """
        mm_items = self._to_mm_items(mm_data)

        mm_hashes = (
            self._hash_mm_items(mm_items, hf_processor_mm_kwargs)
            if return_mm_hashes
            else None
        )

        (
            prompt_ids,
            mm_kwargs,
            mm_hashes,
            is_update_applied,
        ) = self._cached_apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            return_mm_hashes=return_mm_hashes,
        )

        images = mm_items.get_items("image", (ImageEmbeddingItems, ImageProcessorItems))
        if isinstance(images, ImageEmbeddingItems):
            # remove 2 extra <|begin_of_text|> tokens introduced by encoding and decoding multiple times
            i = 0
            while prompt_ids[i] == 200000:
                i += 1
            if i > 1:
                prompt_ids = prompt_ids[i - 1 :]
            placeholder_ranges = [
                PlaceholderRange(offset=0, length=len(prompt_ids), is_embed=None)
            ]  # FIXME need to change it for multi-image
            mm_placeholder_ranges = {"image": placeholder_ranges}
            return MultiModalInputs(
                type="multimodal",
                prompt=prompt,
                prompt_token_ids=prompt_ids,
                mm_kwargs=mm_kwargs,
                mm_hashes=mm_hashes,
                mm_placeholders=mm_placeholder_ranges,
            )
        else:
            assert "pixel_values" in mm_kwargs
            pixel_values = mm_kwargs["pixel_values"]
            single_patch = pixel_values[0].unsqueeze(0)
            max_num_tiles = self.info.get_max_num_tiles() + 1
            while pixel_values.shape[0] < max_num_tiles:  # max_num_patch:
                pixel_values = torch.cat([pixel_values, single_patch], 0)
            mm_kwargs._items_by_modality["image"][0]["pixel_values"].data = pixel_values
            mm_kwargs._data = None
            prompt_ids, prompt, mm_placeholders = self._maybe_apply_prompt_updates(
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                prompt_ids=prompt_ids,
                mm_kwargs=mm_kwargs,
                is_update_applied=is_update_applied,
            )

            mm_placeholder_ranges = {
                modality: [item.to_range() for item in placeholders]
                for modality, placeholders in mm_placeholders.items()
            }

            return MultiModalInputs(
                type="multimodal",
                prompt=prompt,
                prompt_token_ids=prompt_ids,
                mm_kwargs=mm_kwargs,
                mm_hashes=mm_hashes,
                mm_placeholders=mm_placeholder_ranges,
            )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data == {}:
            # If the input contains image embeddings, there's no need to call the HF processor
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)


class Gemma3QaicMultiModalProcessor(Gemma3MultiModalProcessor):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super(Gemma3MultiModalProcessor, self)._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        images_kwargs = self.info._resolve_image_kwargs(
            hf_processor, {
                "do_pan_and_scan"
        })
        do_pan_and_scan = images_kwargs["do_pan_and_scan"]
        assert not do_pan_and_scan, "Qaic does not support Gemma3 with pan and scan"

        if (images := mm_data.get("images")) is not None:
            parsed_images = (self._get_data_parser().parse_mm_data({
                "image":
                images
            }).get_items("image", (ImageEmbeddingItems, ImageProcessorItems)))
            num_crops = [0] * len(parsed_images)
            processed_outputs["num_crops"] = torch.tensor(num_crops)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_crops = hf_inputs.get("num_crops", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_crops + 1),
            num_crops=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.boi_token

        def get_replacement_gemma3(item_idx: int):
            boi_token = hf_processor.boi_token

            image_text = boi_token
            repl_full = image_text.replace(boi_token,
                                           hf_processor.full_image_sequence)

            tokenizer = hf_processor.tokenizer
            vocab = tokenizer.get_vocab()
            image_token_id = vocab[tokenizer.image_token]

            return PromptUpdateDetails.select_token_id(repl_full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_gemma3,
            )
        ]


def register_qaic_custom_mm_processor():
    mllama = MULTIMODAL_REGISTRY.register_processor(
        MllamaQaicMultiModalProcessor,
        info=MllamaProcessingInfo,
        dummy_inputs=MllamaDummyInputsBuilder,
    )
    mllama(MllamaForConditionalGeneration)

    llama4 = MULTIMODAL_REGISTRY.register_processor(
        Mllama4QaicMultiModalProcessor,
        info=Mllama4ProcessingInfo,
        dummy_inputs=Mllama4QaicDummyInputsBuilder,
    )
    llama4(Llama4ForConditionalGeneration)

    gemma3 = MULTIMODAL_REGISTRY.register_processor(
        Gemma3QaicMultiModalProcessor,
        info=Gemma3ProcessingInfo,
        dummy_inputs=Gemma3DummyInputsBuilder,
    )
    gemma3(Gemma3ForConditionalGeneration)
