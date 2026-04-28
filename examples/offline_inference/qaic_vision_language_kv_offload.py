# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal embedding.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from argparse import Namespace
from typing import Literal, NamedTuple, Optional, TypedDict, Union, get_args

from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import torch

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import requests
import io
from urllib.parse import urlparse

# define qpc parameters
vision_bsz = 1 # The vision model in a VLM that outputs vision embeddings should be compiled with a batch size of 1.
decode_bsz = 4 # Except for MLLama, the language model in a VLM that generate language outputs supports continuous batching.


class TextQuery(TypedDict):
    modality: Literal["text"]
    text: str


class TextImageQuery(TypedDict):
    modality: Literal["image+text"]
    text: str
    image: Image.Image


QueryModality = Literal["text", "image+text"]
Query = TextImageQuery


class ModelRequestData(NamedTuple):
    llm_embed: LLM
    llm_gen: LLM
    prompt: str
    image: Optional[Image.Image]
    stop_token_ids: list
    image_grid_thw: Optional[torch.Tensor] = None


# InternVL
def run_internvl(query: Query, args):
    ctx_len = 4096
    seq_len = 128

    model_name = "OpenGVLab/InternVL2_5-1B"

    if query["modality"] == "image+text":
        llm_embed = LLM(
            runner="pooling",
            model=model_name,
            device_group=args.device_group_embed,
            max_num_seqs=vision_bsz, 
            max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
            max_seq_len_to_capture=seq_len,  # seq_len# Preferred option to same KV cache and increase performance
            enable_prefix_caching=False,
            trust_remote_code=True,
            override_qaic_config={
                "num_patches": 13,
                "kv_offload": True,
            },  # In QAic, num_patches is fixed
        )
        image = query["image"]
        messages = [{"role": "user", "content": f"<image>\n{query['text']}"}]
    elif query["modality"] == "text":
        llm_embed = None
        image = None
        messages = [{"role": "user", "content": f"{query['text']}"}]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm_gen = LLM(
        model=model_name,
        device_group=args.device_group_gen,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        enable_prefix_caching=False,
        trust_remote_code=True,
        override_qaic_config={
            "num_patches": 13,
            "kv_offload": True,
        },  # In QAic, num_patches is fixed
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]  # "<|end|>"
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        llm_embed=llm_embed,
        llm_gen=llm_gen,
        prompt=prompt,
        image=image,
        stop_token_ids=stop_token_ids,
    )


def run_qwen_2_5_vl(query: Query, args):
    ctx_len = 4096
    seq_len = 128

    model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Image's height and width should be divisible by the default image factor (28).
    # If a number provided is not divisible,
    # it will be rounded to the nearest integer that is divisible by the factor.
    height = 364
    width = 532
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize)
    height, width = smart_resize(height=height, width=width)

    if query["modality"] == "image+text":
        llm_embed = LLM(
            runner="pooling",
            model=model_name,
            device_group=args.device_group_embed,
            max_num_seqs=decode_bsz,  # determines decode batch size
            max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
            max_seq_len_to_capture=seq_len,  # seq_len
            kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
            enable_prefix_caching=False,
            trust_remote_code=True,
            override_qaic_config={
                "height": height,
                "width": width,
                "kv_offload": True,
            },
        )
        image = query["image"]
        image = image.resize((width, height))
        # Because QAIC uses a fixed image shape for pixel values,
        # the image_grid_thw remains the same for all images.
        # Alternatively, you can use the processor to process a single image and extract the image_grid_thw.
        patch_size = llm_embed.llm_engine.model_config.hf_config.vision_config.patch_size
        grid_h, grid_w = height // patch_size, width // patch_size
        image_grid_thw = torch.tensor([[1, grid_h, grid_w]])
        messages = [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": f"{args.question}"
                    },
                ],
            },
        ]
    elif query["modality"] == "text":
        llm_embed = None
        image = None
        image_grid_thw = None
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{args.question}"
                    },
                ],
            },
        ]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm_gen = LLM(
        model=model_name,
        device_group=args.device_group_gen,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        enable_prefix_caching=False,
        trust_remote_code=True,
        override_qaic_config={
            "height": height,
            "width": width,
            "kv_offload": True,
        },
    )

    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return ModelRequestData(
        llm_embed=llm_embed,
        llm_gen=llm_gen,
        prompt=prompt,
        image=image,
        stop_token_ids=None,
        image_grid_thw=image_grid_thw,
    )


def run_llava(query: Query, args):
    ctx_len = 2048
    seq_len = 128

    model_name = "llava-hf/llava-1.5-7b-hf"

    if query["modality"] == "image+text":
        llm_embed = LLM(
            runner="pooling",
            model=model_name,
            device_group=args.device_group_embed,
            max_num_seqs=vision_bsz,
            max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
            max_seq_len_to_capture=seq_len,  # seq_len Preferred option to same KV cache and increase performance
            enable_prefix_caching=False,
            trust_remote_code=True,
            override_qaic_config={
                "kv_offload": True,
            },
        )
        image = query["image"]
        prompt = f"USER: <image>\n{args.question}\nASSISTANT:"
    elif query["modality"] == "text":
        llm_embed = None
        image = None
        prompt = f"USER: {args.question}\nASSISTANT:"
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm_gen = LLM(
        model=model_name,
        device_group=args.device_group_gen,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        enable_prefix_caching=False,
        trust_remote_code=True,
        override_qaic_config={
            "kv_offload": True,
        },
    )

    stop_token_ids = None

    return ModelRequestData(
        llm_embed=llm_embed,
        llm_gen=llm_gen,
        prompt=prompt,
        image=image,
        stop_token_ids=stop_token_ids,
    )


def run_mllama(query: Query, args):
    ctx_len = 512
    seq_len = 32
    decode_bsz = 1 # mllama does not support continuous batching yet

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    if query["modality"] == "image+text":
        llm_embed = LLM(
            runner="pooling",
            model=model_name,
            device_group=args.device_group_embed,
            max_num_seqs=vision_bsz,
            max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
            max_seq_len_to_capture=seq_len,  # seq_len
            enable_prefix_caching=False,
            override_qaic_config={
                "kv_offload": True,
                "dfs": False,
            },
        )
        image = query["image"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{args.question}"},
                ],
            }
        ]
    elif query["modality"] == "text":
        llm_embed = None
        image = None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{args.question}"},
                ],
            }
        ]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm_gen = LLM(
        model=model_name,
        device_group=args.device_group_gen,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",
        enable_prefix_caching=False,
        trust_remote_code=True,
        override_qaic_config={
            "kv_offload": True,
            "dfs": False,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    stop_token_ids = None

    return ModelRequestData(
        llm_embed=llm_embed,
        llm_gen=llm_gen,
        prompt=prompt,
        image=image,
        stop_token_ids=stop_token_ids,
    )


def run_llama4(query: Query, args):
    ctx_len = 3072
    seq_len = 128

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    if query["modality"] == "image+text":
        llm_embed = LLM(
            runner="pooling",
            model=model_name,
            device_group=args.device_group_embed,
            max_num_seqs=vision_bsz,
            max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
            max_seq_len_to_capture=seq_len,  # seq_len
            kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
            enable_prefix_caching=False,
            override_qaic_config={
                "kv_offload": True,
            },
            # The default value of max_patches is 16,
            # which results in up to 17 tiles after adding a global tile.
            # Override this value if necessary using:
            # mm_processor_kwargs = {"max_patches": 16}
        )
        image = query["image"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{args.question}"},
                ],
            }
        ]
    elif query["modality"] == "text":
        llm_embed = None
        image = None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{args.question}"},
                ],
            }
        ]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm_gen = LLM(
        model=model_name,
        device_group=args.device_group_gen,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        enable_prefix_caching=False,
        trust_remote_code=True,
        override_qaic_config={
            "kv_offload": True,
        },
        # The default value of max_patches is 16,
        # which results in up to 17 tiles after adding a global tile.
        # Override this value if necessary using:
        # mm_processor_kwargs = {"max_patches": 16}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    stop_token_ids = None

    return ModelRequestData(
        llm_embed=llm_embed,
        llm_gen=llm_gen,
        prompt=prompt,
        image=image,
        stop_token_ids=stop_token_ids,
    )


def run_gemma3(query: Query, args):
    ctx_len = 3072
    seq_len = 128

    model_name = "google/gemma-3-4b-it"

    if query["modality"] == "image+text":
        llm_embed = LLM(
            task="embed",
            model=model_name,
            device_group=args.device_group_embed,
            max_num_seqs=vision_bsz,
            max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
            max_seq_len_to_capture=seq_len,  # seq_len
            enable_prefix_caching=False,
            override_qaic_config={
                "kv_offload": True,
                "mos": 1,
                # If the model produces an incorrect result, you should provide the node precision information file.
                # This file can be found in the quic/efficient-transformers repository under examples/gemma3_example/fp32_nodes_gemma3_4b.yaml.
                # To do this, use:
                # "node_precision_info": "path_to_fp32_nodes_gemma3_4b.yaml"
            },
        )
        image = query["image"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{args.question}"},
                ],
            }
        ]
    elif query["modality"] == "text":
        llm_embed = None
        image = None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{args.question}"},
                ],
            }
        ]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm_gen = LLM(
        model=model_name,
        device_group=args.device_group_gen,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        enable_prefix_caching=False,
        override_qaic_config={
            "kv_offload": True,
            "mos": 1,
            # If the model produces an incorrect result, you should provide the node precision information file.
            # This file can be found in the quic/efficient-transformers repository under examples/gemma3_example/fp32_nodes_gemma3_4b.yaml.
            # To do this, use:
            # "node_precision_info": "path_to_fp32_nodes_gemma3_4b.yaml"
        },
    )

    tokenizer = AutoProcessor.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    stop_token_ids = None

    return ModelRequestData(
        llm_embed=llm_embed,
        llm_gen=llm_gen,
        prompt=prompt,
        image=image,
        stop_token_ids=stop_token_ids,
    )


def get_query(args):
    if args.modality == "image+text":
        # Input image and question
        if args.image_url and urlparse(args.image_url).scheme != "":  # url
            image = Image.open(requests.get(args.image_url, stream=True).raw)
        else:
            image = Image.open(args.file_path)
        return TextImageQuery(
            modality="image+text",
            text=args.question,
            image=image,
        )
    elif args.modality == "text":
        return TextQuery(modality="text", text=args.question)

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def run_encode(args, req_data):
    mm_data = {}
    mm_data["image"] = req_data.image

    # Batch inference
    # Use the same image for all prompts
    inputs = [
        {
            "prompt": req_data.prompt
            if isinstance(req_data.prompt, str)
            else req_data.prompt[0],
            "multi_modal_data": mm_data,
        }
        for _ in range(args.num_prompts)
    ]

    outputs = req_data.llm_embed.encode(inputs, pooling_task="encode")

    embeddings = []
    prompt_ids = []

    for output in outputs:
        embed = output.outputs.data
        print(f"Embedding shape: {embed.shape}")
        embeddings.append(embed)
        prompt_ids.append(output.prompt_token_ids)

    return embeddings, prompt_ids


def main(args: Namespace):
    query = get_query(args)
    req_data = model_example_map[args.model_name](query, args)

    if req_data.image is not None:
        embeddings, prompt_token_ids = run_encode(args, req_data)
        gen_inputs = [
            {
                "prompt": req_data.prompt
                if args.model_name != "llama4"
                # For vllm_gen, it only receives embeddings as the multimodal input,
                # without access to the original images.
                # However, LLaMA 4 requires the formatted aspect ratio—calculated by the HF processor
                # to generate the expanded text prompt based on the input images.
                # Since embeddings alone do not contain aspect ratio information,
                # we will provide the expanded text prompt directly to vllm_gen.
                 else prompt_token_ids[i],
                "multi_modal_data": {
                    "image": embeddings[i]
                } if req_data.image_grid_thw is None
                # For models that use mrope, image_grid_thw is required to calculate positional encoding.
                else {
                    "image": {
                        "image_embeds": embeddings[i],
                        "image_grid_thw": req_data.image_grid_thw
                    }
                }
            } for i in range(args.num_prompts)
        ]
    else:
        gen_inputs = [{"prompt": req_data.prompt} for i in range(args.num_prompts)]

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=None, stop_token_ids=req_data.stop_token_ids
    )

    outputs = req_data.llm_gen.generate(
        gen_inputs,
        sampling_params=sampling_params,
    )

    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)


model_example_map = {
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "mllama": run_mllama,
    "llama4": run_llama4,
    "gemma3": run_gemma3,
    "qwenvl": run_qwen_2_5_vl,
}

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for multimodal embedding"
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="internvl_chat",
        choices=model_example_map.keys(),
        help="Huggingface model name.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prompts to run."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image+text",
        choices=get_args(QueryModality),
        help="Modality of the input.",
    )
    parser.add_argument("--image-url", type=str, help="Url for image input.")
    parser.add_argument(
        "--file-path", type=str, help="File path for local image file, e.g. rabbit.jpg"
    )
    parser.add_argument("--question", type=str, help="Question about the image.")
    parser.add_argument(
        "--device-group-embed",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[0],
        help="Define qaic device ids for embed task of kv_offload mode in csv format (e.g.,"
        "--device-id 0,1,2,3).",
    )
    parser.add_argument(
        "--device-group-gen",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[1],
        help="Define qaic device ids for generate task of kv_offload mode in csv format (e.g.,"
        "--device-id 4,5,6,7).",
    )
    args = parser.parse_args()
    if args.image_url is None and args.file_path is None:
        args.modality == "text"
    main(args)
