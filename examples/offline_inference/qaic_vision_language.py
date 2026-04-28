# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

from transformers import AutoTokenizer,AutoProcessor

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

from PIL import Image
import requests
from urllib.parse import urlparse


# define qpc parameters
# Multimodal models do not support batching yet,
# so decode_bsz must be 1
decode_bsz = 1

# LLaVA-1.5
def run_llava(question: str, modality: str, args):
    if modality == "image+text":
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
    elif modality == "text":
        prompt = f"USER: {args.question}\nASSISTANT:"
    else:
        raise ValueError(f"Unsupported query modality: '{modality}'")

    ctx_len = 2048
    seq_len = 128

    model_name = "llava-hf/llava-1.5-7b-hf"

    llm = LLM(
        model=model_name,
        device_group=args.device_group,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        enable_prefix_caching=False,
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question: str, modality: str, args):
    if modality == "image+text":
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
    elif modality == "text":
        messages = [{"role": "user", "content": f"{question}"}]
    else:
        raise ValueError(f"Unsupported query modality: '{modality}'")

    ctx_len = 4096
    seq_len = 128

    model_name = "OpenGVLab/InternVL2_5-1B"

    llm = LLM(
        model=model_name,
        device_group=args.device_group,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        enable_prefix_caching=False,
        trust_remote_code=True,
        override_qaic_config={"num_patches": 13},  # In QAic, num_patches is fixed
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
    return llm, prompt, stop_token_ids


# LLama 3.2
def run_mllama(question: str, modality: str, args):
    if modality == "image+text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    elif modality == "text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    else:
        raise ValueError(f"Unsupported query modality: '{modality}'")

    ctx_len = 512
    seq_len = 32

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    llm = LLM(
        model=model_name,
        device_group=args.device_group,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        enable_prefix_caching=False,
        gpu_memory_utilization=1.0,
        override_qaic_config={"dfs": False},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Llama 4
def run_llama4(question: str, modality: str, args):
    if modality == "image+text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    elif modality == "text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    else:
        raise ValueError(f"Unsupported query modality: '{modality}'")

    ctx_len = 3072
    seq_len = 128

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    llm = LLM(
        model=model_name,
        device_group=args.device_group,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        enable_prefix_caching=False,
        gpu_memory_utilization=1.0,
        # The default value of max_patches is 16,
        # which results in up to 17 tiles after adding a global tile.
        # Override this value if necessary using:
        # mm_processor_kwargs = {"max_patches": 16}
        # Llama 4 can be configured as a text-only model by disabling multimodal processing.
        # This can be done by setting:
        # override_qaic_config = {
        #     "disable_multimodal": True,
        # }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


def run_gemma3(question: str, modality: str, args):
    if modality == "image+text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    elif modality == "text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    else:
        raise ValueError(f"Unsupported query modality: '{modality}'")

    ctx_len = 1024
    seq_len = 128
    model_name = "google/gemma-3-4b-it"

    llm = LLM(
        model=model_name,
        device_group=args.device_group,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to same KV cache and increase performance
        # The device can be automatically detected when Qaic Apps SDK is installed.
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        enable_prefix_caching=False,
        # if gemma3 generates garbage output we need to pass 'node precision inforrmation' file for the model.
        # The npi file location within qefficient : efficient-transformers/examples/gemma3_example/fp32_mm.yaml
        # This can be done by setting:
        # override_qaic_config = {
        #     "node_precision_info": "path_to_fp32_mm.yaml"
        # }
    )

    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

model_example_map = {
    "llava": run_llava,
    "internvl_chat": run_internvl,
    "mllama": run_mllama,
    "llama4": run_llama4,
    "gemma3": run_gemma3,
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    img_question = args.question
    if args.modality == "image+text":
        # Input image and question
        if args.image_url and urlparse(args.image_url).scheme != "":  # url
            image = Image.open(requests.get(args.image_url, stream=True).raw)
        else:
            image = Image.open(args.file_path)

        return {
            "data": image,
            "question": img_question,
        }
    elif args.modality == "text":
        return {
            "data": None,
            "question": img_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def main(args):
    model = args.model_name
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = model_example_map[model](question, modality, args)

    # We set temperature to 0.0 so that outputs can be consistent
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=None, stop_token_ids=stop_token_ids
    )

    assert args.num_prompts > 0
    if data is not None:
        # Batch inference
        # Use the same image for all prompts
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": data},
            }
            for _ in range(args.num_prompts)
        ]
    else:
        inputs = [
            {
                "prompt": prompt,
            }
            for _ in range(args.num_prompts)
        ]

    if args.time_generate:
        import time

        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
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
        choices=["image+text", "text"],
        help="Modality of the input.",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        help="Url for image input.",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        help="File path for local multimodal file, e.g. rabbit.jpg",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question about the image.",
    )
    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal preprocessor/mapper.",
    )
    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[0],
        help="Define qaic device ids in csv format (e.g.," "--device-id 0,1,2,3).",
    )

    args = parser.parse_args()
    if args.image_url is None and args.file_path is None:
        args.modality == "text"
    if args.question is None:
        parser.print_help()
        raise ValueError(f"Question cannot be empty.")
    main(args)
