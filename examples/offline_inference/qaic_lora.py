
# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from huggingface_hub import snapshot_download
# prompts for different adapters
prompts = [
    'My name is',
    # Add more prompts for different adapters here
] * 9
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=None)
# adapters
repo_id = [
    "predibase/gsm8k",
    "predibase/tldr_content_gen",
    "predibase/agnews_explained",
    "predibase/e2e_nlg",
    "predibase/viggo",
    "predibase/hellaswag_processed",
    "predibase/bc5cdr",
    "predibase/conllpp",
    "predibase/tldr_headline_gen",
]
# define qpc parameters
ctx_len = 256
seq_len = 128
decode_bsz = 4
# Create a LLM.
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    device_group=[0],
    max_num_seqs=decode_bsz,
    max_model_len=ctx_len,
    max_seq_len_to_capture=seq_len,
    quantization="mxfp6",
    kv_cache_dtype="mxint8",
    # disable_log_stats=False,
    gpu_memory_utilization=1.0,
    enable_lora=True,
    max_loras=decode_bsz,  # maxium loras to run within a batch
    lora_modules=[
        LoRAModulePath(
            name=repo_id[i].split("/")[1],
            path=snapshot_download(repo_id=repo_id[i]),
        )
        for i in range(len(repo_id))
    ],
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
lora_requests = [
    LoRARequest(
        lora_name=repo_id[i].split("/")[1], lora_int_id=(i + 1), lora_path=snapshot_download(repo_id=repo_id[i])
    )
    for i in range(len(prompts))
]
outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_tokens = output.outputs[0].token_ids
    num_generated_tokens = len(generated_tokens)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Num generated tokens: {num_generated_tokens!r}"
    )