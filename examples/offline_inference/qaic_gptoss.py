# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from vllm import LLM, SamplingParams
import random

# Sample prompts.
prompts = [
    'My name is',
    # Add more prompts here
] * 100

random.shuffle(prompts)
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)
# Only Greedy Sampling (temperature < 1e-5) or
# Random sampling with best_of==1 is supported.
# best_of >1 or beam search not supported in current qaic implementaion
# Beam search not supported.

# define qpc parameters
ctx_len = 256
seq_len = 1
decode_bsz = 1

# Create a LLM.
llm = LLM(
    model="openai/gpt-oss-20b",
    device_group=[0,1,2,3,4,5,6,7],
    max_num_seqs=decode_bsz, # determines decode batch size
    max_model_len=ctx_len, # ctx_len (does not account for padding, but does account for prompt and generated tokens)
    max_seq_len_to_capture=seq_len, # seq_len
    # quantization="mxfp6", # Preferred quantization
    # kv_cache_dtype="mxint8", # Preferred option to same KV cache and increase performance
    disable_log_stats=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=1.0,
    override_qaic_config={'continuous_batching':False, 'mos':1},
    dtype='bfloat16',
    )
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_tokens = output.outputs[0].token_ids
    num_generated_tokens = len(generated_tokens)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Num generated tokens: {num_generated_tokens!r}")