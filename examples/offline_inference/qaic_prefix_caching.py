# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from time import time

from vllm import LLM, SamplingParams

# Common prefix.
prefix = ("") # Add common prefix

# Sample prompts.
# Add prompts here
prompts = [
    "My name is"
]

generating_prompts = [prefix + prefix + prefix + prefix + prompt for prompt in prompts]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

ctx_len = 2048
seq_len = 128
decode_bsz = 16
# Create a LLM.

regular_llm = LLM(  model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    gpu_memory_utilization=1.0,
                    device_group=[0],
                    max_num_seqs=decode_bsz, # determines decode batch size
                    max_model_len=ctx_len, # ctx_len (does not account for padding, but does account for prompt and generated tokens)
                    max_seq_len_to_capture=seq_len, # seq_len
                    quantization="mxfp6", # Preferred quantization
                    kv_cache_dtype="mxint8", # Preferred option to same KV cache and increase performance
                    disable_log_stats=False,
                    enable_prefix_caching=False,)

print("Results without `enable_prefix_caching`")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start_time_regular = time()
outputs = regular_llm.generate(generating_prompts, sampling_params)
outputs = regular_llm.generate(generating_prompts, sampling_params)
outputs = regular_llm.generate(generating_prompts, sampling_params)
outputs = regular_llm.generate(generating_prompts, sampling_params)
duration_regular = time() - start_time_regular

regular_generated_texts = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    regular_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)
del regular_llm
import gc
gc.collect()

prefix_cached_llm = LLM(
                        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        enable_prefix_caching=True,
                        gpu_memory_utilization=1.0,
                        device_group=[0],
                        max_num_seqs=decode_bsz, # determines decode batch size
                        max_model_len=ctx_len, # ctx_len (does not account for padding, but does account for prompt and generated tokens)
                        max_seq_len_to_capture=seq_len, # seq_len
                        quantization="mxfp6", # Preferred quantization
                        kv_cache_dtype="mxint8", # Preferred option to same KV cache and increase performance
                        disable_log_stats=False,
                        num_gpu_blocks_override=16*4
                        )

# Warmup so that the shared prompt's KV cache is computed.
prefix_cached_llm.generate(generating_prompts[0], sampling_params)

# Generate with prefix caching.
start_time_cached = time()
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
duration_cached = time() - start_time_cached

print("Results with `enable_prefix_caching`")

cached_generated_texts = []
# Print the outputs. You should see the same outputs as before.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    cached_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Compare the results and display the speedup
generated_same = all([
    regular_generated_texts[i] == cached_generated_texts[i]
    for i in range(len(prompts))
])
print(f"Generated answers are the same: {generated_same}")

speedup = round(duration_regular / duration_cached, 2)
print(duration_regular, duration_cached)
print(f"Speed up of cached generation compared to the regular is: {speedup}")
