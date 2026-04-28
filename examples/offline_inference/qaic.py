# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from vllm import LLM, SamplingParams
import random
import time

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
seq_len = 128
decode_bsz = 16

# Create a LLM.
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    device_group=[0],
    max_num_seqs=decode_bsz, # determines decode batch size
    max_model_len=ctx_len, # ctx_len (does not account for padding, but does account for prompt and generated tokens)
    max_seq_len_to_capture=seq_len, # seq_len
    quantization="mxfp6", # Preferred quantization
    kv_cache_dtype="mxint8", # Preferred option to same KV cache and increase performance
    disable_log_stats=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=1.0,
    speculative_config={
            "model": "ngram",
            "num_speculative_tokens": 5
        }
    )

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print(f"Generating for {len(prompts)} prompts...")
start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()

total_output_tokens = 0
total_input_tokens = 0
ttfts = []
tpots = []

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_tokens = output.outputs[0].token_ids
    num_generated_tokens = len(generated_tokens)
    num_prompt_tokens = len(output.prompt_token_ids)
    
    total_output_tokens += num_generated_tokens
    total_input_tokens += num_prompt_tokens
    
    metrics = output.metrics
    if metrics:
        if metrics.first_token_time is not None:
            ttft = metrics.first_token_time - metrics.arrival_time
            ttfts.append(ttft)
            if num_generated_tokens > 1:
                tpot = (metrics.finished_time - metrics.first_token_time) / (num_generated_tokens - 1)
                tpots.append(tpot)

    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Num generated tokens: {num_generated_tokens!r}")

total_time = end_time - start_time
print("\n" + "="*40)
print("PERFORMANCE SUMMARY")
print("="*40)
print(f"Total requests:          {len(outputs)}")
print(f"Total input tokens:      {total_input_tokens}")
print(f"Total generated tokens:  {total_output_tokens}")
print(f"Total time:              {total_time:.2f} s")
print(f"Throughput (output):     {total_output_tokens / total_time:.2f} tokens/s")
print(f"Throughput (total):      {(total_input_tokens + total_output_tokens) / total_time:.2f} tokens/s")

if ttfts:
    print(f"Average TTFT:            {sum(ttfts) / len(ttfts) * 1000:.2f} ms")
if tpots:
    print(f"Average TPOT:            {sum(tpots) / len(tpots) * 1000:.2f} ms")
print("="*40)
