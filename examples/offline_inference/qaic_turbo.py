# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""
turbo
"""
from vllm import LLM, SamplingParams
import random
import argparse
import re

parser = argparse.ArgumentParser(description="QAIC TurboLoRA example")
parser.add_argument("--model",
                        type=str,
                        required=True, help="HuggingFace model card or path to checkpoint directory of the TurboLoRA model to run")
parser.add_argument('--device-group',
                    type=lambda device_ids: [int(x) for x in device_ids.split(",")],
                    default=[0],
                    help= 'Define qaic device ids in csv format (e.g.,'
                    '--device-id 0,1,2).')
parser.add_argument("--override-qaic-config",
        type=lambda configs: {
            str(value[0]): value[1] if len(value) > 1 else True
            for value in (
                re.split(r'[:=]', config.strip())
                for config in re.split(r'[ ]+', configs.strip())
            )
        },
        default=None,
    )
args = parser.parse_args()
# Sample prompts.
prompts = ["My name is"]

# Create a sampling params object.
# Only Greedy Sampling (temperature < 1e-5) or
# Random sampling with best_of==1 is supported.
# best_of >1 or beam search not supported in current qaic implementaion
# Beam search not supported.
sampling_params = SamplingParams(temperature=0.0)

# define qpc parameters
seq_len = 128
decode_bsz = 4
ctx_len = 256 # block size is equal to ctx_len

print("Running TurboLoRA inference....\n")
# Create a LLM.
llm = LLM(
    model=args.model,
    device_group=args.device_group,
    max_num_seqs=decode_bsz, # determines decode batch size
    max_model_len=ctx_len, # ctx_len (does not account for padding, but does account for prompt and generated tokens)
    max_seq_len_to_capture=seq_len, # seq_len
    quantization="mxfp6", # Preferred quantization
    kv_cache_dtype="mxint8", # Preferred option to same KV cache and increase performance
    disable_log_stats=False,
    speculative_config = dict(
        method="turbo",
        acceptance_method="typical_acceptance_sampler" # typical acceptance with temperature=0.0 ensures that turbo distribution matches model without projections
    ),
    override_qaic_config=args.override_qaic_config,
    gpu_memory_utilization=1.0,
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
