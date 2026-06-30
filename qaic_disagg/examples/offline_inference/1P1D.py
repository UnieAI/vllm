# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances,
and then transfer the KV cache between them.
"""
import os
import subprocess
import time
import argparse
import sys
from multiprocessing import Event, Process, Queue, set_start_method

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# SIZE = 15360
NUMBER_OF_PROMPTS = 1

# PORT = 5656

prefill_device_group = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

decode_device_group = [16,17,18,19,20,21,22,23]  # For 8x A100 GPUs

prompts = []
thread_failure = False  # Global variable to track thread failure

def parse_args():
    parser = argparse.ArgumentParser(description="Disaggregated Prefill/Decode Example")
    parser.add_argument("--port", type=int, default=5656, help="Port ID for KV transfer")

    # Mutually exclusive group for prefill
    prefill_group = parser.add_mutually_exclusive_group(required=True)
    prefill_group.add_argument("--prefill_qpc", type=str, help="Path to prefill QPC file")
    prefill_group.add_argument("--prefill_mdp", type=str, help="Path to prefill MDP file")

    # Mutually exclusive group for decode
    decode_group = parser.add_mutually_exclusive_group(required=True)
    decode_group.add_argument("--decode_qpc", type=str, help="Path to decode QPC file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # # Use arguments
    PORT = args.port
    PREFILL_QPC_PATH = args.prefill_qpc
    DECODE_QPC_PATH = args.decode_qpc
    PREFILL_MDP_PATH = args.prefill_mdp

    def get_override_qaic_config(prefill_only, stages, **kwargs):
        config = {
            "prefill_only": prefill_only,
            "stages": stages,
            'allow-mxint8-mdp-io': True,
            'split-retained-state-io': True,
            **kwargs,
        }
        if prefill_only and PREFILL_QPC_PATH is not None:
            config["qpc_path"] = PREFILL_QPC_PATH
        if not prefill_only and DECODE_QPC_PATH is not None:
            config["qpc_path"] = DECODE_QPC_PATH
        if prefill_only and PREFILL_MDP_PATH is not None:
            config["mdp_load_partition_config"] = PREFILL_MDP_PATH
        return config

    def run_prefill(prefill_done):
        global prompts, thread_failure

        try:
            while True:
                sampling_params = SamplingParams(temperature=0, max_tokens=1)

                # TODO: Change it to QAIC's implementation
                # Using PyNcclConnector to transmit KV caches between vLLM instances.
                # This instance is the prefill node (kv_producer, rank 0).
                # The number of parallel instances for KV cache transfer is set to 2,
                # as required for PyNcclConnector.
                ktc = KVTransferConfig.from_cli(
                    f'{{"kv_connector":"QaicConnector","kv_role":"kv_producer","kv_rank":0,"kv_port":{PORT}}}'
                )

                llm = LLM(
                    # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    model="meta-llama/Llama-3.3-70B-Instruct",
                    kv_transfer_config=ktc,
                    max_num_seqs=1,  # determines decode batch size
                    max_seq_len_to_capture=256,  # seq_len
                    max_model_len=2048,
                    disable_log_stats=True,
                    device_group=prefill_device_group,
                    enable_prefix_caching=False,
                    gpu_memory_utilization=1.0,
                    override_qaic_config=get_override_qaic_config(True, 16)
                )
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                end = time.perf_counter()
                print("Prefill node is finished.")
                i = 0
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                    print(f"Prompt: {i}, Generated text: {generated_text}")
                    i +=1
                print(f"Total prefill time: {end - start} seconds")
                prefill_done.set()

                break  # Exit after one round of prefill

        except Exception as e:
            thread_failure = True
            print(f"An error occurred in the prefill process: {e}", flush=True)
            prefill_done.set()
            sys.exit(1)


    def run_decode(prefill_done):
        global prompts, thread_failure
        try:
            while True:
                sampling_params = SamplingParams(temperature=0, max_tokens=220)

                # Using PyNcclConnector to transmit KV caches between vLLM instances.
                # This instance is the decode node (kv_consumer, rank 1).
                # The number of parallel instances for KV cache transfer is set to 2,
                # as required for PyNcclConnector.
                ktc = KVTransferConfig.from_cli(
                    f'{{"kv_connector":"QaicConnector","kv_role":"kv_consumer","kv_rank":1,"kv_port":{PORT}}}'
                )

                # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
                # memory. You may need to adjust the value to fit your GPU.
                llm = LLM(
                    # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    model="meta-llama/Llama-3.3-70B-Instruct",
                    kv_transfer_config=ktc,
                    max_num_seqs=1,  # determines decode batch size
                    max_seq_len_to_capture=256,  # seq_len
                    max_model_len=2048,
                    disable_log_stats=True,
                    device_group=decode_device_group,
                    enable_prefix_caching=False,
                    gpu_memory_utilization=1.0,
                    override_qaic_config=get_override_qaic_config(
                        prefill_only=False,
                        stages=1,
                        aic_include_sampler=True,
                        aic_return_pdfs=False,
                        max_top_k_ids=512,
                    )
                )

                # Wait for the producer to start the pipe
                print("Waiting for prefill node to finish...")
                prefill_done.wait()

                # transferred to this decode node, so we can start decoding.
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                end = time.perf_counter()
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                    print(f"Generated text: {generated_text}")

                print(f"Total time: {end - start} seconds")

                break  # Exit after one round of decoding

        except Exception as e:
            thread_failure = True
            print(f"An error occurred in the Decode process: {e}", flush=True)
            time.sleep(1)  # Give some time for the error message to be printed
            sys.exit(1)

    def run_qaic_cache_server(port):
        server_proc = subprocess.Popen([
            "python", "-m", "qaic_disagg.kv_handoff.server", "--port", str(port) , "--size", "64"
        ])
        return server_proc

    prompts = [
        "My name is ",
    ] * NUMBER_OF_PROMPTS

    prefill_done = Event()

    server_process = run_qaic_cache_server(PORT)

    time.sleep(10)

    prefill_process = Process(target=run_prefill, args=(prefill_done, ))
    decode_process = Process(target=run_decode, args=(prefill_done, ))

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()
    # prefill_process.join()

    print("Prefill process PID: ", prefill_process.pid)
    print("Decode process PID: ", decode_process.pid)

    while True:
        if thread_failure:
            print("Exiting the app due to thread failure.", flush=True)
            # prefill_process.join()
            # decode_process.join()
            sys.exit(1)
            break
        if not prefill_process.is_alive():
            print("Prefill process is done...")
        if not decode_process.is_alive():
            print("decode process is done...")
            prefill_process.terminate()
            break
        if not prefill_process.is_alive() and not decode_process.is_alive():
            print("Exiting the app as both threads have completed.")
            break
        time.sleep(1)  # Check for failure periodically

    server_process.terminate()
    while server_process.poll() is None:
        time.sleep(1)

    # # Terminate the prefill node when decode is finished
    # decode_process.join()
