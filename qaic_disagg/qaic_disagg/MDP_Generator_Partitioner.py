# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Utility to generate MDP files."""

import argparse
import os
import subprocess
from QEfficient import QEFFAutoModelForCausalLM
from pathlib import Path

if os.environ.get("HF_TOKEN"):
    hf_token = os.environ.get("HF_TOKEN")
else:
    raise OSError("HF_TOKEN is not set")


def _run_partitioner(
    dump_json_path: str,
    output_json_path: str,
    num_devices: int,
    num_partitions: int,
    layers_per_partition: int,
):
    """
    Runs the Pipeline_partitioner.py script to partition a dumped MDP JSON.
    """
    # Step 2: Partition the dumped JSON using Pipeline_partitioner.py
    partition_cmd = [
        "python3",
        os.path.join(os.path.dirname(__file__), "Pipeline_partitioner.py"),
        dump_json_path,
        output_json_path,
        str(num_devices),
        str(num_partitions),
        str(layers_per_partition),
    ]
    print("Running Pipeline_partitioner.py...")
    subprocess.run(partition_cmd, check=True)
    print(f"Partitioned JSON written to {output_json_path}")


def generate_mdp_with_qeff(
    model_name: str,
    num_devices: int,
    num_partitions: int,
    output_json_path: str,
    prefill_seq_len: int = 32,
    ctx_len: int = 128,
    prefill_max_num_seqs: int = 1,
    use_mxfp6_matmul: bool = False,
    use_mxint8_kv_cache: bool = False,
    trust_remote_code: bool = False,
    continuous_batching: bool = True,
):
    """
    Generates an MDP JSON using QEfficient and then partitions it.
    """
    assert num_partitions <= num_devices, (
        "num_partitions must be less than or equal to num_devices"
    )
    out_dir = Path(output_json_path).absolute().parent
    out_dir.mkdir(exist_ok=True)
    temp_dump_json_path = str(out_dir / "tmp_mdp_dump_partition.json")

    model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name,
        continuous_batching=continuous_batching,
        token=hf_token,
        attn_implementation="eager",
        trust_remote_code=trust_remote_code,
    )

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=prefill_max_num_seqs if continuous_batching else None,
        mdp_dump_partition_config=temp_dump_json_path,
        prefill_only=True,
        mxfp6_matmul=use_mxfp6_matmul,
        mxint8_kv_cache=use_mxint8_kv_cache,
    )

    layers_per_partition = model.num_layers // num_partitions

    _run_partitioner(
        temp_dump_json_path,
        output_json_path,
        num_devices,
        num_partitions,
        layers_per_partition,
    )
    if os.path.exists(temp_dump_json_path):
        os.remove(temp_dump_json_path)


def get_parser():
    """
    Parses command-line arguments for MDP generation and partitioning.
    """
    parser = argparse.ArgumentParser(
        description="Generates Pipeline partitioned MDP based on Huggingface model id."
    )

    parser.add_argument(
        "model_name",
        type=str,
        help="HuggingFace model name e.g., 'meta-llama/Llama-3.1-8B-Instruct'",
    )
    parser.add_argument(
        "num_devices",
        type=int,
        help="Number of devices available for partitioning.",
    )
    parser.add_argument(
        "num_partitions",
        type=int,
        help="Desired number of partitions for the model.",
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Path to the output JSON file where partitioned MDP will be saved.",
    )
    parser.add_argument(
        "--prefill_seq_len",
        type=int,
        default=32,
        help="Prefill sequence length for QEfficient model compilation.",
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=128,
        help="Context length for QEfficient model compilation.",
    )
    parser.add_argument(
        "--prefill_max_num_seqs",
        type=int,
        default=1,
        help="Max number of prefill sequences for QEfficient model compilation.",
    )
    parser.add_argument(
        "--mxfp6_matmul",
        action="store_true",
        help="Enable MXFP6 matmul for QEfficient model compilation.",
    )
    parser.add_argument(
        "--mxint8_kv_cache",
        action="store_true",
        help="Enable MXINT8 KV cache for QEfficient model compilation.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allows execution of arbitrary Python code from a remote  \
            Hugging Face repository. Use only with trusted sources \
            due to security risks.",
    )
    parser.add_argument(
        "--disable_continuous_batching",
        action="store_true",
        help="Disable continuous batching during QEfficient model compilation.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    generate_mdp_with_qeff(
        args.model_name,
        args.num_devices,
        args.num_partitions,
        args.output_json,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        prefill_max_num_seqs=args.prefill_max_num_seqs,
        use_mxfp6_matmul=args.mxfp6_matmul,
        use_mxint8_kv_cache=args.mxint8_kv_cache,
        trust_remote_code=args.trust_remote_code,
        continuous_batching=not args.disable_continuous_batching,
    )
