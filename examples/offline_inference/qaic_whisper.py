# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import time

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import librosa

# define qpc parameters
decode_bsz = 1
device_group = [0]

encoder_ctx_len = 1500 # The maximum length of context for encoder
ctx_len = 150 # The maximum length of context to keep for decoding
seq_len = 1

def main(args):
    # Create a Whisper encoder/decoder model instance
    llm = LLM(
        model="openai/whisper-tiny.en",
        device_group=device_group,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len, # ctx_len (does not account for padding, but does account for prompt and generated tokens)
        max_seq_len_to_capture=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
        hf_overrides={"max_source_positions": encoder_ctx_len},
    )

    audio = librosa.load(args.file_path, sr=None)

    prompt = {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": audio
        }
    }

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200,
    )

    start = time.time()

    # Generate output tokens from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated
    # text, and other information.
    outputs = llm.generate(prompt, sampling_params)

    # Print the outputs.
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    duration = time.time() - start

    print("Duration:", duration)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "whisper model for speech to text"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        help="File path for local audio file, e.g. audio.wav",
    )

    args = parser.parse_args()
    if args.file_path is None:
        parser.print_help()
        raise ValueError(f"File-path must be provided.")
    main(args)