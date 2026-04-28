# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


# Set example specific arguments
# For qaic pooling with model="intfloat/e5-large" pass 'override_qaic_config={"pooling_device":"qaic", "pooling_method":"mean", "normalize": True, "softmax": False}'
# For qaic pooling with model="jinaai/jina-embeddings-v2-base-code" pass 'override_qaic_config={"pooling_device":"qaic", "pooling_method":"mean", "normalize": False, "softmax": False}'
# For cpu pooling pass 'override_qaic_config={"pooling_device":"cpu"}'. The pooling method will be chosen based on HF modules.json for the model. To use pooling methods other than the ones set by model config add 'from vllm.config import PoolerConfig' and pass preferred pooling method in the following format: 'override_pooler_config=PoolerConfig(pooling_type="LAST", normalize=True)'
# For no pooling do not pass any override_qaic_config: Use model.encode instead of model.embed
# To compile for multiple sequence lengths pass multi_seq_lens in the following format: 'override_qaic_config={"multi_seq_lens":[32,512]}'. Always include max_model_len.

def print_embeds(prompts, outputs):
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = ((str(embeds[:16])[:-1] +
                           ", ...]") if len(embeds) > 16 else embeds)
        print(f"Prompt: {prompt!r} \n"
              f"Embeddings: {embeds_trimmed} (size={len(embeds)})")
        print("-" * 60)

def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
    ]*10
    # Create an LLM.
    # You should pass runner="pooling" for embedding models

    print("running single specialization with pooling on cpu") #seq len will be set to max_model_len
    model = LLM(model="intfloat/e5-large",
                runner="pooling",
                enforce_eager=True,
                max_num_seqs=4,
                override_qaic_config={"pooling_device":"cpu"})

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = model.embed(prompts)

    # Print the outputs.
    print_embeds(prompts, outputs)


    print("running multi specialization with pooling on qaic")
    model = LLM(model="intfloat/e5-large",
                runner="pooling",
                enforce_eager=True,
                max_num_seqs=4,
                override_qaic_config={"pooling_device":"qaic", "pooling_method":"avg", "embed_seq_len":[32,512]})  ## always send max model len from override

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = model.embed(prompts)

    # Print the outputs.
    print_embeds(prompts, outputs)


if __name__ == "__main__":
    main()