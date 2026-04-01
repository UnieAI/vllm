CUDA_VISIBLE_DEVICES=2 vllm serve /modelz/Qwen/Qwen3-8B \
--served-model-name test \
--host 0.0.0.0 \
--port 8976 \
--tensor-parallel-size 1 \
--trust-remote-code \
--dtype float16 \
--async-scheduling \
--speculative-config '{"method":"ngram_dsc","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"prompt_lookup_min":3,"prompt_lookup_max":8}'
