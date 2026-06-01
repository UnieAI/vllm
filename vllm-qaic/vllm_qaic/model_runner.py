"""QaicModelRunner — PORT THIS from the fork. *** THE HARD PART ***

Source: /tmp/unie_vllm/vllm/v1/worker/qaic_model_runner.py
        (Qualcomm base, 619 lines) + UnieAI's ngram additions (+222 lines:
        the 7 _qaic_rejection_sample* functions, the 2D decode packing in
        execute_model, and the __init__ ngram gating).

WHY THIS IS THE HARD PART
-------------------------
QaicModelRunner subclasses vllm.v1.worker.gpu_model_runner.GPUModelRunner.
GPUModelRunner was ~2-3k lines in v0.10.1; in vLLM 0.21+ it is 7000+ lines and
heavily refactored (execute_model, _prepare_inputs, KV-cache spec, sampler
wiring all changed). NO amount of plugin machinery fixes this — every override
must be re-fitted by hand against the installed GPUModelRunner.

PORTING STEPS (see README PART 1, step 4):
  1. Copy the fork file here as model_runner.py.
  2. Fix the model-loader import to vllm_qaic.model_loader.load_qaic_model.
  3. Open the *installed* vllm/v1/worker/gpu_model_runner.py and reconcile,
     method by method, every override (start with execute_model, load_model,
     get_kv_cache_spec, initialize_kv_cache, __init__).
  4. The ngram pieces (UnieAI's ~250 lines) are self-contained CPU code and
     port almost verbatim — do them LAST, after the base path runs.

DO IT INCREMENTALLY: first get a non-speculative prefill/decode working, then
re-enable ngram. (See README PART 3.)
"""

raise NotImplementedError(
    "Port vllm/v1/worker/qaic_model_runner.py from the v1_ngram fork and "
    "re-align its GPUModelRunner overrides against your installed vLLM. "
    "See the module docstring and README PART 1, step 4."
)
