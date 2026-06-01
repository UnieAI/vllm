"""QaicWorker — PORT THIS from the fork.

Source: /tmp/unie_vllm/vllm/v1/worker/qaic_worker.py  (254 lines, by Qualcomm)
        (on your machine: vllm/v1/worker/qaic_worker.py in the v1_ngram fork)

It subclasses vllm.v1.worker.worker_base.WorkerBase and overrides:
    init_device(), load_model(), initialize_cache(), get_kv_cache_spec(),
    execute_model(), ...

PORTING STEPS (see README PART 1, step 3):
  1. Copy the fork file here as worker.py.
  2. Change the import of the model runner from
        vllm.v1.worker.qaic_model_runner.QaicModelRunner
     to
        vllm_qaic.model_runner.QaicModelRunner
  3. Diff WorkerBase between the fork (v0.10.1) and your installed vLLM; fix
     any changed __init__ / method signatures.
"""

raise NotImplementedError(
    "Port vllm/v1/worker/qaic_worker.py from the v1_ngram fork into this file. "
    "See the module docstring and README PART 1, step 3."
)
