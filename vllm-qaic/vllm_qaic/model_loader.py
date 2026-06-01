"""Model loader — PORT THIS from the fork. (Low risk: mostly self-contained.)

Sources (copy both into this package):
  * vllm/model_executor/model_loader/qaic_v1.py   -> model_loader.py (this file)
        QaicCausalLM (nn.Module shell) + load_qaic_model(vllm_config, ...)
  * vllm/model_executor/model_loader/qaic.py      -> compile_config.py
        _clean_config() + QEfficient compile-config assembly (num_cores,
        prefill_seq_len, mxfp6_matmul, mxint8_kv_cache, ...).

These talk to QEfficient / qaicrt, not to vLLM internals, so they port with
little change. The one edit: read QAIC knobs from vllm_config.additional_config
(set by QaicPlatform.check_and_update_config) instead of
vllm_config.model_config.override_qaic_config.

SPLIT-ENV NOTE (see README PART 0): the QPC *compile* path imports
`from QEfficient import QEFFAutoModelForCausalLM` and needs torch 2.7. In the
serve environment (torch 2.11) you must hit the *pre-compiled* path only
(pass qpc_path / VLLM_QAIC_QPC_PATH) so this import is never reached. Loading a
prebuilt QPC uses only qaicrt (see session.py).
"""

raise NotImplementedError(
    "Port vllm/model_executor/model_loader/qaic_v1.py (+ qaic.py) from the "
    "v1_ngram fork. See the module docstring and README PART 1, step 5."
)
