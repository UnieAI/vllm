"""Model loader — PORT from the fork via scripts/port_from_fork.sh.

This file is a thin, import-SAFE placeholder so the rest of the package imports
cleanly before porting. Running scripts/port_from_fork.sh OVERWRITES it with the
fork's qaic_v1.py (QaicCausalLM + load_qaic_model), import-rewritten.

Sources (copied by the port script):
  * vllm/model_executor/model_loader/qaic_v1.py  -> this file (model_loader.py)
  * vllm/model_executor/model_loader/qaic.py     -> compile_config.py
  * vllm/model_executor/model_loader/qaic_session_np.py -> session.py

These talk to QEfficient / qaicrt, not vLLM internals, so they port with little
change beyond the two manual edits the port script prints (override_qaic_config
-> additional_config; is_qaic() -> device_type=="qaic").

SPLIT-ENV NOTE (README PART 0): the compile path imports
`from QEfficient import QEFFAutoModelForCausalLM` and needs torch 2.7. In the
serve env (torch 2.11) hit the PRE-COMPILED path only (pass qpc_path /
VLLM_QAIC_QPC_PATH) so that import is never reached.
"""

from typing import Any


def load_qaic_model(vllm_config: Any, speculative_model_type: Any = None):
    raise NotImplementedError(
        "vllm_qaic.model_loader is a placeholder. Run "
        "scripts/port_from_fork.sh <fork> to bring over qaic_v1.py, then apply "
        "the two manual adaptations it prints.")
