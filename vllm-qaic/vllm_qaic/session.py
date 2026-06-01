"""qaicrt session — PORT from the fork via scripts/port_from_fork.sh.

Import-SAFE placeholder. The port script OVERWRITES this with the fork's
vllm/model_executor/model_loader/qaic_session_np.py (657 lines, BSD-3-Clause,
pure numpy + qaicrt + QAicApi_pb2 — NO torch dependency).

This is the host<->device boundary: it wraps qaicrt (Context / Queue / Qpc /
Program / ExecObj) and runs a pre-compiled QPC on the AIC100. Because it does
not import torch, it runs fine under the serve env's torch 2.11 — this is the
technical basis of the split-environment plan (README PART 0). The GO test
tests/gate2_load_qpc.py imports the fork file directly to prove this.
"""


class QAICInferenceSession:  # noqa: D401 - placeholder
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "vllm_qaic.session is a placeholder. Run scripts/port_from_fork.sh "
            "<fork> to bring over qaic_session_np.py.")
