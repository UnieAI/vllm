"""qaicrt session — PORT THIS from the fork. (Pure numpy + qaicrt, no torch.)

Source: vllm/model_executor/model_loader/qaic_session_np.py  (657 lines)

This is the host<->device boundary: it wraps qaicrt (Context / Queue / Qpc /
Program / ExecObj) and runs a pre-compiled QPC on the AIC100. It depends only
on numpy, qaicrt and QAicApi_pb2 — NOT on torch. That is exactly why the
split-environment plan works: this file runs fine under torch 2.11.

It is also what the GO test (tests/gate2_load_qpc.py) imports directly to prove
a prebuilt QPC loads and runs under the target torch, with zero vLLM involved.
"""

raise NotImplementedError(
    "Port vllm/model_executor/model_loader/qaic_session_np.py from the "
    "v1_ngram fork. See the module docstring and README PART 1, step 5."
)
