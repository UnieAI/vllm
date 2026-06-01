#!/usr/bin/env python3
"""GATE 2 (confirms GO): load a PRE-COMPILED QPC and run it under the target
torch (2.11), using ONLY qaicrt — no QEfficient compile path, no vLLM.

This proves the split-environment plan: a QPC compiled offline (torch 2.7 /
QEfficient env) loads and executes in the serve env (torch 2.11).

Usage:
    python gate2_load_qpc.py /path/to/qpc /path/to/qaic_session_np.py

  arg1: a QPC directory (from gate0; contains programqpc.bin).
  arg2: path to the fork's qaic_session_np.py (pure numpy+qaicrt session).

Verdicts (exit code):
  0  GATE2 PASS         — session built AND one inference actually ran on-device.
  2  GATE2 INCONCLUSIVE — session + QPC loaded under torch 2.11 (a STRONG go
                          signal), but this script could not auto-derive input
                          shapes to run inference. Drive one real step via your
                          input-prep to fully confirm. NOT a pass, NOT a no-go.
  1  GATE2 FAIL         — session build / load / run errored (inspect the trace).

Design notes (why this is careful):
  * It NEVER prints PASS without actually running >=1 input buffer on-device
    (running zero buffers is a hard FAIL/INCONCLUSIVE, never a PASS).
  * It feeds inputs from the session's REAL input_names with each binding's real
    shape+dtype, and SKIPS the on-card KV buffers (past_* / *_RetainedState)
    that the session deliberately skip_buffers — feeding those crashes run().
"""

import importlib.util
import sys

import numpy as np


def load_session_module(path):
    spec = importlib.util.spec_from_file_location("qaic_session_np", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_kv_buffer(name):
    return name.startswith("past_") or name.endswith("_RetainedState")


def derive_shape_dtype(session, name):
    """Best-effort (shape, np.dtype) for an input binding, or (None, None).

    QEfficient/qaicrt expose binding metadata in a few shapes across versions;
    try the common ones. Returns (None, None) if not derivable so the caller can
    report INCONCLUSIVE instead of guessing.
    """
    # name -> binding index
    idx = None
    bmap = getattr(session, "binding_index_map", None)
    if isinstance(bmap, dict) and name in bmap:
        idx = bmap[name]
    elif name in getattr(session, "input_names", []):
        idx = list(session.input_names).index(name)
    binding = None
    bindings = getattr(session, "bindings", None)
    if bindings is not None and idx is not None and idx < len(bindings):
        binding = bindings[idx]

    shape = None
    for attr in ("dims", "shape"):
        v = getattr(binding, attr, None)
        if v:
            try:
                shape = tuple(int(x) for x in v)
                break
            except Exception:
                pass

    dtype = None
    type_map = getattr(session, "aic_to_np_dtype_mapping", None)
    btype = getattr(binding, "type", None)
    if isinstance(type_map, dict) and btype in type_map:
        dtype = np.dtype(type_map[btype])
    return shape, dtype


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    qpc_path, session_py = argv[1], argv[2]

    import torch
    print("torch:", torch.__version__)

    qsess = load_session_module(session_py)
    Session = qsess.QAICInferenceSession  # noqa: N806

    # --- session build + QPC load/activate on-device (the core GO signal) ----
    sess = Session(qpc_path, device_ids=[0])
    in_names = list(getattr(sess, "input_names", []))
    out_names = list(getattr(sess, "output_names", []))
    print("input_names :", in_names[:12])
    print("output_names:", out_names[:12])
    if not in_names:
        print("GATE2 FAIL: session exposes no input_names")
        return 1

    runnable = [n for n in in_names if not is_kv_buffer(n)]
    print("runnable inputs (KV buffers skipped):", runnable)
    if not runnable:
        print("GATE2 FAIL: every input looks like a KV buffer; cannot run")
        return 1

    # --- build correctly-shaped zero inputs ----------------------------------
    inputs, unresolved = {}, []
    for n in runnable:
        shape, dtype = derive_shape_dtype(sess, n)
        if shape is None or dtype is None:
            unresolved.append(n)
            continue
        inputs[n] = np.zeros(shape, dtype=dtype)

    if unresolved:
        # Show one binding's attributes so the user can adapt derive_shape_dtype.
        b = getattr(sess, "bindings", [None])
        sample = b[0] if b else None
        print("could not derive shape/dtype for:", unresolved)
        print("sample binding attrs:", [a for a in dir(sample) if not a.startswith("__")][:30])
        print("GATE2 INCONCLUSIVE: session + QPC loaded under torch",
              torch.__version__, "(strong GO signal), but couldn't auto-build",
              "inputs. Drive one step via your real input-prep to confirm.")
        return 2

    if not inputs:
        print("GATE2 FAIL: built zero input buffers (nothing to run)")
        return 1

    # --- run one inference on-device -----------------------------------------
    out = sess.run(inputs)
    if out is None:
        print("GATE2 FAIL: run() returned None")
        return 1
    print(f"ran {len(inputs)} input buffer(s); output keys:",
          list(out)[:8] if hasattr(out, "__iter__") else type(out))
    print("GATE2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
