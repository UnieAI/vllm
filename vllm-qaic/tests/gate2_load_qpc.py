#!/usr/bin/env python3
"""GATE 2 (confirms GO): load a PRE-COMPILED QPC and run one inference under
the target torch (2.11), using ONLY qaicrt — no QEfficient compile path, no
vLLM.

This proves the split-environment design: a QPC compiled offline (torch 2.7 /
QEfficient env) can be loaded and executed in the serve env (torch 2.11).

Usage:
    python gate2_load_qpc.py /path/to/qpc /path/to/qaic_session_np.py

  arg1: a QPC directory (from gate0; contains programqpc.bin).
  arg2: path to the fork's qaic_session_np.py (pure numpy+qaicrt session).
        e.g. <fork>/vllm/model_executor/model_loader/qaic_session_np.py

  PASS -> creates a session, prints input/output names, runs once -> "GATE2 PASS"
  FAIL -> note whether the error is in import (torch/ABI) or in qaicrt
          enqueue/run (usually LD_LIBRARY_PATH missing /opt/qti-aic/dev/lib/...).

NOTE: qaic_session_np.py may import a couple of vLLM symbols at module top. If
that pulls in something heavy/broken, copy the file and comment those lines —
GATE 2 only needs: build session -> set_buffers -> run -> get output.
"""

import importlib.util
import sys

import numpy as np


def load_session_module(path: str):
    spec = importlib.util.spec_from_file_location("qaic_session_np", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    qpc_path, session_py = argv[1], argv[2]

    import torch
    print("torch:", torch.__version__)

    qsess = load_session_module(session_py)
    Session = qsess.QAICInferenceSession  # noqa: N806
    print("Loaded session class:", Session)

    sess = Session(qpc_path, device_ids=[0])
    print("inputs :", list(getattr(sess, "input_names", []))[:8])
    print("outputs:", list(getattr(sess, "output_names", []))[:8])

    # Build a minimal all-ones input matching each binding's shape/dtype.
    # The fork's session exposes bindings; adjust attribute names if needed.
    dummy = {}
    for b in getattr(sess, "bindings", []):
        name = getattr(b, "name", None)
        if name is None:
            continue
        # TODO: derive real shape/dtype from the binding; ones() of size 1 is a
        # placeholder so you can see whether run() reaches the device at all.
        dummy[name] = np.ones((1,), dtype=np.int64)

    out = sess.run(dummy) if dummy else None
    print("run() returned:", type(out))
    print("GATE2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
