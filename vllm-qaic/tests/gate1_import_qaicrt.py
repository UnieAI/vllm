#!/usr/bin/env python3
"""GATE 1 (decisive): can qaicrt be imported under the TARGET torch (2.11)?

Run this INSIDE the new torch-2.11 / vLLM-0.21 venv on the AIC machine.
This is the cheapest, most decisive go/no-go signal: qaicrt is a Cloud AI SDK
C++/pybind module and should be torch-independent. The thing that actually
bites here is the Python ABI version, not torch.

  PASS  -> prints "GATE1 PASS"
  FAIL  -> import error. If it's an ABI/undefined-symbol error, your venv's
           Python version must match the one the SDK built qaicrt for (check
           the path printed by gate0: .../pyXYZ/...). Recreate the venv with
           that Python (uv venv --python 3.X) and retry.

If qaicrt is not on sys.path, the SDK default locations are appended (same as
the fork's detection).
"""

import platform
import sys


def main() -> int:
    import torch
    print("torch:", torch.__version__)

    try:
        import qaicrt
    except ImportError:
        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt
    print("qaicrt OK ->", getattr(qaicrt, "__file__", "<builtin>"))

    try:
        import QAicApi_pb2  # noqa: F401
    except ImportError:
        sys.path.append("/opt/qti-aic/dev/python")
        import QAicApi_pb2  # noqa: F401
    print("QAicApi_pb2 OK")

    print("python:", sys.version.split()[0])
    print("GATE1 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
