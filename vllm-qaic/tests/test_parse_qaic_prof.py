# SPDX-License-Identifier: Apache-2.0
"""Unit test for tools/parse_qaic_prof.py (dependency-free; runs on any CPU)."""

import importlib.util
import math
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "parse_qaic_prof.py"
_spec = importlib.util.spec_from_file_location("parse_qaic_prof", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_lines = _mod.parse_lines

_PURE = ("INFO 06-08 10:00:00 model_runner.py:466] QAIC-PROF reqs=90 decodes=90 prefills=0 "
         "mixed=0 decode_tokens=90 prefill_tokens=0 | pack=0.5ms decode_qpc=200.0ms "
         "prefill_qpc=0.0ms merge=0.2ms logits=2.0ms sample=1.0ms bookkeeping=0.3ms")
# step_ms = 0.5+200+0+0.2+2+1+0.3 = 204.0
_MIXED = ("INFO 06-08 10:00:01 model_runner.py:466] QAIC-PROF reqs=96 decodes=90 prefills=6 "
          "mixed=1 decode_tokens=90 prefill_tokens=2048 | pack=0.5ms decode_qpc=200.0ms "
          "prefill_qpc=1150.0ms merge=0.3ms logits=2.0ms sample=1.0ms bookkeeping=0.3ms")
# step_ms = 0.5+200+1150+0.3+2+1+0.3 = 1354.1


def test_parse_mixed_ratio_and_step_split():
    lines = [_PURE, _MIXED, _PURE, _MIXED, "some unrelated log line", ""]
    s = parse_lines(lines)
    assert s["steps"] == 4
    assert s["mixed_steps"] == 2
    assert math.isclose(s["mixed_ratio"], 0.5)
    assert math.isclose(s["mean_step_ms_pure_decode"], 204.0, abs_tol=1e-6)
    assert math.isclose(s["mean_step_ms_mixed"], 1354.1, abs_tol=1e-6)
    assert math.isclose(s["mean_step_ms"], (204.0 * 2 + 1354.1 * 2) / 4, abs_tol=1e-6)
    assert math.isclose(s["mean_decode_qpc_ms"], 200.0, abs_tol=1e-6)
    assert math.isclose(s["mean_prefill_qpc_ms"], 1150.0, abs_tol=1e-6)


def test_parse_empty_input():
    s = parse_lines([])
    assert s["steps"] == 0
    assert s["mixed_ratio"] == 0.0
    assert s["mean_step_ms"] == 0.0


def test_ignores_non_prof_lines():
    s = parse_lines(["hello", "QAIC-PROF malformed without fields", _PURE])
    assert s["steps"] == 1  # malformed (missing reqs=..) skipped


if __name__ == "__main__":
    test_parse_mixed_ratio_and_step_split()
    test_parse_empty_input()
    test_ignores_non_prof_lines()
    print("PARSE QAIC-PROF: ALL PASS")
