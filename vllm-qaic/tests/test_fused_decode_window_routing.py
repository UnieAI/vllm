# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the EXPERIMENTAL decode-window short-prefill admission contract.

Dependency-free: loads vllm_qaic/fused_decode_window.py directly (without importing
the vllm_qaic package __init__, which needs vLLM), so this runs on any CPU.
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[1] / "vllm_qaic" / "fused_decode_window.py"
_spec = importlib.util.spec_from_file_location("fused_decode_window", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
eligible = _mod.is_short_prefill_decode_window_eligible


def _base(**over):
    # A canonical eligible call (K=4, exactly K free, decode batch present, enabled).
    kw = dict(prompt_len=4, decode_window_size=4, available_slots=4,
              has_decode_batch=True, enabled=True)
    kw.update(over)
    return eligible(**kw)


def test_prompt_len_equals_K_eligible():
    assert _base(prompt_len=4, available_slots=4, decode_window_size=4) is True


def test_prompt_len_greater_than_K_not_eligible():
    # prompt can't exceed the window; available_slots can't exceed K either.
    assert _base(prompt_len=5, available_slots=4, decode_window_size=4) is False


def test_prompt_len_zero_not_eligible():
    assert _base(prompt_len=0) is False


def test_no_decode_batch_not_eligible():
    # A lone short prompt must NOT sneak into the decode graph (wastes the K-window).
    assert _base(has_decode_batch=False) is False


def test_disabled_not_eligible():
    assert _base(enabled=False) is False


def test_available_slots_less_than_prompt_not_eligible():
    assert _base(prompt_len=3, available_slots=2) is False


def test_available_slots_equals_prompt_eligible():
    # Window partly taken (only 2 free); a 2-token prompt still fits exactly.
    assert _base(prompt_len=2, available_slots=2, decode_window_size=4) is True


def test_decode_window_size_le_1_not_eligible():
    # Plain single-token decode graph cannot absorb a prefill.
    assert _base(prompt_len=1, available_slots=1, decode_window_size=1) is False


if __name__ == "__main__":
    test_prompt_len_equals_K_eligible()
    test_prompt_len_greater_than_K_not_eligible()
    test_prompt_len_zero_not_eligible()
    test_no_decode_batch_not_eligible()
    test_disabled_not_eligible()
    test_available_slots_less_than_prompt_not_eligible()
    test_available_slots_equals_prompt_eligible()
    test_decode_window_size_le_1_not_eligible()
    print("FUSED DECODE WINDOW ROUTING: ALL PASS")
