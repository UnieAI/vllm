# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Rust-accelerated scheduler helpers.

This module provides a Python-fallback-safe interface to the Rust
scheduler acceleration functions.  If the ``vllm._rs`` native module is
available it will be used; otherwise identical pure-Python fallbacks run
transparently.

Usage in scheduler.py:
    from vllm.v1.core.sched.rust_accelerated import (
        compute_running_tokens_batch,
        batch_check_stop_reasons,
        batch_precompute_stop_reasons,
        batch_apply_spec_decode,
    )
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)

try:
    from vllm._rs import (  # type: ignore[import-untyped]
        batch_apply_generated_tokens as _rs_batch_apply,
        batch_check_stop as _rs_batch_check_stop,
        compute_running_tokens as _rs_compute_running,
    )

    _HAS_RUST = True
except ImportError:
    try:
        from _rs import (  # type: ignore[import-untyped]
            batch_apply_generated_tokens as _rs_batch_apply,
            batch_check_stop as _rs_batch_check_stop,
            compute_running_tokens as _rs_compute_running,
        )

        _HAS_RUST = True
    except ImportError:
        _HAS_RUST = False

if _HAS_RUST:
    logger.info("Rust scheduler acceleration enabled")
else:
    logger.info(
        "vllm._rs not available, using pure-Python scheduler (install "
        "vllm-scheduler-rs for ~100x faster scheduling loops)"
    )


# ── compute_running_tokens ──────────────────────────────────────────

def compute_running_tokens_batch(
    running: list[Request],
    token_budget: int,
    long_prefill_threshold: int,
    max_model_len: int,
) -> npt.NDArray[np.int64]:
    """Compute num_new_tokens for all running requests in one call.

    Returns an int64 array of length len(running), where each element is
    the number of tokens to schedule for the corresponding request.
    """
    n = len(running)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    spec = np.empty(n, dtype=np.int64)
    placeholders = np.empty(n, dtype=np.int64)
    computed = np.empty(n, dtype=np.int64)
    prompt = np.empty(n, dtype=np.int64)
    max_tok = np.empty(n, dtype=np.int64)

    for i, req in enumerate(running):
        spec[i] = req.num_tokens_with_spec
        placeholders[i] = req.num_output_placeholders
        computed[i] = req.num_computed_tokens
        prompt[i] = req.num_prompt_tokens
        max_tok[i] = req.max_tokens

    if _HAS_RUST:
        return _rs_compute_running(
            spec, placeholders, computed, prompt, max_tok,
            token_budget, long_prefill_threshold, max_model_len,
        )

    # Pure-Python fallback.
    result = np.zeros(n, dtype=np.int64)
    budget = token_budget
    for i in range(n):
        if budget <= 0:
            break
        if (placeholders[i] > 0
                and (computed[i] + 2 - placeholders[i])
                >= (prompt[i] + max_tok[i])):
            continue
        num_new = int(spec[i] + placeholders[i] - computed[i])
        if long_prefill_threshold > 0 and num_new > long_prefill_threshold:
            num_new = long_prefill_threshold
        num_new = min(num_new, budget)
        num_new = min(num_new, max_model_len - 1 - int(computed[i]))
        num_new = max(num_new, 0)
        result[i] = num_new
        budget -= num_new
    return result


# ── batch_check_stop ────────────────────────────────────────────────

# Stop reason codes (must match Rust).
STOP_NONE = 0
STOP_EOS = 1
STOP_TOKEN = 2
STOP_LENGTH = 3


def _build_stop_token_2d(
    stop_lists: list[list[int]],
    n: int,
) -> npt.NDArray[np.int64]:
    """Build a padded 2-D numpy array from per-request stop token lists.

    Shape: ``(n, max_stop_len)``, padded with ``-1`` (sentinel).
    When no request has stop tokens, returns shape ``(n, 0)``.
    """
    max_stop_len = max((len(sl) for sl in stop_lists), default=0)
    if max_stop_len == 0:
        return np.empty((n, 0), dtype=np.int64)
    stop_arr = np.full((n, max_stop_len), -1, dtype=np.int64)
    for i, sl in enumerate(stop_lists):
        for j, tok_id in enumerate(sl):
            stop_arr[i, j] = tok_id
    return stop_arr


def batch_check_stop_reasons(
    requests: list[Request],
    max_model_len: int,
) -> npt.NDArray[np.int32]:
    """Batch-check stop conditions for multiple requests.

    Returns an int32 array of stop reason codes (see STOP_* constants).
    """
    n = len(requests)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    last_tok = np.empty(n, dtype=np.int64)
    num_tok = np.empty(n, dtype=np.int64)
    num_out = np.empty(n, dtype=np.int64)
    min_tok = np.empty(n, dtype=np.int64)
    max_tok = np.empty(n, dtype=np.int64)
    eos = np.empty(n, dtype=np.int64)
    stop_lists: list[list[int]] = []

    for i, req in enumerate(requests):
        last_tok[i] = req.output_token_ids[-1] if req.output_token_ids else -1
        num_tok[i] = req.num_tokens
        num_out[i] = req.num_output_tokens
        sp = req.sampling_params
        if sp is not None:
            min_tok[i] = sp.min_tokens
            max_tok[i] = req.max_tokens
            eos[i] = sp.eos_token_id if sp.eos_token_id is not None else -1
            stop_lists.append(
                list(sp.stop_token_ids) if sp.stop_token_ids else []
            )
        else:
            min_tok[i] = 0
            max_tok[i] = req.max_tokens
            eos[i] = -1
            stop_lists.append([])

    stop_arr = _build_stop_token_2d(stop_lists, n)

    if _HAS_RUST:
        return _rs_batch_check_stop(
            last_tok, num_tok, num_out, min_tok, max_tok, eos,
            stop_arr, max_model_len,
        )

    # Pure-Python fallback.
    result = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if num_out[i] < min_tok[i]:
            continue
        if eos[i] >= 0 and last_tok[i] == eos[i]:
            result[i] = STOP_EOS
            continue
        if last_tok[i] in stop_lists[i]:
            result[i] = STOP_TOKEN
            continue
        if num_tok[i] >= max_model_len or num_out[i] >= max_tok[i]:
            result[i] = STOP_LENGTH
            continue
    return result


def batch_precompute_stop_reasons(
    requests: list[Request],
    new_token_ids: npt.NDArray[np.int64],
    max_model_len: int,
) -> npt.NDArray[np.int32]:
    """Pre-compute stop conditions BEFORE tokens are appended.

    ``new_token_ids[i]`` is the single token that **will be** appended to
    ``requests[i]``.  Token and output counts are adjusted by +1 to
    simulate post-append state.

    Returns an int32 array of stop reason codes.
    """
    n = len(requests)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    num_tok = np.empty(n, dtype=np.int64)
    num_out = np.empty(n, dtype=np.int64)
    min_tok = np.empty(n, dtype=np.int64)
    max_tok = np.empty(n, dtype=np.int64)
    eos = np.empty(n, dtype=np.int64)
    stop_lists: list[list[int]] = []

    for i, req in enumerate(requests):
        # +1: we are checking as if the token has already been appended.
        num_tok[i] = req.num_tokens + 1
        num_out[i] = req.num_output_tokens + 1
        sp = req.sampling_params
        if sp is not None:
            min_tok[i] = sp.min_tokens
            max_tok[i] = req.max_tokens
            eos[i] = sp.eos_token_id if sp.eos_token_id is not None else -1
            stop_lists.append(
                list(sp.stop_token_ids) if sp.stop_token_ids else []
            )
        else:
            min_tok[i] = 0
            max_tok[i] = req.max_tokens
            eos[i] = -1
            stop_lists.append([])

    stop_arr = _build_stop_token_2d(stop_lists, n)

    if _HAS_RUST:
        return _rs_batch_check_stop(
            new_token_ids, num_tok, num_out, min_tok, max_tok, eos,
            stop_arr, max_model_len,
        )

    # Pure-Python fallback.
    result = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if num_out[i] < min_tok[i]:
            continue
        tok = int(new_token_ids[i])
        if eos[i] >= 0 and tok == eos[i]:
            result[i] = STOP_EOS
            continue
        if tok in stop_lists[i]:
            result[i] = STOP_TOKEN
            continue
        if num_tok[i] >= max_model_len or num_out[i] >= max_tok[i]:
            result[i] = STOP_LENGTH
            continue
    return result


# ── batch_apply_spec_decode ─────────────────────────────────────────

def batch_apply_spec_decode(
    requests: list[Request],
    num_generated: list[int],
    num_draft: list[int],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Batch-compute spec decode acceptance/rejection.

    Returns (adjusted_computed, adjusted_placeholders, num_accepted,
    num_rejected) as four int64 arrays.
    """
    n = len(requests)
    computed = np.array(
        [r.num_computed_tokens for r in requests], dtype=np.int64,
    )
    placeholders = np.array(
        [r.num_output_placeholders for r in requests], dtype=np.int64,
    )
    gen = np.array(num_generated, dtype=np.int64)
    draft = np.array(num_draft, dtype=np.int64)

    if _HAS_RUST:
        return _rs_batch_apply(computed, placeholders, gen, draft)

    # Pure-Python fallback.
    adj_c = np.copy(computed)
    adj_p = np.copy(placeholders)
    accepted = np.copy(gen)
    rejected = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if draft[i] > 0 and gen[i] > 0:
            na = gen[i] - 1
            nr = draft[i] - na
            if computed[i] > 0:
                adj_c[i] = computed[i] - nr
            if placeholders[i] > 0:
                adj_p[i] = placeholders[i] - nr
            accepted[i] = na
            rejected[i] = nr
    return adj_c, adj_p, accepted, rejected
