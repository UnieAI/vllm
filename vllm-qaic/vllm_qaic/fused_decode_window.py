# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENTAL admission contract for "decode-window absorbs short prefill".

Pure decision function only — NOT wired into the scheduler or runner. Step A
(external/efficient-transformers .../test_experimental_fused_decode_window.py) proved
the MODEL contract: a single seq_len=K forward can mix a short-prefill row with decode
rows, bit-exact vs the separate paths. THIS file fixes the ADMISSION contract that the
scheduler/runner will later consult, so the rule can't drift when production wiring
lands. Keep it dependency-free (no vLLM import) so it stays unit-testable on any CPU.
"""

from __future__ import annotations


def is_short_prefill_decode_window_eligible(
    *,
    prompt_len: int,
    decode_window_size: int,
    available_slots: int,
    has_decode_batch: bool,
    enabled: bool,
) -> bool:
    """Whether a prompt may ride the decode graph's K-wide query window this step
    instead of triggering a separate (expensive) prefill QPC.

    Args:
        prompt_len: uncomputed prompt tokens to schedule for this request.
        decode_window_size: the compiled decode graph's query width K (its seq_len).
            K <= 1 is an ordinary single-token decode graph that cannot absorb prefill.
        available_slots: query slots still FREE in this step's decode window after
            spec tokens / other short-prefill rows have taken theirs. This — not K
            alone — bounds what can ride along, because the window is shared; using K
            directly would let two short prefills overcommit the same window.
        has_decode_batch: whether this step already runs the decode graph for real
            decode rows. If it does not, routing a lone short prompt through the decode
            graph just runs the K-wide decode graph for nothing — wasteful; prefill it
            normally instead.
        enabled: feature flag (off => always ineligible).

    Eligible iff ALL hold:
        enabled
        and has_decode_batch
        and decode_window_size > 1
        and 0 < prompt_len <= available_slots
        and available_slots <= decode_window_size

    The chained bound (0 < prompt_len <= available_slots <= decode_window_size) is the
    contract: a positive prompt that fits the slots actually free this step, which in
    turn never exceed the compiled window width.
    """
    return (
        enabled
        and has_decode_batch
        and decode_window_size > 1
        and 0 < prompt_len <= available_slots
        and available_slots <= decode_window_size
    )
