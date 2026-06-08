#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate VLLM_QAIC_PROFILE ("QAIC-PROF ...") log lines into the metrics that
decide the decode-priority scheduler A/B on the box.

Each engine step emits (model_runner.py, under VLLM_QAIC_PROFILE=1):

    QAIC-PROF reqs=.. decodes=.. prefills=.. mixed=0|1 decode_tokens=.. prefill_tokens=.. |
      pack=..ms decode_qpc=..ms prefill_qpc=..ms merge=..ms logits=..ms sample=..ms bookkeeping=..ms

Since TPOT ~= the engine step time, the key signals are:
  * mixed-step ratio  (how often a step runs BOTH the decode and prefill QPC),
  * mean step time on PURE-DECODE steps  (~ best-case TPOT),
  * mean step time on MIXED steps        (~ TPOT inflated by prefill_qpc),
  * mean prefill_qpc                      (the per-mixed-step decode penalty).

A good scheduler run drops the mixed ratio and the mean (esp. mixed) step time.

Usage:
    parse_qaic_prof.py run.log
    cat run.log | parse_qaic_prof.py
    parse_qaic_prof.py off.log on.log          # compare two runs side by side
"""

from __future__ import annotations

import re
import sys

_PHASES = ("pack", "decode_qpc", "prefill_qpc", "merge", "logits", "sample", "bookkeeping")
_LINE = re.compile(r"QAIC-PROF\b")
_INT = {k: re.compile(rf"\b{k}=(\d+)\b") for k in ("reqs", "decodes", "prefills", "mixed")}
_MS = {k: re.compile(rf"\b{k}=([\d.]+)ms") for k in _PHASES}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def parse_lines(lines):
    """Return a summary dict from an iterable of log lines."""
    steps = []
    for line in lines:
        if not _LINE.search(line):
            continue
        row = {}
        ok = True
        for k, rx in _INT.items():
            m = rx.search(line)
            if not m:
                ok = False
                break
            row[k] = int(m.group(1))
        if not ok:
            continue
        for k, rx in _MS.items():
            m = rx.search(line)
            row[k] = float(m.group(1)) if m else 0.0
        row["step_ms"] = sum(row[p] for p in _PHASES)
        steps.append(row)

    n = len(steps)
    mixed = [s for s in steps if s["mixed"]]
    pure_dec = [s for s in steps if not s["mixed"] and s["decodes"] > 0]
    return {
        "steps": n,
        "mixed_steps": len(mixed),
        "mixed_ratio": (len(mixed) / n) if n else 0.0,
        "mean_step_ms": _mean([s["step_ms"] for s in steps]),
        "mean_step_ms_pure_decode": _mean([s["step_ms"] for s in pure_dec]),
        "mean_step_ms_mixed": _mean([s["step_ms"] for s in mixed]),
        "mean_decode_qpc_ms": _mean([s["decode_qpc"] for s in steps if s["decodes"] > 0]),
        "mean_prefill_qpc_ms": _mean([s["prefill_qpc"] for s in steps if s["prefills"] > 0]),
    }


def _fmt(summary: dict) -> str:
    s = summary
    return (
        f"  steps                     : {s['steps']}\n"
        f"  mixed-step ratio          : {s['mixed_ratio'] * 100:.1f}%  ({s['mixed_steps']}/{s['steps']})\n"
        f"  mean step time            : {s['mean_step_ms']:.1f} ms   (~ mean TPOT)\n"
        f"    on pure-decode steps    : {s['mean_step_ms_pure_decode']:.1f} ms   (~ best-case TPOT)\n"
        f"    on mixed steps          : {s['mean_step_ms_mixed']:.1f} ms   (~ inflated TPOT)\n"
        f"  mean decode_qpc           : {s['mean_decode_qpc_ms']:.1f} ms\n"
        f"  mean prefill_qpc          : {s['mean_prefill_qpc_ms']:.1f} ms   (per-mixed-step penalty)\n"
    )


def main(argv):
    paths = argv[1:]
    if not paths:
        print(_fmt(parse_lines(sys.stdin)))
        return 0
    for p in paths:
        with open(p) as f:
            print(f"=== {p} ===")
            print(_fmt(parse_lines(f)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
