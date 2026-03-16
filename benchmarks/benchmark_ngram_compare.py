# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the current N-gram proposer implementation against a git baseline.

This benchmark is CPU-only and avoids model/config downloads by constructing a
minimal config stub for the proposer. It can be used to compare:

1. The current worktree implementation against a previous git revision.
2. The current implementation with full-history search vs a bounded search
   window.
3. Runner-side integration overhead from precomputing valid request indices.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tabulate import tabulate

from vllm.config import SpeculativeConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.spec_decode.ngram_proposer import NgramProposalInputs, NgramProposer


def _load_baseline_proposer(repo_root: Path, baseline_ref: str):
    baseline_source = subprocess.check_output(
        ["git", "show", f"{baseline_ref}:vllm/v1/spec_decode/ngram_proposer.py"],
        cwd=repo_root,
        text=True,
    )
    baseline_module = types.ModuleType("baseline_ngram_proposer")
    exec(baseline_source, baseline_module.__dict__)
    return baseline_module.NgramProposer


def _make_config(
    *,
    min_ngram: int,
    max_ngram: int,
    num_spec_token: int,
    max_model_len: int,
    max_num_seqs: int,
    search_window: int | None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=SpeculativeConfig(
            prompt_lookup_min=min_ngram,
            prompt_lookup_max=max_ngram,
            prompt_lookup_window=search_window,
            num_speculative_tokens=num_spec_token,
            method="ngram",
        ),
    )


def _percentile(sorted_samples: list[float], pct: float) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    index = (len(sorted_samples) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_samples) - 1)
    weight = index - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


def _measure_us(fn, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000.0)

    samples.sort()
    return {
        "avg": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "max": samples[-1],
    }


def _build_sampled_token_ids(
    token_ids_cpu: np.ndarray,
    *,
    empty_every: int,
) -> list[list[int]]:
    sampled_token_ids: list[list[int]] = []
    for i in range(token_ids_cpu.shape[0]):
        if empty_every > 0 and i % empty_every == 0:
            sampled_token_ids.append([])
        else:
            sampled_token_ids.append([int(token_ids_cpu[i, -1])])
    return sampled_token_ids


class _OldRunner:
    def __init__(
        self,
        proposer,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> None:
        self.proposer = proposer
        self.sampled_token_ids = sampled_token_ids
        self.num_tokens_no_spec = num_tokens_no_spec
        self.token_ids_cpu = token_ids_cpu

    def run(self) -> list[list[int]]:
        return self.proposer.propose(
            self.sampled_token_ids,
            self.num_tokens_no_spec,
            self.token_ids_cpu,
        )


class _NewRunner:
    def __init__(
        self,
        proposer: NgramProposer,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> None:
        self.proposer = proposer
        self.sampled_token_ids = sampled_token_ids
        self.num_tokens_no_spec = num_tokens_no_spec
        self.token_ids_cpu = token_ids_cpu

    def get_ngram_proposal_inputs(self) -> NgramProposalInputs:
        valid_ngram_requests = np.empty(len(self.sampled_token_ids), dtype=np.int32)
        num_valid_requests = 0
        for i, sampled_ids in enumerate(self.sampled_token_ids):
            if self.proposer.should_propose_for_request(
                i,
                sampled_ids,
                self.num_tokens_no_spec[i],
            ):
                valid_ngram_requests[num_valid_requests] = i
                num_valid_requests += 1

        return NgramProposalInputs(
            sampled_token_ids=self.sampled_token_ids,
            num_tokens_no_spec=self.num_tokens_no_spec,
            token_ids_cpu=self.token_ids_cpu,
            valid_ngram_requests=valid_ngram_requests[:num_valid_requests],
        )

    def run(self) -> list[list[int]]:
        inputs = self.get_ngram_proposal_inputs()
        return self.proposer.propose(
            inputs.sampled_token_ids,
            inputs.num_tokens_no_spec,
            inputs.token_ids_cpu,
            valid_ngram_requests=inputs.valid_ngram_requests,
        )


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Compare old and new ngram proposer implementations."
    )
    parser.add_argument(
        "--baseline-ref",
        type=str,
        default="HEAD",
        help="Git ref used as the baseline proposer implementation.",
    )
    parser.add_argument("--num-req", type=int, default=64)
    parser.add_argument("--num-token", type=int, default=1024)
    parser.add_argument("--min-ngram", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, default=5)
    parser.add_argument("--num-spec-token", type=int, default=3)
    parser.add_argument(
        "--compare-search-window",
        type=int,
        default=None,
        help="Optional bounded search window to compare against full-history.",
    )
    parser.add_argument(
        "--empty-every",
        type=int,
        default=4,
        help="Make every Nth request ineligible for ngram drafting.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    baseline_cls = _load_baseline_proposer(repo_root, args.baseline_ref)

    np.random.seed(args.seed)
    token_ids_cpu = np.random.randint(
        0,
        2000,
        (args.num_req, args.num_token),
        dtype=np.int32,
    )
    num_tokens_no_spec = np.full(args.num_req, args.num_token, dtype=np.int32)
    sampled_token_ids = _build_sampled_token_ids(
        token_ids_cpu,
        empty_every=args.empty_every,
    )
    max_model_len = args.num_token + args.num_spec_token

    baseline = baseline_cls(
        _make_config(
            min_ngram=args.min_ngram,
            max_ngram=args.max_ngram,
            num_spec_token=args.num_spec_token,
            max_model_len=max_model_len,
            max_num_seqs=args.num_req,
            search_window=None,
        )
    )
    candidate_full = NgramProposer(
        _make_config(
            min_ngram=args.min_ngram,
            max_ngram=args.max_ngram,
            num_spec_token=args.num_spec_token,
            max_model_len=max_model_len,
            max_num_seqs=args.num_req,
            search_window=None,
        )
    )
    candidate_window = None
    if args.compare_search_window is not None:
        candidate_window = NgramProposer(
            _make_config(
                min_ngram=args.min_ngram,
                max_ngram=args.max_ngram,
                num_spec_token=args.num_spec_token,
                max_model_len=max_model_len,
                max_num_seqs=args.num_req,
                search_window=args.compare_search_window,
            )
        )

    rows: list[list[object]] = []

    baseline_propose = _measure_us(
        lambda: baseline.propose(
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rows.append([
        "baseline_propose",
        "full",
        *[round(baseline_propose[k], 3) for k in ("avg", "p50", "p95", "p99", "max")],
    ])

    candidate_full_propose = _measure_us(
        lambda: candidate_full.propose(
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rows.append([
        "candidate_propose",
        "full",
        *[
            round(candidate_full_propose[k], 3)
            for k in ("avg", "p50", "p95", "p99", "max")
        ],
    ])

    baseline_runner = _measure_us(
        _OldRunner(
            baseline,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        ).run,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rows.append([
        "baseline_runner",
        "full",
        *[round(baseline_runner[k], 3) for k in ("avg", "p50", "p95", "p99", "max")],
    ])

    candidate_full_runner = _measure_us(
        _NewRunner(
            candidate_full,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        ).run,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rows.append([
        "candidate_runner",
        "full",
        *[
            round(candidate_full_runner[k], 3)
            for k in ("avg", "p50", "p95", "p99", "max")
        ],
    ])

    if candidate_window is not None:
        candidate_window_propose = _measure_us(
            lambda: candidate_window.propose(
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        rows.append([
            "candidate_propose",
            str(args.compare_search_window),
            *[
                round(candidate_window_propose[k], 3)
                for k in ("avg", "p50", "p95", "p99", "max")
            ],
        ])

        candidate_window_runner = _measure_us(
            _NewRunner(
                candidate_window,
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
            ).run,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        rows.append([
            "candidate_runner",
            str(args.compare_search_window),
            *[
                round(candidate_window_runner[k], 3)
                for k in ("avg", "p50", "p95", "p99", "max")
            ],
        ])

    print(
        f"baseline_ref={args.baseline_ref}, num_req={args.num_req}, "
        f"num_token={args.num_token}, min_ngram={args.min_ngram}, "
        f"max_ngram={args.max_ngram}, num_spec_token={args.num_spec_token}, "
        f"empty_every={args.empty_every}"
    )
    print(
        tabulate(
            rows,
            headers=["Case", "Search Window", "Avg (us)", "P50 (us)", "P95 (us)",
                     "P99 (us)", "Max (us)"],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )

    improvements: list[list[object]] = []
    improvements.append([
        "propose_full_vs_baseline",
        round(baseline_propose["avg"] - candidate_full_propose["avg"], 3),
        round(
            ((baseline_propose["avg"] - candidate_full_propose["avg"])
             / baseline_propose["avg"]) * 100.0,
            2,
        ),
    ])
    improvements.append([
        "runner_full_vs_baseline",
        round(baseline_runner["avg"] - candidate_full_runner["avg"], 3),
        round(
            ((baseline_runner["avg"] - candidate_full_runner["avg"])
             / baseline_runner["avg"]) * 100.0,
            2,
        ),
    ])
    if candidate_window is not None:
        improvements.append([
            "propose_window_vs_baseline",
            round(baseline_propose["avg"] - candidate_window_propose["avg"], 3),
            round(
                ((baseline_propose["avg"] - candidate_window_propose["avg"])
                 / baseline_propose["avg"]) * 100.0,
                2,
            ),
        ])
        improvements.append([
            "runner_window_vs_baseline",
            round(baseline_runner["avg"] - candidate_window_runner["avg"], 3),
            round(
                ((baseline_runner["avg"] - candidate_window_runner["avg"])
                 / baseline_runner["avg"]) * 100.0,
                2,
            ),
        ])

    print(
        tabulate(
            improvements,
            headers=["Comparison", "Avg Improvement (us)", "Avg Improvement (%)"],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


if __name__ == "__main__":
    main()  # pragma: no cover
