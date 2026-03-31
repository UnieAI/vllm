#!/usr/bin/env python3
"""CPU-side benchmark for Rust scheduler acceleration.

Simulates the EngineCore hot loop with mock data to measure
the CPU overhead reduction from Rust acceleration, without GPU.

Usage:
    python benchmarks/benchmark_rust_cpu.py
    python benchmarks/benchmark_rust_cpu.py --num-requests 500 --iterations 200
"""
from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Try to import Rust module ──
try:
    from _rs import (
        RustFreeBlockQueue,
        StopStringMatcher,
        batch_apply_generated_tokens,
        batch_check_stop,
        batch_hash_blocks,
        compute_running_tokens,
    )
    HAS_RUST = True
except ImportError:
    try:
        from vllm._rs import (
            RustFreeBlockQueue,
            StopStringMatcher,
            batch_apply_generated_tokens,
            batch_check_stop,
            batch_hash_blocks,
            compute_running_tokens,
        )
        HAS_RUST = True
    except ImportError:
        HAS_RUST = False


# ── Mock objects ──

@dataclass
class MockSamplingParams:
    min_tokens: int = 0
    max_tokens: int = 1024
    eos_token_id: int = 2
    stop_token_ids: frozenset[int] = field(default_factory=frozenset)
    stop: list[str] = field(default_factory=list)
    include_stop_str_in_output: bool = False
    repetition_detection: Any = None


@dataclass
class MockRequest:
    request_id: str
    num_tokens: int
    num_output_tokens: int
    num_computed_tokens: int
    num_prompt_tokens: int
    num_tokens_with_spec: int
    num_output_placeholders: int = 0
    max_tokens: int = 1024
    sampling_params: MockSamplingParams = field(
        default_factory=MockSamplingParams
    )
    output_token_ids: list[int] = field(default_factory=list)
    all_token_ids: list[int] = field(default_factory=list)
    block_hashes: list[bytes] = field(default_factory=list)


def make_mock_requests(n: int, rng: np.random.Generator) -> list[MockRequest]:
    """Create N realistic mock requests."""
    requests = []
    for i in range(n):
        prompt_len = int(rng.integers(50, 500))
        output_len = int(rng.integers(10, 200))
        total = prompt_len + output_len
        output_ids = rng.integers(0, 32000, size=output_len).tolist()
        all_ids = rng.integers(0, 32000, size=total).tolist()

        stop_token_ids = frozenset(
            rng.integers(0, 32000, size=rng.integers(0, 4)).tolist()
        )
        stop_strings = []
        if rng.random() < 0.3:
            stop_strings = ["</s>", "\n\n"]

        requests.append(MockRequest(
            request_id=f"req-{i}",
            num_tokens=total,
            num_output_tokens=output_len,
            num_computed_tokens=total - int(rng.integers(1, 10)),
            num_prompt_tokens=prompt_len,
            num_tokens_with_spec=total,
            max_tokens=int(rng.integers(256, 2048)),
            sampling_params=MockSamplingParams(
                eos_token_id=2,
                stop_token_ids=stop_token_ids,
                stop=stop_strings,
            ),
            output_token_ids=output_ids,
            all_token_ids=all_ids,
        ))
    return requests


# ── Benchmark functions ──

def bench_compute_running_tokens(
    requests: list[MockRequest], iters: int
) -> dict[str, float]:
    """Benchmark: schedule() token budget computation."""
    n = len(requests)
    spec = np.array([r.num_tokens_with_spec for r in requests], dtype=np.int64)
    ph = np.array([r.num_output_placeholders for r in requests], dtype=np.int64)
    comp = np.array([r.num_computed_tokens for r in requests], dtype=np.int64)
    prompt = np.array([r.num_prompt_tokens for r in requests], dtype=np.int64)
    maxt = np.array([r.max_tokens for r in requests], dtype=np.int64)

    results = {}

    if HAS_RUST:
        # Warmup
        for _ in range(50):
            compute_running_tokens(spec, ph, comp, prompt, maxt, 100000, 0, 4096)
        t0 = time.perf_counter()
        for _ in range(iters):
            compute_running_tokens(spec, ph, comp, prompt, maxt, 100000, 0, 4096)
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline
    def py_compute():
        result = np.zeros(n, dtype=np.int64)
        budget = 100000
        for i in range(n):
            if budget <= 0:
                break
            num_new = int(spec[i] + ph[i] - comp[i])
            num_new = min(num_new, budget, 4095 - int(comp[i]))
            num_new = max(num_new, 0)
            result[i] = num_new
            budget -= num_new

    for _ in range(10):
        py_compute()
    t0 = time.perf_counter()
    for _ in range(iters):
        py_compute()
    results["python"] = (time.perf_counter() - t0) / iters

    return results


def bench_batch_check_stop(
    requests: list[MockRequest], iters: int
) -> dict[str, float]:
    """Benchmark: update_from_output() stop condition checking."""
    n = len(requests)
    last_tok = np.array(
        [r.output_token_ids[-1] if r.output_token_ids else -1 for r in requests],
        dtype=np.int64,
    )
    num_tok = np.array([r.num_tokens for r in requests], dtype=np.int64)
    num_out = np.array([r.num_output_tokens for r in requests], dtype=np.int64)
    min_tok = np.array(
        [r.sampling_params.min_tokens for r in requests], dtype=np.int64
    )
    max_tok = np.array([r.max_tokens for r in requests], dtype=np.int64)
    eos = np.array(
        [r.sampling_params.eos_token_id for r in requests], dtype=np.int64
    )
    stop_lists = [list(r.sampling_params.stop_token_ids) for r in requests]
    max_stop_len = max((len(sl) for sl in stop_lists), default=0)
    if max_stop_len == 0:
        stop_arr = np.empty((n, 0), dtype=np.int64)
    else:
        stop_arr = np.full((n, max_stop_len), -1, dtype=np.int64)
        for i, sl in enumerate(stop_lists):
            for j, tok_id in enumerate(sl):
                stop_arr[i, j] = tok_id

    results = {}

    if HAS_RUST:
        for _ in range(50):
            batch_check_stop(
                last_tok, num_tok, num_out, min_tok, max_tok, eos, stop_arr, 4096
            )
        t0 = time.perf_counter()
        for _ in range(iters):
            batch_check_stop(
                last_tok, num_tok, num_out, min_tok, max_tok, eos, stop_arr, 4096
            )
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline
    def py_stop():
        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if num_out[i] < min_tok[i]:
                continue
            if eos[i] >= 0 and last_tok[i] == eos[i]:
                result[i] = 1
                continue
            found = False
            for j in range(max_stop_len):
                if j >= len(stop_lists[i]):
                    break
                if last_tok[i] == stop_lists[i][j]:
                    result[i] = 2
                    found = True
                    break
            if found:
                continue
            if num_tok[i] >= 4096 or num_out[i] >= max_tok[i]:
                result[i] = 3

    for _ in range(10):
        py_stop()
    t0 = time.perf_counter()
    for _ in range(iters):
        py_stop()
    results["python"] = (time.perf_counter() - t0) / iters

    return results


def bench_batch_apply_spec_decode(
    requests: list[MockRequest], iters: int
) -> dict[str, float]:
    """Benchmark: spec decode accept/reject computation."""
    n = len(requests)
    computed = np.array([r.num_computed_tokens for r in requests], dtype=np.int64)
    placeholders = np.array(
        [r.num_output_placeholders for r in requests], dtype=np.int64
    )
    generated = np.array(
        [int(np.random.randint(1, 6)) for _ in requests], dtype=np.int64
    )
    draft = np.array(
        [int(np.random.randint(0, 6)) for _ in requests], dtype=np.int64
    )

    results = {}

    if HAS_RUST:
        for _ in range(50):
            batch_apply_generated_tokens(computed, placeholders, generated, draft)
        t0 = time.perf_counter()
        for _ in range(iters):
            batch_apply_generated_tokens(computed, placeholders, generated, draft)
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline
    def py_apply():
        for i in range(n):
            if draft[i] > 0 and generated[i] > 0:
                na = generated[i] - 1
                nr = draft[i] - na
                _ = computed[i] - nr if computed[i] > 0 else computed[i]

    for _ in range(10):
        py_apply()
    t0 = time.perf_counter()
    for _ in range(iters):
        py_apply()
    results["python"] = (time.perf_counter() - t0) / iters

    return results


def bench_batch_hash_blocks(n_blocks: int, iters: int) -> dict[str, float]:
    """Benchmark: prefix caching block hash computation."""
    block_size = 16
    tokens = list(range(n_blocks * block_size))
    parent = b"\x00" * 16

    results = {}

    if HAS_RUST:
        for _ in range(50):
            batch_hash_blocks(parent, tokens, block_size)
        t0 = time.perf_counter()
        for _ in range(iters):
            batch_hash_blocks(parent, tokens, block_size)
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline (sha256 chain)
    def py_hash():
        prev = parent
        for i in range(0, len(tokens), block_size):
            blk = tokens[i : i + block_size]
            data = prev + b"".join(
                t.to_bytes(4, "little", signed=True) for t in blk
            )
            prev = hashlib.sha256(data).digest()

    for _ in range(5):
        py_hash()
    t0 = time.perf_counter()
    for _ in range(max(1, iters // 10)):
        py_hash()
    results["python"] = (time.perf_counter() - t0) / max(1, iters // 10)

    return results


def bench_free_block_queue(capacity: int, iters: int) -> dict[str, float]:
    """Benchmark: FreeKVCacheBlockQueue popleft_n + append_n cycle."""
    results = {}

    if HAS_RUST:
        q = RustFreeBlockQueue(list(range(capacity)), capacity)
        for _ in range(10):
            ids = q.popleft_n(capacity)
            q.append_n(ids)
        t0 = time.perf_counter()
        for _ in range(iters):
            ids = q.popleft_n(capacity)
            q.append_n(ids)
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline: attribute-based linked list
    class Node:
        __slots__ = ("bid", "prev", "nxt")
        def __init__(self, b: int):
            self.bid = b
            self.prev: Node | None = None
            self.nxt: Node | None = None

    nodes = [Node(i) for i in range(capacity)]

    def rebuild():
        for i in range(capacity - 1):
            nodes[i].nxt = nodes[i + 1]
            nodes[i + 1].prev = nodes[i]
        nodes[0].prev = None
        nodes[-1].nxt = None

    def py_cycle():
        rebuild()
        # popleft_n
        popped = []
        cur = nodes[0]
        for _ in range(capacity):
            popped.append(cur)
            nxt = cur.nxt
            cur.prev = cur.nxt = None
            cur = nxt
        # append_n
        for i in range(len(popped) - 1):
            popped[i].nxt = popped[i + 1]
            popped[i + 1].prev = popped[i]

    for _ in range(3):
        py_cycle()
    t0 = time.perf_counter()
    for _ in range(max(1, iters // 10)):
        py_cycle()
    results["python"] = (time.perf_counter() - t0) / max(1, iters // 10)

    return results


def bench_stop_string_matcher(iters: int) -> dict[str, float]:
    """Benchmark: Aho-Corasick vs Python str.find() for stop strings."""
    import random
    import string

    random.seed(42)
    text = "".join(random.choices(string.ascii_letters + " ", k=10000))
    stops = ["ENDOFTEXT", "</s>", "###", "STOP", "END"]

    results = {}

    if HAS_RUST:
        matcher = StopStringMatcher(stops)
        for _ in range(100):
            matcher.check(text, 50, False)
        t0 = time.perf_counter()
        for _ in range(iters):
            matcher.check(text, 50, False)
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline
    def py_check():
        for s in stops:
            text.find(s, max(0, len(text) - 50 - len(s) + 1))

    for _ in range(100):
        py_check()
    t0 = time.perf_counter()
    for _ in range(iters):
        py_check()
    results["python"] = (time.perf_counter() - t0) / iters

    return results


def bench_simulated_step(
    requests: list[MockRequest], iters: int
) -> dict[str, float]:
    """Simulate a full EngineCore step (CPU side only).

    This combines all the operations that happen in one scheduling step:
    1. compute_running_tokens (schedule)
    2. batch_check_stop (update_from_output)
    3. batch_apply_generated_tokens (spec decode)

    Measures total CPU overhead per step.
    """
    n = len(requests)
    rng = np.random.default_rng(123)

    # Prepare arrays (simulating the collection phase)
    spec = np.array([r.num_tokens_with_spec for r in requests], dtype=np.int64)
    ph = np.array([r.num_output_placeholders for r in requests], dtype=np.int64)
    comp = np.array([r.num_computed_tokens for r in requests], dtype=np.int64)
    prompt = np.array([r.num_prompt_tokens for r in requests], dtype=np.int64)
    maxt_arr = np.array([r.max_tokens for r in requests], dtype=np.int64)
    last_tok = rng.integers(0, 32000, size=n, dtype=np.int64)
    num_tok = np.array([r.num_tokens for r in requests], dtype=np.int64)
    num_out = np.array([r.num_output_tokens for r in requests], dtype=np.int64)
    min_tok = np.zeros(n, dtype=np.int64)
    eos = np.full(n, 2, dtype=np.int64)
    stop_arr = np.full((n, 3), -1, dtype=np.int64)
    gen = rng.integers(1, 4, size=n, dtype=np.int64)
    draft = rng.integers(0, 4, size=n, dtype=np.int64)

    results = {}

    if HAS_RUST:
        # Warmup
        for _ in range(20):
            compute_running_tokens(
                spec, ph, comp, prompt, maxt_arr, 100000, 0, 4096
            )
            batch_check_stop(
                last_tok, num_tok, num_out, min_tok, maxt_arr, eos, stop_arr, 4096
            )
            batch_apply_generated_tokens(comp, ph, gen, draft)

        t0 = time.perf_counter()
        for _ in range(iters):
            compute_running_tokens(
                spec, ph, comp, prompt, maxt_arr, 100000, 0, 4096
            )
            batch_check_stop(
                last_tok, num_tok, num_out, min_tok, maxt_arr, eos, stop_arr, 4096
            )
            batch_apply_generated_tokens(comp, ph, gen, draft)
        results["rust"] = (time.perf_counter() - t0) / iters

    # Python baseline
    def py_step():
        # 1. compute running tokens
        budget = 100000
        for i in range(n):
            if budget <= 0:
                break
            num_new = int(spec[i] + ph[i] - comp[i])
            num_new = min(num_new, budget, 4095 - int(comp[i]))
            num_new = max(num_new, 0)
            budget -= num_new

        # 2. check stop
        for i in range(n):
            if num_out[i] < min_tok[i]:
                continue
            if eos[i] >= 0 and last_tok[i] == eos[i]:
                continue
            if num_tok[i] >= 4096 or num_out[i] >= maxt_arr[i]:
                continue

        # 3. spec decode
        for i in range(n):
            if draft[i] > 0 and gen[i] > 0:
                _ = gen[i] - 1
                _ = draft[i] - (gen[i] - 1)

    for _ in range(10):
        py_step()
    t0 = time.perf_counter()
    for _ in range(iters):
        py_step()
    results["python"] = (time.perf_counter() - t0) / iters

    return results


# ── Main ──

def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:>8.1f} μs"
    return f"{seconds * 1e3:>8.2f} ms"


def print_result(name: str, results: dict[str, float], width: int = 40):
    py = results.get("python", 0)
    rs = results.get("rust")
    if rs is not None:
        speedup = py / rs if rs > 0 else float("inf")
        print(f"  {name:<{width}} Rust {fmt(rs)} | Py {fmt(py)} | {speedup:>6.1f}x")
    else:
        print(f"  {name:<{width}} Rust {'N/A':>11} | Py {fmt(py)} |    N/A")


def main():
    parser = argparse.ArgumentParser(
        description="CPU-side benchmark for Rust scheduler acceleration"
    )
    parser.add_argument(
        "-n", "--num-requests", type=int, default=1000,
        help="Number of simulated concurrent requests (default: 1000)",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=1000,
        help="Number of benchmark iterations (default: 1000)",
    )
    args = parser.parse_args()

    N = args.num_requests
    ITERS = args.iterations
    rng = np.random.default_rng(42)
    requests = make_mock_requests(N, rng)

    print(f"vllm._rs CPU Benchmark")
    print(f"  Rust module:  {'available' if HAS_RUST else 'NOT FOUND'}")
    print(f"  Requests:     {N}")
    print(f"  Iterations:   {ITERS}")
    print("=" * 75)

    # Individual benchmarks
    print("\n── Individual Operations ──\n")

    r = bench_compute_running_tokens(requests, ITERS)
    print_result(f"compute_running_tokens (N={N})", r)

    r = bench_batch_check_stop(requests, ITERS)
    print_result(f"batch_check_stop (N={N})", r)

    r = bench_batch_apply_spec_decode(requests, ITERS)
    print_result(f"batch_apply_generated (N={N})", r)

    r = bench_batch_hash_blocks(N, ITERS)
    print_result(f"batch_hash_blocks ({N} blocks)", r)

    r = bench_free_block_queue(min(N, 10000), ITERS)
    print_result(f"FreeBlockQueue ({min(N, 10000)} blocks)", r)

    r = bench_stop_string_matcher(ITERS * 10)
    print_result("StopStringMatcher (5 patterns)", r)

    # Simulated full step
    print("\n── Simulated Full Step (schedule + stop + spec) ──\n")

    r = bench_simulated_step(requests, ITERS)
    print_result(f"Full step CPU overhead (N={N})", r)

    py_step = r.get("python", 0)
    rs_step = r.get("rust")

    print("\n── Impact Estimate ──\n")
    gpu_times = [2, 5, 10, 25, 50]
    print(f"  {'GPU forward':>12}  {'Step (Py)':>12}  {'Step (Rust)':>12}  {'Saved':>8}  {'Speedup':>8}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")
    for gpu_ms in gpu_times:
        gpu_s = gpu_ms / 1000
        total_py = gpu_s + py_step
        total_rs = gpu_s + (rs_step if rs_step else py_step)
        saved_pct = (1 - total_rs / total_py) * 100 if total_py > 0 else 0
        speedup = total_py / total_rs if total_rs > 0 else 1
        print(
            f"  {gpu_ms:>10} ms  {total_py*1e3:>10.2f} ms  "
            f"{total_rs*1e3:>10.2f} ms  {saved_pct:>6.1f}%  {speedup:>6.2f}x"
        )

    print("\n" + "=" * 75)
    if not HAS_RUST:
        print(
            "NOTE: Rust module not found. Install with:\n"
            "  cd rust && maturin develop --release"
        )


if __name__ == "__main__":
    main()
