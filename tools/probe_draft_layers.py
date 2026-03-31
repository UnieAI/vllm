#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Probe tool for Self-Speculative Decoding.

Tests different layer depths as draft models and reports quality metrics.
Use this to find the optimal `self_draft_depth` for your model.

Usage:
    python tools/probe_draft_layers.py \\
        --model meta-llama/Llama-3.1-8B \\
        --depths 4 8 12 16 24 \\
        --num-prompts 50

Output:
    A table showing, for each depth:
    - Draft speed (tokens/sec of the draft-only forward pass)
    - Agreement rate (how often the draft's top-1 matches the full model)
    - Recommended k (max draft length before quality drops below threshold)
"""

import argparse
import time
from itertools import islice

import torch
import torch.nn.functional as F


def load_model_and_tokenizer(model_name: str, dtype: str = "auto"):
    """Load model via HuggingFace transformers (not vLLM, for probing)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if dtype == "auto" else getattr(torch, dtype),
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def find_inner_model(model):
    """Find the transformer model with layer list."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer
    raise ValueError(f"Cannot find inner model in {type(model).__name__}")


def get_num_layers(model) -> int:
    inner = find_inner_model(model)
    if hasattr(inner, "layers"):
        return len(inner.layers)
    if hasattr(inner, "h"):
        return len(inner.h)
    raise ValueError("Cannot determine number of layers")


@torch.inference_mode()
def probe_depth(
    model,
    tokenizer,
    depth: int,
    prompts: list[str],
    max_new_tokens: int = 20,
) -> dict:
    """Measure draft quality at a given layer depth.

    Returns dict with:
        - agreement_rate: fraction of tokens where draft top-1 == full top-1
        - draft_time_per_token: average time for one draft forward pass
        - full_time_per_token: average time for one full forward pass
        - speedup: full_time / draft_time
        - recommended_k: max consecutive agreements before quality drops
    """
    inner = find_inner_model(model)

    # Save original layer bounds.
    if hasattr(inner, "end_layer"):
        orig_end = inner.end_layer
        set_end = lambda d: setattr(inner, "end_layer", d)  # noqa: E731
    elif hasattr(inner, "h"):
        # GPT-2 style
        orig_layers = list(inner.h)
        set_end = None  # handled differently
    else:
        raise ValueError("Unsupported model architecture")

    total_layers = get_num_layers(model)
    device = next(model.parameters()).device

    total_agree = 0
    total_tokens = 0
    consecutive_agree_counts = []
    draft_times = []
    full_times = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        for step in range(max_new_tokens):
            # ── Full model forward ──
            t0 = time.perf_counter()
            full_out = model(input_ids=input_ids)
            full_time = time.perf_counter() - t0
            full_logits = full_out.logits[:, -1, :]
            full_token = full_logits.argmax(dim=-1)

            # ── Draft model forward (first `depth` layers) ──
            if set_end is not None:
                set_end(depth)
            t0 = time.perf_counter()
            draft_out = model(input_ids=input_ids)
            draft_time = time.perf_counter() - t0
            if set_end is not None:
                set_end(orig_end)

            draft_logits = draft_out.logits[:, -1, :]
            draft_token = draft_logits.argmax(dim=-1)

            # ── Compare ──
            agree = (draft_token == full_token).item()
            total_agree += agree
            total_tokens += 1
            draft_times.append(draft_time)
            full_times.append(full_time)

            # Track consecutive agreements for recommended_k.
            if agree:
                if consecutive_agree_counts and consecutive_agree_counts[-1] > 0:
                    consecutive_agree_counts[-1] += 1
                else:
                    consecutive_agree_counts.append(1)
            else:
                consecutive_agree_counts.append(0)

            # Append the full model's token for next step.
            input_ids = torch.cat(
                [input_ids, full_token.unsqueeze(0)], dim=1
            )

            # Stop at EOS.
            if full_token.item() == tokenizer.eos_token_id:
                break

    agreement_rate = total_agree / max(total_tokens, 1)
    avg_draft = sum(draft_times) / max(len(draft_times), 1)
    avg_full = sum(full_times) / max(len(full_times), 1)

    # Recommended k: median consecutive agreement streak.
    streaks = [c for c in consecutive_agree_counts if c > 0]
    if streaks:
        streaks.sort()
        recommended_k = streaks[len(streaks) // 2]  # median
    else:
        recommended_k = 0

    return {
        "depth": depth,
        "total_layers": total_layers,
        "ratio": f"{depth}/{total_layers}",
        "agreement_rate": agreement_rate,
        "draft_ms": avg_draft * 1000,
        "full_ms": avg_full * 1000,
        "speedup": avg_full / max(avg_draft, 1e-9),
        "recommended_k": recommended_k,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe optimal draft depth for self-speculative decoding"
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="Layer depths to probe (default: auto-select 25%%, 50%%, 75%%)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to test",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Max tokens to generate per prompt",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype (auto, float16, bfloat16)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    total_layers = get_num_layers(model)
    print(f"Model has {total_layers} layers")

    # Default depths: 25%, 50%, 75% of total layers.
    if args.depths is None:
        args.depths = sorted(set([
            max(1, total_layers // 4),
            max(1, total_layers // 2),
            max(1, total_layers * 3 // 4),
        ]))

    # Filter invalid depths.
    args.depths = [d for d in args.depths if 1 <= d < total_layers]
    if not args.depths:
        print(f"No valid depths for a {total_layers}-layer model")
        return

    # Simple test prompts.
    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "In machine learning, the gradient descent algorithm",
        "The quick brown fox jumps over the",
        "To implement a binary search tree in Python,",
        "According to the latest research in quantum computing,",
        "The HTTP protocol defines several request methods including",
        "In the year 2024, artificial intelligence has",
        "The Fibonacci sequence starts with 0, 1,",
        "To optimize database queries, you should consider",
    ] * (args.num_prompts // 10 + 1)
    prompts = prompts[: args.num_prompts]

    # ── Probe each depth ──
    results = []
    for depth in args.depths:
        print(f"\nProbing depth={depth}/{total_layers}...")
        result = probe_depth(
            model, tokenizer, depth, prompts, args.max_new_tokens
        )
        results.append(result)
        print(
            f"  Agreement: {result['agreement_rate']:.1%}  "
            f"Draft: {result['draft_ms']:.1f}ms  "
            f"Full: {result['full_ms']:.1f}ms  "
            f"Speedup: {result['speedup']:.2f}x  "
            f"Rec. k: {result['recommended_k']}"
        )

    # ── Summary table ──
    print("\n" + "=" * 80)
    print(f"{'Depth':>8} {'Ratio':>8} {'Agree%':>8} {'Draft':>8} "
          f"{'Full':>8} {'Speedup':>8} {'Rec. k':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['depth']:>8} {r['ratio']:>8} "
            f"{r['agreement_rate']:>7.1%} "
            f"{r['draft_ms']:>7.1f}ms "
            f"{r['full_ms']:>7.1f}ms "
            f"{r['speedup']:>7.2f}x "
            f"{r['recommended_k']:>8}"
        )
    print("=" * 80)

    # ── Recommendation ──
    # Find the depth with best (agreement * speedup) product.
    best = max(results, key=lambda r: r["agreement_rate"] * r["speedup"])
    print(f"\nRecommendation: --self-draft-depth {best['depth']}")
    print(f"  {best['ratio']} layers, {best['agreement_rate']:.0%} agreement, "
          f"{best['speedup']:.1f}x draft speedup, k={best['recommended_k']}")

    if best["agreement_rate"] < 0.3:
        print(
            "\n⚠️  Low agreement rate. Self-speculative decoding may not be "
            "beneficial for this model. Consider using EAGLE or a dedicated "
            "draft model instead."
        )


if __name__ == "__main__":
    main()
