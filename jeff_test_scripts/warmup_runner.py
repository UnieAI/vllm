#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup runner: sends requests from neural-bridge/rag-dataset-12000 to vLLM.

Downloads the dataset from HuggingFace (cached after first run), formats
each sample as "{context}\n{question}", and sends v1/completions requests
concurrently using asyncio + aiohttp.
"""

import argparse
import asyncio
import time

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp")
    raise SystemExit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets is required. Install with: pip install datasets")
    raise SystemExit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm is required. Install with: pip install tqdm")
    raise SystemExit(1)


def load_prompts(max_requests: int) -> list[str]:
    """Load prompts from neural-bridge/rag-dataset-12000 train split."""
    print("Loading dataset neural-bridge/rag-dataset-12000 (train)...")
    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    prompts = []
    for i, sample in enumerate(ds):
        if i >= max_requests:
            break
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n{question}"
        prompts.append(prompt)
    print(f"Loaded {len(prompts)} prompts.")
    return prompts


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send a single completion request."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    async with semaphore:
        try:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                return {"status": resp.status, "ok": resp.status == 200}
        except Exception as e:
            return {"status": 0, "ok": False, "error": str(e)}


async def run_warmup(
    base_url: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
):
    """Send all prompts concurrently with bounded concurrency."""
    # Discover model name from the server
    models_url = f"{base_url}/v1/models"
    async with aiohttp.ClientSession() as session, session.get(models_url) as resp:
        models_data = await resp.json()
        model = models_data["data"][0]["id"]
    print(f"Using model: {model}")

    completions_url = f"{base_url}/v1/completions"
    semaphore = asyncio.Semaphore(concurrency)

    total = len(prompts)
    errors = 0
    start_time = time.monotonic()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        tasks = []
        for prompt in prompts:
            task = asyncio.create_task(
                send_request(
                    session, completions_url, prompt, model, max_tokens, semaphore
                )
            )
            tasks.append(task)

        # Process results with tqdm progress bar
        pbar = tqdm(total=total, desc="Warmup", unit="req")
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if not result["ok"]:
                errors += 1
            pbar.update(1)
            elapsed = time.monotonic() - start_time
            rps = pbar.n / elapsed if elapsed > 0 else 0
            pbar.set_postfix(rps=f"{rps:.1f}", errors=errors)
        pbar.close()

    elapsed = time.monotonic() - start_time
    print("\nSummary:")
    print(f"  Total requests:  {total}")
    print(f"  Successful:      {total - errors}")
    print(f"  Errors:          {errors}")
    print(f"  Duration:        {elapsed:.1f}s")
    print(f"  Throughput:      {total / elapsed:.1f} req/s")


def main():
    parser = argparse.ArgumentParser(description="Warmup vLLM with RAG dataset")
    parser.add_argument("--base-url", default="http://localhost:2167")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=9600)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    prompts = load_prompts(args.max_requests)
    asyncio.run(
        run_warmup(
            base_url=args.base_url,
            prompts=prompts,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
