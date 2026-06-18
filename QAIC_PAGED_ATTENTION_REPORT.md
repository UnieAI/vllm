# QAIC Paged Attention — Implementation Report

**Date:** 2026-06-07  **Author:** roy-shih (AI-assisted, Claude)
**Scope:** Bring vLLM-style paged attention to Qualcomm Cloud AI 100 (QAIC), as the
prerequisite for Mooncake KV-transfer integration on the UnieAI vLLM fork.

---

## 1. Executive summary

QAIC did **not** support paged attention: every sequence reserved a full
`max_model_len` KV slot on-card, wasting memory and capping concurrency. This work
adds true **block-table paged KV** to the Qualcomm stack and wires the vLLM-QAIC
plugin to use it.

**Status:** All paged-attention code that can be written/verified without the QAIC
card is **complete and verified on CPU** — across two repositories. The only
remaining work is the AIC binary compile + on-card accuracy/throughput measurement,
which fundamentally require the hardware.

**Headline result (measured, CPU):** paged KV is **bit-exact** vs the contiguous
path (eager + onnxruntime) and uses **~7.5× less KV memory** (86.7% saving) for a
short-sequences-in-long-context workload — the memory that, on-card, converts into a
larger compilable `full_batch_size` (more decode lanes → higher throughput).

---

## 2. The core problem (why this was non-trivial)

- QAIC runs a **pre-compiled QPC** (Qualcomm Program Container) built by **QEfficient**
  (`quic/efficient-transformers`). The vLLM plugin only loads the `.qpc`; it cannot
  change the attention graph.
- Confirmed in the plugin source: `platform.py` set `block_size = max_model_len`,
  disabled prefix caching, `num_gpu_blocks = max_num_seqs+1`; `worker.py` literally
  comments `# QAIC does not support paged attention`.
- Therefore **true paged attention requires changing QEfficient** (the graph), not
  just the plugin.

**Key research finding that made it feasible:** QEfficient's KV read/write already
lower to ONNX `GatherND`/`ScatterND` with **data-dependent index tensors** (the
`CtxGatherCB`/`CtxScatterCB` ops used by continuous batching). The AIC compiler thus
already supports indexed gather/scatter — the hard primitive of paged attention. What
was missing was a **block-pool layout + block_table indirection** built on top.

---

## 3. What was built

### Repo A — QEfficient (`external/efficient-transformers`, branch `unieai/paged-kv`, 17 commits)

| Component | File | What |
|---|---|---|
| Paged ops | `QEfficient/customop/ctx_paged_scatter_gather.py` | `CtxScatterPagedFunc` / `CtxGatherPagedFunc`: address a block pool `[num_blocks, heads, page_size, head_dim]` via per-token `(block, offset)` indices; lower to ScatterND/GatherND |
| Paged cache | `QEfficient/transformers/cache_utils.py` | `QEffPagedDynamicLayer` / `QEffPagedDynamicCache`: resolve positions via `block_table`, route padding to a reserved null block, CCL-aware gather |
| Model threading | `models/qwen2/modeling_qwen2.py`, `blocking/attention_blocking.py` | thread optional `block_table` through the Qwen2 forward chain + shared `past_key_value_update`; select the paged cache when `block_table` is supplied |
| Export | `models/modeling_auto.py` `export(paged_kv=True, paged_block_size, paged_num_blocks)` | exports block-pool KV + `block_table`/`attention_mask` graph inputs (dyn axes `num_blocks`/`page_size`/`max_num_blocks`) |
| Compile | `models/modeling_auto.py` `compile(paged_kv=True, page_size, num_blocks)` | injects the block-pool dims into every specialization; doesn't leak paged kwargs to the AIC compiler |

**Architecture (before → after):**
```
BEFORE  KV pool [fbs, heads, ctx_len, dim]   per-seq full-ctx slot; waste = ctx_len
AFTER   KV pool [num_blocks, heads, page, dim] shared pool; waste <= page_size
        phys = block_table[seq][pos//page]*page + pos%page  (same ScatterND/GatherND)
```
Only the index computation and one `block_table` input change; the attention math and
op vocabulary are unchanged.

### Repo B — vLLM-QAIC plugin (`vllm`, branch `roy/qaic-paged-kv`, 1 commit)

Gated by `--additional-config '{"paged_kv":true,"page_size":N,"num_blocks":M}'`
(non-paged path byte-for-byte unchanged):

| File | Change |
|---|---|
| `platform.py` | paged → `block_size = page_size`, allow prefix caching |
| `worker.py` | paged → `num_gpu_blocks` = block-pool size; skip the `max_num_seqs+1` assumption |
| `model_runner.py` | paged → feed the **full** per-request `block_table` rows to the model |
| `model_loader.py` | thread `block_table` into `_run_prefill`/`_run_decode` session inputs (key `"block_table"`) |

---

## 4. Verification

**8 CPU test suites (QEfficient), all green** — every step reports time / peak-RSS / precision:

| Test | Validates | Result |
|---|---|---|
| `test_paged_kv_parity` | ops vs contiguous (prefill+decode, shuffled block_table, isolation, CCL) | max_abs_diff 0.0 |
| `test_paged_cache_layer` | paged cache layer vs proven CB layer (+ padding/null-block) | 0.0 |
| `test_paged_qwen2_e2e` | full Qwen2 logits, paged vs contiguous, prefill + 3-step decode | 0.0 |
| `test_paged_onnx_export` | **ops run under onnxruntime vs eager** (symbolic path) | 0.0 |
| `test_paged_qwen2_onnx` | full model exports with block_table as a graph input | structural ✓ |
| `test_paged_export_plumbing` | real `export(paged_kv=True)` produces the paged graph | ✓ |
| `test_paged_compile_spec` | `compile(paged_kv=True)` injects pool dims into specs | ✓ |
| `bench_paged_vs_contiguous` | time / precision / memory | 0.0 prec; 7.5× mem |

vLLM plugin: syntax-checked (can't import off-card) + reviewed.

**Review process:** 6 independent clean-subagent code reviews (5 on QEfficient, 1 on
the plugin). **Two real bugs were caught that the eager tests could not** — both on
the ONNX symbolic path the AIC compiler consumes:
1. ScatterND index `Concat` mixed int32/int64 → fixed (all paged indices int64).
2. `Range` used 1-D args; ONNX spec needs scalars → fixed (scalar Range).
These were surfaced specifically by the eager-vs-onnxruntime numeric test.

---

## 5. What remains (box-gated — needs a QAIC host)

Only steps that require the Qualcomm AIC compiler / card:
1. **AIC binary compile** of the paged QPC (the compiler that turns ONNX+specs → `.qpc`).
2. **On-card accuracy** vs the contiguous baseline.
3. **Throughput / FBS go/no-go** (plan Step 3): paging frees KV memory → compile a
   larger `full_batch_size` → measure tok/s and TTFT.

There is **no remaining CPU-writable paged code**; the full path (`export` → `compile`
→ plugin runtime) is wired.

### How to run on the box
```python
m   = QEFFAutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-...")
onnx = m.export(paged_kv=True, paged_block_size=128)
qpc  = m.compile(onnx_path=onnx, paged_kv=True, page_size=128, num_blocks=N, ctx_len=CTX)
# serve:
# vllm serve Qwen/Qwen2.5-... \
#   --additional-config '{"paged_kv":true,"page_size":128,"num_blocks":N}'
# measure: accuracy vs non-paged; throughput at the larger full_batch_size paging affords.
```

---

## 6. Environment notes
- CPU dev/test venv: `external/efficient-transformers/.venv-cputest` (torch 2.12,
  onnx/onnxscript/onnxruntime, transformers 4.57.3, diffusers, peft) so `import
  QEfficient` works on macOS. The box uses torch 2.7 (classic ONNX exporter, default);
  tests force the classic exporter where needed to match.
- `gh` CLI is not installed; opening the QEfficient PR needs a UnieAI fork + remote.

---

## 7. Recommended next steps
1. **Open the QEfficient PR**: fork `quic/efficient-transformers` → UnieAI, then
   `git push -u unieai unieai/paged-kv` (description in `PAGED_KV_PR.md`).
2. **Box validation** (Steps 1–3 above) when the QAIC host is reachable.
3. **Mooncake-QAIC** (the original goal, now **unblocked**): paged mode re-enables
   prefix caching, so `Request.block_hashes` regenerate — the blocker for
   `MooncakeStoreConnector` is gone. Implement `QaicMooncakeStoreConnector` + staging
   arena + P2P + MultiConnector (all CPU-writable). See the plan file
   `~/.claude/plans/pr-mooncake-sunny-wreath.md`.

---

## Appendix A — architecture diagrams

**Two repos: A = "how the card computes", B = "what data to feed at runtime".**
```
 vLLM engine (schedules; KVCacheManager -> block_table[req] = [physical block ids])
        │ block_table
        ▼  Repo B: vLLM-QAIC plugin
   platform(block_size=page_size, prefix cache) → worker(pool size)
     → model_runner(extract FULL block_table rows) → model_loader(feed "block_table" to QPC)
        │ numpy: input_ids / positions / block_table
        ▼  on-card QPC  (compiled by Repo A)
   attention uses block_table to gather KV from a scattered pool
        ▲ export(paged_kv=True) → compile(paged_kv=True)
        │  Repo A: QEfficient (paged ops + cache + Qwen2 threading)
```

**KV memory: BEFORE vs AFTER** (page_size=4, ctx_len=16; req A=6 tokens, B=3):
```
BEFORE  [num_seqs, heads, ctx_len=16, dim]
  slot0(A): A0 A1 A2 A3 A4 A5 . . . . . . . . . .   (6 used, 10 wasted)
  slot1(B): B0 B1 B2 . . . . . . . . . . . . .       (3 used, 13 wasted)
  waste = ctx_len - actual_len  (huge for short seqs)

AFTER   [num_blocks, heads, page_size=4, dim]  + a block_table
  pool: blk0:[A0 A1 A2 A3] blk1:[A4 A5 _ _] blk2:[B0 B1 B2 _] blk3:[free] ...
  block_table: A.logical[0,1]->blk0,blk1 ;  B.logical[0]->blk2
  waste <= page_size  →  same memory fits MANY more concurrent sequences (~7.5x)
```

**Index math (read A's token at pos=5):**
```
logical_block = 5 // 4 = 1 ;  physical = block_table[A][1] = blk1 ;  offset = 5 % 4 = 1
KV = pool[blk1][..][1][..] = A5
formula:  physical = block_table[seq][pos//page]*page + pos%page
(same GatherND/ScatterND op as QEfficient continuous batching — only the index differs)
```

**Runtime data flow (one decode step):**
```
vLLM block_table → [B]model_runner: full_block_table = block_table[:num_reqs, :]
  → [B]model_loader: decode_inputs["block_table"] = [decode_bsz, max_blocks]; session.run(...)
  → [A-compiled QPC]: phys = block_table[seq][pos//page]*page + pos%page; CtxGatherPaged(pool, phys)
  → logits → vLLM samples next token
```

## Appendix B — branches & key paths
- QEfficient: `external/efficient-transformers @ unieai/paged-kv` (17 commits); PR notes `PAGED_KV_PR.md`.
- vLLM plugin: `vllm @ roy/qaic-paged-kv` (1 commit).
- Tests: `external/efficient-transformers/tests/customop/test_paged_*.py`, `bench_paged_vs_contiguous.py`.
- Plan: `~/.claude/plans/pr-mooncake-sunny-wreath.md`.
