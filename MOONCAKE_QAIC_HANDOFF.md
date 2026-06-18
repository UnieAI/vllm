# QAIC Paged Attention + Mooncake — Box-side Handoff Checklist

Everything that can be written/verified **without** a QAIC card is done and committed.
This is the precise execution list for the **hardware-gated** remainder, with exact
files/functions, contracts, and how to verify each step on the box.

## Branches to bring to the box
- QEfficient: `external/efficient-transformers @ unieai/paged-kv` (paged ops, cache,
  Qwen2 threading, `export(paged_kv=True)`, `compile(paged_kv=True)`).
- vLLM plugin: `vllm @ roy/qaic-paged-kv` (paged plugin wiring; Mooncake staging arena
  + `QaicMooncakeStoreConnector` skeleton).

Install on the box (Linux, torch 2.7, qaicrt available):
```
uv pip install -e ./external/efficient-transformers
uv pip install -e ./vllm-qaic
```

---

## Critical contracts (read first)
- **`num_blocks` is a dynamic axis**: the `export(paged_kv=True)` dummy value does not
  fix it — `compile(..., num_blocks=N)` is authoritative (it injects the spec dim).
  Pass the SAME `N` to `compile()` and to the plugin's `--additional-config num_blocks`.
- **Null block**: `export`'s `paged_num_blocks` defaults to `base_bs*max_blocks + 1`
  (the +1 is the reserved padding null block the cache layer needs); the cache routes
  `position_ids<0` to block `num_blocks-1`. Size the pool with that spare block.
- **Plugin `num_blocks`**: `worker.py` reads it as the pool size (and requires it for
  paged); it does not currently cross-check against the QPC — keep them equal manually.
- **KV dtype / mxint8**: `compile(..., mxint8_kv_cache=True)` for paged KV is untested
  here; if used, ensure the staging arena dtype matches the on-card pool dtype.
- **Where the arena is wired**: `QaicModelRunner.initialize_kv_cache`
  (`model_runner.py:~814`) is currently a stub ("KV-transfer registration left out") —
  that is the hook to build+attach the arena and call `register_kv_caches()` (gated on
  `has_kv_transfer_group()`, `model_runner.py:~261`).

## Phase 1 — Paged attention on the card (unblocks everything else)

### 1.1 Compile a paged QPC (QEfficient)
```python
m   = QEFFAutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
onnx = m.export(paged_kv=True, paged_block_size=128)
qpc  = m.compile(onnx_path=onnx, paged_kv=True, page_size=128,
                 num_blocks=N, ctx_len=CTX, num_cores=16)   # N = pool size you choose
```
**Verify:** export produced `block_table`+`attention_mask` graph inputs and
`past_key.*` with a `page_size` dim (already CPU-checked by
`tests/customop/test_paged_export_plumbing.py`); the AIC compile must now succeed.
If the compiler rejects the block-pool `GatherND`/`ScatterND`, capture the error —
that is the one real unknown (perf/lowering of a large-pool gather).

### 1.2 Serve with paging (plugin)
```
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"paged_kv":true,"page_size":128,"num_blocks":N}'
```
`num_blocks` MUST equal the QPC's compiled pool size (`worker.py` reads it as the
pool size; keep it equal manually — see Critical contracts).

### 1.3 Step-3 go/no-go gate (the decision)
- **Accuracy:** same prompts, paged vs non-paged QPC → outputs must match (greedy)
  or be within sampling noise.
- **Throughput / FBS:** paging frees per-seq memory → recompile with a larger
  `full_batch_size`/`max_num_seqs`; measure tok/s + TTFT vs non-paged at the largest
  batch each fits. Expectation: more concurrent seqs → higher tok/s. (CPU memory model
  predicted ~7.5× KV saving for short-seq workloads.)
- **Perf of the gather:** confirm the block-pool gather doesn't dominate decode
  latency vs the contiguous path.

### 1.4 First Mooncake prerequisite (free here)
With paging on, prefix caching is enabled → confirm `Request.block_hashes` is
populated (was the original Mooncake-Store blocker). Quick check: log
`request.block_hashes` in the scheduler for a repeated-prefix workload.

---

## Phase 2 — Mooncake Store on QAIC (cross-instance KV reuse)

The host-side staging arena (`vllm_qaic/kv_connector/qaic_kv_staging.py`,
`QaicKVStagingArena`) is implemented + CPU-unit-tested. The connector
(`QaicMooncakeStoreConnector`) is a gated skeleton. Box-side TODOs:

### 2.1 Real `CardKVSessionAdapter` (the only QAIC-coupled I/O)
Implement an adapter (replacing `InMemoryCardKV`) with:
```python
read_blocks(layer_idx, block_ids)  -> (k, v) np.ndarray [len(block_ids), kv_heads, page, dim]
write_blocks(layer_idx, block_ids, k, v) -> None
```
over the QAIC session. The paged KV pool is a retained-state binding; read via
`session.create_numpy_buffers(..., direction="out", ...)` and write via
`session.set_data_for_kv_handoff(..., slicing_parameters=[(... block ...)])`
(see `vllm_qaic/session.py`). **Verify:** round-trip a known block on the card equals
the CPU arena (mirror of `test_qaic_kv_staging.py::test_roundtrip_integrity`).

### 2.2 Wire the arena into the model runner
In `QaicModelRunner.initialize_kv_cache` (when `has_kv_transfer_group()` and the
connector is a `QaicMooncakeStoreConnector`): build `QaicKVStagingArena(num_layers,
num_blocks, kv_heads, page_size, head_dim, dtype, real_adapter)`, then
`connector.attach_staging_arena(arena)` and `connector.register_kv_caches()` (which
registers the host mirror with Mooncake). **Verify:** `register_buffer` succeeds on
host memory (no card needed for the registration itself).

### 2.3 Block-id derivation (`_qaic_save_block_ids` / `_qaic_recv_block_ids`)
Override in `QaicMooncakeStoreConnector` to return the physical pool block ids being
saved/received this step, from the bound connector metadata (the same ids the base
computes Mooncake keys/addresses for — `store/data.py:prepare_value`,
`store/worker.py`). Then call `connector.enable_qaic_transfers()` (fails fast until
these are overridden). **Verify with a mock metadata unit test** (pure function:
metadata → block ids), like the arena tests.

### 2.4 Remove CUDA-event use on the CPU-only QAIC host
The base store worker uses `torch.cuda.Event()`/`.synchronize()`
(`vllm/distributed/.../mooncake/store/worker.py:575,1125`). Override `get_finished`
/ the save path in `QaicMooncakeStoreConnector` so QAIC copies (synchronous via the
session) don't hit CUDA. Also sweep `.record/.wait/.synchronize/current_stream`.

### 2.5 E2E (needs a running Mooncake store + 2 instances)
`MOONCAKE_CONFIG_PATH=mooncake.json` (master_server_address/protocol/...). Start two
QAIC instances with
`--kv-transfer-config '{"kv_connector":"QaicMooncakeStoreConnector","kv_role":"kv_both"}'`.
Send a long prompt to A, then the same prefix to B → B's TTFT drops, store get hits
(`VLLM_MOONCAKE_STORE_TIER_LOG=1`), output matches recompute.

---

## Phase 3 — P2P + MultiConnector (disaggregated prefill)

- `QaicMooncakeConnector(MooncakeConnector)`: same staging-arena pattern, but the P2P
  base has **deep CUDA assumptions** (`mooncake_connector.py:741-744,805-810,
  1351-1355,1410,1437`: `torch.accelerator.current_device_index`, thread-device
  binding, `cache.data_ptr()` registration). Plan a **larger override** than the Store
  case: bypass device binding, register the CPU staging arena, drop CUDA lifecycle.
- `MultiConnector` chain (P2P first, Store second) via
  `--kv-transfer-config '{"kv_connector":"MultiConnector","kv_connector_extra_config":
  {"connectors":[{...QaicMooncakeConnector...},{...QaicMooncakeStoreConnector...}]}}'`.
  **Verify** P2P-miss → Store-takes-over with a mock/unit test before e2e.

(Full design + risk order in `~/.claude/plans/pr-mooncake-sunny-wreath.md` PR#2.)

---

## What is already verified (no box needed)
- QEfficient paged: 8 CPU suites green; ops bit-exact under onnxruntime; full Qwen2
  logits bit-exact (prefill+decode); export+compile plumbing.
- Plugin paged wiring: syntax + review (gated; non-paged unchanged).
- Mooncake staging arena: CPU unit tests (block copy/isolation/roundtrip/dtype/range).
- 7 clean-subagent review rounds; all actionable findings fixed.

## Known pre-existing (upstream, not introduced here)
- `modeling_qwen2.py` blocking branch `comp_ctx_length=` typo (drops CCL in the
  blocked path); left untouched.
