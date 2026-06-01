# QAIC → vLLM 0.21+ migration: the COMPLETE change-set

This is the full work-list for porting the Qualcomm QAIC backend (fork
`UnieAI/vllm@v1_ngram`, based on upstream **v0.10.1**) onto **vLLM 0.21/0.22**,
re-packaged as the out-of-tree plugin `vllm-qaic`.

It covers **every file** the Qualcomm "qaic patch" (`bd90d0d`) touched
(45 new + 40 modified core files), plus UnieAI's own changes, plus the deep
`GPUModelRunner` old/new comparison.

> **Confidence:** categorizations marked `⚠confirm` are inferred from file
> name + diff size and have NOT been line-read; verify before relying on them.
> Everything else is grounded in code we read.

## Legend (the 5 buckets)

| Tag | Meaning | Where it goes |
|---|---|---|
| 🟢 **PLUGIN** | expressible via an official extension point — no core edit | `vllm-qaic` package |
| 🔵 **PORT** | QAIC logic that must be carried over (subclass / copy) | `vllm-qaic` package |
| 🟠 **CORE-PATCH** | small unavoidable edit to vLLM core, or upstream PR | a thin patch / PR |
| ⚪ **DROP-V0** | only used by the V0 engine; irrelevant for a V1-only deployment | not ported |
| ⚫ **OPTIONAL** | a separate feature (gpt-oss, disagg, pooling, MM) — port only if you need it | per-feature |

---

## 1. Strategy at a glance

The fork modifies ~40 core files. The plugin model + V1-only scope collapses
that dramatically:

- **~half are ⚪ DROP-V0** (V0 scheduler / block manager / sequence / engine /
  spec-decode infra). Gone for a V1-only deployment.
- **Most of the rest are 🟢 PLUGIN** (platform, quant, KV connector, model
  registry, CLI knobs → `--additional-config`).
- **The genuinely unavoidable core edits (🟠) are a short list** — see §6.
- **The real engineering is 🔵 PORT**, concentrated in `QaicModelRunner` (§4).

---

## 2. NEW files added by the qaic patch (45)

### 2a. Port into the plugin

| File | Bucket | Notes |
|---|---|---|
| `vllm/v1/worker/qaic_model_runner.py` | 🔵 PORT | → `vllm_qaic/model_runner.py`. **The hard one** (§4). |
| `vllm/v1/worker/qaic_worker.py` | 🔵 PORT | → `vllm_qaic/worker.py`. Subclass `WorkerBase`. |
| `vllm/model_executor/model_loader/qaic_v1.py` | 🔵 PORT | → `vllm_qaic/model_loader.py`. QPC load/compile. |
| `vllm/model_executor/model_loader/qaic.py` | 🔵 PORT | → `vllm_qaic/compile_config.py`. `_clean_config()`. |
| `vllm/model_executor/model_loader/qaic_session_np.py` | 🔵 PORT | → `vllm_qaic/session.py`. **BSD-3-Clause** (others are proprietary). Pure numpy+qaicrt. |
| `vllm/platforms/qaic.py` | 🟢→🔵 | → `vllm_qaic/platform.py` (already drafted, OOT enum). |
| `vllm/model_executor/layers/quantization/qaic_quant.py` | 🟢 PLUGIN | → `vllm_qaic/quant.py` via `register_quantization_config`. |
| `vllm/distributed/kv_transfer/kv_connector/qaic_connector.py` | ⚫ OPTIONAL | only for disaggregated prefill/decode. Register via factory. |
| `vllm/model_executor/models/qaic_custom_mm_processor.py` | ⚫ OPTIONAL | only if serving multimodal models. |

### 2b. V0 QAIC files — DROP for a V1-only deployment

| File | Bucket |
|---|---|
| `vllm/worker/qaic_model_runner.py` | ⚪ DROP-V0 |
| `vllm/worker/qaic_worker.py` | ⚪ DROP-V0 |
| `vllm/worker/qaic_pooling_model_runner.py` | ⚪ DROP-V0 (⚫ if you need pooling) |
| `vllm/spec_decode/qaic_multi_step_worker.py` | ⚪ DROP-V0 (this is V0 draft-model SPD, not the V1 ngram path) |
| `vllm/core/block/qaic_prefix_caching_block.py` | ⚪ DROP-V0 |

### 2c. V0 spec-decode infrastructure the patch re-added — DROP for V1

`vllm/spec_decode/*` (re-added: `spec_decode_worker.py`, `multi_step_worker.py`,
`ngram_worker.py`, `top1_proposer.py`, `batch_expansion.py`, `mqa_scorer.py`,
`draft_model_runner.py`, `target_model_runner.py`, `medusa_worker.py`,
`mlp_speculator_worker.py`, `smaller_tp_proposer_worker.py`,
`proposer_worker_base.py`, `interfaces.py`, `metrics.py`, `util.py`,
`__init__.py`), plus `vllm/model_executor/layers/rejection_sampler.py`,
`spec_decode_base_sampler.py`, `typical_acceptance_sampler.py`, and
`vllm/engine/output_processor/multi_step.py` — all **⚪ DROP-V0**. V1 speculative
decoding lives in `vllm/v1/spec_decode/` and is provided by upstream.

### 2d. Examples — reference only (⚪/⚫, adapt as needed)
`examples/offline_inference/qaic*.py` (11 files).

---

## 3. MODIFIED core files (40) — full categorization

### 3a. 🟢 PLUGIN — replace with an extension point, no core edit

| File | +lines | What it did | Plugin replacement |
|---|---|---|---|
| `vllm/platforms/__init__.py` | 33 | qaic detection | `vllm.platform_plugins` entry point |
| `vllm/engine/arg_utils.py` | 74 | `--override-qaic-config`, `--device-group` | `--additional-config` dict |
| `vllm/config/__init__.py` | 126 | `override_qaic_config` / `device_group` fields + validation | read from `additional_config` |
| `vllm/model_executor/layers/quantization/__init__.py` | 9 | register `mxfp6` | `register_quantization_config("mxfp6")` |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 34 | register qaic connector | `KVConnectorFactory.register_connector()` |
| `vllm/model_executor/models/registry.py` | 10 | register qaic models | `ModelRegistry.register_model()` in `general_plugins` |
| `vllm/transformers_utils/configs/__init__.py` | 10 | register custom HF configs | register in `general_plugins` |
| `vllm/envs.py` | 8 | `VLLM_QAIC_*` env vars | plugin reads `os.environ` itself |

### 3b. 🟠 CORE-PATCH — small unavoidable edits (or upstream PR). See §6.

| File | +lines | Why a plugin can't do it |
|---|---|---|
| `vllm/config/cache.py` | 12 | adds `"mxint8"` to the **closed `CacheDType` Literal** + validation. Can't extend an enum from outside. |
| `vllm/platforms/interface.py` | 10 | adds Platform base methods/enum. ⚠confirm: most already exist upstream (`is_kv_cache_dtype_supported`, `OOT`); may be a no-op now. |
| `vllm/_custom_ops.py` | 8 | ⚠confirm: likely makes custom-op import degrade gracefully when no CUDA. May be unnecessary on 0.21. |
| `vllm/transformers_utils/config.py` | 15 | ⚠confirm: HF config-loading hook for qaic models. May be pluginable via the configs registry instead. |

### 3c. ⚪ DROP-V0 — not needed for a V1-only deployment

| File | +lines | Note |
|---|---|---|
| `vllm/core/scheduler.py` | 37 | V0 scheduler (V1 uses `vllm/v1/core`) |
| `vllm/core/block_manager.py` | 17 | V0 |
| `vllm/core/block/cpu_gpu_block_allocator.py` | 46 | V0 block allocator |
| `vllm/engine/llm_engine.py` | 54 | V0 engine |
| `vllm/engine/output_processor/interfaces.py` | 28 | V0 multi-step output |
| `vllm/sequence.py` | 36 | V0 sequence structures |
| `vllm/worker/worker_base.py` | 4 | V0 worker base |
| `vllm/model_executor/layers/sampler.py` | 6 | V0 sampler hook |
| `vllm/engine/metrics.py` / `metrics_types.py` | 87 / 14 | V0 metrics (⚫ if you want QAIC metrics on V1, re-do via V1 logging) |
| `vllm/config/scheduler.py` | 15 | ⚠confirm: check whether any line is V1-relevant |

### 3d. VERIFIED (each diff line-read against `bd90d0d`)

**Root-cause insight:** almost every QAIC-specific *core* touch reduces to TWO
mechanisms. Handle these two and most of the list disappears:
1. **"is this the QAIC platform?"** — the patch added `current_platform.is_qaic()`
   + a `QAIC` enum. On 0.21 the OOT platform has no such enum; use
   `current_platform.device_type == "qaic"` instead (no core edit).
2. **"QAIC has no torch custom ops / no CUDA `_C`"** — the patch makes
   `supports_custom_op()` return `False` on QAIC and skips `vllm._C`. Once that
   is true, the mxfp4 / gguf / bitsandbytes "graceful-degradation" fallbacks
   **trigger automatically** — they are not separate work.

#### QAIC-required core touches (small)

| File | +lines | Verified: what it does | Bucket |
|---|---|---|---|
| `vllm/utils/__init__.py` | 10 | (1) add `"mxint8": torch.uint8` dtype; (2) `supports_custom_op()` → `False` on QAIC | 🟠 CORE-PATCH (root cause #2) — or replicate in plugin init |
| `vllm/_custom_ops.py` | 8 | skip `import vllm._C` on QAIC (no CUDA ops) | 🟢 PLUGIN/🟠 small — likely unneeded if OOT platform never builds `_C` |
| `vllm/platforms/interface.py` | 10 | add `QAIC` enum + `is_qaic()` | 🟢 AVOIDED — use `device_type=="qaic"` on the OOT platform |
| `vllm/env_override.py` | 10 | skip `torch._inductor` thread config on QAIC | ⚪ covered by `TORCH_COMPILE_DISABLE=1` already in your launch |
| `vllm/transformers_utils/config.py` | 15 | mllama: force `is_encoder_decoder=False` on QAIC (no cross-attn) + 2 new model configs | ⚫ OPTIONAL(mllama) — new configs likely already upstream |

#### Optional features — NOT needed for basic V1 chat (Qwen2.5, mxfp6, fp8)

| File | +lines | Verified: what it does | Bucket |
|---|---|---|---|
| `vllm/entrypoints/openai/serving_chat.py` | 252 | kimi_k2 tool-call IDs + harmony actions + `return_token_ids` debug streaming | ⚫ OPTIONAL(tool-calling / gpt-oss / debug) |
| `vllm/entrypoints/openai/protocol.py` | 72 | optional fields: `return_token_ids`, timing, embedding-bytes | ⚫ OPTIONAL — backwards-compat, likely upstream |
| `vllm/entrypoints/openai/serving_pooling.py` | 104 | embedding encoding formats (float/base64/**bytes**) | ⚫ OPTIONAL(embeddings) |
| `vllm/entrypoints/openai/api_server.py` | 9 | dispatch `PoolingBytesResponse` | ⚫ OPTIONAL(embeddings) |
| `vllm/entrypoints/chat_utils.py` | 17 | kimi_k2 tool-call-id helpers | ⚫ OPTIONAL(tool-calling) |
| `vllm/entrypoints/openai/tool_parsers/__init__.py` | 5 | register `OpenAIToolParser` | 🟢 PLUGIN (ToolParserManager) |
| `vllm/entrypoints/harmony_utils.py` | 106 | gpt-oss function-calling | ⚫ OPTIONAL(gpt-oss) |
| `vllm/reasoning/gptoss_reasoning_parser.py` | 45 | gpt-oss reasoning parse | ⚫ OPTIONAL(gpt-oss) |
| `vllm/model_executor/models/config.py` | 6 | rename gpt-oss reasoning backend | ⚫ IGNORE — likely upstream, not QAIC |
| `vllm/distributed/kv_transfer/kv_connector/base.py` | 142 | re-add V0 `KVConnectorBase` (disagg) | ⚪ DROP-V0 / ⚫ OPTIONAL(disagg) |
| `vllm/distributed/kv_transfer/kv_transfer_state.py` | 9 | V0 connector init path | ⚪ DROP-V0 / ⚫ OPTIONAL(disagg) |
| `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` | 22 | pure-Python fallback when no custom op | ⚫ OPTIONAL(mxfp4) — auto via root cause #2 |
| `vllm/model_executor/layers/quantization/gguf.py` | 29 | pure-Python fallback when no custom op | ⚫ OPTIONAL(gguf) — auto via root cause #2 |
| `vllm/model_executor/layers/quantization/bitsandbytes.py` | 10 | skip bnb custom op on QAIC | ⚫ OPTIONAL(bnb) — auto via root cause #2 |
| `setup.py` | 40 | QAIC build detection / SDK version / `qaic.txt` | 🟢 N/A — plugin has its own `pyproject.toml` |

---

## 4. `GPUModelRunner`: v0.10.1 (fork base) vs 0.21/0.22 (target) — DEEP DIVE

`QaicModelRunner` subclasses `GPUModelRunner`, refactored from ~2–3k lines to
7000+. This is the core 🔵 PORT work. Source files:
- OLD: fork `vllm/v1/worker/qaic_model_runner.py` (819 lines).
- NEW: installed `vllm/v1/worker/gpu_model_runner.py`.

### 4.A Class & constructor

| | OLD | NEW | Action |
|---|---|---|---|
| Base classes | `GPUModelRunner(...)` | `GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin)` | inherits fine; be aware of mixins |
| `__init__` sig | `(vllm_config, device, speculative_model_type=None)` | `(vllm_config, device)` | **drop 3rd param**; derive internally (done) |

### 4.B Input-prep data model — **the biggest break**

| Fork attribute (used by QAIC) | In 0.21? | 0.21 replacement |
|---|---|---|
| `self.positions_np` | ❌ | `self.positions` / recompute from `num_computed_tokens_cpu` |
| `self.cu_num_tokens` | ❌ | computed in `_prepare_inputs`; `self.query_start_loc` |
| `self.num_decodes` | ❌ | recompute via `reorder_batch_to_split_decodes_and_prefills` |
| `self.input_ids_cpu` | ❌ | `self.input_ids` (`CpuGpuBuffer`) + `InputBatch.token_ids_cpu` |
| `self.batch_indices` | ❌ | `InputBatch.block_table[...]` |
| `self.arange_np` | ❌ | rebuild locally |
| input batch fields | different | `InputBatch` (`gpu_input_batch.py`): `token_ids_cpu`, `num_tokens_no_spec`, `num_computed_tokens_cpu`, `block_table`, `sampling_metadata` |

**Consequence:** rewrite `_prepare_qaic_inputs()` / `_postprocess_tensors()` to
source from `scheduler_output` + `InputBatch`. Most time goes here.

### 4.C Forward + sample — **architectural split**

| | OLD | NEW | Action |
|---|---|---|---|
| `execute_model` | one method: prep → QPC forward → sample → `ModelRunnerOutput` | may return `None` and stash `self.execute_model_state`; `sample_tokens(grammar_output)` does sampling | QAIC forward is synchronous/host-driven → do the full QAIC path inside `execute_model`, return `ModelRunnerOutput` directly, bypass `sample_tokens`. Verify base permits this. |

### 4.D Speculative decoding

| | OLD | NEW | Action |
|---|---|---|---|
| `_calc_spec_decode_metadata` | fork-local, numpy | base method `gpu_model_runner.py:2698`, torch tensors | prefer base; drop fork copy |
| `SpecDecodeMetadata` | fewer fields | `draft_token_ids`, `num_draft_tokens`(list), `cu_num_draft_tokens`(tensor), `cu_num_sampled_tokens`, `target_logits_indices`, `bonus_logits_indices`, `logits_indices` | UnieAI sampler uses fields that all still exist |
| `NgramProposer.propose` | `propose(context_tokens)` | `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)` | update call / delegate to base |
| `propose_draft_token_ids` | `(scheduler_output, sampled_token_ids)` | `(+hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata, common_attn_metadata, slot_mappings)` | re-fit or delegate to `super()` |
| `RejectionSampler` | Triton-based (UnieAI bypassed with CPU impl) | `nn.Module.forward(metadata, draft_probs, logits, sampling_metadata)` (`v1/sample/rejection_sampler.py`) | **check Triton dependency.** If still GPU/Triton → keep UnieAI's CPU `_qaic_rejection_sample`. |

### 4.E Lower-risk overrides

| Method | OLD → NEW | Action |
|---|---|---|
| `load_model` | `(*args, **kwargs)` → `(load_dummy_weights=False)` | align sig; call `load_qaic_model` |
| `initialize_kv_cache` | `(kv_cache_config)` → `(kv_cache_config, is_profiling=False)` | add param |
| `get_kv_cache_spec` | dict of `FullAttentionSpec` → same shape | low risk; verify field names |
| `_init_device_properties` / `_sync_device` | empty stubs | keep if base still calls them |
| `_may_reorder_batch` | sets `self.num_decodes` | rework around removed field |

### 4.F What does NOT change

UnieAI's seven `_qaic_rejection_sample*` helpers are pure `torch` +
`sampling_metadata` + `SpecDecodeMetadata` → port **verbatim** (already done in
`vllm_qaic/model_runner.py`); only glance at `SpecDecodeMetadata` field names
(all present on 0.21).

### 4.G UnieAI's ngram sampler — what / why / how

**What we changed.** Three things, all on the host (the QPC math is untouched):
1. *Un-gated* speculative decoding on the V1+QAIC path (Qualcomm shipped it
   `raise ValueError("...not yet supported...")`), restricting it to `ngram`.
2. *Packed* the decode batch as 2-D `[num_decodes, N+1]` so the target QPC
   verifies the previous token + up to N proposals in **one** card pass.
3. *Wrote a CPU rejection sampler* — the seven `_qaic_rejection_sample*`
   functions — that decides which proposals to accept.

**Why we had to.** vLLM's built-in V1 rejection sampler runs on the GPU via
**Triton** kernels. The QAIC host executes on **CPU** and may have no Triton
driver, so the built-in verifier cannot run there. Without a CPU verifier,
speculative decoding simply cannot function on QAIC — which is exactly why
Qualcomm had disabled it. Our CPU sampler implements the *same, mathematically
faithful* acceptance rule (accept-with-probability, recover-on-reject, greedy
fast-path), so output is identical in distribution to non-speculative decoding;
only tokens-per-card-pass increases.

**How it survives the 0.21 port.** The seven functions depend ONLY on
`torch`, the existing `sampling_metadata` (temperature / top_k / top_p /
generators / all_greedy), and `SpecDecodeMetadata`
(`num_draft_tokens`, `max_spec_len`, `cu_num_draft_tokens`, `draft_token_ids`).
They touch **no** removed `GPUModelRunner` internals (no `positions_np`,
`cu_num_tokens`, `num_decodes`, no `InputBatch` fields). We verified every one of
those `SpecDecodeMetadata` fields still exists on 0.21 (one, `cu_num_draft_tokens`,
is now a tensor, which the existing `.item()` calls already handle). That is why
this block ports **verbatim** while the surrounding input-prep does not — it was
written against the stable sampling API, not the volatile runner internals.

---

## 5. UnieAI's own changes (the only non-Qualcomm, non-upstream code)

Carried in commit `909a809`. See `docs/UnieAI_Quic_integrated.md` for the full
narrative. Summary:

| File (fork) | Change | Port status |
|---|---|---|
| `vllm/v1/worker/qaic_model_runner.py` | +222: 7 ngram CPU rejection-sampler fns + 2D decode packing + ngram gate | ✅ ported into `vllm_qaic/model_runner.py` (verbatim helpers + `_pack_decode_batch`) |
| `vllm/model_executor/model_loader/qaic_v1.py` | +27: target QPC emits `N+1` logits | port within `vllm_qaic/model_loader.py` |
| `docs/qaic-v1-ngram-speculative.md` | design doc | reference |

---

## 6. The 🟠 must-patch-core shortlist (everything else is plugin/port/drop)

If you want a truly fork-free core, only these need a decision:

1. **`vllm/config/cache.py` — `"mxint8"` in `CacheDType`. → PROBABLY NOT NEEDED.**
   `mxint8` and `fp8` are different things at different layers:
   - `fp8` = an 8-bit **float** KV dtype at the **vLLM** layer (`--kv-cache-dtype`).
   - `mxint8` = microscaling int8, a **QEfficient compiler** flag deciding how the
     QPC stores KV **on the card** — passed via `--additional-config`, NOT via
     `--kv-cache-dtype`.

   The production launch uses `--kv-cache-dtype fp8` and carries `mxint8_kv_cache`
   in the override/additional config. So the vLLM-level dtype is `fp8` (no core
   patch needed) and mxint8 is handled by the plugin's additional_config. Only if
   you want to ALSO expose `--kv-cache-dtype mxint8` as a vLLM option (the launch
   does not) would you need the core enum edit. The plugin allows fp8
   (`platform.py::is_kv_cache_dtype_supported`); that is correct for this stack.
2. **`vllm/platforms/interface.py`** — ⚠confirm whether anything is still
   needed; likely a no-op on 0.21 (methods/enum already upstream).
3. **`vllm/_custom_ops.py`** — ⚠confirm; may be unnecessary on 0.21.
4. **`vllm/transformers_utils/config.py`** — ⚠confirm; may be replaced by the
   configs registry (`general_plugins`).

**Bottom line: this list is probably EMPTY.** Item 1 (mxint8) is not needed —
QAIC's on-card KV is mxint8 regardless of the vLLM `--kv-cache-dtype` value
(fp8/fp16 at the vLLM layer doesn't change the on-card format). Item 2 is not
needed (OOT platform replaces it; move to DROP). Item 3 is almost certainly not
needed. Item 4 cannot be decided yet — it depends on which models you run
(only needed for mllama); confirm on-machine. (The Chinese `MIGRATION_zh.md` is
the more readable, fully revised version of this document.)

---

## 7. Recommended order (de-risked)

1. **GO test first** (README PART 0) — torch wall (gate 1/2). No-go ⇒ stop.
2. Stand up the plugin shell: 🟢 platform + quant + model/config registration;
   confirm `current_platform == QaicPlatform`.
3. 🔵 Port low-risk: `session.py`, `model_loader.py`, `get_kv_cache_spec`,
   `load_model`, `initialize_kv_cache`.
4. 🔵 Rewrite input-prep (§4.B) — get **non-speculative** prefill/decode working.
5. 🔵 Re-enable ngram: wire `_pack_decode_batch` + `_qaic_rejection_sample`
   (already present) into the rebuilt `execute_model`; resolve §4.D RejectionSampler.
6. 🟠 Decide on `mxint8` (§6.1).
7. ⚫ Add optional features (pooling / disagg / gpt-oss / MM) only as needed.
