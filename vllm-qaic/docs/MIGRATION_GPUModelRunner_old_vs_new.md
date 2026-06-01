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

### 3d. ⚫ OPTIONAL — separate features; port only if needed

| File | +lines | Feature it serves |
|---|---|---|
| `vllm/entrypoints/openai/serving_chat.py` | 252 | ⚠ on-device sampling + gpt-oss/harmony. Not needed for basic serving. |
| `vllm/entrypoints/openai/protocol.py` | 72 | ⚠ API fields for on-device sampling / gpt-oss |
| `vllm/entrypoints/harmony_utils.py` | 106 | gpt-oss harmony format |
| `vllm/reasoning/gptoss_reasoning_parser.py` | 45 | gpt-oss reasoning |
| `vllm/entrypoints/openai/serving_pooling.py` | 104 | embeddings/pooling on QAIC |
| `vllm/distributed/kv_transfer/kv_connector/base.py` | 142 | disaggregated prefill/decode |
| `vllm/distributed/kv_transfer/kv_transfer_state.py` | 9 | disaggregated serving |
| `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` | 22 | mxfp4 quant |
| `vllm/model_executor/layers/quantization/gguf.py` | 29 | ⚠confirm gguf interaction |
| `vllm/model_executor/layers/quantization/bitsandbytes.py` | 10 | ⚠confirm |
| `vllm/entrypoints/chat_utils.py` | 17 | ⚠ chat template (MM?) |
| `vllm/entrypoints/openai/api_server.py` | 9 | ⚠ minor server hook |
| `vllm/entrypoints/openai/tool_parsers/__init__.py` | 5 | register a tool parser |
| `vllm/model_executor/models/config.py` | 6 | ⚠ model config tweak |
| `vllm/env_override.py` | 10 | ⚠ env overrides (could move to plugin `register()`) |
| `vllm/utils/__init__.py` | 10 | ⚠ helper utils (copy needed ones into package) |
| `setup.py` | 40 | build/packaging — N/A (plugin has its own `pyproject.toml`) |

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

1. **`vllm/config/cache.py` — add `"mxint8"` to `CacheDType`.** Options:
   (a) ship a 1-line core patch; (b) use `fp8` and skip mxint8; (c) upstream PR.
   The plugin currently allows fp8 only (`platform.py::is_kv_cache_dtype_supported`).
2. **`vllm/platforms/interface.py`** — ⚠confirm whether anything is still
   needed; likely a no-op on 0.21 (methods/enum already upstream).
3. **`vllm/_custom_ops.py`** — ⚠confirm; may be unnecessary on 0.21.
4. **`vllm/transformers_utils/config.py`** — ⚠confirm; may be replaced by the
   configs registry (`general_plugins`).

Items 2–4 are "confirm, probably not needed". Item 1 is the one real core gap.

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
