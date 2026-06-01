# `GPUModelRunner`: v0.10.1 (fork base) vs 0.21/0.22 (target)

`QaicModelRunner` subclasses `GPUModelRunner`. This is the single hardest part
of the migration, because `GPUModelRunner` was refactored heavily between the
two releases. This table is the work-list for `vllm_qaic/model_runner.py`.

Sources:
- OLD: `UnieAI/vllm@v1_ngram` `vllm/v1/worker/qaic_model_runner.py` (819 lines)
  subclassing `GPUModelRunner` (~2–3k lines).
- NEW: installed `vllm/v1/worker/gpu_model_runner.py` (7000+ lines).

## A. Class & constructor

| | OLD (v0.10.1) | NEW (0.21/0.22) | Action |
|---|---|---|---|
| Base classes | `GPUModelRunner(...)` | `GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin)` | inherits fine; just be aware mixins exist |
| `__init__` sig | `(vllm_config, device, speculative_model_type=None)` | `(vllm_config, device)` | **drop the 3rd param**; derive `speculative_model_type` internally (done) |
| super call | `super().__init__(vllm_config, device)` | same | OK |

## B. Input-prep data model — **the biggest break**

The fork drove the QPC from host-side NumPy arrays kept on the runner. In 0.21
these were removed; data now lives in a persistent `InputBatch` object and in
`CpuGpuBuffer`/GPU tensors.

| Fork attribute (used by QAIC) | Exists in 0.21? | 0.21 replacement |
|---|---|---|
| `self.positions_np` | ❌ removed | `self.positions` (tensor) / recompute from `num_computed_tokens_cpu` |
| `self.cu_num_tokens` | ❌ removed | computed inside `_prepare_inputs`; `self.query_start_loc` (CpuGpuBuffer) |
| `self.num_decodes` | ❌ removed | recompute via `reorder_batch_to_split_decodes_and_prefills` locally |
| `self.input_ids_cpu` | ❌ removed | `self.input_ids` (`CpuGpuBuffer`, int32) + `InputBatch.token_ids_cpu` |
| `self.batch_indices` | ❌ removed | `InputBatch.block_table[...]` |
| `self.arange_np` | ❌ removed | rebuild locally (`np.arange`) |
| input batch | `self.input_batch` (different fields) | `InputBatch` (`gpu_input_batch.py`): `token_ids_cpu`, `num_tokens_no_spec`, `num_computed_tokens_cpu`, `block_table`, `sampling_metadata` |

**Consequence:** the fork's `_prepare_qaic_inputs()` / `_postprocess_tensors()`
must be rewritten to source from `scheduler_output` + `InputBatch`. This is
hand-work; budget the most time here.

## C. Forward + sample — **architectural split**

| | OLD | NEW | Action |
|---|---|---|---|
| `execute_model` | one method: prep → QPC forward → sample → `ModelRunnerOutput` | may return `None` and stash `self.execute_model_state`; a second call `sample_tokens(grammar_output)` does sampling | QAIC forward is synchronous/host-driven → easiest is to do the full QAIC path inside `execute_model` and return `ModelRunnerOutput` directly, bypassing `sample_tokens`. Verify the base permits this on your build. |
| `sample_tokens` | n/a | `(grammar_output) -> ModelRunnerOutput \| ...` | only relevant if you adopt the 2-phase model |

## D. Speculative decoding

| | OLD | NEW | Action |
|---|---|---|---|
| `_calc_spec_decode_metadata` | fork-local, numpy | base method at `gpu_model_runner.py:2698`, torch tensors | prefer calling base; drop fork copy |
| `SpecDecodeMetadata` | fewer fields | `draft_token_ids`, `num_draft_tokens`(list), `cu_num_draft_tokens`(tensor), `cu_num_sampled_tokens`, `target_logits_indices`, `bonus_logits_indices`, `logits_indices` | UnieAI rejection sampler uses `num_draft_tokens`, `max_spec_len`, `cu_num_draft_tokens`, `draft_token_ids` — all present; `cu_num_draft_tokens` is now a tensor (`.item()` calls already handle that) |
| `NgramProposer.propose` | `propose(context_tokens)` | `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)` | update call site / delegate to base |
| `propose_draft_token_ids` | `(scheduler_output, sampled_token_ids)` | `(... + hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata, common_attn_metadata, slot_mappings)` | re-fit override or delegate to `super()` |
| `RejectionSampler` | Triton-based (UnieAI bypassed it with a CPU impl) | `nn.Module.forward(metadata, draft_probs, logits, sampling_metadata) -> SamplerOutput` (`v1/sample/rejection_sampler.py`) | **check if it still requires GPU/Triton.** If yes → keep UnieAI's `_qaic_rejection_sample` CPU path. If a CPU path now exists → you may be able to drop it. |

## E. Other overrides (lower risk)

| Method | OLD → NEW | Action |
|---|---|---|
| `load_model` | `(*args, **kwargs)` → `(load_dummy_weights=False)` | align sig; call `load_qaic_model` |
| `initialize_kv_cache` | `(kv_cache_config)` → `(kv_cache_config, is_profiling=False)` | add param |
| `get_kv_cache_spec` | dict of `FullAttentionSpec` → same shape | low risk; verify field names |
| `_init_device_properties` / `_sync_device` | empty stubs | keep as stubs if base still calls them |
| `_may_reorder_batch` | sets `self.num_decodes` | rework around removed field |

## Recommended porting order (de-risked)

1. `get_kv_cache_spec`, `load_model`, `initialize_kv_cache` (signatures only).
2. Rewrite input-prep (section B) — get a **non-speculative** prefill/decode
   working end-to-end first.
3. Re-enable ngram: wire `_pack_decode_batch` + `_qaic_rejection_sample` (both
   already in `model_runner.py`) into the rebuilt `execute_model`.
4. Confirm `RejectionSampler` Triton dependency (section D) to decide whether
   the CPU sampler stays.

## What does NOT change

UnieAI's seven `_qaic_rejection_sample*` helpers are pure
`torch` + `sampling_metadata` + `SpecDecodeMetadata`. They port **verbatim**
(already done in `model_runner.py`); only the `SpecDecodeMetadata` field names
need a glance, and they all still exist on 0.21.
