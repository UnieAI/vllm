# UnieAI × Qualcomm Cloud AI 100 — Technical Integration Note

> **Want the overview first?** Read **[`EXPLAINER_plain_zh.md`](EXPLAINER_plain_zh.md)**
> (中文,资工硕士可读版): a concise "what we did" with the two pillars
> (porting/upgrade + ngram) and a per-item status table. This document is the
> precise version with provenance and line-level attribution.

**Subject:** UnieAI's engineering on the Qualcomm Cloud AI 100 (QAIC) vLLM
backend — (A) the porting/upgrade work that brings the QAIC backend onto
successive open-source vLLM bases and the V1 engine, and (B) an n-gram
speculative-decoding optimization added on top.

**Audience:** Qualcomm engineering (technical exchange) and reviewing counsel.
This document is written so that a non-engineer can follow the *what* and *why*
in plain language (Sections 1–3 and the Glossary), while engineers can follow
the *how* in precise detail (Sections 4–7).

**Purpose:** A technical exchange describing UnieAI's own work. It is **not** a
release, an open-source publication, or a product distribution.

**Status of the underlying code:** The base QAIC backend files in the fork are
marked *"Confidential and Proprietary — Qualcomm Technologies"*. UnieAI's work
described here was implemented as additions *inside / on top of* those files.
Section 3 states exactly which lines are UnieAI's.

**Status of the vLLM 0.21+ migration:** The v0.10.1 fork lineage and the
vLLM 0.21+ plugin migration are separate version axes. The current local plugin
branch has the runtime path ported into `vllm-qaic` (platform, worker, loader,
qaicrt session, qserve dependency, and model-runner execute path) and passes
local syntax compilation. It has **not** been validated on an AIC host. Do not
describe this as exact `v0.21.0` support until a branch based on the `v0.21.0`
tag passes the GO tests and a serve smoke test.

---

## 1. Summary (plain language) — UnieAI's work has TWO parts

**Part A — Porting / upgrade engineering (the bulk of the work).** The
open-source inference server **vLLM** changes fast, and Qualcomm's QAIC support
is distributed as a patch tied to particular versions. UnieAI did the
engineering to **bring the QAIC backend onto successive open-source vLLM bases
and keep it running in production**, specifically:
1. **Align to the v0.10.1 base** — make the Qualcomm QAIC patch apply and run on
   open-source vLLM v0.10.1.
2. **Adapt to the "V1" engine and productionize it** — vLLM has an older (V0) and
   a newer, faster (V1) execution engine. Qualcomm's official guide demonstrates
   **V0**; UnieAI's production runs on the **V1** engine (`VLLM_USE_V1=1`). V1 is
   internally very different — this is an engine swap, not a recompile.
3. **Integrate the QEfficient / Cloud AI SDK versions** — wire up
   efficient-transformers **v1.21** + Cloud AI SDK **1.21** so the
   compile-model→run-on-card path works end to end.
4. **Migrate to vLLM 0.2x (in progress)** — bring the whole stack onto a much
   newer vLLM, where the engine was refactored again. This is not yet a
   completed runtime claim for exact vLLM 0.21.0.

> Note on attribution: many QAIC files carry Qualcomm or upstream copyright
> (Section 3). Part A is **integration/upgrade *labor*** — getting that code to
> run on these specific bases/engines/SDKs and migrating it forward — not a claim
> of authorship over Qualcomm's or upstream's source. Code provenance and
> integration labor are distinct; both are stated explicitly in Section 3.

**Part B — An n-gram speculative-decoding optimization (added on top).** On the
ported stack, UnieAI added one speed-up: the model **guesses several likely next
tokens from text it has already seen, then verifies them in a single pass on the
card** — more tokens per step, identical output. Qualcomm had **switched this
off** on the V1 path; UnieAI switched it on and wrote the missing verifier (a
CPU rejection sampler — see §4). This is **originally authored UnieAI code**,
about **250 lines in one file** plus a design document, and is **one optimization
item within speculative decoding**, not the whole of UnieAI's work.

---

## 2. Background primer (plain language)

- **vLLM** — an open-source server that runs LLMs efficiently (batching many
  user requests, managing memory). Licensed Apache-2.0.
- **Cloud AI 100 / QAIC** — Qualcomm's inference accelerator. To run a model on
  it, the model is first **compiled** (by Qualcomm's *QEfficient* toolchain)
  into a binary called a **QPC**, which then executes on the card through
  Qualcomm's runtime library, **`qaicrt`**.
- **Division of labour** — vLLM stays on the host CPU and handles *scheduling*
  (which requests run, in what order) and *sampling* (choosing the next word
  from the model's scores). The actual heavy math (the neural-network forward
  pass) runs on the card via the QPC. The two sides exchange plain numeric
  arrays.
- **Version separation** — host vLLM and the AIC QPC/runtime are separate
  compatibility axes. Moving host vLLM from 0.10.x to 0.21.x is primarily a
  host-side API port; the card still executes the QPC through `qaicrt`. The
  SDK/runtime/Python ABI/QPC format must still match on the AIC host.
- **"V1" engine** — a rewritten, faster execution core inside vLLM. UnieAI's
  work targets this engine only.
- **Speculative decoding** — a standard technique: cheaply *propose* several
  candidate next tokens, then have the real model *verify* them in one pass.
  Accepted candidates are emitted together, so more tokens come out per step.
- **N-gram (prompt-lookup) speculation** — the cheapest way to propose
  candidates: look for where the recent text repeats earlier text, and reuse
  what followed last time. No second "draft" model is needed; the main model
  does the verification.
- **Why a special verification step was needed** — vLLM's built-in verifier
  ("rejection sampler") is written to run on a GPU using *Triton* kernels. The
  QAIC host runs on CPU and may have no Triton driver, so that built-in verifier
  cannot run there. UnieAI therefore wrote an **equivalent verifier in plain
  CPU PyTorch**.

---

## 3. Provenance — what is whose (the boundary of "our work")

**Two distinct kinds of UnieAI contribution — keep them separate:**

- **(i) Integration / upgrade labor (Part A).** Engineering effort to make
  existing code (Qualcomm's QAIC patch + upstream vLLM) run on specific bases and
  engines and to migrate it forward (v0.10.1 → V1 engine → SDK v1.21 → 0.2x).
  This is *work performed on* code authored by Qualcomm/upstream; it is **not** a
  claim of copyright over that underlying source.
- **(ii) Originally authored code (Part B).** New source written by UnieAI — the
  n-gram speculative-decoding logic (~250 lines), enumerated to the line in §7.

The fork `github.com/UnieAI/vllm`, branch `v1_ngram`, is three code-provenance
layers:

| Layer | Origin | Licensing marking | Contents |
|---|---|---|---|
| **L0** | upstream vLLM **v0.10.1** | Apache-2.0 | the open-source engine |
| **L1** | Qualcomm "qaic patch" | files marked **"Confidential and Proprietary — Qualcomm"** (one file, `qaic_session_np.py`, is BSD-3-Clause) | the entire QAIC backend: platform, worker, model runner skeleton, model loader, qaicrt session, mxfp6 quant, KV connector |
| **L2** | **UnieAI** | additions made within L1 files | **n-gram speculative decoding on V1+QAIC** |

**Git evidence (UnieAI commits, author `weimin023`):**

| Commit | Date | What it is |
|---|---|---|
| `909a809` "update v1 engine + ngram" | 2026-05-11 | UnieAI's substantive contribution |
| `84a8d43` "update" | 2026-05-04 | a one-file example tweak (not functional) |
| `bd90d0d` "v0.10.1 + qaic patch" | 2026-04-28 | the Qualcomm patch (L1), not UnieAI's authorship |

**Exactly what UnieAI changed (commit `909a809`):**

| File | Change | Nature |
|---|---|---|
| `vllm/v1/worker/qaic_model_runner.py` | **+222 lines** | the feature (see Section 4) — added inside a Qualcomm-proprietary file |
| `vllm/model_executor/model_loader/qaic_v1.py` | **+27 lines** | make the target QPC emit `N+1` score rows (Section 4.4) — inside a Qualcomm-proprietary file |
| `docs/qaic-v1-ngram-speculative.md` | new, 534 lines | UnieAI's design document |
| `qaic_compare_ngram_20260506_080201/` | new fixtures | a benchmark prompt + config used for A/B comparison |

> **Plain-language note for counsel:** UnieAI's original contribution is the
> *logic* of n-gram speculative decoding for this backend. That logic was
> written *into* files that Qualcomm marks proprietary. The contribution is
> intellectually separable (it is a well-delimited block of functions and two
> small edits, listed line-by-line in Section 7), but as it physically sits it
> is interleaved with Qualcomm-owned code.

---

## 4. The technical change, in detail (the "how")

UnieAI's feature has four parts. All four live on the **host CPU**; none change
the compiled QPC math, only how the host drives and verifies it.

### 4.1 Enabling the feature (the gate)

Qualcomm's V1 model runner explicitly refused speculative decoding:

```python
# Qualcomm original (bd90d0d):
if self.speculative_config:
    raise ValueError("Speculative decoding is not yet suppoerted "
                     "on qaic backend when using vllm v1.")
```

UnieAI replaced this with a gate that **allows n-gram only**, and marks the
model as the speculation *target* (the model that verifies, since there is no
separate draft model):

```python
# UnieAI (909a809):
if self.speculative_config is not None:
    if self.speculative_config.method != "ngram":
        raise ValueError("Only ngram speculative decoding is supported ...")
    if speculative_model_type in (None, "default"):
        speculative_model_type = "target"
```

### 4.2 Packing the decode batch as 2-D (verify N+1 positions at once)

Normally each request decodes one token per step. With speculation, each request
carries the previously-sampled token **plus up to N proposed tokens**. UnieAI
reshapes the per-request decode input from a flat list into a 2-D block of shape
`[num_decode_requests, N+1]`, recording the true length of each row in
`decode_lengths`. This lets the target model score all `N+1` positions of every
request in a single card pass. (Code: the `_pack_decode_batch` logic, ported in
`vllm_qaic/model_runner.py`.)

### 4.3 Verifying the guesses on the CPU (the rejection sampler)

This is the core of the contribution and the reason the feature could not simply
be turned on. vLLM's built-in verifier uses GPU Triton kernels; QAIC's host is
CPU and may lack Triton. UnieAI wrote a **CPU-only, PyTorch equivalent** as seven
small functions:

| Function | What it does (plain language) |
|---|---|
| `_qaic_rejection_sample` | top-level: for each request, run greedy or random verification and assemble the accepted tokens (+1 "bonus" token if all guesses were accepted) |
| `_qaic_is_greedy_request` | decide whether this request uses greedy or random sampling |
| `_qaic_rejection_sample_greedy_req` | greedy case: accept each guess while it equals the model's top choice; stop at the first mismatch |
| `_qaic_rejection_sample_random_req` | random case: accept each guess with the probability the model assigns it; otherwise stop and re-sample a replacement token |
| `_qaic_target_probs_for_req` | turn the model's scores into probabilities for one request (apply temperature, top-k, top-p) |
| `_qaic_apply_top_k_top_p` | the standard top-k / top-p filtering, in place |
| `_qaic_sample_from_probs` | draw one token from a probability distribution (exponential-noise / Gumbel-style trick) |

These functions implement the **standard, mathematically faithful speculative-
decoding acceptance rule** (accept-with-probability, recover-on-reject), just on
CPU. They depend only on PyTorch tensors and vLLM's existing sampling parameters
— not on any Qualcomm-specific internals — which is why they port cleanly across
vLLM versions.

### 4.4 Making the target model emit `N+1` score rows

For verification to work, the compiled target model must output scores for all
`N+1` positions, not just the last one. UnieAI's 27-line change in the model
loader sets the loader to keep `num_speculative_tokens + 1` score rows and sizes
the output buffers accordingly (the QPC is selected/compiled as `..._N+1`).

---

## 5. End-to-end flow (one decode step with speculation)

```
1. The host proposes up to N candidate tokens by n-gram lookup in the
   request's own recent text.                       (vLLM's standard proposer)
2. The host packs each request as [prev_token, cand_1..cand_N]  -> 2-D batch.   (4.2)
3. The card runs the target model once and returns N+1 score rows per request.  (4.4)
4. The host verifies the candidates against those scores on the CPU, deciding
   how many to accept and what token to emit next.                              (4.3)
5. Accepted tokens (+ one bonus token if all accepted) are returned to the
   scheduler; the loop repeats.
```

Output is **identical in distribution** to non-speculative decoding; only the
number of tokens produced per card pass increases.

---

## 6. Verification / benchmarking artifact

UnieAI included `qaic_compare_ngram_20260506_080201/` (a prompt set
`prompt.jsonl` and a `config.txt`) used to A/B compare throughput/output with the
feature on vs off. The design rationale is documented in
`docs/qaic-v1-ngram-speculative.md` (534 lines, UnieAI-authored).

---

## 7. Line-level manifest of UnieAI's changes (for precise location)

In commit `909a809`, file `vllm/v1/worker/qaic_model_runner.py`:

| Region | Lines (fork) | Description |
|---|---|---|
| ngram gate in `__init__` | ~42–54 | allow ngram only; set `speculative_model_type="target"` |
| 2-D decode packing in `execute_model` | ~241–289 | build `[num_decodes, N+1]` decode batch + `decode_lengths` |
| `_qaic_rejection_sample` | 476–514 | CPU verifier (top level) |
| `_qaic_is_greedy_request` | 517–523 | greedy/random decision |
| `_qaic_rejection_sample_greedy_req` | 526–547 | greedy acceptance |
| `_qaic_rejection_sample_random_req` | 549–589 | random acceptance + recovery |
| `_qaic_target_probs_for_req` | 591–604 | scores → probabilities |
| `_qaic_apply_top_k_top_p` | 607–627 | top-k / top-p filtering |
| `_qaic_sample_from_probs` | 630–639 | draw one token |

In commit `909a809`, file `vllm/model_executor/model_loader/qaic_v1.py`: a
27-line change to keep `N+1` logits and size decode buffers accordingly.

All other QAIC code in the fork is Layer L1 (Qualcomm) or L0 (upstream vLLM).

---

## 8. Glossary (for non-engineers)

| Term | Meaning |
|---|---|
| **vLLM** | open-source software that serves LLMs to users efficiently |
| **V1 engine** | the newer, faster execution core inside vLLM |
| **QAIC / Cloud AI 100** | Qualcomm's AI accelerator card |
| **QEfficient** | Qualcomm's toolchain that compiles a model for the card |
| **QPC** | the compiled model binary that runs on the card |
| **qaicrt** | Qualcomm's runtime library that loads/runs a QPC |
| **Token** | a piece of a word; models read and write text as tokens |
| **Decoding** | the step-by-step generation of output tokens |
| **Speculative decoding** | guess several tokens cheaply, verify in one pass — a speed-up that does not change results |
| **N-gram / prompt lookup** | the simplest way to guess: reuse what followed the same recent text earlier |
| **Rejection sampler** | the component that decides which guesses to accept; UnieAI wrote a CPU version |
| **Triton** | a GPU programming framework; vLLM's built-in verifier needs it, the QAIC host may not have it |
| **Logits / scores** | the model's numeric preference for each possible next token |
| **Temperature / top-k / top-p** | standard knobs controlling randomness of token choice |

---

*Prepared by: UnieAI. For technical exchange with Qualcomm Technologies, Inc.*
