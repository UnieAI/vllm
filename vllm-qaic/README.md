# vllm-qaic — running vLLM 0.21+ on Qualcomm Cloud AI 100, as an out-of-tree plugin

This package is the **Route B** integration: instead of forking vLLM and
patching ~45 core files (what the `UnieAI/vllm` `v1_ngram` branch does today on
v0.10.1), QAIC support lives in **this separate pip package** and registers
itself through vLLM's official plugin entry points. vLLM core is installed
unmodified (`pip install vllm==0.21.x`).

> **Version note:** there are two separate version axes. The source QAIC patch
> is from the v0.10.1 fork lineage; the target serve environment is vLLM
> 0.21.x/0.22.x. The current local branch is newer than the exact v0.21.0 tag
> (`git describe` reports `v0.21.1rc0-162-g...`), so do not describe it as
> "the v0.21.0 tag" unless it is rebased or cherry-picked onto that tag.

> **Status:** runtime path ported, not yet AIC-validated. The plugin wiring,
> QAIC platform, worker, model loader, qaicrt session, compile config,
> qserve runner dependency, and `QaicModelRunner.execute_model()` are now in
> `vllm_qaic/` and pass Python syntax compilation against the local 0.21-series
> tree. The remaining proof is on an AIC host: Gate 1/2 plus a `vllm serve`
> smoke test with a precompiled QPC.

## Version/commit compatibility map

| Axis | What it means | Current evidence | Status |
| --- | --- | --- | --- |
| Source patch lineage | The QAIC backend/patch being ported from the fork | `v0.10.1 + qaic patch` in `docs/UnieAI_Quic_integrated.md`; commits `bd90d0d` and `909a809` in the fork notes | Historical source, not the target runtime version |
| Target vLLM API | The upstream vLLM version whose plugin APIs and V1 runner internals this package targets | `vLLM 0.21+` docs; local branch describes as `v0.21.1rc0-162-g...` | 0.21-series port in progress |
| Exact v0.21.0 tag | The released `v0.21.0` tag specifically | `v0.21.0` is not an ancestor of the current local branch | Not yet proven against exact `v0.21.0` |
| Runtime viability | Whether a precompiled QPC can load and run under the target serve env | Gate 1/Gate 2 in PART 0 | Must be confirmed on the AIC host |
| Functional serve path | Whether `vllm serve` works via the plugin | Runtime path is ported; needs AIC Gate 2 + serve smoke | Not yet AIC-validated |

---

## 0. The one-paragraph picture

vLLM stays on the **host/CPU** (scheduling, KV-block bookkeeping, sampling).
The Transformer forward pass is **not** run by torch — it is a pre-compiled
Qualcomm **QPC** graph executed on the AIC100 card through the **`qaicrt`**
runtime. NumPy is the host↔device interchange; the KV cache lives on-card. Of
the whole stack, **UnieAI's own contribution is only the ngram speculative
decoding** (~250 lines in the model runner); everything else is upstream vLLM
plus Qualcomm's QEfficient patch.

This also means the host vLLM version and the AIC card software are separate
axes:

- Upgrading host vLLM from 0.10.x to 0.21.x is a host-side software port:
  scheduler/input-batch/model-runner APIs, sampling, config, and plugin wiring.
- The AIC still executes whatever QPC was produced by QEfficient + Cloud AI SDK.
  The card kernels/operators are fixed at QPC compile time.
- The two sides meet through `qaicrt` buffers. If the prebuilt QPC can be loaded
  by the same Cloud AI SDK runtime under the vLLM 0.21 serve env, the card
  firmware/kernel/SDK does not need to change just because host vLLM changed.
- This independence is not a blanket compatibility guarantee: Python ABI,
  `qaicrt` library path, QPC format, and SDK runtime version still have to match
  on the AIC host. Gate 1/2 below are the proof.

---

## 1. Prerequisites & the version wall (read this)

| Component | Serve env (this plugin) | Compile env (QEfficient) |
| --- | --- | --- |
| vLLM | `0.21.x` (or target) | – |
| torch | **2.11.0** (vLLM pins it) | **2.7.0** (QEfficient 1.21 pins it) |
| transformers | ≥4.56 | 4.57.3 (OK for both) |
| Cloud AI SDK | runtime libs (`qaicrt`) | full SDK (compiler) |
| Python | match the SDK's `qaicrt` build (e.g. 3.12) | 3.12 |

**The wall:** torch `2.7.0` (QEfficient) vs `2.11.0` (vLLM 0.21) cannot coexist
in one venv. So we **split** the work:

```
COMPILE (offline, once)              SERVE (online, this plugin)
torch 2.7 + QEfficient 1.21          torch 2.11 + vLLM 0.21 + vllm-qaic
  model.compile() ──► QPC binary ──►   qaicrt loads the prebuilt QPC
                                       (never imports QEfficient compile path)
```

Everything below runs **on the AIC Linux machine** (where `qaicrt` + the card
live). Your Mac is only for editing this package.

---

## PART 0 — GO / NO-GO test (do this before anything else)

Goal: prove a **pre-compiled QPC loads and runs under torch 2.11**, using only
`qaicrt`. If yes → the split design works → GO. If no → stop and reassess.

### Gate 0 — collect two paths (in your CURRENT `vllm-aic` env)

```bash
source /workspace/weiminc/Unie/vllm/vllm-aic/bin/activate

# (a) where the SDK's qaicrt lives (note the pyXYZ in the path = required Python)
python -c "import qaicrt, QAicApi_pb2; print('QAICRT:', qaicrt.__file__); print('PROTO:', QAicApi_pb2.__file__)"

# (b) a ready-made QPC (you already ran the model, so one is cached)
find / -name "programqpc.bin" 2>/dev/null | head
ls -d ~/.cache/qeff_models/*/qpc* 2>/dev/null
echo "VLLM_QAIC_QPC_PATH=$VLLM_QAIC_QPC_PATH"
```

Write down: the **qaicrt directory** and **one QPC directory**. Note the Python
version baked into the qaicrt path (e.g. `py312`).

### Gate 1 — torch 2.11 + `import qaicrt`  (decisive, ~5 min)

```bash
cd /workspace/weiminc/Unie
uv venv --python 3.12 vllm-021-test      # MUST match the SDK's qaicrt Python
source vllm-021-test/bin/activate
uv pip install "vllm==0.21.1"            # pulls torch==2.11.0

# expose the SDK's qaicrt/proto into this venv (NO QEfficient here)
SDK_DIR=$(dirname "$(/workspace/weiminc/Unie/vllm/vllm-aic/bin/python -c 'import qaicrt;print(qaicrt.__file__)')")
echo "$SDK_DIR"            > "$VIRTUAL_ENV/lib/python3.12/site-packages/qaic_sdk.pth"
echo "/opt/qti-aic/dev/python" >> "$VIRTUAL_ENV/lib/python3.12/site-packages/qaic_sdk.pth"

python /path/to/this/repo/vllm-qaic/tests/gate1_import_qaicrt.py
```

- ✅ `GATE1 PASS` → qaicrt is torch-independent. Continue.
- ❌ `undefined symbol` / ABI error → Python version mismatch. Recreate the
  venv with the Python the SDK built qaicrt for, retry.

### Gate 2 — load a QPC + run once, under torch 2.11

```bash
# still in vllm-021-test; also export the SDK runtime libs for qaicrt
export LD_LIBRARY_PATH=/opt/qti-aic/dev/lib/$(uname -m):$LD_LIBRARY_PATH

python /path/to/this/repo/vllm-qaic/tests/gate2_load_qpc.py \
    /path/to/your/qpc \
    /workspace/weiminc/Unie/vllm/vllm/model_executor/model_loader/qaic_session_np.py
```

- ✅ `GATE2 PASS` (session built, `run()` returned) → **GO. The migration is
  viable.** Proceed to PART 1.
- ❌ fails on import → torch/ABI (revisit Gate 1).
  fails on enqueue/run → usually `LD_LIBRARY_PATH` missing the SDK libs.

> Only after **GATE2 PASS** is it worth building the plugin.

---

## PART 1 — Build the plugin (port the fork's code into this package)

Each stub file names its source file and the exact edits. Port in this order
(low-risk first); test-import after each.

| Step | This file | Port from (fork) | Risk |
| --- | --- | --- | --- |
| 5a | `vllm_qaic/session.py` | `model_loader/qaic_session_np.py` | 🟢 low (numpy+qaicrt) |
| 5b | `vllm_qaic/quant.py` | `layers/quantization/qaic_quant.py` | 🟢 low |
| 5c | `vllm_qaic/model_loader.py` (+`compile_config.py`, `qserve_model_runner.py`) | `model_loader/qaic_v1.py` (+`qaic.py`, `qserve_model_runner.py`) | 🟢 low–🟡 |
| 3  | `vllm_qaic/worker.py` | `v1/worker/qaic_worker.py` | 🟡 medium |
| 4  | `vllm_qaic/model_runner.py` | `v1/worker/qaic_model_runner.py` | 🔴 **hard** |

Global edits while porting:
- Replace intra-fork imports (`vllm.v1.worker.qaic_*`,
  `vllm.model_executor.model_loader.qaic_*`) with `vllm_qaic.*`.
- Read QAIC knobs from `vllm_config.additional_config` (set by the platform)
  instead of `vllm_config.model_config.override_qaic_config`.

### Can `qserve_model_runner.py` just be copied?

Mostly yes, but "copy it" is necessary, not sufficient. `qaic.py` /
`qaic_v1.py` import the fork's `qserve_model_runner.py`, and upstream vLLM does
not have that file, so the plugin must carry it. The file is mostly a
QEfficient/qaicrt runner helper rather than a vLLM core extension, so a direct
copy with import rewrites is the right shape:

- `vllm.model_executor.model_loader.qaic_session_np` → `vllm_qaic.session`
- `QEfficient.generation.cloud_infer` top-level session imports → local
  `vllm_qaic.session`
- package path imports → `vllm_qaic.*`

What direct copy does **not** prove: that the QPC shapes, `qaicrt` runtime, and
QEfficient-generated artifacts match the target host. That still requires Gate 2
and a serve smoke test.

### Step 4 is the real work — re-aligning against the new `GPUModelRunner`

`QaicModelRunner` subclasses `GPUModelRunner`, which went from ~2–3k lines
(v0.10.1) to **7000+ lines** (0.21). The plugin can't help here; you must
reconcile each override by hand. Recommended:

```bash
# put both side by side
sed -n '1,80p'  /tmp/unie_vllm/vllm/v1/worker/qaic_model_runner.py     # fork base
# vs your installed copy:
python -c "import vllm.v1.worker.gpu_model_runner as m; print(m.__file__)"
```

Re-fit in this priority: `__init__` → `execute_model` → `get_kv_cache_spec`
→ `initialize_kv_cache` → `load_model`. **Leave UnieAI's ngram code for last**
(the `_qaic_rejection_sample*` functions + 2D decode packing are self-contained
and port nearly verbatim once the base path runs).

---

## PART 2 — Install & verify the wiring

```bash
source /workspace/weiminc/Unie/vllm/vllm-021-test/bin/activate   # the GO env
uv pip install -e /path/to/this/repo/vllm-qaic

# confirm vLLM discovers the plugin and selects QaicPlatform
python - <<'PY'
from vllm.platforms import current_platform
print("platform:", type(current_platform).__name__, current_platform.device_type)
PY
# expect: QaicPlatform qaic
```

If it prints `UnspecifiedPlatform`/`CpuPlatform`, the entry point isn't being
found (reinstall `-e`) or `register_platform()` returned `None` (qaicrt not
importable in this env — fix `qaic_sdk.pth` / `LD_LIBRARY_PATH`).

---

## PART 3 — Run, incrementally

Always V1, eager, pre-compiled QPC, knobs via `--additional-config`:

```bash
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export VLLM_QAIC_QPC_PATH=/path/to/your/qpc      # force pre-compiled path

# 3a) FIRST: no speculative decoding — prove the base path works
vllm serve "Qwen/Qwen2.5-7B-Instruct" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 2048 --max-seq-len-to-capture 32 --max-num-seqs 256 \
  --quantization mxfp6 --kv-cache-dtype fp8 \
  --additional-config '{"device_group":[0],"num_cores":16,"mxfp6":true,"mxint8_kv_cache":true,"prefill_seq_len":32,"aic_enable_depth_first":true}' \
  --disable-frontend-multiprocessing

# 3b) THEN: turn ngram back on (UnieAI's feature)
#   add: --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":1}'
```

Smoke test:
```bash
curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Hello","max_tokens":16}'
```

---

## Known core gaps (things the plugin genuinely cannot do)

1. **Exact old CLI flags.** `--device-group` and `--override-qaic-config` are
   fork-only flags. Upstream 0.21 does not have them; use `--additional-config`.
2. **`mxint8` as a vLLM cache dtype.** Upstream `CacheDType` has no `mxint8`,
   but the QAIC path does not need it as a vLLM CLI dtype. Use
   `--kv-cache-dtype fp8` and pass `mxint8_kv_cache=true` through
   `--additional-config`; QEfficient/SDK decide the on-card KV format.
3. **torch wall** — handled by the split env (PART 0).
4. **AIC validation** — syntax has been checked locally, but QPC load/run must
   be tested on the AIC host.
5. **V0 path** — intentionally unsupported; ~half the fork's core patch was V0
   (scheduler/block_manager/sequence/llm_engine/multi_step) and is dropped.

---

## Troubleshooting quick table

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `import qaicrt` undefined symbol | venv Python ≠ SDK qaicrt Python | recreate venv with matching `--python` |
| platform = CPU/Unspecified | entry point not found / qaicrt not importable | `uv pip install -e .` again; check `qaic_sdk.pth` |
| enqueue/run fails in qaicrt | SDK runtime libs not on path | `export LD_LIBRARY_PATH=/opt/qti-aic/dev/lib/$(uname -m):$LD_LIBRARY_PATH` |
| `mxint8` rejected | not in CacheDType Literal | use fp8, or patch core (see gaps) |
| QEfficient/torch 2.7 gets pulled into serve env | compile path imported | set `VLLM_QAIC_QPC_PATH`; use `vllm_qaic.session` (qaicrt-only) |

---

## Checklist

- [ ] Gate 0: have qaicrt path + a QPC path; noted required Python version
- [ ] Gate 1: `GATE1 PASS` (qaicrt imports under torch 2.11)
- [ ] Gate 2: `GATE2 PASS` (prebuilt QPC runs under torch 2.11) ← **GO line**
- [x] Port session / quant / loader / qserve dependency (steps 5a–5c)
- [x] Port worker (step 3)
- [x] Re-align model runner vs new GPUModelRunner enough for the base path
- [x] Local syntax check with `uv run --no-sync python -m py_compile`
- [ ] PART 2: `current_platform` == `QaicPlatform`
- [ ] PART 3a: serve without speculation works
- [ ] Port + enable ngram; PART 3b works
- [x] Decide on `mxint8`: use vLLM `fp8` label plus QAIC
      `mxint8_kv_cache=true` compile knob
