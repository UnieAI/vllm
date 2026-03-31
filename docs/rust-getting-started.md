# vLLM Rust 加速模組 — 啟動教學

本文件說明如何從零開始建構、安裝、驗證、啟用 `vllm._rs` Rust 加速模組，
以及如何在開發中新增 Rust 函數。

---

## 目錄

1. [架構總覽](#1-架構總覽)
2. [前置需求](#2-前置需求)
3. [建構與安裝](#3-建構與安裝)
4. [驗證安裝](#4-驗證安裝)
5. [在 vLLM 中啟用](#5-在-vllm-中啟用)
6. [效能基準測試](#6-效能基準測試)
7. [開發流程](#7-開發流程)
8. [常見問題](#8-常見問題)
9. [TODO — 已完成與未完成項目](#9-todo--已完成與未完成項目)

---

## 1. 架構總覽

### 1.1 為什麼需要 Rust 加速

vLLM V1 的 EngineCore 忙碌迴圈在每步推論中執行：

```
schedule() → execute_model() → update_from_output()
   CPU           GPU               CPU
```

GPU 計算佔推論延遲的 90%+，但在高併發（100-1000+ 同時請求）場景下，
CPU 端的 Python 迴圈成為瓶頸。以 N=1000 requests 為例：

| 操作 | Python 延遲 | Rust 延遲 | 加速比 |
|------|-------------|-----------|--------|
| Token 預算計算 | 1,064 μs | 5.5 μs | 194x |
| Stop 條件批次檢查 | 3,313 μs | 11 μs | 301x |
| Spec decode 接受/拒絕 | 767 μs | 4.0 μs | 192x |
| Block hash 計算（1K blocks）| 4,646 μs | 339 μs | 13.7x |
| Free block queue（10K blocks）| 2,943 μs | 494 μs | 6.0x |
| Stop string 匹配（5 patterns）| 2.5 μs | 0.3 μs | 7.3x |

> **效能數據來源**：上述數字均在 Apple Silicon (aarch64-apple-darwin)、
> CPython 3.12、Rust 1.93 release profile 下，用 `time.perf_counter()`
> 取多次迭代平均值實測。測試方法見[第 6 節](#6-效能基準測試)。
> 報告中的「端到端吞吐量預估」（如「小模型 3.4x step 加速」）為
> 算術推算，**未經 GPU 端到端驗證**。

### 1.2 模組結構

```
rust/
├── Cargo.toml              # pyo3 0.23, numpy 0.23, xxhash-rust 0.8, aho-corasick 1
├── pyproject.toml           # maturin 建構配置，module-name = "vllm._rs"
├── .cargo/config.toml       # macOS aarch64 linker 設定
└── src/
    ├── lib.rs               # PyO3 模組入口（註冊所有函數和類別）
    │
    │   ── Scheduler 加速 ──
    ├── schedule.rs          # compute_running_tokens, compute_waiting_tokens
    ├── stop_check.rs        # batch_check_stop（2D numpy stop_token_ids）
    ├── update_output.rs     # batch_apply_generated_tokens
    ├── ngram.rs             # batch_ngram_propose（並行 KMP）
    │
    │   ── Phase 2 加速 ──
    ├── block_hash.rs        # hash_block_tokens_rust, batch_hash_blocks（xxh3_128）
    ├── block_pool.rs        # RustFreeBlockQueue（index-based linked list）
    ├── stop_strings.rs      # StopStringMatcher（Aho-Corasick 自動機）
    └── serial_helpers.rs    # encode_int_array_as_bytes（IPC 輔助）
```

### 1.3 Python 整合點

| Python 檔案 | 使用的 Rust 元件 | 啟用方式 |
|-------------|-----------------|---------|
| `vllm/v1/core/sched/rust_accelerated.py` | compute_running_tokens, batch_check_stop, batch_apply_generated_tokens | 自動（`_HAS_RUST`） |
| `vllm/v1/core/sched/scheduler.py` | `_batch_precompute_stops()` → batch_check_stop | 自動 |
| `vllm/v1/core/kv_cache_utils.py` | RustFreeBlockQueue, batch_hash_blocks | 自動 / `--prefix-caching-hash-algo builtin` |
| `vllm/v1/engine/detokenizer.py` | StopStringMatcher | 自動（有 stop strings 時） |
| `vllm/v1/spec_decode/ngram_proposer.py` | batch_ngram_propose | 自動（`_HAS_RUST_NGRAM`） |
| `vllm/utils/hashing.py` | hash_block_tokens_rust | `--prefix-caching-hash-algo builtin` |

### 1.4 設計原則

| 原則 | 做法 |
|------|------|
| **零侵入** | `vllm._rs` 不存在時自動 fallback 到 Python，功能不受影響 |
| **SoA 介面** | Rust 函數只接收 numpy array / bytes / int，不接觸 Python 物件 |
| **不碰 GPU** | KV cache 管理、block allocation、FlashAttention 留在 Python/CUDA |
| **增量建構** | 修改 `.rs` 後 `maturin develop --release` 只需 1-5 秒 |

---

## 2. 前置需求

| 工具 | 最低版本 | 安裝方式 |
|------|---------|---------|
| Python | 3.10+ | 系統自帶或 pyenv |
| Rust toolchain | 1.70+ | `rustup`（見下方）|
| maturin | 1.0+ | `pip install maturin` |
| numpy | 1.24+ | vLLM 依賴自動安裝 |

### 2.1 安裝 Rust

```bash
# Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 驗證
rustc --version   # 期望: rustc 1.7x.0+
cargo --version   # 期望: cargo 1.7x.0+
```

### 2.2 安裝 maturin

```bash
pip install maturin
```

---

## 3. 建構與安裝

### 3.1 開發模式（推薦）

```bash
cd /path/to/vllm/rust
maturin develop --release
```

**首次建構**約 15-20 秒（下載依賴 + 編譯），後續增量建構 1-5 秒。

預期輸出：

```
🔗 Found pyo3 bindings
🐍 Found CPython 3.12 at /path/to/python
📡 Using build options features from pyproject.toml
   Compiling vllm-scheduler-rs v0.1.0
    Finished `release` profile [optimized] target(s) in 3.03s
📦 Built wheel for CPython 3.12 to /tmp/.../vllm_scheduler_rs-0.1.0-....whl
✏️ Setting installed package as editable
🛠 Installed vllm-scheduler-rs-0.1.0
```

### 3.2 建構 wheel（CI / 分發）

```bash
cd /path/to/vllm/rust
maturin build --release

# 產出位置
ls target/wheels/
# vllm_scheduler_rs-0.1.0-cp312-cp312-linux_x86_64.whl

# 在目標機器上安裝
pip install target/wheels/vllm_scheduler_rs-0.1.0-*.whl
```

### 3.3 搭配 vLLM 完整安裝

```bash
# 1. 安裝 vLLM（需要 GPU 環境）
cd /path/to/vllm
VLLM_USE_PRECOMPILED=1 pip install -e . --torch-backend=auto

# 2. 安裝 Rust 模組
cd rust
maturin develop --release

# 此時 from vllm._rs import ... 可用
```

> **無 GPU 環境**：如果只想開發/測試 Rust 模組本身，不需要安裝 vLLM。
> `maturin develop` 後可以 `from _rs import ...` 直接使用。

---

## 4. 驗證安裝

### 4.1 檢查模組載入

```bash
python3 -c "
try:
    from vllm._rs import compute_running_tokens
    print('OK: vllm._rs loaded')
except ImportError:
    try:
        from _rs import compute_running_tokens
        print('OK: _rs loaded (dev mode, vllm not installed)')
    except ImportError:
        print('FAIL: module not installed')
"
```

### 4.2 列出所有匯出

```bash
python3 -c "
try:
    import _rs as m
except ImportError:
    from vllm import _rs as m

for name in sorted(dir(m)):
    if not name.startswith('_'):
        obj = getattr(m, name)
        kind = 'class' if isinstance(obj, type) else 'function'
        print(f'  {kind:8s}  {name}')
"
```

預期輸出：

```
  class     RustFreeBlockQueue
  class     StopStringMatcher
  function  batch_apply_generated_tokens
  function  batch_check_stop
  function  batch_encode_int32_arrays
  function  batch_hash_blocks
  function  batch_ngram_propose
  function  compute_running_tokens
  function  compute_waiting_tokens
  function  encode_int_array_as_bytes
  function  hash_block_tokens_rust
```

### 4.3 Smoke test（各模組快速驗證）

```bash
python3 << 'PYEOF'
import numpy as np

try:
    from vllm._rs import (
        compute_running_tokens, batch_check_stop,
        batch_apply_generated_tokens, batch_hash_blocks,
        RustFreeBlockQueue, StopStringMatcher,
    )
except ImportError:
    from _rs import (
        compute_running_tokens, batch_check_stop,
        batch_apply_generated_tokens, batch_hash_blocks,
        RustFreeBlockQueue, StopStringMatcher,
    )

# ── Scheduler: compute_running_tokens ──
r = compute_running_tokens(
    np.array([100], dtype=np.int64),
    np.zeros(1, dtype=np.int64),
    np.array([90], dtype=np.int64),
    np.array([100], dtype=np.int64),
    np.array([100], dtype=np.int64),
    1000, 0, 4096,
)
assert list(r) == [10]
print("[PASS] compute_running_tokens")

# ── Scheduler: batch_check_stop (numpy 2D) ──
r = batch_check_stop(
    np.array([2, 42], dtype=np.int64),
    np.array([100, 100], dtype=np.int64),
    np.array([10, 10], dtype=np.int64),
    np.array([0, 0], dtype=np.int64),
    np.array([100, 100], dtype=np.int64),
    np.array([2, -1], dtype=np.int64),
    np.array([[-1], [42]], dtype=np.int64),
    4096,
)
assert list(r) == [1, 2]  # EOS, STOP_TOKEN
print("[PASS] batch_check_stop")

# ── Scheduler: batch_apply_generated_tokens ──
ac, ap, aa, ar = batch_apply_generated_tokens(
    np.array([105], dtype=np.int64),
    np.array([5], dtype=np.int64),
    np.array([3], dtype=np.int64),
    np.array([5], dtype=np.int64),
)
assert list(ac) == [102]
print("[PASS] batch_apply_generated_tokens")

# ── Block Hash: batch_hash_blocks ──
parent = b'\x00' * 16
hashes = batch_hash_blocks(parent, [1, 2, 3, 4, 5, 6], 3)
assert len(hashes) == 2
assert hashes[0] != hashes[1]
# Chain: hash[1] depends on hash[0]
hashes2 = batch_hash_blocks(hashes[0], [4, 5, 6], 3)
assert hashes2[0] == hashes[1]
print("[PASS] batch_hash_blocks (chain hashing)")

# ── Block Pool: RustFreeBlockQueue ──
q = RustFreeBlockQueue([0, 1, 2, 3, 4], 5)
assert len(q) == 5
assert q.popleft() == 0
q.remove(2)
assert q.get_all() == [1, 3, 4]
q.append(2)
assert q.get_all() == [1, 3, 4, 2]
popped = q.popleft_n(2)
assert popped == [1, 3]
print("[PASS] RustFreeBlockQueue")

# ── Stop Strings: StopStringMatcher ──
m = StopStringMatcher(["</s>", "STOP"])
r = m.check("Hello</s>", 9, False)
assert r is not None and r[0] == "</s>" and r[1] == 5
r = m.check("Hello world", 11, False)
assert r is None
r = m.check("Hello STOP", 10, True)
assert r is not None and r[0] == "STOP" and r[1] == -1  # no truncation
print("[PASS] StopStringMatcher")

print()
print("=== All 6 smoke tests passed ===")
PYEOF
```

### 4.4 跑完整測試套件

```bash
# 從 vllm 根目錄（如果 conftest.py 有 vllm 依賴衝突，複製到 /tmp 跑）
cp tests/v1/core/test_rust_scheduler.py /tmp/
cp tests/v1/spec_decode/test_ngram_rust.py /tmp/
cd /tmp && python3 -m pytest test_rust_scheduler.py test_ngram_rust.py -v

# 如果 vLLM 已安裝，直接在 repo 內跑
cd /path/to/vllm
python3 -m pytest tests/v1/core/test_rust_scheduler.py tests/v1/spec_decode/test_ngram_rust.py -v
```

預期結果：**25 tests passed**。

---

## 5. 在 vLLM 中啟用

### 5.1 自動啟用（預設行為）

安裝 `vllm-scheduler-rs` 後，vLLM 啟動時**自動偵測並啟用**。
不需要任何配置。日誌中會出現：

```
INFO: Rust scheduler acceleration enabled (vllm._rs)
```

### 5.2 各元件啟用狀態

| 元件 | 自動啟用 | 額外配置 | 說明 |
|------|---------|---------|------|
| Scheduler batch stop 預計算 | ✅ 自動 | 無 | `_batch_precompute_stops()` |
| FreeKVCacheBlockQueue Rust backend | ✅ 自動 | 無 | 偵測 `RustFreeBlockQueue` 可用性 |
| StopStringMatcher (Aho-Corasick) | ✅ 自動 | 無 | 請求有 stop strings 時建構 |
| N-gram proposer Rust path | ✅ 自動 | 無 | 偵測 `batch_ngram_propose` 可用性 |
| Block hash Rust 快速路徑 | ❌ 需配置 | `--prefix-caching-hash-algo builtin` | 見下方 |

### 5.3 啟用 Rust block hash

Block hash 使用新的 `"builtin"` 算法（Rust xxh3_128，跳過 cbor2/pickle 序列化）：

```bash
vllm serve <model> --prefix-caching-hash-algo builtin
```

> **注意**：`"builtin"` 與 `"sha256"` / `"xxhash_cbor"` 產生不同的 hash 值。
> 切換算法後，已有的 prefix cache 會失效（首次請求會重新計算）。
> 單一 vLLM 實例內部是自洽的，不影響正確性。

### 5.4 確認 Rust 加速正在使用

```bash
# 方法 1：啟動時搜尋日誌
vllm serve <model> 2>&1 | grep -i rust

# 方法 2：Python 中檢查
python3 -c "
from vllm.v1.core.sched.rust_accelerated import _HAS_RUST
print(f'Scheduler Rust: {_HAS_RUST}')
"
```

---

## 6. 效能基準測試

以下腳本可在**任何安裝了 Rust 模組的環境**中執行，不需要 GPU。

### 6.1 完整基準測試腳本

```bash
python3 << 'PYEOF'
"""vllm._rs performance benchmark.

Measures each Rust function against a comparable Python baseline.
All numbers are wall-clock time (time.perf_counter), averaged over many iterations.
"""
import time
import numpy as np

try:
    from vllm._rs import (
        compute_running_tokens, batch_check_stop,
        batch_apply_generated_tokens, batch_hash_blocks,
        RustFreeBlockQueue, StopStringMatcher,
    )
except ImportError:
    from _rs import (
        compute_running_tokens, batch_check_stop,
        batch_apply_generated_tokens, batch_hash_blocks,
        RustFreeBlockQueue, StopStringMatcher,
    )

N = 1000
rng = np.random.default_rng(42)

def bench(name, fn, iters=5000, warmup=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters

print(f"Benchmark: N={N} requests")
print("=" * 65)

# ── 1. compute_running_tokens ──
spec = rng.integers(100, 4096, size=N, dtype=np.int64)
ph = np.zeros(N, dtype=np.int64)
comp = spec - rng.integers(1, 50, size=N, dtype=np.int64)
prompt = rng.integers(50, 2000, size=N, dtype=np.int64)
maxt = np.full(N, 1024, dtype=np.int64)

rust_s = bench("", lambda: compute_running_tokens(
    spec, ph, comp, prompt, maxt, 100000, 0, 4096))

def py_compute():
    result = np.zeros(N, dtype=np.int64)
    budget = 100000
    for i in range(N):
        if budget <= 0: break
        n = int(spec[i] + ph[i] - comp[i])
        n = min(n, budget, 4095 - int(comp[i]))
        n = max(n, 0)
        result[i] = n
        budget -= n

py_s = bench("", py_compute, iters=500)
print(f"compute_running_tokens:  Rust {rust_s*1e6:>8.1f} μs | Py {py_s*1e6:>8.1f} μs | {py_s/rust_s:>6.1f}x")

# ── 2. batch_check_stop ──
last_tok = rng.integers(0, 50000, size=N, dtype=np.int64)
num_tok = rng.integers(100, 4096, size=N, dtype=np.int64)
num_out = rng.integers(0, 500, size=N, dtype=np.int64)
min_tok = np.zeros(N, dtype=np.int64)
max_tok2 = np.full(N, 1024, dtype=np.int64)
eos = np.full(N, 2, dtype=np.int64)
stop_arr = np.full((N, 5), -1, dtype=np.int64)
for i in range(N):
    ns = rng.integers(0, 6)
    stop_arr[i, :ns] = rng.integers(0, 50000, size=ns)

rust_s = bench("", lambda: batch_check_stop(
    last_tok, num_tok, num_out, min_tok, max_tok2, eos, stop_arr, 4096))

def py_stop():
    result = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if num_out[i] < min_tok[i]: continue
        if eos[i] >= 0 and last_tok[i] == eos[i]:
            result[i] = 1; continue
        for j in range(5):
            if stop_arr[i, j] < 0: break
            if last_tok[i] == stop_arr[i, j]:
                result[i] = 2; break
        if result[i] > 0: continue
        if num_tok[i] >= 4096 or num_out[i] >= max_tok2[i]:
            result[i] = 3

py_s = bench("", py_stop, iters=500)
print(f"batch_check_stop:        Rust {rust_s*1e6:>8.1f} μs | Py {py_s*1e6:>8.1f} μs | {py_s/rust_s:>6.1f}x")

# ── 3. batch_hash_blocks ──
parent = b'\x00' * 16
tokens = list(range(N * 16))

rust_s = bench("", lambda: batch_hash_blocks(parent, tokens, 16), iters=1000)

import hashlib
def py_hash():
    prev = parent
    for i in range(0, len(tokens), 16):
        blk = tokens[i:i+16]
        data = prev + b''.join(t.to_bytes(4, 'little', signed=True) for t in blk)
        prev = hashlib.sha256(data).digest()

py_s = bench("", py_hash, iters=100)
print(f"batch_hash_blocks:       Rust {rust_s*1e6:>8.1f} μs | Py {py_s*1e6:>8.1f} μs | {py_s/rust_s:>6.1f}x")

# ── 4. FreeBlockQueue ──
CAP = 10000
q = RustFreeBlockQueue(list(range(CAP)), CAP)
rust_s = bench("", lambda: (q.append_n(q.popleft_n(CAP))), iters=1000)
print(f"FreeBlockQueue (10K):    Rust {rust_s*1e6:>8.1f} μs |")

# ── 5. StopStringMatcher ──
import string, random
random.seed(42)
text = ''.join(random.choices(string.ascii_letters + ' ', k=10000))
stops = ['ENDOFTEXT', '</s>', '###', 'STOP', 'END']
matcher = StopStringMatcher(stops)

rust_s = bench("", lambda: matcher.check(text, 50, False), iters=10000)

def py_ss():
    for s in stops:
        text.find(s, max(0, len(text) - 50 - len(s) + 1))

py_s = bench("", py_ss, iters=10000)
print(f"StopStringMatcher:       Rust {rust_s*1e6:>8.2f} μs | Py {py_s*1e6:>8.2f} μs | {py_s/rust_s:>6.1f}x")

print("=" * 65)
PYEOF
```

### 6.2 效能數據的來源說明

| 欄位 | 說明 |
|------|------|
| **Rust / Python 延遲** | `time.perf_counter()` 實測，warmup 後取 N 次迭代平均值 |
| **加速比** | `Python 延遲 / Rust 延遲` |
| **端到端預估** | 算術推算，**未經 GPU 端到端驗證** |
| **硬體** | 實測環境：Apple Silicon M-series, CPython 3.12, Rust 1.93 release |

> 在不同硬體（Linux x86_64 + CUDA）上數字可能不同。
> 建議在目標部署硬體上重跑上方基準測試。

### 6.3 端到端驗證（需要 GPU）

```bash
# 1. 安裝 vLLM + Rust 模組
cd /path/to/vllm
VLLM_USE_PRECOMPILED=1 pip install -e . --torch-backend=auto
cd rust && maturin develop --release && cd ..

# 2. 啟動 server
vllm serve meta-llama/Llama-3.1-8B &

# 3. 跑 serving benchmark
python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3.1-8B \
    --num-prompts 1000 \
    --request-rate inf \
    --backend vllm

# 4. 對比：移除 Rust 模組再跑一次
pip uninstall -y vllm-scheduler-rs
# 重啟 server，再跑同樣的 benchmark
```

---

## 7. 開發流程

### 7.1 修改 → 建構 → 測試

```bash
# 1. 修改 Rust 原始碼
vim rust/src/schedule.rs

# 2. 快速語法檢查（不安裝，~1 秒）
cd rust && cargo check

# 3. 建構 + 安裝（~1-5 秒增量）
maturin develop --release

# 4. 測試
python3 -m pytest tests/v1/core/test_rust_scheduler.py -v
```

### 7.2 新增一個 Rust 函數

以新增 `my_fast_function(arr: numpy.int64[]) -> int` 為例：

**Step 1：建立 Rust 實作**

```rust
// rust/src/my_module.rs
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
pub fn my_fast_function(arr: PyReadonlyArray1<'_, i64>) -> i64 {
    arr.as_slice().unwrap().iter().sum()
}
```

**Step 2：在 lib.rs 註冊**

```rust
// rust/src/lib.rs
mod my_module;  // ← 新增

fn vllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... 既有的註冊 ...
    m.add_function(wrap_pyfunction!(my_module::my_fast_function, m)?)?;
    Ok(())
}
```

**Step 3：建構**

```bash
cd rust && maturin develop --release
```

**Step 4：Python 中使用（含 fallback）**

```python
try:
    from vllm._rs import my_fast_function
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

def my_function(arr):
    if _HAS_RUST:
        return my_fast_function(arr)
    return sum(arr)  # Pure Python fallback
```

**Step 5：撰寫測試**

```python
# tests/v1/core/test_my_module.py
import numpy as np
import pytest

try:
    from vllm._rs import my_fast_function
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

@pytest.mark.skipif(not HAS_RUST, reason="vllm._rs not installed")
def test_my_fast_function():
    arr = np.array([1, 2, 3], dtype=np.int64)
    assert my_fast_function(arr) == 6
```

### 7.3 新增一個 Rust pyclass

以有狀態、可重複使用的物件為例：

```rust
// rust/src/my_class.rs
use pyo3::prelude::*;

#[pyclass]
pub struct MyMatcher {
    patterns: Vec<String>,
}

#[pymethods]
impl MyMatcher {
    #[new]
    fn new(patterns: Vec<String>) -> Self {
        Self { patterns }
    }

    fn check(&self, text: &str) -> Option<String> {
        self.patterns.iter().find(|p| text.contains(p.as_str())).cloned()
    }

    fn __len__(&self) -> usize {
        self.patterns.len()
    }
}
```

在 `lib.rs` 中用 `m.add_class::<my_class::MyMatcher>()?;` 註冊。

---

## 8. 常見問題

### Q: `maturin develop` 找不到 `Cargo.toml`

```
💥 maturin failed
  Caused by: Can't find Cargo.toml
```

**解法**：確保在 `rust/` 目錄下執行：

```bash
cd /path/to/vllm/rust
maturin develop --release
```

### Q: macOS 上 `cargo test` linker 錯誤

```
ld: Undefined symbols: _PyBaseObject_Type ...
```

**原因**：PyO3 extension module 的測試需要 Python runtime。

**解法**：用 Python pytest 跑測試，不要用 `cargo test`。

### Q: `from vllm._rs import ...` 失敗

**原因**：vllm 未安裝，maturin 將 `.so` 裝為頂層 `_rs` 模組。

**解法**：

```python
# 開發模式：直接用 _rs
from _rs import compute_running_tokens

# 生產模式：先安裝 vllm
pip install -e /path/to/vllm
cd /path/to/vllm/rust && maturin develop --release
# 之後 from vllm._rs import ... 即可
```

> vLLM 程式碼中已有 fallback：依序嘗試 `vllm._rs` → `_rs` → 純 Python。

### Q: 修改 Rust 後 Python 沒有更新

```bash
cd rust && maturin develop --release
# 然後重啟 Python / vLLM server
```

### Q: 如何確認用的是 Rust 而非 fallback

```python
python3 -c "
from vllm.v1.core.sched.rust_accelerated import _HAS_RUST
print(f'Scheduler:   {_HAS_RUST}')
from vllm.v1.spec_decode.ngram_proposer import _HAS_RUST_NGRAM
print(f'N-gram:      {_HAS_RUST_NGRAM}')
"
```

### Q: 想建構多個 Python 版本的 wheel

```bash
maturin build --release --interpreter python3.10 python3.11 python3.12
```

### Q: 效能比預期慢

1. **確認 release 模式**：`maturin develop --release`（沒有 `--release` 慢 10-50x）
2. **確認 Rust 路徑**：`_HAS_RUST == True`
3. **跑基準測試**：見[第 6 節](#6-效能基準測試)
4. **高併發才有效**：Rust 加速在 N=100+ requests 時明顯，N=1-10 時 GPU 為主要瓶頸

### Q: Linux x86_64 需要特殊設定嗎

不需要。`.cargo/config.toml` 中的 macOS linker flag 在 Linux 上自動忽略。
直接 `maturin develop --release` 即可。

---

## 9. TODO — 已完成與未完成項目

### 已完成 ✅

| 項目 | Rust 檔案 | Python 整合位置 | 效能 |
|------|----------|----------------|------|
| Token 預算計算 | `schedule.rs` | `rust_accelerated.py` | 194x vs Python |
| 批次 stop 檢查 | `stop_check.rs` | `scheduler.py` `update_from_output()` | 301x vs Python |
| Spec decode 接受/拒絕 | `update_output.rs` | `rust_accelerated.py` | 192x vs Python |
| N-gram KMP 並行 | `ngram.rs` | `ngram_proposer.py` | 2.1x vs Numba-1T |
| N-gram Numba 多線程解鎖 | — | `ngram_proposer.py` | 1.8x（P0 改 1 行）|
| N-gram set 查找 | — | `ngram_proposer.py` | O(n)→O(1)（P0 改 1 行）|
| Block hash 加速 | `block_hash.rs` | `kv_cache_utils.py`, `utils/hashing.py` | xxHash + batch |
| Free block queue | `block_pool.rs` | `kv_cache_utils.py` | Arena O(1) 雙向鏈結串列 |
| Stop string 匹配 | `stop_strings.rs` | `detokenizer.py` | Aho-Corasick 多模式匹配 |
| 序列化輔助 | `serial_helpers.rs` | （Rust 已實作） | int32 array 編碼 |
| 模組名遷移 | `lib.rs` + `pyproject.toml` | 全部 import | `vllm._rs` + `_rs` fallback |

### 未完成

| 優先級 | 項目 | 說明 | 難度 |
|--------|------|------|------|
| **P0** | `"builtin"` 加入 `PrefixCachingHashAlgo` | `config/cache.py` 的 `Literal` 型別未包含 `"builtin"`，CLI `--prefix-caching-hash-algo builtin` 會被 argparse 拒絕。需加入 type 定義 + 測試 | 低 |
| **P1** | `schedule()` running 迴圈完整接入 | `compute_running_tokens` 已可用，但排程迴圈中穿插了 KV cache allocation、encoder scheduling、preemption，需拆解 budget 回收邏輯後才能用 Rust 結果驅動迴圈 | 中 |
| **P1** | `batch_apply_spec_decode` 接入 `update_from_output()` | `batch_apply_generated_tokens` Rust 函數已就緒（192x），但 `update_from_output()` 的 spec decode 調整段仍逐 request 處理。可用 batch precompute 模式（類似 batch stop check）在迴圈前批次計算 | 中 |
| **P1** | vllm build system 整合 | 將 `maturin build` 嵌入 vllm 的 `pyproject.toml`，讓 `pip install vllm` 自動編譯 Rust 模組（或作為 optional dependency） | 中 |
| **P1** | `serial_helpers.rs` Python 端接入 | Rust 函數已寫好，尚未接入 `serial_utils.py` 的 `MsgpackEncoder.enc_hook()` tensor 編碼路徑。注意：numpy `tobytes()` 已是 C 最佳化，Rust 可能無額外收益，需先 profile 確認瓶頸 | 低 |
| **P2** | CUDA n-gram kernel | GPU 版 n-gram 用 PyTorch `unfold` 做 O(n×m) 暴力匹配 + Python `for` 遍歷 ngram 長度；寫 fused CUDA kernel 做 O(n) KMP 可快 2-5x | 高 |
| **P2** | Rust SoA request metadata 鏡像 | 在 Rust 側維護 running requests 的 struct-of-arrays 鏡像（num_computed_tokens 等），避免每步從 Python 物件收集到 numpy 的開銷 | 高 |
| **P3** | Lock-free IPC queue | 替換 EngineCore 的 `queue.Queue`（每次 put/get 需 GIL + mutex 雙重鎖）為 Rust SPSC ring buffer，透過 PyO3 暴露 | 高 |
| **P3** | Shared memory IPC | 用 Rust 管理的 memory-mapped ring buffer 替換 ZMQ（同機器內），消除 socket + kernel copy 開銷 | 高 |
| **P3** | CI 自動建構 | GitHub Actions：PR 中自動 `maturin build` + `pytest` Rust 測試 + 效能回歸檢查 | 低 |

### 驗證待辦

- [ ] Linux x86_64 + NVIDIA GPU 環境下建構並驗證所有 Rust 模組
- [ ] `benchmark_serving.py` 做有/無 Rust 的 A/B 對比（不同併發數 1/10/100/500/1000）
- [ ] `py-spy` 或 `scalene` profile 確認 CPU 熱點是否已從 scheduler 迴圈轉移
- [ ] 新增 `block_hash` Python 整合測試（`"builtin"` + chain hashing 正確性）
- [ ] 新增 `RustFreeBlockQueue` 與原 Python 版本的行為交叉驗證測試
- [ ] 新增 `StopStringMatcher` 邊界測試（Unicode、空 stop list、跨 token boundary）
