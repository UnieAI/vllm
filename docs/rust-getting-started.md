# vLLM Rust 加速模組 — 啟動教學

本文件說明如何從零開始建構、安裝、驗證 `vllm._rs` Rust 加速模組。

---

## 目錄

1. [前置需求](#1-前置需求)
2. [環境準備](#2-環境準備)
3. [建構與安裝](#3-建構與安裝)
4. [驗證安裝](#4-驗證安裝)
5. [在 vLLM 中啟用](#5-在-vllm-中啟用)
6. [開發流程](#6-開發流程)
7. [常見問題](#7-常見問題)

---

## 1. 前置需求

| 工具 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.10+ | vLLM 運行環境 |
| Rust toolchain | 1.70+ | 編譯 Rust 程式碼 |
| maturin | 1.0+ | Rust→Python 建構工具 |
| numpy | 1.24+ | Rust↔Python 資料交換 |

### 1.1 檢查現有環境

```bash
# 確認 Python 版本
python3 --version
# 期望輸出: Python 3.10.x 或更高

# 確認 Rust 是否已安裝
rustc --version
# 期望輸出: rustc 1.7x.0 或更高

# 確認 cargo（Rust 套件管理器）
cargo --version
```

---

## 2. 環境準備

### 2.1 安裝 Rust（如果尚未安裝）

```bash
# 官方安裝腳本（Linux / macOS）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安裝完成後，載入環境變數
source $HOME/.cargo/env

# 驗證
rustc --version
cargo --version
```

> **Windows 使用者**：到 https://rustup.rs/ 下載安裝程式。

### 2.2 安裝 maturin

```bash
pip install maturin
```

### 2.3 確認 numpy 已安裝

```bash
python3 -c "import numpy; print(numpy.__version__)"
# 需要 1.24 以上
```

---

## 3. 建構與安裝

### 3.1 切換到 Rust crate 目錄

```bash
cd /path/to/vllm/rust
```

目錄結構應該是：

```
rust/
├── Cargo.toml          # Rust 依賴定義
├── Cargo.lock          # 鎖定的依賴版本
├── pyproject.toml      # maturin 建構配置
├── .cargo/
│   └── config.toml     # macOS linker 設定
└── src/
    ├── lib.rs           # PyO3 模組入口（註冊所有函數）
    ├── schedule.rs      # compute_running_tokens, compute_waiting_tokens
    ├── stop_check.rs    # batch_check_stop
    ├── update_output.rs # batch_apply_generated_tokens
    ├── ngram.rs         # batch_ngram_propose（並行 KMP）
    ├── block_hash.rs    # hash_block_tokens_rust, batch_hash_blocks
    ├── block_pool.rs    # RustFreeBlockQueue
    ├── stop_strings.rs  # StopStringMatcher
    └── serial_helpers.rs # batch_encode_int32_arrays
```

### 3.2 開發模式安裝（推薦）

```bash
maturin develop --release
```

這會：
1. 編譯所有 Rust 原始碼（release 模式，含最佳化）
2. 產生 `.so` / `.dylib` / `.pyd` 檔案
3. 安裝到目前的 Python 環境中

首次建構約需 **15-20 秒**（下載並編譯依賴），後續增量建構約 **1-5 秒**。

**預期輸出：**

```
🔗 Found pyo3 bindings
🐍 Found CPython 3.12 at /path/to/python
📡 Using build options features from pyproject.toml
   Compiling pyo3 v0.23.5
   Compiling numpy v0.23.0
   Compiling vllm-scheduler-rs v0.1.0
    Finished `release` profile [optimized] target(s) in 17.05s
📦 Built wheel for CPython 3.12 to /tmp/.../vllm_scheduler_rs-0.1.0-....whl
✏️ Setting installed package as editable
🛠 Installed vllm-scheduler-rs-0.1.0
```

### 3.3 建構 wheel（用於分發 / CI）

```bash
maturin build --release
```

產出的 `.whl` 檔案在 `target/wheels/` 目錄中，可以用 `pip install` 安裝到任何相容的環境：

```bash
pip install target/wheels/vllm_scheduler_rs-0.1.0-cp312-cp312-linux_x86_64.whl
```

---

## 4. 驗證安裝

### 4.1 基本 import 測試

```bash
python3 -c "
try:
    from vllm._rs import batch_ngram_propose
    print('OK: vllm._rs 載入成功')
except ImportError:
    try:
        from _rs import batch_ngram_propose
        print('OK: _rs 載入成功（開發模式）')
    except ImportError:
        print('FAIL: 模組未安裝')
"
```

### 4.2 列出所有可用函數

```python
python3 -c "
try:
    import _rs as mod
except ImportError:
    from vllm import _rs as mod

for name in sorted(dir(mod)):
    if not name.startswith('_'):
        print(f'  {name}')
"
```

預期輸出：

```
  RustFreeBlockQueue
  StopStringMatcher
  batch_apply_generated_tokens
  batch_check_stop
  batch_encode_int32_arrays
  batch_hash_blocks
  batch_ngram_propose
  compute_running_tokens
  compute_waiting_tokens
  encode_int_array_as_bytes
  hash_block_tokens_rust
```

### 4.3 功能驗證（smoke test）

```python
python3 -c "
import numpy as np

try:
    from vllm._rs import (
        compute_running_tokens,
        batch_check_stop,
        batch_apply_generated_tokens,
        batch_ngram_propose,
    )
except ImportError:
    from _rs import (
        compute_running_tokens,
        batch_check_stop,
        batch_apply_generated_tokens,
        batch_ngram_propose,
    )

# 1. compute_running_tokens
result = compute_running_tokens(
    np.array([100], dtype=np.int64),   # num_tokens_with_spec
    np.zeros(1, dtype=np.int64),       # num_output_placeholders
    np.array([90], dtype=np.int64),    # num_computed_tokens
    np.array([100], dtype=np.int64),   # num_prompt_tokens
    np.array([100], dtype=np.int64),   # max_tokens_per_req
    1000,                              # token_budget
    0,                                 # long_prefill_threshold (0=disabled)
    4096,                              # max_model_len
)
assert list(result) == [10], f'FAIL: expected [10], got {list(result)}'
print('[PASS] compute_running_tokens')

# 2. batch_check_stop
stop_result = batch_check_stop(
    np.array([2], dtype=np.int64),     # last_token_ids (EOS=2)
    np.array([100], dtype=np.int64),   # num_tokens
    np.array([10], dtype=np.int64),    # num_output_tokens
    np.array([0], dtype=np.int64),     # min_tokens
    np.array([100], dtype=np.int64),   # max_tokens_per_req
    np.array([2], dtype=np.int64),     # eos_token_ids
    np.empty((1, 0), dtype=np.int64),  # stop_token_ids (2D, no stop tokens)
    4096,                              # max_model_len
)
assert stop_result[0] == 1, f'FAIL: expected 1 (EOS), got {stop_result[0]}'
print('[PASS] batch_check_stop')

# 3. batch_apply_generated_tokens
ac, ap, aa, ar = batch_apply_generated_tokens(
    np.array([105], dtype=np.int64),   # num_computed_tokens
    np.array([5], dtype=np.int64),     # num_output_placeholders
    np.array([3], dtype=np.int64),     # num_generated
    np.array([5], dtype=np.int64),     # num_draft_tokens
)
assert list(ac) == [102], f'FAIL: expected [102], got {list(ac)}'
print('[PASS] batch_apply_generated_tokens')

# 4. batch_ngram_propose
token_ids = np.zeros((1, 100), dtype=np.int32)
token_ids[0, :8] = [1, 2, 3, 4, 5, 1, 2, 3]
draft, num_drafts = batch_ngram_propose(
    token_ids,                         # token_ids [batch, max_len]
    np.array([8], dtype=np.int32),     # num_tokens [batch]
    [0],                               # valid_indices
    2,                                 # min_n
    5,                                 # max_n
    100,                               # max_model_len
    3,                                 # k (num draft tokens)
)
assert num_drafts[0] == 3, f'FAIL: expected 3 drafts, got {num_drafts[0]}'
print('[PASS] batch_ngram_propose')

print()
print('=== All 4 smoke tests passed ===')
"
```

### 4.4 跑完整測試套件

```bash
# 從 vllm 根目錄
# 注意：需要在 conftest.py 不干擾的環境下跑
# 方法 1：複製到 /tmp 跑
cp tests/v1/core/test_rust_scheduler.py /tmp/
cp tests/v1/spec_decode/test_ngram_rust.py /tmp/
cd /tmp && python3 -m pytest test_rust_scheduler.py test_ngram_rust.py -v
```

預期輸出：

```
test_rust_scheduler.py::TestRustComputeRunningTokens::test_basic PASSED
test_rust_scheduler.py::TestRustComputeRunningTokens::test_budget_clamp PASSED
test_rust_scheduler.py::TestRustComputeRunningTokens::test_long_prefill_threshold PASSED
test_rust_scheduler.py::TestRustComputeRunningTokens::test_max_model_len_clamp PASSED
test_rust_scheduler.py::TestRustComputeRunningTokens::test_async_scheduling_skip PASSED
test_rust_scheduler.py::TestRustComputeRunningTokens::test_empty PASSED
test_rust_scheduler.py::TestRustBatchCheckStop::test_eos PASSED
test_rust_scheduler.py::TestRustBatchCheckStop::test_stop_token PASSED
test_rust_scheduler.py::TestRustBatchCheckStop::test_length_cap PASSED
test_rust_scheduler.py::TestRustBatchCheckStop::test_min_tokens_suppresses_stop PASSED
test_rust_scheduler.py::TestRustBatchCheckStop::test_multiple_requests_mixed PASSED
test_rust_scheduler.py::TestRustBatchApplyGenerated::test_no_spec PASSED
test_rust_scheduler.py::TestRustBatchApplyGenerated::test_with_spec PASSED
test_rust_scheduler.py::TestRustComputeWaitingTokens::test_basic PASSED
test_rust_scheduler.py::TestRustComputeWaitingTokens::test_chunked_prefill_disabled PASSED
test_rust_scheduler.py::TestRustPerformance::test_compute_running_is_fast PASSED
test_rust_scheduler.py::TestRustPerformance::test_batch_check_stop_is_fast PASSED
test_ngram_rust.py::TestRustNgramProposer::test_basic_match PASSED
test_ngram_rust.py::TestRustNgramProposer::test_no_match PASSED
test_ngram_rust.py::TestRustNgramProposer::test_min_ngram_filter PASSED
test_ngram_rust.py::TestRustNgramProposer::test_batch_processing PASSED
test_ngram_rust.py::TestRustNgramProposer::test_valid_indices_subset PASSED
test_ngram_rust.py::TestRustNgramProposer::test_max_model_len_limit PASSED
test_ngram_rust.py::TestRustNgramProposer::test_cross_validate_with_numba PASSED
test_ngram_rust.py::TestRustNgramProposer::test_performance PASSED

======================== 25 passed in 1.98s ========================
```

### 4.5 效能基準測試

```python
python3 -c "
import numpy as np
import time

try:
    from vllm._rs import compute_running_tokens, batch_check_stop, batch_ngram_propose
except ImportError:
    from _rs import compute_running_tokens, batch_check_stop, batch_ngram_propose

N = 1000
rng = np.random.default_rng(42)
ITERS = 5000

# --- compute_running_tokens ---
spec = rng.integers(100, 4096, size=N, dtype=np.int64)
ph = np.zeros(N, dtype=np.int64)
comp = spec - rng.integers(1, 50, size=N, dtype=np.int64)
prompt = rng.integers(50, 2000, size=N, dtype=np.int64)
maxt = np.full(N, 1024, dtype=np.int64)

# warmup
compute_running_tokens(spec, ph, comp, prompt, maxt, 100000, 0, 4096)

t0 = time.perf_counter()
for _ in range(ITERS):
    compute_running_tokens(spec, ph, comp, prompt, maxt, 100000, 0, 4096)
sched_us = (time.perf_counter() - t0) / ITERS * 1e6

# --- batch_check_stop ---
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

batch_check_stop(last_tok, num_tok, num_out, min_tok, max_tok2, eos, stop_arr, 4096)

t0 = time.perf_counter()
for _ in range(ITERS):
    batch_check_stop(last_tok, num_tok, num_out, min_tok, max_tok2, eos, stop_arr, 4096)
stop_us = (time.perf_counter() - t0) / ITERS * 1e6

# --- batch_ngram_propose ---
BATCH = 128; SL = 1500; K = 5
ti = np.zeros((BATCH, 100000), dtype=np.int32)
ti[:, :SL] = rng.integers(0, 50, size=(BATCH, SL), dtype=np.int32)
nt = np.full(BATCH, SL, dtype=np.int32)
vi = list(range(BATCH))

batch_ngram_propose(ti, nt, vi, 3, 7, 100000, K)

t0 = time.perf_counter()
for _ in range(200):
    batch_ngram_propose(ti, nt, vi, 3, 7, 100000, K)
ngram_ms = (time.perf_counter() - t0) / 200 * 1e3

print(f'Benchmark Results (N={N} requests)')
print(f'  compute_running_tokens:  {sched_us:6.1f} μs   (target: < 20 μs)')
print(f'  batch_check_stop:        {stop_us:6.1f} μs   (target: < 30 μs)')
print(f'  batch_ngram_propose:     {ngram_ms:6.2f} ms   (target: < 1 ms)')
print()
if sched_us < 20 and stop_us < 30 and ngram_ms < 1:
    print('Performance OK')
else:
    print('WARNING: Some functions slower than expected')
"
```

---

## 5. 在 vLLM 中啟用

### 5.1 自動啟用

安裝 `vllm-scheduler-rs` 後，vLLM 啟動時會**自動偵測並啟用** Rust 加速。你會在日誌中看到：

```
INFO: Rust scheduler acceleration enabled (vllm._rs)
```

如果模組不存在，會看到：

```
INFO: vllm._rs not available, using pure-Python scheduler
```

**不需要任何額外配置。**

### 5.2 影響的元件

| 元件 | 加速的操作 | 預期加速 |
|------|-----------|---------|
| Scheduler `schedule()` | batch stop 預計算 | 301x |
| Scheduler `update_from_output()` | batch stop + spec decode | 192x |
| N-gram proposer（CPU 版） | KMP 匹配 + 並行 | 2.1x |

### 5.3 確認正在使用 Rust

啟動 vLLM 後，在日誌中搜尋 `Rust scheduler acceleration`：

```bash
vllm serve <model> 2>&1 | grep -i rust
```

---

## 6. 開發流程

### 6.1 修改 Rust 程式碼後重新建構

```bash
cd rust/

# 修改 src/*.rs 檔案後：
maturin develop --release

# 通常只需 1-5 秒（增量編譯）
```

### 6.2 只檢查編譯（不安裝）

```bash
cd rust/
cargo check
```

這比 `maturin develop` 快（不需要打包和安裝），適合開發中頻繁檢查語法。

### 6.3 跑 Rust 單元測試

由於 PyO3 extension module 的限制，`cargo test` 在 macOS 上需要特殊 linker 設定（已在 `.cargo/config.toml` 中配置）。但 Rust 測試仍然需要 Python runtime，建議改用 Python 測試：

```bash
cp tests/v1/core/test_rust_scheduler.py /tmp/
cp tests/v1/spec_decode/test_ngram_rust.py /tmp/
cd /tmp && python3 -m pytest test_rust_scheduler.py test_ngram_rust.py -v
```

### 6.4 新增 Rust 函數的流程

1. **在 `rust/src/` 新增或修改 `.rs` 檔案**
2. **在 `rust/src/lib.rs` 註冊**：
   ```rust
   m.add_function(wrap_pyfunction!(your_module::your_function, m)?)?;
   // 或
   m.add_class::<your_module::YourClass>()?;
   ```
3. **建構**：`maturin develop --release`
4. **在 Python 中使用**：
   ```python
   try:
       from vllm._rs import your_function
   except ImportError:
       from _rs import your_function
   ```
5. **撰寫測試**放在 `tests/v1/` 對應目錄

---

## 7. 常見問題

### Q: `maturin develop` 失敗，找不到 `Cargo.toml`

```
💥 maturin failed
  Caused by: Can't find Cargo.toml
```

**解法**：確保你在 `rust/` 目錄下執行，不是 vllm 根目錄。

```bash
cd /path/to/vllm/rust
maturin develop --release
```

### Q: macOS 上 `cargo test` linker 錯誤

```
ld: Undefined symbols: _PyBaseObject_Type ...
```

**解法**：這是正常的。PyO3 extension module 的測試需要透過 Python 執行，不能直接用 `cargo test`。請使用 Python pytest 跑測試。

### Q: `import vllm._rs` 失敗，`No module named 'vllm'`

**原因**：vllm 套件本身沒有安裝到 Python 環境中。maturin 會嘗試把 `.so` 放到 `vllm/` 套件目錄下，但如果 vllm 未安裝，它會放到 `_rs/` 頂層目錄。

**解法**：改用 `from _rs import ...`，或安裝 vllm：

```bash
# 方法 1：用 _rs 匯入（開發模式）
from _rs import batch_ngram_propose

# 方法 2：安裝 vllm（生產模式）
cd /path/to/vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
cd rust && maturin develop --release
# 此時 from vllm._rs import ... 就能用了
```

> **注意**：vLLM 的 Python 程式碼中已經有 fallback，會依序嘗試 `vllm._rs` → `_rs` → 純 Python，不需要手動處理。

### Q: 修改 Rust 後 Python 沒有更新

**原因**：需要重新跑 `maturin develop --release`。

```bash
cd rust/
maturin develop --release
# 然後重新啟動 Python 進程
```

### Q: 如何確認正在使用 Rust 而非 Python fallback？

```python
python3 -c "
from vllm.v1.core.sched.rust_accelerated import _HAS_RUST
print(f'Rust acceleration: {_HAS_RUST}')
"
```

或檢查 ngram：

```python
python3 -c "
from vllm.v1.spec_decode.ngram_proposer import _HAS_RUST_NGRAM
print(f'Rust ngram: {_HAS_RUST_NGRAM}')
"
```

### Q: 想要建構不同 Python 版本的 wheel

```bash
# 指定 Python 路徑
maturin build --release --interpreter /path/to/python3.10

# 建構多版本
maturin build --release --interpreter python3.10 python3.11 python3.12
```

### Q: 效能比預期慢

1. 確認用 `--release` 建構（debug 模式會慢 10-50x）
2. 確認在 `Rust` 路徑而非 fallback：檢查 `_HAS_RUST` 是否為 `True`
3. 跑基準測試確認函數級效能（見 4.5 節）
