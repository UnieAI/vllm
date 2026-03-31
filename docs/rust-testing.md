# vLLM Rust 模組 — 測試、Benchmark、開發流程

本文件涵蓋驗證、效能測試、開發流程。啟動方式見 [`rust-getting-started.md`](rust-getting-started.md)。

---

## 1. Smoke Test

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
assert list(r) == [1, 2]
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

# ── Block Hash ──
parent = b'\x00' * 16
hashes = batch_hash_blocks(parent, [1, 2, 3, 4, 5, 6], 3)
assert len(hashes) == 2 and hashes[0] != hashes[1]
hashes2 = batch_hash_blocks(hashes[0], [4, 5, 6], 3)
assert hashes2[0] == hashes[1]
print("[PASS] batch_hash_blocks")

# ── Block Pool ──
q = RustFreeBlockQueue([0, 1, 2, 3, 4], 5)
assert q.popleft() == 0
q.remove(2)
assert q.get_all() == [1, 3, 4]
print("[PASS] RustFreeBlockQueue")

# ── Stop Strings ──
m = StopStringMatcher(["</s>", "STOP"])
r = m.check("Hello</s>", 9, False)
assert r is not None and r[0] == "</s>"
assert m.check("Hello world", 11, False) is None
print("[PASS] StopStringMatcher")

print("\n=== All 6 smoke tests passed ===")
PYEOF
```

---

## 2. 完整測試套件

```bash
# 如果 vLLM conftest 有依賴衝突，複製到 /tmp 跑
cp tests/v1/core/test_rust_scheduler.py /tmp/
cp tests/v1/spec_decode/test_ngram_rust.py /tmp/
cd /tmp && python3 -m pytest test_rust_scheduler.py test_ngram_rust.py -v

# 如果 vLLM 已安裝
python3 -m pytest tests/v1/core/test_rust_scheduler.py tests/v1/spec_decode/test_ngram_rust.py -v
```

預期：**25 tests passed**。

---

## 3. 效能 Benchmark

```bash
python3 << 'PYEOF'
import time, numpy as np

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

def bench(fn, iters=5000, warmup=200):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters

print(f"Benchmark: N={N} requests")
print("=" * 60)

# 1. compute_running_tokens
spec = rng.integers(100, 4096, size=N, dtype=np.int64)
ph = np.zeros(N, dtype=np.int64)
comp = spec - rng.integers(1, 50, size=N, dtype=np.int64)
prompt = rng.integers(50, 2000, size=N, dtype=np.int64)
maxt = np.full(N, 1024, dtype=np.int64)

t = bench(lambda: compute_running_tokens(spec, ph, comp, prompt, maxt, 100000, 0, 4096))
print(f"compute_running_tokens:  {t*1e6:>8.1f} μs  (target: < 20 μs)")

# 2. batch_check_stop
lt = rng.integers(0, 50000, size=N, dtype=np.int64)
nt = rng.integers(100, 4096, size=N, dtype=np.int64)
no = rng.integers(0, 500, size=N, dtype=np.int64)
mt = np.zeros(N, dtype=np.int64)
mx = np.full(N, 1024, dtype=np.int64)
eos = np.full(N, 2, dtype=np.int64)
sa = np.full((N, 5), -1, dtype=np.int64)
for i in range(N):
    ns = rng.integers(0, 6)
    sa[i, :ns] = rng.integers(0, 50000, size=ns)

t = bench(lambda: batch_check_stop(lt, nt, no, mt, mx, eos, sa, 4096))
print(f"batch_check_stop:        {t*1e6:>8.1f} μs  (target: < 30 μs)")

# 3. batch_hash_blocks
parent = b'\x00' * 16
tokens = list(range(N * 16))
t = bench(lambda: batch_hash_blocks(parent, tokens, 16), iters=1000)
print(f"batch_hash_blocks:       {t*1e6:>8.1f} μs")

# 4. FreeBlockQueue
q = RustFreeBlockQueue(list(range(10000)), 10000)
t = bench(lambda: q.append_n(q.popleft_n(10000)), iters=1000)
print(f"FreeBlockQueue (10K):    {t*1e6:>8.1f} μs")

# 5. StopStringMatcher
import string, random
random.seed(42)
text = ''.join(random.choices(string.ascii_letters + ' ', k=10000))
matcher = StopStringMatcher(['ENDOFTEXT', '</s>', '###', 'STOP', 'END'])
t = bench(lambda: matcher.check(text, 50, False), iters=10000)
print(f"StopStringMatcher:       {t*1e6:>8.2f} μs")

print("=" * 60)
PYEOF
```

### 端到端驗證（需要 GPU）

```bash
# 1. 安裝
cd /path/to/vllm
pip install -e .   # 自動建構 Rust
# 或手動: cd rust && maturin develop --release && cd ..

# 2. Benchmark with Rust
vllm serve meta-llama/Llama-3.1-8B &
python benchmarks/benchmark_serving.py --model meta-llama/Llama-3.1-8B --num-prompts 1000 --request-rate inf

# 3. Benchmark without Rust
pip uninstall -y vllm-scheduler-rs
# 重啟 server，再跑同樣的 benchmark
```

---

## 4. 開發流程

### 4.1 修改 → 建構 → 測試

```bash
vim rust/src/schedule.rs        # 修改 Rust
cd rust && cargo check          # 語法檢查（~1 秒）
maturin develop --release       # 建構 + 安裝（~3 秒增量）
cd .. && python3 -m pytest tests/v1/core/test_rust_scheduler.py -v
```

### 4.2 新增 Rust 函數

**Step 1** — `rust/src/my_module.rs`:

```rust
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
pub fn my_fast_function(arr: PyReadonlyArray1<'_, i64>) -> i64 {
    arr.as_slice().unwrap().iter().sum()
}
```

**Step 2** — `rust/src/lib.rs` 註冊:

```rust
mod my_module;
// in vllm_rs():
m.add_function(wrap_pyfunction!(my_module::my_fast_function, m)?)?;
```

**Step 3** — 建構: `cd rust && maturin develop --release`

**Step 4** — Python 使用（含 fallback）:

```python
try:
    from vllm._rs import my_fast_function
except ImportError:
    def my_fast_function(arr):
        return sum(arr)  # Pure Python fallback
```

### 4.3 新增 Rust pyclass

```rust
// rust/src/my_class.rs
use pyo3::prelude::*;

#[pyclass]
pub struct MyMatcher { patterns: Vec<String> }

#[pymethods]
impl MyMatcher {
    #[new]
    fn new(patterns: Vec<String>) -> Self { Self { patterns } }
    fn check(&self, text: &str) -> Option<String> {
        self.patterns.iter().find(|p| text.contains(p.as_str())).cloned()
    }
}
```

在 `lib.rs` 中: `m.add_class::<my_class::MyMatcher>()?;`

---

## 5. 常見問題

| 問題 | 解法 |
|------|------|
| `maturin develop` 找不到 `Cargo.toml` | 確保在 `rust/` 目錄下執行 |
| macOS `cargo test` linker 錯誤 | 用 pytest 跑測試，不用 `cargo test` |
| `from vllm._rs import` 失敗 | vllm 未安裝時用 `from _rs import`；代碼已有 fallback |
| 修改 Rust 後 Python 沒更新 | `cd rust && maturin develop --release`，重啟 Python |
| 效能比預期慢 | 確認 `--release`、確認 `_HAS_RUST == True`、高併發才有效 |
| Linux x86_64 特殊設定 | 不需要，macOS linker flag 在 Linux 自動忽略 |
| 多 Python 版本 wheel | `maturin build --release --interpreter python3.10 python3.11` |
| 確認 Rust 路徑 | `python3 -c "from vllm.v1.core.sched.rust_accelerated import _HAS_RUST; print(_HAS_RUST)"` |
| 強制跳過 Rust 建構 | `VLLM_SKIP_RUST=1 pip install -e .` |
