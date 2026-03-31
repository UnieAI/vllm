# vLLM Rust 加速模組 — 啟動與 Roadmap

---

## 目錄

1. [快速啟動](#1-快速啟動)
2. [啟動方式詳解](#2-啟動方式詳解)
3. [驗證](#3-驗證)
4. [TODO Roadmap](#4-todo-roadmap)

---

## 1. 快速啟動

```bash
# 安裝 Rust（如果沒有）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 方法 A：隨 vLLM 一起安裝（推薦）
cd /path/to/vllm
pip install maturin
pip install -e .    # setup.py 自動偵測 cargo，自動建構 vllm._rs

# 方法 B：單獨安裝 Rust 模組
cd /path/to/vllm/rust
pip install maturin
maturin develop --release
```

安裝完成後 vLLM 啟動時**自動啟用**，日誌顯示：

```
INFO: Rust scheduler acceleration enabled
```

沒有 Rust 時自動 fallback 到純 Python，**不影響任何功能**。

---

## 2. 啟動方式詳解

### 2.1 方法 A：`pip install -e .` 自動建構

`setup.py` 中的 `_build_rust_extension()` 會在 vLLM 安裝時自動建構 Rust 模組：

```
pip install -e .
  └── setup.py cmake_build_ext.run()
       ├── super().run()             # C/CUDA 編譯（原有）
       └── _build_rust_extension()   # Rust 編譯（新增）
            ├── 檢查 cargo 是否存在
            ├── 檢查 maturin 是否存在（沒有就自動 pip install）
            └── maturin develop --release
```

**行為**：
- 有 cargo → 自動編譯（首次 ~15s，增量 ~3s）
- 沒有 cargo → 跳過，印提示
- 編譯失敗 → 印 WARNING，**不中斷安裝**
- `VLLM_SKIP_RUST=1` → 強制跳過

### 2.2 方法 B：手動 `maturin develop`

```bash
cd /path/to/vllm/rust
maturin develop --release
```

適用於：
- 只想開發/測試 Rust 模組，不想裝完整 vLLM
- `maturin develop` 後可以 `from _rs import ...` 使用（不需要 vLLM）

### 2.3 方法 C：建構 wheel 分發

```bash
cd /path/to/vllm/rust
maturin build --release
# 產出: target/wheels/vllm_scheduler_rs-0.1.0-cp312-cp312-linux_x86_64.whl

# 在目標機器上安裝
pip install target/wheels/vllm_scheduler_rs-*.whl
```

### 2.4 環境需求

| 工具 | 最低版本 | 必要性 |
|------|---------|--------|
| Python | 3.10+ | 必要 |
| Rust (cargo) | 1.70+ | 可選（沒有就 fallback） |
| maturin | 1.0+ | 可選（setup.py 會自動安裝） |
| numpy | 1.24+ | 必要（vLLM 依賴） |

### 2.5 各元件啟用狀態

| 元件 | 啟用方式 |
|------|---------|
| Scheduler batch stop 預計算 | 自動 |
| `schedule()` Rust token 預算 | 自動 |
| FreeKVCacheBlockQueue | 自動 |
| StopStringMatcher (Aho-Corasick) | 自動（有 stop strings 時） |
| N-gram Rust proposer | 自動 |
| Block hash (xxh3_128) | 需設定 `--prefix-caching-hash-algo builtin` |

---

## 3. 驗證

```bash
# 快速檢查
python3 -c "
try:
    from vllm._rs import compute_running_tokens
    print('OK: vllm._rs')
except ImportError:
    try:
        from _rs import compute_running_tokens
        print('OK: _rs (dev mode)')
    except ImportError:
        print('FAIL: not installed')
"
```

完整測試和 benchmark 見 [`docs/rust-testing.md`](rust-testing.md)。

---

## 4. TODO Roadmap

### 已完成 ✅（15 項）

| # | 項目 | Rust 檔案 | Python 整合 | 效能 |
|---|------|----------|-------------|------|
| 1 | Token 預算計算 | `schedule.rs` | `rust_accelerated.py` | 194x |
| 2 | 批次 stop 檢查（2D numpy） | `stop_check.rs` | `scheduler.py` | 301x |
| 3 | Spec decode 接受/拒絕 | `update_output.rs` | `scheduler.py` | 192x |
| 4 | `schedule()` running 迴圈接入 | `schedule.rs` | `scheduler.py` | Rust 預算 + Python 調整 |
| 5 | `update_from_output()` batch spec decode | `update_output.rs` | `scheduler.py` | batch 快速路徑 |
| 6 | `"builtin"` hash 算法 | `block_hash.rs` | `config/cache.py` + `hashing.py` | — |
| 7 | N-gram KMP 並行 | `ngram.rs` | `ngram_proposer.py` | 2.1x vs Numba |
| 8 | N-gram Numba 多線程解鎖 | — | `ngram_proposer.py` | 1.8x |
| 9 | N-gram set 查找 | — | `ngram_proposer.py` | O(n)→O(1) |
| 10 | Block hash（xxh3_128） | `block_hash.rs` | `kv_cache_utils.py` | 13.7x |
| 11 | Free block queue | `block_pool.rs` | `kv_cache_utils.py` | 6.0x |
| 12 | Stop string（Aho-Corasick） | `stop_strings.rs` | `detokenizer.py` | 7.3x |
| 13 | 模組名 `vllm._rs` | `lib.rs` | 全部 import | fallback chain |
| 14 | vllm build system 整合 | — | `setup.py` | best-effort auto |
| 15 | 序列化輔助 | `serial_helpers.rs` | Rust 可用 | — |

### 不做（已驗證不值得）

| 項目 | 原因 |
|------|------|
| `serial_helpers.rs` Python 端接入 | Profile 結果：numpy `tobytes()` 比 Rust 快 3x（numpy 直接回傳 buffer pointer，零拷貝；Rust 需逐元素 `to_le_bytes` + 分配 PyBytes）。Rust 函數保留在 crate 中但不接入 Python。 |

### 未完成（5 項）

| 優先級 | 項目 | 說明 | 難度 |
|--------|------|------|------|
| **P2** | CUDA n-gram kernel | GPU 版用 `unfold` O(n×m)；fused CUDA KMP 可快 2-5x | 高 |
| **P2** | Rust SoA request metadata 鏡像 | Rust 側維護 running requests SoA，避免每步 Python→numpy 收集開銷 | 高 |
| **P3** | Lock-free IPC queue | Rust SPSC ring buffer 替換 `queue.Queue` | 高 |
| **P3** | Shared memory IPC | mmap ring buffer 替換 ZMQ | 高 |
| **P3** | CI 自動建構 | GitHub Actions + 效能回歸 | 低 |

### 驗證待辦

- [ ] Linux x86_64 + NVIDIA GPU 環境建構驗證
- [ ] `benchmark_serving.py` 有/無 Rust A/B 對比（併發 1/10/100/500/1000）
- [ ] `py-spy` profile 確認 CPU 熱點轉移
- [ ] `block_hash` / `RustFreeBlockQueue` / `StopStringMatcher` 整合測試

### Next Step

1. **驗證優先**：Linux GPU 機器跑端到端 benchmark，取得實際吞吐量數據
2. **P2**：CUDA n-gram kernel（如果 GPU 版 n-gram 使用率高）
3. **P3**：Lock-free IPC（架構性改動，投入大回報大）
