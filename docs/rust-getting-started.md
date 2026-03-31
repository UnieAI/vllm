# vLLM Rust 加速模組 — 啟動與 Roadmap

---

## Overview

### 問題

vLLM V1 的 EngineCore 在每步推論中執行 `schedule() → execute_model() → update_from_output()`。GPU 計算佔延遲的 90%+，但在高併發（1000+ 同時請求）時，CPU 端的 Python 排程迴圈累積 ~2-5ms 開銷，在小模型場景（GPU forward ~2-5ms）下佔 step 時間的 **50%+**。

### 解決方案

透過 Rust + PyO3 加速 CPU 熱點路徑，將排程開銷從 ~2ms 降至 ~18μs（**118x**）。模組名稱 `vllm._rs`，安裝後**自動啟用**，未安裝時自動 fallback 到純 Python。

### 效果（mock data 實測）

| 場景 | GPU forward | Step (Python) | Step (Rust) | 加速 |
|------|-------------|---------------|-------------|------|
| 小模型 1-3B × 1000 req | 2 ms | 4.1 ms | 2.0 ms | **2.04x** |
| 7-8B × 1000 req | 5 ms | 7.1 ms | 5.0 ms | **1.42x** |
| 70B × 1000 req | 25 ms | 27.1 ms | 25.0 ms | 1.08x |

> 大模型（70B+）的瓶頸在 GPU forward，CPU 加速效果有限。
> 大模型的進一步優化需要從 **speculative decoding draft 品質** 和 **GPU kernel** 層面著手，見 [Roadmap §4.3](#43-draft-品質優化大模型吞吐量的關鍵)。

### 模組清單

```
rust/src/
├── schedule.rs        Token 預算計算        194x
├── stop_check.rs      批次 stop 檢查        301x
├── update_output.rs   Spec decode 接受/拒絕  192x
├── ngram.rs           N-gram KMP 並行       2.1x
├── block_hash.rs      Prefix cache hash     13.7x
├── block_pool.rs      Free block queue      6.0x
├── stop_strings.rs    Aho-Corasick 匹配     7.3x
└── serial_helpers.rs  序列化輔助            (保留)
```

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

### 已完成 ✅（22 項）

| # | 項目 | 檔案 | 整合位置 | 效能 |
|---|------|------|---------|------|
| 1 | Token 預算計算 | `schedule.rs` | `rust_accelerated.py` | 194x |
| 2 | 批次 stop 檢查（2D numpy） | `stop_check.rs` | `scheduler.py` | 301x |
| 3 | Spec decode 接受/拒絕 | `update_output.rs` | `scheduler.py` | 192x |
| 4 | `schedule()` running 迴圈接入 | `schedule.rs` | `scheduler.py` | Rust 預算 + Python 調整 |
| 5 | `update_from_output()` batch spec decode | `update_output.rs` | `scheduler.py` | batch 快速路徑 |
| 6 | `"builtin"` hash 算法 | `block_hash.rs` | `config/cache.py` + `hashing.py` | — |
| 7 | N-gram KMP 並行（CPU） | `ngram.rs` | `ngram_proposer.py` | 2.1x vs Numba |
| 8 | N-gram Numba 多線程解鎖 | — | `ngram_proposer.py` | 1.8x |
| 9 | N-gram set 查找 | — | `ngram_proposer.py` | O(n)→O(1) |
| 10 | Block hash（xxh3_128） | `block_hash.rs` | `kv_cache_utils.py` | 13.7x |
| 11 | Free block queue | `block_pool.rs` | `kv_cache_utils.py` | 6.0x |
| 12 | Stop string（Aho-Corasick） | `stop_strings.rs` | `detokenizer.py` | 7.3x |
| 13 | 模組名 `vllm._rs` | `lib.rs` | 全部 import | fallback chain |
| 14 | vllm build system 整合 | — | `setup.py` | best-effort auto |
| 15 | 序列化輔助 | `serial_helpers.rs` | Rust 可用 | — |
| 16 | CUDA n-gram kernel（GPU） | `csrc/ngram_kernels.cu` | `ngram_proposer_gpu.py` | Fused KMP O(n)，需 GPU 驗證 |
| 17 | CI 自動建構 | `.github/workflows/rust-ci.yml` | GitHub Actions | cargo check + pytest + 效能回歸 |
| 18 | D1: Adaptive k | — | `scheduler.py` `_adaptive_k()` | 按接受率縮減 draft |
| 19 | D2: Low-confidence filter | — | `scheduler.py` `_is_low_confidence_draft()` | 接受率<20% 減半 |
| 20 | D3: Gradual DSC | — | `scheduler.py` `_get_effective_spec_k()` | 線性降級取代 binary |
| 21 | D4: Relaxed retry | — | `ngram_proposer.py` `_run_propose_backend()` | min_n=1 fallback |
| 22 | D5: Position-aware decay | — | `scheduler.py` `_position_decay_k()` | 截斷低接受率位置 |

### 不做（已驗證不值得）

| 項目 | 原因 |
|------|------|
| `serial_helpers.rs` Python 端接入 | numpy `tobytes()` 比 Rust 快 3x（零拷貝 vs 逐元素複製）。Rust 函數保留但不接入。 |

### 未完成（3 項）

| 優先級 | 項目 | 說明 | 難度 |
|--------|------|------|------|
| **P2** | Rust SoA request metadata 鏡像 | Rust 側維護 running requests SoA，避免每步 Python→numpy 收集開銷 | 高 |
| **P3** | Lock-free IPC queue | Rust SPSC ring buffer 替換 `queue.Queue` | 高 |
| **P3** | Shared memory IPC | mmap ring buffer 替換 ZMQ | 高 |

### 驗證待辦

- [ ] Linux x86_64 + NVIDIA GPU 環境建構驗證
- [ ] `benchmark_serving.py` 有/無 Rust A/B 對比（併發 1/10/100/500/1000）
- [ ] `py-spy` profile 確認 CPU 熱點轉移
- [ ] `block_hash` / `RustFreeBlockQueue` / `StopStringMatcher` 整合測試

### 4.3 Draft 品質優化（大模型吞吐量的關鍵）

大模型（70B+）的 GPU forward 佔 25-50ms，CPU 加速只省 ~8%。真正提升大模型吞吐量的方向是 **speculative decoding 的 draft 品質**——每步 GPU forward 時間固定，但如果 draft 接受率從 60% → 80%，等效吞吐量提升 ~33%。

#### 現有 N-gram 演算法的限制

| 限制 | 說明 | 影響 |
|------|------|------|
| **純語法匹配** | KMP 只匹配 token 序列，不理解語義 | "The dog ran" vs "A canine sprinted" → 無法匹配 |
| **單一匹配** | 只回傳最長 match（一條路徑） | 無法生成多樣化 draft candidates |
| **末尾匹配** | 只搜尋序列末尾的 n-gram | 錯過前面出現過的重複模式 |
| **固定 n-gram 範圍** | `[min_n, max_n]` 靜態配置 | 重複性文本（code）適合大 n，多樣文本適合小 n |
| **無回饋機制** | Draft 品質沒有回饋給 proposer | 持續提出低品質 draft，浪費 GPU 驗證時間 |
| **二元 DSC 開關** | 高負載時完全關閉 n-gram speculation | 應該漸進降級（減少 k）而非全關 |

#### 優化方向（按可行性排序）

**可在目前架構內做的（不改模型）：**

| # | 方向 | 做法 | 預期效果 | 難度 |
|---|------|------|---------|------|
| D1 | **Adaptive k** | 根據每個 request 的歷史接受率動態調整 k。高接受率 → 增加 k，低接受率 → 減少 k 或跳過 | 減少無效 GPU 驗證，提升吞吐量 5-15% | 低 |
| D2 | **Frequency-weighted match** | 對多個 n-gram match 候選，選最高頻出現的而非最長的。高頻模式更可能被接受 | 提升接受率 5-10% | 低 |
| D3 | **Gradual DSC** | 將 DSC 從 binary（全開/全關）改為 `k = max(1, base_k * (1 - load_ratio))`，高負載時減少 draft 數量但不完全關閉 | 高負載下仍保留部分 spec decode 收益 | 低 |
| D4 | **Multi-match candidates** | 回傳 top-N 個 n-gram match（而非只回傳最長），用簡單啟發式選分數最高的 | 增加匹配多樣性，提升接受率 | 中 |
| D5 | **Position-aware draft decay** | 位置越後的 draft token 越不可能被接受。根據歷史 per-position 接受率，動態決定每個位置是否值得 draft | 減少尾部無效 draft，節省 GPU 驗證時間 | 中 |

**需要額外模型/資源的（長期方向）：**

| # | 方向 | 做法 | 預期效果 | 難度 |
|---|------|------|---------|------|
| D6 | **Token embedding 相似度匹配** | 用 embedding cosine similarity 擴展 n-gram 搜尋範圍，匹配語義近似的 token 序列 | 大幅提升非重複性文本的接受率 | 高 |
| D7 | **Lightweight draft model** | 用小模型（如 EAGLE head）生成 draft token，而非純 n-gram 匹配 | 接受率 70-90%（vs n-gram 40-60%） | 高 |
| D8 | **Tree-based speculation** | 同時 draft 多條分支（樹狀），一次 GPU forward 驗證多條路徑 | 等效吞吐量 2-3x | 很高 |

#### 建議路線

```
D1 (adaptive k) → D3 (gradual DSC) → D5 (position decay) → D2 (freq match)
       低難度              低難度             中難度              低難度
```

D1-D3 可以在 1-2 週內完成，且完全不改動 Rust 或 CUDA code，只修改 Python 端的 `ngram_proposer.py` 和 `scheduler.py`。

### Next Step

1. **D1 Adaptive k**：在 `ngram_proposer.py` 加入 per-request 接受率追蹤，動態調整 draft 數量
2. **驗證**：Linux GPU 機器跑 `benchmark_serving.py` A/B 對比
3. **D3 Gradual DSC**：將 binary DSC 開關改為漸進式降級
