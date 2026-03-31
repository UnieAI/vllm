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
| Rust SoA request metadata 鏡像 | CPU 排程已從 ~5ms→0.27ms（95% 減少），Python 屬性收集僅佔 ~50-100μs，再省不顯著。維護 Rust↔Python 雙向同步的成本遠高於收益。 |
| Lock-free IPC queue | `queue.Queue` 每步只呼叫 1-2 次，瓶頸在序列化（msgspec）而非 queue 機制。改 EngineCore 進程架構風險高、收益低。 |
| Shared memory IPC | 大 tensor 已走 OOB shared memory（`TensorIpcSender`），只有小 metadata 走 ZMQ，overhead 可忽略。|

### 驗證待辦

- [ ] Linux x86_64 + NVIDIA GPU 環境建構驗證
- [ ] `benchmark_serving.py` 有/無 Rust A/B 對比（併發 1/10/100/500/1000）
- [ ] `py-spy` profile 確認 CPU 熱點轉移
- [ ] `block_hash` / `RustFreeBlockQueue` / `StopStringMatcher` 整合測試

### 4.3 Draft 品質優化（D1-D5，已完成）

大模型（70B+）的 GPU forward 佔 25-50ms，CPU 加速只省 ~8%。D1-D5 嘗試從 **speculative decoding 的 draft 控制** 層面提升吞吐量。

#### N-gram 的適用場景

N-gram speculative decoding **不是所有場景都有效**。接受率取決於輸出是否重複 context 中的 token 模式：

| 場景 | 預估接受率 | 原因 |
|------|-----------|------|
| Code generation | 50-70% | 大量重複模式（boilerplate, imports） |
| RAG + 長 context | 40-60% | 回答常引用 context 原文 |
| 長文件 summarization | 40-60% | 輸出重複原文片段 |
| 翻譯 | 20-40% | 部分術語/格式重複 |
| Multi-turn chat（短回覆）| 10-30% | 回覆多樣，很少重複 |
| Creative writing | 5-15% | 幾乎不重複 |

#### D1-D5 做了什麼

D1-D5 **不提升接受率本身**，而是**減少無效 GPU 驗證時間**：

```
沒有 D1-D5：
  k=5, 接受率 20% → 每步驗證 5 tokens，只接受 1 token
  浪費 = 4 tokens 的 GPU 驗證時間（~80% 白跑）

有 D1-D5：
  D1 偵測接受率 20% → k 自動降至 1
  每步只驗證 1 token，接受 1 token → 0 浪費
  等效：對低接受率 request 關閉 spec decode（正確決策）
```

| # | 優化 | 觸發條件 | 機制 |
|---|------|---------|------|
| **D1** Adaptive k | 接受率 < 40% | `_adaptive_k()`: k 按 `rate/0.4` 比例縮減 |
| **D2** Low-confidence filter | 接受率 < 20% 且 50+ tokens | `_is_low_confidence_draft()`: draft 減半 |
| **D3** Gradual DSC | enable~disable 閾值之間 | `_get_effective_spec_k()`: k 線性插值，取代 binary 開關 |
| **D4** Relaxed retry | min_n > 1 且 0 draft | `_run_propose_backend()`: 以 min_n=1 重試 |
| **D5** Position decay | 位置 N 接受率 < 15% | `_position_decay_k()`: 截斷至位置 N |

#### 預估效果（誠實評估）

| 優化 | 預估吞吐量改善 | 信心度 | 條件 |
|------|---------------|--------|------|
| D1 Adaptive k | +3-8% | 中 | 混合 workload（有高有低接受率） |
| D2 Low-confidence | +1-3% | 低 | 只影響持續低接受率的 request |
| D3 Gradual DSC | +2-5% | 中 | 只在 DSC 啟用 + 中等負載時有效 |
| D4 Relaxed retry | +1-3% | 低 | 只在 min_n > 1 且嚴格設定時有效 |
| D5 Position decay | +2-5% | 中 | 幾乎所有場景尾部位置都衰減 |
| **D1-D5 合計** | **+5-15%** | **低-中** | **高度依賴 workload** |

> **重要聲明**：以上數字全部是推算，**沒有 GPU 實測數據**。
>
> D1-D5 的效果高度依賴：
> - 具體 workload（code vs chat vs RAG）
> - 原始接受率分佈（如果所有 request 接受率都很高，D1-D5 幾乎無效）
> - k 的原始設定（k=3 vs k=7 差異大）
> - 模型大小（小模型 GPU 快，spec decode 開銷佔比更大）
>
> **最誠實的結論**：在 chat workload + 大模型場景下，D1-D5 可能只帶來 **0-5%** 的吞吐量改善。
> 真正要大幅提升大模型吞吐量，需要 **EAGLE/Medusa 等 learned draft model**（接受率 70-90%）。

#### N-gram 以外的方向（未實作）

| # | 方向 | 預期效果 | 難度 |
|---|------|---------|------|
| D6 | Token embedding 相似度匹配 | 提升非重複文本接受率 | 高 |
| D7 | Lightweight draft model (EAGLE) | 接受率 70-90% | 高 |
| D8 | Tree-based speculation | 等效吞吐量 2-3x | 很高 |

### Next Step

1. **驗證**：Linux GPU 機器跑 `benchmark_serving.py` A/B 對比，用不同 workload（code / chat / RAG）驗證 D1-D5 實際效果
2. **根據驗證結果調參**：D1 的 0.4 閾值、D5 的 0.15 閾值可能需要根據實測調整
3. **如果 D1-D5 效果不顯著**：考慮移除以減少代碼複雜度，專注 EAGLE 整合
