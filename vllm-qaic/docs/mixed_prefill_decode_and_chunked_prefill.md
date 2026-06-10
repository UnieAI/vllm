# vLLM Mixed Prefill/Decode Batch 與 Chunked Prefill 比較

本文件說明 vLLM 中 **Mixed Prefill/Decode Batch** 和 **Chunked Prefill** 兩種排程策略的原理、優缺點及適用場景。

---

## 背景

在 LLM 推論中，每個請求經歷兩個階段：

1. **Prefill（預填充）**：處理完整 prompt，計算所有 token 的 KV cache。計算量大、延遲高。
2. **Decode（解碼）**：逐 token 生成回覆，每步只計算一個 token。計算量小但對延遲敏感。

傳統做法將 prefill 和 decode 分開排程（例如先完成所有 prefill 再做 decode），但這會導致 GPU 利用率低或延遲不穩定。vLLM 引入了以下兩種策略來解決這個問題。

---

## Mixed Prefill/Decode Batch

### 原理

將正在 prefill 的請求與正在 decode 的請求**混合放入同一個 batch**，一次 forward pass 同時處理兩種類型的 token。

### 優點

| 優點 | 說明 |
|------|------|
| **提高 GPU 利用率** | Decode 階段單個 token 計算量小、memory-bound；與 prefill 混合後，能填補 compute 空隙，提升整體吞吐量。 |
| **降低排隊延遲** | 新請求不需要等所有 decode 完成才能開始 prefill，減少首 token 延遲（TTFT）。 |
| **簡化排程邏輯** | 不需要將兩個階段分成獨立的 iteration，排程器邏輯更統一。 |

### 缺點

| 缺點 | 說明 |
|------|------|
| **Decode 延遲抖動** | 長 prompt 的 prefill 會佔用大量計算資源，導致同 batch 中 decode 請求的 TPOT（time per output token）增加。 |
| **Padding 浪費** | 若 prefill 序列長度與 decode token 數差異大，attention kernel 需要處理不規則形狀，可能產生 padding 或降低 kernel 效率。 |
| **KV cache 記憶體壓力** | 長 prefill 一次性分配大量 KV cache block，可能擠壓 decode 請求可用的 block 數。 |
| **硬體相容性挑戰** | 部分 NPU/加速器（如 QAIC）不支援動態形狀或不規則 batch，難以直接實作 mixed batch。 |

---

## Chunked Prefill

### 原理

將一個 prefill 請求的 prompt **切成固定大小的 chunk**（例如 512 或 1024 tokens），每個 iteration 只處理一個 chunk。在每個 iteration 中，prefill chunk 與 decode token 一起排程，形成大小可控的 mixed batch。

### 優點

| 優點 | 說明 |
|------|------|
| **Decode 延遲可控** | 每個 iteration 的 prefill 計算量被 chunk size 上限約束，decode token 不會被長 prefill 嚴重拖延。 |
| **TPOT 穩定性** | 由於每步計算量可預測，decode 的 inter-token latency 更穩定，有利於串流輸出體驗。 |
| **記憶體管理更平滑** | KV cache 逐 chunk 分配，避免一次性佔用大量 block，降低 OOM 風險。 |
| **適合固定形狀硬體** | Chunk size 固定後，attention kernel 的輸入形狀確定，有利於 NPU/ASIC 等需要靜態形狀的加速器。 |
| **長 prompt 友好** | 超長 prompt（數萬 token）可以漸進式處理，不會阻塞整個系統。 |

### 缺點

| 缺點 | 說明 |
|------|------|
| **Prefill 總延遲增加** | 原本一步能完成的 prefill 被拆成多步，增加了 TTFT（首 token 時間）。 |
| **排程複雜度提升** | 需要追蹤每個請求的 prefill 進度（已完成多少 chunk），排程器狀態更複雜。 |
| **Chunk size 調參** | Chunk size 是關鍵超參：太大則失去保護 decode 延遲的效果；太小則 prefill 吞吐量下降，TTFT 過高。 |
| **Cross-chunk attention 開銷** | 後續 chunk 需要 attend 到前面 chunk 已計算的 KV cache，隨 chunk 推進，attention 計算量逐步增加。 |

---

## 比較總結

| 維度 | Mixed Prefill/Decode | Chunked Prefill |
|------|---------------------|-----------------|
| TTFT（首 token 時間） | ✅ 較低（一次完成 prefill） | ❌ 較高（多步完成 prefill） |
| TPOT（token 間延遲） | ❌ 不穩定（受 prefill 長度影響） | ✅ 穩定（受 chunk size 約束） |
| 吞吐量 | ✅ 高（一次填滿 compute） | ✅ 高（穩定 pipeline） |
| 長 prompt 支援 | ❌ 可能阻塞 decode | ✅ 漸進處理 |
| 實作複雜度 | 中等 | 較高 |
| 硬體適配性 | 需要動態形狀支援 | 固定形狀友好（NPU/ASIC） |
| 記憶體可預測性 | 較低 | 較高 |

---

## 適用場景建議

- **追求最低 TTFT**：使用 Mixed Prefill/Decode（不切 chunk，一次完成 prefill）。
- **追求穩定 TPOT / 串流體驗**：使用 Chunked Prefill，設定合理的 chunk size。
- **NPU/ASIC 部署（如 QAIC）**：優先考慮 Chunked Prefill，因為固定 chunk size 可對齊硬體的靜態編譯需求。
- **超長 context（>8K tokens）**：Chunked Prefill 避免單次 prefill 佔滿資源。

---

## 在 vLLM 中的設定

```python
# 啟用 chunked prefill（vLLM v0.4+）
engine_args = EngineArgs(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,  # chunk size 上限
)
```

- `enable_chunked_prefill=True`：啟用 chunked prefill 排程。
- `max_num_batched_tokens`：控制每個 iteration 最大 token 數，間接決定 chunk size。

若不啟用 chunked prefill，vLLM 預設使用 mixed prefill/decode batch。

---

## 參考

- [vLLM Chunked Prefill 設計文件](https://docs.vllm.ai/en/latest/design/chunked_prefill.html)
- Agrawal et al., *Sarathi-Serve: Stall-Free LLM Serving with Chunked Prefills*, 2024
- vLLM scheduler 原始碼：`vllm/core/scheduler.py`
