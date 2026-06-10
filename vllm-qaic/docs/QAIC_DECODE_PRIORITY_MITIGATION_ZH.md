# v1_vllm021: QAIC decode-priority mitigation 說明

本文以 `v1_vllm021` 分支為主，說明從 `roy/qaic-paged-kv` backport 進來的 decode-priority scheduler mitigation。這次只處理高並行 TPOT 被 mixed prefill/decode step 拉高的問題，沒有引入 paged attention / paged KV runtime。

## 先確認：v1_vllm021 沒有 paged attention

`v1_vllm021` 的 QAIC runtime 仍是 non-paged KV 模式：

- `vllm-qaic/vllm_qaic/worker.py` 仍寫著 QAIC 不支援 paged attention。
- `vllm-qaic/vllm_qaic/model_runner.py` 只從 `InputBatch.block_table` 取第 0 個 block id，轉成 contiguous `batch_indices`。
- 沒有 `paged_kv` additional config path。
- 沒有把完整 `block_table` row 傳給 QPC。
- 沒有 `vllm_qaic/kv_connector/` 相關 paged KV / Mooncake staging 檔案。

因此，這份 mitigation 是 scheduler 行為調整，不是 paged attention backport。

## 為什麼需要 mitigation

vLLM V1 scheduler 預設啟用 chunked prefill，會把正在 decode 的 request 和新進來的 prefill request 排在同一個 engine step。

在 GPU backend，mixed prefill/decode batch 通常是吞吐優化。但 QAIC runtime 裡 prefill 和 decode 是不同 QPC path；同一個 step 同時有 decode 和 prefill 時，`QaicModelRunner.execute_model()` 會先跑 decode，再跑 prefill，兩段時間串行相加。

高並行下如果 mixed step 比例很高，decode token 的 TPOT 就容易變成：

```text
TPOT ~= decode_qpc_time + prefill_qpc_time + host overhead
```

這個 mitigation 的目標是讓 decode 壓力高時，新 prefill 先不要進來，讓大多數 decode step 盡量保持純 decode。

## 這次在 v1_vllm021 改了什麼

### 1. 新增 QAIC decode-priority scheduler

新增：

```text
vllm-qaic/vllm_qaic/scheduler.py
```

核心類別：

```python
QaicDecodePriorityScheduler
```

它繼承 vLLM V1 的 stock `Scheduler`，只改 waiting prefill 的 admission 時機。當 running request 中有 decode backlog 時，scheduler 會暫時延後新的 prefill，讓這個 step 優先服務 decode。

### 2. 在 QAIC platform 預設掛載 scheduler

修改：

```text
vllm-qaic/vllm_qaic/platform.py
```

在 `QaicPlatform.check_and_update_config()` 中，如果使用者沒有自訂 `scheduler_cls`，且沒有設定關閉開關，會自動設定：

```python
scheduler_config.scheduler_cls = "vllm_qaic.scheduler.QaicDecodePriorityScheduler"
```

關閉開關：

```text
QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER=1
```

### 3. defer prefill 的方式

這個 scheduler 不清空 `self.waiting`，也不改 preemption。

需要 defer prefill 時，它只會暫時把 `max_num_scheduled_tokens` cap 到 running request 需要的 token 數。stock scheduler 會先排 running request；當 running request 用完 token budget 後，waiting queue admission loop 看到剩餘 budget 為 0，就不會把新 prefill 拉進同一個 step。

排程結束後，`max_num_scheduled_tokens` 會在 `finally` 還原。

### 4. starvation guard

為避免新 request 長期排不到 prefill，有兩個放行條件：

| 變數 | 預設 | 作用 |
|---|---:|---|
| `QAIC_PREFILL_EVERY_N_STEPS` | `8` | 連續 defer prefill 達 N 步後，強制允許一次 prefill step |
| `QAIC_PREFILL_RESUME_FRAC` | `0.5` | decode running request 數量低於 `max_num_running_reqs * frac` 時，允許 prefill |

當 scheduler 判斷可以放行 prefill 時，會直接走原本 `super().schedule()`，不壓低 token budget。這讓 prefill 有機會一次做完或少切幾段，避免被切成很多小 chunk。

## 這次沒有改什麼

這次 backport 沒有引入 paged attention。

這次沒有改 QAIC QPC 編譯圖。

這次沒有改 decode / prefill numerical path。

這次沒有關掉 chunked prefill 本身，而是在 decode 壓力高時少讓新 prefill 混入 decode step。

這次沒有 backport `roy/qaic-paged-kv` 上更細的 profiler split。`v1_vllm021` 目前已有 `VLLM_QAIC_PROFILE=1`，但 log 仍是較粗的 `card=...ms / logits=...ms / sample=...ms / bookkeeping=...ms`，沒有 `mixed=1`、`decode_qpc=...ms`、`prefill_qpc=...ms` 這些欄位。

## 怎麼做 A/B

### 關閉 mitigation

```bash
QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER=1 VLLM_QAIC_PROFILE=1 \
  vllm serve <model> --additional-config '<qaic config>'
```

### 開啟 mitigation

```bash
VLLM_QAIC_PROFILE=1 \
  vllm serve <model> --additional-config '<qaic config>'
```

建議觀察：

1. client 端 TPOT 是否下降。
2. TTFT 是否退化但仍可接受。
3. throughput 是否維持合理。
4. 是否有 request 長時間排不到或 starvation。
5. server log 中 QAIC `card=...ms` 的長尾是否下降。

如果需要更直接看 mixed-step ratio，還要另外 backport `roy/qaic-paged-kv` 上的 profiler split，或在 `v1_vllm021` 的 `model_runner.py` 補上 `mixed / decode_qpc / prefill_qpc` 分段 log。

## 測試覆蓋

新增：

```text
vllm-qaic/tests/test_qaic_scheduler.py
```

測試重點：

- 沒有 decode 時不 defer prefill。
- running 中只有 prefill 時不 defer。
- decode backlog 時 defer 新 prefill。
- 達 `QAIC_PREFILL_EVERY_N_STEPS` 後放行。
- decode 負載低於 `QAIC_PREFILL_RESUME_FRAC` 時放行。
- defer step 會 cap token budget，且 schedule 後還原。
- `super().schedule()` 丟 exception 時也會還原 budget。

## 一句話總結

`v1_vllm021` 這次只加 decode-priority scheduler mitigation：讓 QAIC 在 decode 壓力高時少混入新 prefill，用一部分 TTFT 換更乾淨的 decode step，目標是降低高並行 TPOT。
