# QAIC Decode-Priority Scheduler 設計解析

> 原始碼路徑：`vllm-qaic/vllm_qaic/scheduler.py`

本文件解析 `QaicDecodePriorityScheduler` 的設計動機、運作原理及調優方式。

---

## 目錄

1. [為什麼需要這個排程器](#為什麼需要這個排程器)
2. [核心思路](#核心思路)
3. [類別繼承關係](#類別繼承關係)
4. [決策邏輯 — `_qaic_should_defer_prefill()`](#決策邏輯--_qaic_should_defer_prefill)
5. [排程覆寫 — `schedule()`](#排程覆寫--schedule)
6. [Token Budget 截斷技巧](#token-budget-截斷技巧)
7. [調優參數](#調優參數)
8. [殘留混合問題](#殘留混合問題)
9. [與上游 v1 Scheduler 的關係](#與上游-v1-scheduler-的關係)
10. [流程圖](#流程圖)

---

## 為什麼需要這個排程器

### GPU 上的 Mixed Batch

vLLM v1 排程器預設使用 chunked prefill：在同一個 engine step 中混合 prefill token 和 decode token，一次 forward pass 全部處理。GPU 的動態形狀能力讓這很高效。

### QAIC 上的問題

QAIC 使用靜態編譯的 QPC（Qualcomm Program Container），prefill 和 decode 是**獨立的固定形狀計算圖**：

```
GPU：一次 forward = prefill + decode（混合 kernel）
QAIC：一步 mixed batch = decode QPC 執行 + prefill QPC 執行（串行！）
```

因此，每當一個 step 包含 prefill token 時：

```
TPOT = decode_time + prefill_time
```

在高併發場景下，幾乎每一步都會帶 prefill，導致 **TPOT 暴增 ~5 倍**。

### 解法：Decode-Priority 排程

**核心原則**：當有 decode 積壓時，不允許新的 prefill 進入，保持 decode step 純淨（只跑 decode QPC），讓 TPOT 保持穩定。定期或在負載低時，允許 prefill「爆發」。

---

## 核心思路

| 情境 | 行為 | 效果 |
|------|------|------|
| 有 decode 積壓，未到 cadence | **延遲 prefill** — 截斷 token budget | TPOT 穩定 |
| 達到 cadence（每 N 步一次） | **允許 prefill** — 使用完整 budget | 避免 TTFT 餓死 |
| Decode 數量低於閾值 | **允許 prefill** — decode 影響小 | TPOT 影響可控 |
| 無 decode 在 running 中 | **允許 prefill** — 沒什麼要保護的 | 正常流程 |

**關鍵設計**：不修改 waiting 佇列、不干涉搶佔邏輯，僅透過**暫時縮小 token budget** 來阻止新 prefill 被排程。

---

## 類別繼承關係

```
vllm.v1.core.sched.scheduler.Scheduler     ← 上游 v1 排程器
        │
        └── QaicDecodePriorityScheduler    ← QAIC 覆寫版本
                │
                ├── __init__()              新增計數器和參數
                ├── _qaic_num_decode_running()  計算 decode 中的請求數
                ├── _qaic_should_defer_prefill()  決策是否延遲 prefill
                └── schedule()              覆寫：截斷 budget 後呼叫 super()
```

這是一個**最小侵入設計**——只覆寫 `schedule()` 方法，所有其他功能（搶佔、KV cache 管理、output 處理等）都繼承自上游。

---

## 決策邏輯 — `_qaic_should_defer_prefill()`

```python
def _qaic_should_defer_prefill(self) -> bool:
    num_decode = self._qaic_num_decode_running()
    
    # 條件 1：沒有 decode → 不延遲
    if num_decode == 0:
        return False
    
    # 條件 2：已延遲 N 步 → 強制允許 prefill（防餓死）
    if self._qaic_steps_since_prefill >= self._qaic_prefill_every_n:
        return False
    
    # 條件 3：decode 數量少 → headroom 足夠，prefill 影響小
    if num_decode < self._qaic_resume_frac * self.max_num_running_reqs:
        return False
    
    # 否則：有 decode 積壓 → 延遲 prefill
    return True
```

### 如何判斷「decode 中」

```python
def _qaic_num_decode_running(self) -> int:
    return sum(
        1 for r in self.running
        if r.num_computed_tokens >= r.num_prompt_tokens)
```

一個請求的 `num_computed_tokens >= num_prompt_tokens` 表示它已完成 prefill，正在 decode 階段。

---

## 排程覆寫 — `schedule()`

```python
def schedule(self):
    # 不需要延遲 → 正常排程
    if not self._qaic_should_defer_prefill():
        self._qaic_steps_since_prefill = 0
        return super().schedule()

    # 計算 running 請求所需的最小 token 數
    running_budget = sum(
        max(0, r.num_tokens_with_spec + r.num_output_placeholders
            - r.num_computed_tokens)
        for r in self.running)
    
    if running_budget <= 0:
        self._qaic_steps_since_prefill = 0
        return super().schedule()

    # 暫時截斷 token budget
    saved = self.max_num_scheduled_tokens
    self.max_num_scheduled_tokens = min(saved, running_budget)
    try:
        output = super().schedule()
    finally:
        self.max_num_scheduled_tokens = saved  # 恢復原值
    
    self._qaic_steps_since_prefill += 1
    return output
```

---

## Token Budget 截斷技巧

這是整個設計最巧妙的部分。回顧上游 `schedule()` 的 waiting 排程邏輯：

```python
# 上游 scheduler.py 中的 waiting 排程循環
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    ...
    num_new_tokens = min(num_new_tokens, token_budget)
    ...
```

QAIC 排程器的做法：

1. 計算 running 請求這一步總共需要多少 token（decode 通常每個請求只需 1 token）
2. 將 `max_num_scheduled_tokens` 設為這個值
3. 呼叫 `super().schedule()`
4. 上游排程器先排程完所有 running 請求後，`token_budget` 恰好為 0
5. Waiting 排程循環因為 `token_budget <= 0` 而不執行 → **新 prefill 被阻擋**

**安全性**：
- 不修改任何佇列狀態
- 不影響搶佔邏輯（搶佔發生在 running 排程階段）
- `finally` 確保 budget 一定會恢復

---

## 調優參數

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `QAIC_PREFILL_EVERY_N_STEPS` | `8` | 最多連續延遲 N 步後強制允許 prefill（TTFT 上限 = N × TPOT） |
| `QAIC_PREFILL_RESUME_FRAC` | `0.5` | 當 decode 數 < `frac × max_num_seqs` 時停止延遲 |
| `QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER` | `0`（未設定） | 設為 `1` 完全停用此排程器 |

### 調優建議

| 場景 | 調整方向 |
|------|----------|
| TTFT 太高（使用者等首 token 太久） | 降低 `PREFILL_EVERY_N_STEPS`（如 4） |
| TPOT 不穩定（生成卡頓） | 提高 `PREFILL_EVERY_N_STEPS`（如 16） |
| 低併發時 prefill 不必要被延遲 | 提高 `RESUME_FRAC`（如 0.7） |
| 高併發且 TPOT 敏感 | 降低 `RESUME_FRAC`（如 0.3） |

### TTFT 與 TPOT 的 Trade-off

```
TTFT_worst_case ≈ PREFILL_EVERY_N_STEPS × single_decode_step_time + prefill_time
TPOT_pure_decode ≈ single_decode_step_time（不含 prefill 干擾）
```

---

## 殘留混合問題

### 問題描述

如果一個長 prompt 在 burst step 沒有完成 prefill（prompt 太長，一步裝不下），它會留在 `running` 佇列中。下一步即使是 defer 模式，這個請求的剩餘 chunk 仍然會被排程（因為它已經在 running 中），導致 prefill + decode 混合。

### 緩解方法

```
max_num_batched_tokens ≥ 最大 prompt 長度
```

將 `max_num_batched_tokens` 設得足夠大，確保任何 prompt 在一個 burst step 內完成 prefill。

如果 prompt 非常長無法一步完成，則在殘留 chunk 完成前，TPOT 會暫時升高。

---

## 與上游 v1 Scheduler 的關係

| 面向 | 上游 v1 Scheduler | QAIC 覆寫 |
|------|-------------------|-----------|
| Prefill/Decode 混合 | 鼓勵（提高 GPU 利用率） | 避免（保護 TPOT） |
| Chunked Prefill | 每步切一小 chunk | burst step 盡量一次完成 |
| Token Budget | 固定 `max_num_batched_tokens` | 動態截斷以排除 prefill |
| 搶佔 | ✅ | ✅ 完全繼承 |
| Prefix Caching | ✅ | ✅ 完全繼承 |
| Spec Decoding | ✅ | ✅ 完全繼承 |
| KV Connector | ✅ | ✅ 完全繼承 |

---

## 流程圖

```
schedule() 被呼叫
    │
    ▼
_qaic_should_defer_prefill()?
    │
    ├── False（允許 prefill）
    │       │
    │       ▼
    │   _qaic_steps_since_prefill = 0
    │   super().schedule()  ← 完整 budget，可排新 prefill
    │
    └── True（延遲 prefill）
            │
            ▼
        計算 running_budget = Σ(running 請求需要的 token)
            │
            ├── running_budget <= 0
            │       → super().schedule()（正常）
            │
            └── running_budget > 0
                    │
                    ▼
                暫時設定 max_num_scheduled_tokens = running_budget
                    │
                    ▼
                super().schedule()
                    │
                    ├── Running 排程：所有 decode 正常排程
                    └── Waiting 排程：token_budget = 0 → 不排新 prefill
                    │
                    ▼
                恢復 max_num_scheduled_tokens
                _qaic_steps_since_prefill += 1
```

---

## 時序範例

假設 `PREFILL_EVERY_N_STEPS=4`，3 個 decode 在跑，1 個 prefill 在等待：

```
Step 1: [DEFER] 3 decode → TPOT ≈ decode_time ✓
Step 2: [DEFER] 3 decode → TPOT ≈ decode_time ✓
Step 3: [DEFER] 3 decode → TPOT ≈ decode_time ✓
Step 4: [BURST] 3 decode + 1 prefill → TPOT ≈ decode_time + prefill_time ✗
Step 5: [DEFER] 4 decode → TPOT ≈ decode_time ✓ (新請求已加入 decode)
...
```

**效果**：4 步中只有 1 步 TPOT 受影響，相比不做 defer（每步都 TPOT 爆炸）改善顯著。

---

## 安裝方式

此排程器由 `platform.py` 透過 `scheduler_config.scheduler_cls` 自動安裝：

```python
# platform.py 中
scheduler_config.scheduler_cls = "vllm_qaic.scheduler.QaicDecodePriorityScheduler"
```

停用：
```bash
export QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER=1
```

---

## 參考

- 上游排程器設計：[`v1_scheduler_design_zh.md`](./v1_scheduler_design_zh.md)
- Mixed Prefill/Decode 與 Chunked Prefill 比較：[`mixed_prefill_decode_and_chunked_prefill.md`](./mixed_prefill_decode_and_chunked_prefill.md)
- QAIC Decode Priority 緩解策略（早期文件）：[`QAIC_DECODE_PRIORITY_MITIGATION_ZH.md`](./QAIC_DECODE_PRIORITY_MITIGATION_ZH.md)
