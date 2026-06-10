# vLLM v1 Scheduler 設計解析

> 原始碼路徑：`vllm/v1/core/sched/scheduler.py`

本文件深入解析 vLLM v1 排程器的設計理念、核心資料結構、排程演算法及生命週期管理。

---

## 目錄

1. [設計哲學](#設計哲學)
2. [核心資料結構](#核心資料結構)
3. [排程演算法 — `schedule()` 方法](#排程演算法--schedule-方法)
4. [請求生命週期](#請求生命週期)
5. [KV Cache 管理](#kv-cache-管理)
6. [Preemption（搶佔）機制](#preemption搶佔機制)
7. [Speculative Decoding 支援](#speculative-decoding-支援)
8. [KV Connector（P/D 分離 & Offloading）](#kv-connectorpd-分離--offloading)
9. [Encoder 相關排程](#encoder-相關排程)
10. [LoRA 支援](#lora-支援)
11. [輸出處理 — `update_from_output()`](#輸出處理--update_from_output)
12. [關鍵設定參數](#關鍵設定參數)
13. [對 QAIC 的啟示](#對-qaic-的啟示)

---

## 設計哲學

vLLM v1 Scheduler 的核心設計理念來自原始碼中的一段註解：

> There's no "decoding phase" nor "prefill phase" in the scheduler.
> Each request just has `num_computed_tokens` and `num_tokens_with_spec`.
> At each step, the scheduler tries to assign tokens to the requests
> so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.

**關鍵洞察**：排程器不區分 prefill / decode 階段，而是統一看待每個請求：
- `num_computed_tokens`：已經計算過的 token 數量
- `num_tokens_with_spec`：`len(prompt) + len(output) + len(spec_tokens)`

每一步排程的目標是：讓 `num_computed_tokens` 追上 `num_tokens_with_spec`。

這種抽象足夠通用，可以統一支援：
- **Chunked Prefill**：每步只計算一部分 prompt token
- **Prefix Caching**：已 cache 的 token 不需重算
- **Speculative Decoding**：一次驗證多個投機 token
- **Mixed Prefill/Decode Batch**：自然地在同一 batch 混合不同階段的請求

---

## 核心資料結構

### 請求佇列

```
┌─────────────────────────────────────────────────────┐
│ Scheduler                                           │
│                                                     │
│  requests: dict[str, Request]  ← 所有活躍請求的索引  │
│                                                     │
│  waiting: RequestQueue         ← 等待排程的新請求    │
│  skipped_waiting: RequestQueue ← 被暫時跳過的請求    │
│  running: list[Request]        ← 正在執行的請求      │
│                                                     │
│  finished_req_ids: set[str]    ← 本步結束的請求 ID   │
└─────────────────────────────────────────────────────┘
```

### 排程策略（SchedulingPolicy）

| 策略 | 實作 | 說明 |
|------|------|------|
| `FCFS` | `FCFSRequestQueue`（基於 `deque`） | 先到先排程 |
| `PRIORITY` | `PriorityRequestQueue`（基於 `heapq`） | 按 `(priority, arrival_time)` 排序 |

### SchedulerOutput

排程器的輸出物件，傳遞給 ModelRunner：

| 欄位 | 類型 | 說明 |
|------|------|------|
| `scheduled_new_reqs` | `list[NewRequestData]` | 首次排程的請求（含完整 token IDs、block IDs） |
| `scheduled_cached_reqs` | `CachedRequestData` | 已在 running 中的請求（差異更新） |
| `num_scheduled_tokens` | `dict[str, int]` | 每個請求本步排程的 token 數 |
| `total_num_scheduled_tokens` | `int` | 本步總 token 數 |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | 投機 token IDs |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | 需計算的 encoder 輸入 |
| `preempted_req_ids` | `set[str]` | 被搶佔的請求 |
| `finished_req_ids` | `set[str]` | 已結束的請求 |

---

## 排程演算法 — `schedule()` 方法

每次 engine step 呼叫 `schedule()`，分兩大階段：

### 階段 1：排程 RUNNING 請求

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
    num_new_tokens = min(num_new_tokens, token_budget)
    
    # 分配 KV cache block
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
    
    if new_blocks is None:
        # 搶佔低優先級請求以釋放記憶體
        preempt_lowest_priority_request()
    else:
        # 排程成功
        token_budget -= num_new_tokens
```

重點邏輯：
1. **Long prefill threshold**：若 `num_new_tokens` 超過 `long_prefill_token_threshold`，截斷為閾值（chunked prefill 的一種）。
2. **Block 分配失敗 → 搶佔**：如果 KV cache 不夠，搶佔低優先權或最後加入的請求。
3. **`continue` 而非 `break`**：某個 running 請求暫時無法排程（如 encoder budget 耗盡），不會阻塞後面的請求。

### 階段 2：排程 WAITING 請求

只在沒有發生搶佔且系統未暫停時執行：

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break
    
    request = select_next_waiting_request()
    
    # 計算已 cache 的 token（本地 prefix cache + 遠端 KV connector）
    num_computed_tokens = get_computed_blocks(request) + external_tokens
    num_new_tokens = request.num_tokens - num_computed_tokens
    
    # Chunked prefill 控制
    if not enable_chunked_prefill and num_new_tokens > token_budget:
        break  # 不允許切割 → 停止排程新請求
    num_new_tokens = min(num_new_tokens, token_budget)
    
    # 分配 KV cache block
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is None:
        break  # waiting 請求分配失敗 → 停止（不搶佔）
    
    # 移入 running 佇列
    self.running.append(request)
    token_budget -= num_new_tokens
```

重點邏輯：
1. **Prefix Caching**：透過 `kv_cache_manager.get_computed_blocks()` 取得已 cache 的 token 數，避免重複計算。
2. **Chunked Prefill 開關**：若未啟用，不允許將一個 prefill 請求拆成多步。
3. **Waiting 佇列不觸發搶佔**：與 running 階段不同，新請求分配失敗就直接停止。
4. **LoRA 約束**：同時活躍的 LoRA 數量不超過 `max_loras`。

### Token Budget 控制

```
max_num_scheduled_tokens = max_num_batched_tokens（預設值）
```

這是每步最大排程 token 數，直接控制 GPU 每步的計算量。所有 running + 新 waiting 請求共享這個 budget。

---

## 請求生命週期

```
                        add_request()
                            │
                            ▼
                    ┌───────────────┐
                    │   WAITING     │◄──── preempt
                    └───────┬───────┘          ▲
                            │ schedule()       │
                            ▼                  │
                    ┌───────────────┐          │
                    │   RUNNING     │──────────┘
                    └───────┬───────┘   (KV cache 不足)
                            │
                            │ check_stop()
                            ▼
              ┌─────────────────────────────┐
              │  FINISHED_STOPPED /         │
              │  FINISHED_LENGTH_CAPPED /   │
              │  FINISHED_ABORTED /         │
              │  FINISHED_ERROR             │
              └─────────────────────────────┘
```

### 特殊狀態

| 狀態 | 觸發條件 | 所在佇列 |
|------|----------|----------|
| `WAITING_FOR_REMOTE_KVS` | KV Connector 非同步載入中 | `skipped_waiting` |
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | 等待結構化輸出 grammar 初始化 | `skipped_waiting` |
| `WAITING_FOR_STREAMING_REQ` | 串流輸入的請求等待下一段 | `skipped_waiting` |
| `PREEMPTED` | 被搶佔後重入 waiting | `waiting` |

---

## KV Cache 管理

排程器透過 `KVCacheManager` 管理所有 KV cache block：

```
Scheduler ──► KVCacheManager
                 │
                 ├── allocate_slots(request, num_new_tokens)
                 │     → 分配新 block，回傳 KVCacheBlocks
                 │
                 ├── get_computed_blocks(request)
                 │     → 查詢已有 prefix cache，回傳 (blocks, num_cached_tokens)
                 │
                 ├── free(request)
                 │     → 釋放請求佔用的 block
                 │
                 └── get_num_common_prefix_blocks(req_id)
                       → 用於 cascade attention 最佳化
```

### Prefix Caching 流程

1. 新請求進入 waiting 佇列
2. 排程時呼叫 `get_computed_blocks()` → 比對已有 block hash
3. 已 cache 的 token 不佔 token budget
4. 只排程未計算的 token

### 記憶體壓力處理

- Running 階段：搶佔低優先級請求以釋放 block
- Waiting 階段：直接停止排程新請求（不搶佔）

---

## Preemption（搶佔）機制

當 running 請求需要更多 KV cache block 但記憶體不足時：

```python
def _preempt_request(self, request, timestamp):
    kv_cache_manager.free(request)         # 釋放所有 block
    encoder_cache_manager.free(request)    # 釋放 encoder cache
    request.status = PREEMPTED
    request.num_computed_tokens = 0        # 重置：需要重新 prefill
    request.num_preemptions += 1
    self.waiting.prepend_request(request)  # 放回 waiting 佇列前端
```

**搶佔策略**：
- PRIORITY 策略：搶佔 `max(priority, arrival_time)` 最大的（最低優先權/最晚到達）
- FCFS 策略：搶佔 `self.running.pop()`（最後加入的）

**代價**：被搶佔的請求 `num_computed_tokens` 歸零，需要重新 prefill。

---

## Speculative Decoding 支援

排程器原生支援投機解碼：

1. **排程階段**：`num_tokens_with_spec = num_tokens + len(spec_token_ids)`，投機 token 佔用 token budget。
2. **Lookahead tokens**：分配 block 時多預留 `num_lookahead_tokens` 個 block。
3. **驗證後調整**：`update_from_output()` 中根據被拒絕的 token 數回退 `num_computed_tokens`。

```python
# 投機解碼的拒絕處理
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
```

### Eagle 模式

Eagle（自回歸投機解碼）有特殊處理：
- 設定 `num_lookahead_tokens = num_spec_tokens`
- KV cache hash 計算需考慮 Eagle 的 block 修剪行為

---

## KV Connector（P/D 分離 & Offloading）

排程器支援透過 `KVConnectorBase_V1` 進行遠端 KV cache 存取：

### 流程

1. **新請求到達** → `connector.get_num_new_matched_tokens()` 查詢遠端已有的 token 數
2. **非同步載入** → 若 `load_kv_async=True`，請求進入 `WAITING_FOR_REMOTE_KVS` 狀態
3. **載入完成** → 回到 waiting 佇列正常排程
4. **請求結束** → `connector.build_connector_meta()` 可選擇將 KV push 到遠端

### 失敗處理

```python
self.recompute_kv_load_failures = True  # 載入失敗 → 本地重算
# 或
self.recompute_kv_load_failures = False # 載入失敗 → 直接報錯
```

---

## Encoder 相關排程

對於多模態模型（Vision-Language 等）和 Encoder-Decoder 模型：

### 資源約束

| 資源 | 說明 |
|------|------|
| `encoder_compute_budget` | 每步最多處理的 encoder token 數 |
| `EncoderCacheManager` | 管理 encoder 輸出的 cache |

### 排程邏輯

`_try_schedule_encoder_inputs()` 決定本步需要計算哪些 encoder 輸入：
- 只排程那些 decoder 即將 attend 到的 encoder 輸入
- encoder cache 已有的不重複計算
- 計算完成後，decoder 處理過的 encoder 輸出會被釋放

---

## LoRA 支援

排程器追蹤當前活躍的 LoRA adapter：

```python
if len(scheduled_loras) == max_loras and request.lora_int_id not in scheduled_loras:
    # 跳過此請求，放入 step_skipped_waiting
    continue
```

約束：同一步中活躍的 LoRA 數量不超過 `max_loras`。

---

## 輸出處理 — `update_from_output()`

`schedule()` 之後，ModelRunner 執行 forward pass，然後排程器透過 `update_from_output()` 處理結果：

```
ModelRunnerOutput
    │
    ├── sampled_token_ids[req_index]   → 取得新 token
    ├── logprobs[req_index]            → 取得 logprobs
    └── prompt_logprobs_dict[req_id]   → 取得 prompt logprobs
```

### 處理流程

1. **處理 KV 載入失敗** → `_handle_invalid_blocks()`
2. **逐請求處理**：
   - 取得 `generated_token_ids`
   - 處理投機解碼拒絕（調整 `num_computed_tokens`）
   - 釋放已處理的 encoder 輸入
   - 追加 output token → 檢查停止條件
   - 結構化輸出 grammar 驗證
   - 構建 `EngineCoreOutput` 回傳給前端
3. **移除已停止的請求**
4. **發佈 KV cache 事件**
5. **聚合統計資訊**

### 停止條件（`check_stop()`）

- 達到 `max_tokens` 上限
- 達到 `max_model_len`
- 命中 stop token / stop string
- 外部 abort

---

## 關鍵設定參數

| 參數 | 來源 | 說明 |
|------|------|------|
| `max_num_seqs` | `SchedulerConfig` | 同時 running 的最大請求數 |
| `max_num_batched_tokens` | `SchedulerConfig` | 每步最大 token 數（token budget） |
| `max_num_scheduled_tokens` | `SchedulerConfig` | 覆寫 token budget（若設定） |
| `enable_chunked_prefill` | `SchedulerConfig` | 是否允許 prefill 切割 |
| `long_prefill_token_threshold` | `SchedulerConfig` | 超過此閾值的 prefill 強制截斷 |
| `policy` | `SchedulerConfig` | `"fcfs"` 或 `"priority"` |
| `enable_prefix_caching` | `CacheConfig` | 是否啟用 prefix cache |
| `block_size` | 初始化參數 | KV cache block 大小 |
| `scheduler_reserve_full_isl` | `SchedulerConfig` | 是否為 waiting 請求預留完整序列長度的 block |

---

## 對 QAIC 的啟示

理解 v1 Scheduler 的設計，對 QAIC 後端整合有以下重要意義：

### 1. Token Budget 即 Batch 形狀

排程器透過 `max_num_scheduled_tokens` 控制每步總 token 數。QAIC 可以將此設為 NPU 支援的固定 batch token 數，使每步計算量可預測。

### 2. Chunked Prefill 天然適配

排程器已內建 chunked prefill 邏輯：
- `enable_chunked_prefill=True` → 允許切割
- `long_prefill_token_threshold` → 每 chunk 最大 token 數

QAIC 可設定 threshold 為 NPU 的最大輸入序列長度。

### 3. 排程輸出是 Block-Based

`SchedulerOutput` 提供 `block_ids`，而非連續的 KV tensor。QAIC 的 paged attention 實作需要能根據 block_ids 索引 KV cache。

### 4. Preemption 需要考慮

被搶佔的請求 `num_computed_tokens` 歸零。若 QAIC 的 KV cache 在 device memory 上，需要確保 `free()` 能正確釋放 NPU 記憶體。

### 5. `num_scheduled_tokens` 是 ModelRunner 的合約

ModelRunner 必須為每個 `req_id` 準確處理 `num_scheduled_tokens[req_id]` 個 token。這是排程器與 worker 之間的核心介面。

---

## 流程圖總覽

```
Engine Step
    │
    ▼
scheduler.schedule()
    │
    ├── 1. 排程 RUNNING 請求
    │       ├── 計算 num_new_tokens
    │       ├── allocate_slots()
    │       └── 必要時搶佔
    │
    ├── 2. 排程 WAITING 請求
    │       ├── get_computed_blocks()（prefix cache）
    │       ├── connector.get_num_new_matched_tokens()（遠端 KV）
    │       ├── chunked prefill 截斷
    │       └── allocate_slots()
    │
    ├── 3. 構建 SchedulerOutput
    │       └── _update_after_schedule()（推進 num_computed_tokens）
    │
    ▼
ModelRunner.execute_model(scheduler_output)
    │
    ▼
scheduler.update_from_output(scheduler_output, model_runner_output)
    │
    ├── 處理投機解碼拒絕
    ├── 追加 output tokens
    ├── 檢查停止條件
    ├── 釋放已結束請求的資源
    └── 回傳 EngineCoreOutputs 給前端
```

---

## 參考

- 原始碼：`vllm/v1/core/sched/scheduler.py`
- 排程輸出定義：`vllm/v1/core/sched/output.py`
- 請求佇列：`vllm/v1/core/sched/request_queue.py`
- KV Cache 管理：`vllm/v1/core/kv_cache_manager.py`
- Request 定義：`vllm/v1/request.py`
