# QAIC V1 Engine 支援 ngram Speculative Decoding

這份文件說明如何讓 QAIC backend 在 vLLM V1 engine 中支援 `ngram`
speculative decoding。本文聚焦在 `ngram`，不包含 draft model、Eagle、
Medusa、Turbo 等其他 speculative decoding 形式。

## 背景

原本 QAIC 在 V1 engine 中會直接拒絕 speculative decoding：

```text
ValueError: Speculative decoding is not yet suppoerted on qaic backend when using vllm v1.
```

但 QAIC V1 runner 其實已經有部分 ngram speculative decoding 需要的基礎：

- `QaicModelRunner` 繼承 `GPUModelRunner`
- `GPUModelRunner` 已經會在 V1 中初始化 `NgramProposer`
- `QaicModelRunner` 已經有 `propose_draft_token_ids()`
- `QaicModelRunner` 已經有 `SpecDecodeMetadata` 和 rejection sampler 相關流程

缺少的是 QAIC target QPC 路徑、V1 scheduler decode token layout，以及
QAIC CPU 執行環境下的 rejection sampler fallback。

## 啟用方式

啟動 QAIC V1 + ngram speculative decoding 時，需要：

```bash
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
```

`VLLM_WORKER_MULTIPROC_METHOD=fork` 是為了避開 `spawn` 子程序中
`cv2/typing` 汙染 `typing` import path 的問題。沒有這個設定時，可能會看到：

```text
ImportError: cannot import name 'NamedTuple' from partially initialized module 'typing'
```

範例 command：

```bash
source /opt/vllm-env/bin/activate
cd /workspace/weiminc/Unie/vllm

export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=fork

python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --host 127.0.0.1 \
  --port 8000 \
  --device-group 0 \
  --max-model-len 128 \
  --max-seq-len-to-capture 32 \
  --max-num-seqs 32 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --override-qaic-config "num_cores=16 mxfp6=True mxint8_kv_cache=True prefill_seq_len=32 aic_enable_depth_first=True" \
  --speculative-config '{"model":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":1}'
```

成功啟動時應該看到類似：

```text
Initializing a V1 LLM engine
Transforming and compiling model[target] using QEfficient library
Spec model type target_4
Application startup complete.
```

其中 `target_4` 代表 `num_speculative_tokens=3`，target model 需要輸出
`3 + 1` 個 logits，用來做 draft tokens verification 和 bonus token sampling。

## ngram proposer 演算法

ngram speculative decoding 的 draft token 不是來自另一個 draft model，而是來自
目前 request 自己的歷史 token。這種方法也常被稱為 prompt lookup decoding。

實作位置：

```text
vllm/v1/spec_decode/ngram_proposer.py
```

`NgramProposer` 使用三個主要參數：

- `prompt_lookup_min`：最短要匹配的 ngram 長度
- `prompt_lookup_max`：最長要匹配的 ngram 長度
- `num_speculative_tokens`：最多提出多少個 draft tokens

以這個設定為例：

```json
{
  "model": "ngram",
  "num_speculative_tokens": 3,
  "prompt_lookup_max": 3,
  "prompt_lookup_min": 1
}
```

流程如下：

1. 取得目前 request 的 context tokens，也就是 prompt tokens 加上已生成 tokens。
2. 從 context 尾端取 suffix ngram，長度範圍是
   `[prompt_lookup_min, prompt_lookup_max]`。
3. 在更早的 context token 中尋找相同 ngram。
4. 優先找最長匹配。
5. 找到匹配後，取該匹配片段後面的 tokens 作為 draft tokens。
6. draft tokens 最多取 `num_speculative_tokens` 個。
7. 如果 context 太短、找不到匹配、或已達 `max_model_len`，則不提出 draft tokens。

例子：

```text
context = [1, 2, 3, 4, 2, 3]
prompt_lookup_min = 2
prompt_lookup_max = 3
num_speculative_tokens = 4
```

演算法會先嘗試匹配尾端長度 3 的 ngram：

```text
[4, 2, 3]
```

如果前文找不到，就嘗試長度 2：

```text
[2, 3]
```

前文中有：

```text
[1, 2, 3, 4, ...]
    ^^^^^
```

所以匹配 `[2, 3]` 後，取它後面的 tokens：

```text
[4, 2, 3]
```

這些 tokens 就會成為 draft tokens。

實作細節上，`_find_longest_matched_ngram_and_propose_tokens()` 會把 token
序列反轉，然後用類似 KMP/LPS 的方式找「尾端 suffix」在歷史 token 中的最長匹配。
它不是暴力枚舉每個 ngram window，因此能避免簡單寫法的重複掃描。

```python
draft_token_ids = NgramProposer(vllm_config).propose(context_token_ids)
```

ngram proposer 只負責「猜」draft tokens。這些 draft tokens 還不能直接回傳給使用者，
必須再交給 target model 驗證。

在 QAIC V1 的流程中：

```text
context tokens
  -> NgramProposer 產生 draft tokens
  -> QAIC target QPC 同時計算 target/draft/bonus logits
  -> rejection sampler 決定接受幾個 draft tokens
  -> 回傳 accepted tokens + recovered token 或 bonus token
```

這也是為什麼 QAIC V1 ngram 需要 target QPC 輸出
`num_speculative_tokens + 1` 個 logits。

## 主要改動

### 1. 只允許 ngram speculative decoding

檔案：

```text
vllm/v1/worker/qaic_model_runner.py
```

原本 QAIC V1 runner 只要看到 `self.speculative_config` 就直接報錯。
改成：

- `speculative_config is None`：維持原本非 speculative path
- `speculative_config.method == "ngram"`：允許
- 其他 method：明確報錯

概念上是：

```python
if self.speculative_config is not None:
    if self.speculative_config.method != "ngram":
        raise ValueError(...)
```

注意要用 `is not None`，不要寫 `if self.speculative_config:`。這個物件在目前路徑
truthiness 不可靠，可能導致 speculative config 存在卻沒有進入分支。

### 2. ngram 時改用 target QPC

QAIC model loader 需要知道目前是 target speculative model，才會編出支援
`num_speculative_tokens + 1` logits 的 QPC。

在 `QaicModelRunner.__init__()` 中，當 method 是 `ngram` 時，把
`speculative_model_type` 從 `None` 或 `"default"` 改成 `"target"`：

```python
if speculative_model_type in (None, "default"):
    speculative_model_type = "target"
```

這會讓 loader 走：

```text
model[target]
num_speculative_tokens = N
num_logits_to_keep = N + 1
```

### 3. 打包 V1 scheduler 的 decode tokens

在 V1 speculative decoding 中，一個 decode request 可能不是只有 1 個 token，
而是：

```text
1 target token + N draft tokens
```

QAIC target QPC 則需要固定形狀：

```text
[num_decodes, num_speculative_tokens + 1]
```

因此在 `QaicModelRunner.execute_model()` 裡，spec decode decode phase 要把
flat input ids/positions 改成 2D tensor：

```text
decode_input_ids:  [num_decodes, max_decode_tokens]
decode_positions:  [num_decodes, max_decode_tokens]
decode_lengths:    每個 request 實際 token 數
```

`decode_lengths` 由 `cu_num_tokens` 算出，用來記錄每個 decode request 實際有
幾個 token。這很重要，因為不是每個 request 都一定會有完整的
`num_speculative_tokens + 1` 個 token。

同時 prefill 的 slice offset 不能再用 `self.num_decodes`，因為 decode token 數
已經不是 request 數，而是所有 decode request 的實際 token 總和。需要改成：

```python
decode_token_count = sum(decode_lengths)
prefill_input_ids = input_ids[decode_token_count:total_num_scheduled_tokens]
```

### 4. QAIC model wrapper 接受 decode_lengths

檔案：

```text
vllm/model_executor/model_loader/qaic_v1.py
```

`QaicCausalLM.forward()` 新增 `decode_lengths` 參數，並傳給 `_run_decode()`。

target QPC 載入時，如果 `num_logits_to_keep is not None`，decode input buffer
需要改成寬度 `num_logits_to_keep`：

```python
self.decode_batch_inputs["input_ids"] = np.full(
    (self.decode_bsz, self.num_logits_to_keep), -1, dtype=np.int64)
self.decode_batch_inputs["position_ids"] = np.full(
    (self.decode_bsz, self.num_logits_to_keep), -1, dtype=np.int64)
```

### 5. 攤平 target QPC logits

QAIC target QPC 回傳：

```text
[decode_batch_size, num_logits_to_keep, vocab_size]
```

V1 rejection sampler 期待的是 flat token logits，順序要對應
`SpecDecodeMetadata.target_logits_indices` 和 `bonus_logits_indices`。

所以 `_run_decode()` 在收到 `decode_lengths` 時，要把每個 request 的有效 logits
攤平成：

```text
[sum(decode_lengths), vocab_size]
```

概念上：

```python
return np.concatenate(
    [logits[i, :num_tokens] for i, num_tokens in enumerate(decode_lengths)],
    axis=0)
```

### 6. 處理 warmup / dummy run

target QPC 的 decode logits 維度是 `[batch, N + 1, vocab]`，但 warmup 或非
spec decode path 仍可能只需要第一個 logits。

如果沒有 `decode_lengths`，但目前是 target QPC，可以回傳第 0 個 logits：

```python
if self.is_spec_decode_target_model:
    return logits[:num_decodes, 0]
```

否則舊的：

```python
return logits[:num_decodes].squeeze(1)
```

會因為第 1 維大小不是 1 而報錯：

```text
ValueError: cannot select an axis to squeeze out which has size not equal to one
```

### 7. QAIC 專用 CPU rejection sampler fallback

檔案：

```text
vllm/v1/worker/qaic_model_runner.py
```

V1 預設的 `RejectionSampler` 會呼叫 Triton kernel：

```python
rejection_greedy_sample_kernel[(batch_size, )](...)
rejection_random_sample_kernel[(batch_size, )](...)
```

QAIC 這條路徑是在 CPU tensor 上執行，而且環境可能沒有 active Triton driver。
這時 Triton kernel 可能退化成普通 function，呼叫時會報：

```text
TypeError: 'function' object is not subscriptable
```

因此 QAIC V1 ngram 不再呼叫 V1 預設 Triton rejection sampler，而是在
`QaicModelRunner` 裡新增 `_qaic_rejection_sample()`。這個 fallback 只處理
ngram case，也就是 `draft_probs is None`。

主要行為：

- 建立 `[batch_size, max_spec_len + 1]` 的 `output_token_ids`
- 使用同一個 `PLACEHOLDER_TOKEN_ID`，讓既有 `parse_output()` 可以重用
- greedy request 使用 target argmax 驗證 draft tokens
- random request 根據 target probabilities 接受或拒絕 draft token
- random request 支援 temperature、top-k、top-p
- 如果所有 draft tokens 都接受，append bonus token
- 如果遇到第一個 rejected token，輸出 recovered token，後續位置保持 placeholder

greedy ngram 的概念：

```python
target_argmax = target_logits.argmax(dim=-1)
for pos in range(num_draft_tokens):
    output_token_ids[req_idx, pos] = target_argmax[token_idx]
    if draft_token_id != target_argmax[token_idx]:
        rejected = True
        break
if not rejected:
    output_token_ids[req_idx, num_draft_tokens] = bonus_token_id
```

random ngram 的概念：

```python
target_probs = softmax(filtered_target_logits)
if target_probs[draft_token_id] >= uniform_random:
    output_token_ids[req_idx, pos] = draft_token_id
else:
    recovered_probs = target_probs.clone()
    recovered_probs[draft_token_id] = 0
    output_token_ids[req_idx, pos] = sample(recovered_probs)
    break
```

這個 fallback 是 QAIC-specific，避免影響 GPU backend 的 Triton fast path。

## 驗證

語法檢查：

```bash
python -m py_compile \
  vllm/v1/worker/qaic_model_runner.py \
  vllm/model_executor/model_loader/qaic_v1.py
```

啟動 server 後送一個 completion request：

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Hello","max_tokens":4,"temperature":0}'
```

也要測一個 random/default sampling request，確認 CPU rejection sampler fallback
沒有走到 Triton kernel：

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Hello","max_tokens":8}'
```

成功時會回傳正常 completion，例如：

```json
{
  "choices": [
    {
      "text": " there! I'm",
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 4,
    "total_tokens": 5
  }
}
```

## 限制

目前這個 patch 只支援：

- QAIC backend
- V1 engine
- `speculative_config.method == "ngram"`
- 單一 target model QPC scoring path
- QAIC CPU fallback rejection sampling

目前不支援：

- draft model speculative decoding
- Eagle / Eagle3
- Medusa
- Turbo
- multimodal
- LoRA + speculative decoding 的完整驗證
- disaggregated serving + speculative decoding 的完整驗證

## Debug checklist

如果看到 fallback 到 V0：

```text
qaic is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.
```

確認有明確設定：

```bash
export VLLM_USE_V1=1
```

如果看到 `cv2/typing` / `NamedTuple` circular import：

```text
ImportError: cannot import name 'NamedTuple' from partially initialized module 'typing'
```

確認有設定：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=fork
```

如果看到仍然是 `model[default]`：

```text
Transforming and compiling model[default]
```

代表 `speculative_model_type` 沒有被切到 `"target"`。ngram speculative path
應該看到：

```text
Transforming and compiling model[target]
Spec model type target_4
```

如果看到 logits squeeze error：

```text
ValueError: cannot select an axis to squeeze out which has size not equal to one
```

代表 target QPC 的 `[batch, N + 1, vocab]` logits 沒有被正確處理。需要確認
`_run_decode()` 中有 `self.is_spec_decode_target_model` 的 fallback。

如果看到 Triton kernel launch error：

```text
TypeError: 'function' object is not subscriptable
```

常見位置會在：

```text
vllm/v1/sample/rejection_sampler.py
rejection_greedy_sample_kernel[(batch_size, )](...)
```

代表 QAIC V1 ngram 仍然走到了 V1 預設 Triton rejection sampler。需要確認
`QaicModelRunner.execute_model()` 裡 spec decode path 呼叫的是：

```python
output_token_ids = self._qaic_rejection_sample(...)
```

而不是：

```python
output_token_ids = self.rejection_sampler(...)
```
