# QAIC 上的 vLLM V1 Engine Review

這份文件 review 下面這條 command，trace 它是否正確開啟 QAIC 上的
vLLM V1 engine，以及開啟 V1 之後是否支援 ngram speculative decoding。

```bash
VLLM_USE_V1=1 TORCH_COMPILE_DISABLE=1 vllm serve "Qwen/Qwen2.5-7B-Instruct" \
  --host 127.0.0.1 \
  --port 8000 \
  --device-group 0 \
  --max-model-len 2048 \
  --max-seq-len-to-capture 32 \
  --max-num-seqs 256 \
  --quantization mxfp6 \
  --kv-cache-dtype fp8 \
  --override-qaic-config "num_cores=16 mxfp6=True mxint8_kv_cache=True prefill_seq_len=32 aic_enable_depth_first=True" \
  --disable-frontend-multiprocessing
```

## 簡短結論

是，這條 command 會明確開啟 vLLM V1 engine。

在 QAIC 上，V1 是有支援的，但目前不是 default，所以
`VLLM_USE_V1=1` 是關鍵開關。設定這個 env var 之後，vLLM 預期會選到：

- V1 engine core
- V1 QAIC worker：`vllm.v1.worker.qaic_worker.QaicWorker`
- QAIC V1 model runner：`vllm.v1.worker.qaic_model_runner.QaicModelRunner`
- QAIC compile config 中的 `mxfp6_matmul=True`
- QAIC compile config 中的 `mxint8_kv_cache=True`，這是由 `--kv-cache-dtype fp8`
  映射而來

QAIC V1 支援 ngram speculative decoding。更精確地說，QAIC V1 目前只允許
ngram speculative decoding；如果使用其他 speculative method，會在 V1 QAIC
model runner 裡被拒絕。

## Runtime Log 確認訊號

實際跑 server 時，可以在 log 裡找這行：

```text
Initializing a V1 LLM engine
```

這行是 V1 `EngineCore` 啟動時印出的：

- `vllm/v1/engine/core.py:74`

如果 `./0601/logs/server_kv_fp8.log` 裡有這行，就代表 process 正在使用
V1 engine。

## V1 選擇流程

### 1. 使用者強制指定 V1

`VLLM_USE_V1=1` 會在建立 engine config 時被讀取。

相關程式碼：

- `vllm/engine/arg_utils.py:1114`
- `vllm/engine/arg_utils.py:1120`
- `vllm/engine/arg_utils.py:1124`

重要行為：

- 如果沒有設定 `VLLM_USE_V1`，vLLM 可能會自動選 V0 或 V1。
- 如果設定 `VLLM_USE_V1=1`，vLLM 會嘗試使用 V1，並 assert 最終選擇結果
  必須和使用者要求一致。
- 如果有 feature 不支援 V1，startup 應該會失敗，而不是 silently fallback 到 V0。

### 2. QAIC Platform 宣告支援 V1

QAIC 宣告支援 V1：

- `vllm/platforms/qaic.py:130`

```python
def supports_v1(cls, model_config):
    return True
```

但 QAIC 不會 default 開啟 V1：

- `vllm/platforms/qaic.py:134`

```python
def default_v1(cls, model_config):
    return False
```

所以現階段如果要在 QAIC 上使用 V1，需要明確設定 `VLLM_USE_V1=1`。

### 3. QAIC Worker Class 切到 V1 Worker

V1 被選中之後，QAIC 會更新 worker class：

- `vllm/platforms/qaic.py:48`

當 `envs.VLLM_USE_V1` 為 true 時：

```python
parallel_config.worker_cls = "vllm.v1.worker.qaic_worker.QaicWorker"
```

這是從一般 QAIC worker code 進入 V1 QAIC worker code 的主要轉接點。

### 4. V1 QAIC Worker 建立 V1 QAIC Model Runner

V1 QAIC worker 會建立：

- `vllm/v1/worker/qaic_model_runner.py`

相關程式碼：

- `vllm/v1/worker/qaic_worker.py:126`

```python
self.model_runner = QaicModelRunner(
    self.vllm_config, self.device, self.speculative_model_type
)
```

這裡還有一個 QAIC 特有的細節：QAIC 會透過 CPU-facing torch device plumbing
來執行。

- `vllm/config/__init__.py:1922`

對 QAIC 而言，`DeviceConfig.device` 會變成 `torch.device("cpu")`，但實際的
QAIC device ids 仍然由 `device_group` 指定。

## Command 參數 Trace

### `--device-group 0`

CLI parsing：

- `vllm/engine/arg_utils.py:891`

parser 會把 CSV 格式的 device ids 轉成 int list。所以：

```text
--device-group 0
```

會變成：

```python
[0]
```

Device config 會在 QAIC 上保留這個設定：

- `vllm/config/__init__.py:1915`

V1 QAIC worker 會檢查指定的 device id 是否存在：

- `vllm/v1/worker/qaic_worker.py:114`

### `--max-model-len 2048`

這個參數會變成 scheduling 和 compile config 使用的 context length。

QAIC-specific cache config 會使用 `max_model_len` 當 block size：

- `vllm/engine/arg_utils.py:1164`

QAIC platform 之後會設定：

- `vllm/platforms/qaic.py:91`

```python
cache_config.block_size = vllm_config.model_config.max_model_len
```

所以以這條 command 來看，QAIC V1 cache block size 預期是 `2048`。

### `--max-seq-len-to-capture 32`

這個參數會 mapping 到 `model_config.max_seq_len_to_capture`。

QAIC compile config 會用它當 default prefill sequence length：

- `vllm/model_executor/model_loader/qaic.py:636`

```python
"prefill_seq_len": vllm_config.model_config.max_seq_len_to_capture
```

你的 override 裡也明確設定：

```text
prefill_seq_len=32
```

所以這兩條路徑的設定是一致的。

### `--max-num-seqs 256`

這個參數會 mapping 到 `scheduler_config.max_num_seqs`。

QAIC compile config 會把它用在：

- `vllm/model_executor/model_loader/qaic.py:639`

```python
"full_batch_size": vllm_config.scheduler_config.max_num_seqs
```

V1 QAIC worker 內部會使用 `max_num_seqs + 1` 個 blocks，因為 V1 block pool
會保留一個 null block：

- `vllm/v1/worker/qaic_worker.py:167`

### `--quantization mxfp6`

QAIC 宣告 `mxfp6` 是支援的 quantization：

- `vllm/platforms/qaic.py:25`

QAIC compile config 會偵測這個設定並設：

- `vllm/model_executor/model_loader/qaic.py:612`

```python
mxfp6_en = True
```

接著寫入：

- `vllm/model_executor/model_loader/qaic.py:644`

```python
"mxfp6_matmul": mxfp6_en
```

所以 `--quantization mxfp6` 應該會以 `mxfp6_matmul=True` 的形式傳到
QEfficient compile call。

你的 override 裡也包含：

```text
mxfp6=True
```

override parser 也會把它 normalize 成 `mxfp6_matmul=True`：

- `vllm/model_executor/model_loader/qaic.py:563`

### `--kv-cache-dtype fp8`

V1 support oracle 會檢查自訂 KV cache dtype：

- `vllm/engine/arg_utils.py:1463`

QAIC 接受：

- `vllm/platforms/qaic.py:151`

```python
return kv_cache_dtype in ["fp8", "mxint8"]
```

QAIC 會把 vLLM 的 `fp8` cache dtype 映射成 QAIC 的 `mxint8` cache compression：

- `vllm/model_executor/model_loader/qaic.py:34`

```python
"fp8": "mxint8"
```

接著 compile config 會啟用：

- `vllm/model_executor/model_loader/qaic.py:617`

```python
mxint8_en = True
```

並寫入：

- `vllm/model_executor/model_loader/qaic.py:645`

```python
"mxint8_kv_cache": mxint8_en
```

所以在 QAIC 上，`--kv-cache-dtype fp8` 實際上代表 QAIC MXINT8 KV cache
compression。

你的 override 裡也包含：

```text
mxint8_kv_cache=True
```

override parser 會直接識別這個 key：

- `vllm/model_executor/model_loader/qaic.py:567`

### `--override-qaic-config`

CLI 會把以空白分隔的 `key=value` 或 `key:value` 轉成 dict：

- `vllm/engine/arg_utils.py:880`

接著 `_clean_config()` 會 normalize QAIC-specific aliases：

- `num_cores` / `aic_num_cores` 轉成 integer `num_cores`
- `mxfp6` / `mxfp6_matmul` / `mxfp6_en` 轉成 boolean `mxfp6_matmul`
- `mxint8` / `mxint8_en` / `mxint8_kv_cache` 轉成 boolean `mxint8_kv_cache`
- `dfs` / `aic_enable_depth_first` 轉成 boolean `aic_enable_depth_first`

相關程式碼：

- `vllm/model_executor/model_loader/qaic.py:557`
- `vllm/model_executor/model_loader/qaic.py:563`
- `vllm/model_executor/model_loader/qaic.py:567`
- `vllm/model_executor/model_loader/qaic.py:571`

重要 caveat：

`_clean_config()` 有一個 ignore list：

- `vllm/model_executor/model_loader/qaic.py:538`

它會忽略直接 override 下面這些值：

```python
["prefill_seq_len", "ctx_len", "batch_size", "full_batch_size",
 "num_speculative_tokens"]
```

在你的 command 中這沒有問題，因為 `--max-seq-len-to-capture 32` 已經把 default
`prefill_seq_len` 設為 `32`。但是 `--override-qaic-config` 裡面的
`prefill_seq_len=32` 不是實際生效的那一段。

### `--disable-frontend-multiprocessing`

這個參數影響 API server frontend process 行為，不是 V1 engine selection 本身。

V1 engine 的選擇是在 worker/model runner 路徑之前完成的，關鍵是
`VLLM_USE_V1=1` 和 V1 support oracle。

## QAIC V1 Cache 行為

QAIC V1 會自動關閉 prefix caching：

- `vllm/platforms/qaic.py:79`

```python
if envs.VLLM_USE_V1 and cache_config.enable_prefix_caching:
    cache_config.enable_prefix_caching = False
```

這點很重要，因為 generic V1 常常會 default 開啟 prefix caching，但 QAIC V1
會明確把它關掉。

當 prefix caching 關閉時，QAIC 會設定：

- `cache_config.block_size = max_model_len`

相關程式碼：

- `vllm/platforms/qaic.py:85`
- `vllm/platforms/qaic.py:91`

接著 V1 QAIC worker 預期：

- `num_gpu_blocks == max_num_seqs + 1`

相關程式碼：

- `vllm/v1/worker/qaic_worker.py:80`
- `vllm/v1/worker/qaic_worker.py:96`

## QEfficient Compile Flow

QAIC compile config 會在這裡建立：

- `vllm/model_executor/model_loader/qaic.py`

V1 的實際 compile call 是：

- `vllm/model_executor/model_loader/qaic_v1.py:657`

```python
qeff_model.compile(**qaic_compile_config.cfg)
```

接著 loaded QPC 會被掛到 QAIC devices：

- `vllm/model_executor/model_loader/qaic_v1.py:693`

```python
model.load_model(
    qpc_path=qpc_path,
    device_id=qaic_compile_config.device_group,
    ...
)
```

所以預期會傳到 QEfficient 的重要 compile settings 是：

```python
{
    "prefill_seq_len": 32,
    "ctx_len": 2048,
    "full_batch_size": 256,
    "num_cores": 16,
    "mxfp6_matmul": True,
    "mxint8_kv_cache": True,
    "aic_enable_depth_first": True,
}
```

實際 final dict 在 return `QaicCompileConfig` 前會被印出：

- `vllm/model_executor/model_loader/qaic.py:816`

如果你要確認最後值，可以在 server log 裡找這個 printed dict。

## Spec Decoding Ngram 支援

### Generic V1 支援

V1 明確支援 ngram speculative decoding：

- `vllm/engine/arg_utils.py:1504`

```python
# V1 supports N-gram, Medusa, and Eagle speculative decoding.
```

generic V1 GPU model runner 會在 method 是 `ngram` 時建立 `NgramProposer`：

- `vllm/v1/worker/gpu_model_runner.py:186`

```python
if self.speculative_config.method == "ngram":
    self.drafter = NgramProposer(self.vllm_config)
```

proposer 實作在：

- `vllm/v1/spec_decode/ngram_proposer.py:11`

它需要：

- `speculative_config.prompt_lookup_min`
- `speculative_config.prompt_lookup_max`
- `speculative_config.num_speculative_tokens`

相關檢查：

- `vllm/v1/spec_decode/ngram_proposer.py:14`
- `vllm/v1/spec_decode/ngram_proposer.py:15`
- `vllm/v1/spec_decode/ngram_proposer.py:25`

### QAIC V1 支援

QAIC V1 import 並使用 `NgramProposer`：

- `vllm/v1/worker/qaic_model_runner.py:23`

QAIC V1 會明確拒絕任何非 `ngram` 的 speculative method：

- `vllm/v1/worker/qaic_model_runner.py:42`

```python
if self.speculative_config.method != "ngram":
    raise ValueError(
        "Only ngram speculative decoding is supported on qaic "
        "backend when using vllm v1.")
```

這是最強的支援訊號：在 QAIC 上開啟 V1 之後，ngram 是被支援的，而非 ngram
的 speculative methods 是刻意不支援。

QAIC V1 會透過 ngram path 產生 draft tokens：

- `vllm/v1/worker/qaic_model_runner.py:648`

```python
assert isinstance(self.drafter, NgramProposer)
spec_token_ids = self.propose_ngram_draft_token_ids(...)
```

QAIC V1 也包含 CPU rejection sampler fallback：

- `vllm/v1/worker/qaic_model_runner.py:476`

code 裡有說明原因：

```python
V1's default rejection sampler launches Triton kernels. QAIC runs this
path on CPU and can have no active Triton driver, so use a small
PyTorch/CPU implementation for ngram, where draft_probs is None.
```

也就是說，QAIC V1 的 ngram speculative decoding 不是只有 config 接受而已；
它還有 QAIC-specific 的 rejection sampling execution path。

### 如何開啟 Ngram Spec Decoding

你目前這條 command 沒有開啟 speculative decoding。它只開啟 V1。

要開啟 ngram speculative decoding，需要加上 `--speculative-config`。可用欄位
由 `SpeculativeConfig` 定義：

- `vllm/config/__init__.py:1941`
- `vllm/config/__init__.py:2000`
- `vllm/config/__init__.py:2003`

範例格式：

```bash
--speculative-config '{
  "method": "ngram",
  "num_speculative_tokens": 4,
  "prompt_lookup_min": 1,
  "prompt_lookup_max": 5
}'
```

在 shell script 裡建議使用 single-line JSON string：

```bash
--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":1,"prompt_lookup_max":5}'
```

開啟後預期行為：

- V1 engine 仍然會啟動。
- QAIC V1 model runner 會接受這個 config，因為 method 是 `ngram`。
- ngram draft tokens 會根據 CPU-side token history 產生。
- 需要 rejection sampling 時，會使用 QAIC-specific CPU rejection sampler。

## Review Checklist

- 確認 `server_kv_fp8.log` 包含 `Initializing a V1 LLM engine`。
- 確認 log 包含 QAIC platform detection。
- 確認 log 印出的 QAIC compile config 包含 `mxfp6_matmul: True`。
- 確認 log 印出的 QAIC compile config 包含 `mxint8_kv_cache: True`。
- 確認 log 印出的 QAIC compile config 包含 `num_cores: 16`。
- 確認 log 印出的 QAIC compile config 包含 `aic_enable_depth_first: True`。
- 確認 log 印出的 QAIC compile config 包含 `prefill_seq_len: 32`。
- 如果要測 spec decoding，加入 method 為 `ngram` 的 `--speculative-config`。
- 如果要測 spec decoding，確認沒有出現 only ngram is supported 的錯誤。
- 如果在 QAIC V1 使用非 ngram method，預期 startup 會失敗，這是設計行為。

## 最終判斷

這條 command 正確開啟了 QAIC 上的 vLLM V1。

`mxfp6` path 有支援，並會以 `mxfp6_matmul=True` 傳到 QAIC。

`fp8` KV cache path 在 QAIC V1 上有支援，但會映射成
`mxint8_kv_cache=True`。

V1 開啟後支援 ngram speculative decoding，但目前這條 command 沒有啟用
speculative decoding。若要啟用，需要加入 method 為 `ngram` 的
`--speculative-config`。
