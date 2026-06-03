# QAIC Commit 變更摘要

這份文件整理 `roy/qaic-plugin` branch 上最近 QAIC 相關 commit 的改動。

目前 branch 已更新到：

```text
6f5a832cf Port QAIC runtime path to vLLM 0.21 plugin
```

## Branch 從哪裡分出來

`roy/qaic-plugin` 是接在 `origin/roy/tool-use-stream` 後面長出來的。

直接分叉點是：

```text
ca4ea2172 Fix DeepSeek DSML streaming argument deltas
```

也就是說，下面列出的 QAIC 改動，都是在這個 commit 之後新增的。改動範圍主要集中在新的 out-of-tree plugin package：

```text
vllm-qaic/
```

## 整體改了什麼

這個 branch 新增了一個獨立的 QAIC plugin package：

```text
vllm-qaic/
```

目標是把 Qualcomm Cloud AI 100 / QAIC 支援，從原本需要 patch vLLM core 的 fork，搬到一個可以獨立 pip install 的 vLLM plugin 裡。

整體變更量：

```text
20 files changed, 6548 insertions(+)
```

目前沒有修改主 vLLM core 檔案；改動都集中在 `vllm-qaic/`。

## 新增的主要內容

### 1. Plugin package 架構

相關檔案：

```text
vllm-qaic/pyproject.toml
vllm-qaic/vllm_qaic/__init__.py
vllm-qaic/vllm_qaic/platform.py
```

改動重點：

- 新增 package：`vllm-qaic`。
- 透過 `vllm.platform_plugins` 註冊 QAIC platform plugin。
- 透過 `vllm.general_plugins` 註冊一般 plugin 初始化流程。
- 新增 `QaicPlatform`，作為 QAIC 的 out-of-tree platform class。
- 透過 `qaicrt` 和 `QAicApi_pb2` 判斷目前機器是否支援 QAIC。
- 在 vLLM config wiring 裡設定 QAIC worker。

### 2. QAIC runtime path

相關檔案：

```text
vllm-qaic/vllm_qaic/compile_config.py
vllm-qaic/vllm_qaic/model_loader.py
vllm-qaic/vllm_qaic/session.py
vllm-qaic/vllm_qaic/worker.py
vllm-qaic/vllm_qaic/qserve_model_runner.py
```

改動重點：

- 新增 QAIC compile config 組裝邏輯。
- 新增 QPC path 檢查與處理。
- 新增 `QaicCausalLM` 和 `load_qaic_model()`，讓 plugin 可以載入 QAIC model。
- 新增 `DisAgg_QAICInferenceSession`，處理 QAIC runtime session。
- 新增 `QaicWorker`，接上 vLLM worker path。
- 新增 `QServeModelRunner`，處理 qserve / QPC execution path。
- 把 runtime 相關邏輯往 vLLM 0.21 plugin 架構移植。

### 3. Model runner 與 ngram speculative decoding

相關檔案：

```text
vllm-qaic/vllm_qaic/model_runner.py
```

改動重點：

- 新增 `QaicModelRunner`。
- 把舊 fork 裡的 QAIC model runner 結構往 vLLM 0.21 移植。
- 加入 UnieAI 的 ngram speculative decoding helper。
- 加入 QAIC 用的 CPU rejection sampling。
- 加入 speculative decoding 需要的 2D decode packing helper。
- 文件中標註舊 vLLM 0.10.1 `GPUModelRunner` 和新 vLLM 0.21 `GPUModelRunner` 的主要差異。

### 4. Quantization

相關檔案：

```text
vllm-qaic/vllm_qaic/quant.py
```

改動重點：

- 新增 QAIC quantization config registration。
- 透過 vLLM quantization registration path 註冊 `mxfp6`。
- 把 platform check 改成適合 out-of-tree QAIC platform 的方式。
- 說明目前 upstream vLLM 對 `mxint8` KV cache dtype 的限制。

### 5. 測試與 Gate check

相關檔案：

```text
vllm-qaic/tests/gate1_import_qaicrt.py
vllm-qaic/tests/gate2_load_qpc.py
```

改動重點：

- 新增 Gate 1：確認 target serve environment 可以 import `qaicrt`。
- 新增 Gate 2：確認 precompiled QPC 可以透過 `qaicrt` 載入並執行。
- 強化 Gate 2：必須真的跑一次 on-device inference 才能回報 PASS。
- 從 QAIC session binding metadata 自動推導 input shape / dtype。
- 建立測試 input 時會跳過 KV retained-state buffer，避免餵錯 buffer。

### 6. Porting script

相關檔案：

```text
vllm-qaic/scripts/port_from_fork.sh
```

改動重點：

- 新增 helper script，用來從舊 QAIC fork port 部分檔案到 plugin package。
- 自動改寫 plugin-local import path。
- 標出 config migration 時需要人工處理的地方。

### 7. 文件

相關檔案：

```text
vllm-qaic/README.md
vllm-qaic/docs/EXPLAINER_plain_zh.md
vllm-qaic/docs/MIGRATION_GPUModelRunner_old_vs_new.md
vllm-qaic/docs/MIGRATION_zh.md
vllm-qaic/docs/REBUILD_input_prep_4B.md
vllm-qaic/docs/UnieAI_Quic_integrated.md
```

改動重點：

- 說明 out-of-tree plugin 策略。
- 說明 split environment 設計：
  - Compile environment：QEfficient / torch 2.7。
  - Serve environment：vLLM 0.21 / torch 2.11 / `qaicrt`。
- 說明為什麼 QPC compilation 和 QPC serving 要分開。
- 說明如何從舊 QAIC fork 遷移到 plugin。
- 比較舊版與新版 `GPUModelRunner` 的差異。
- 說明哪些部分是 Qualcomm-derived runtime support，哪些部分是 UnieAI ngram speculative decoding。
- 新增中文 migration / explainer 文件，方便溝通。

## Commit-by-commit 摘要

### `6f5a832cf` - Port QAIC runtime path to vLLM 0.21 plugin

這是最近最大、也最關鍵的一個 commit。

它把 `vllm-qaic` 從「plugin scaffold / migration guide」往前推進成「已經有大量 QAIC runtime implementation 的 plugin package」。

主要改動：

- 新增 `compile_config.py`。
- 新增 `qserve_model_runner.py`。
- 大幅擴充 `model_loader.py`，從 placeholder 變成實際 QAIC model loading path。
- 大幅擴充 `session.py`，加入 QAIC inference session 邏輯。
- 大幅擴充 `worker.py`，加入 QAIC worker 邏輯。
- 更新 `model_runner.py` 和 `platform.py`，讓 vLLM 0.21 runtime wiring 更完整。
- 更新 README 和 migration docs，反映目前已經 port 進 runtime path。

改動檔案：

```text
M vllm-qaic/README.md
M vllm-qaic/docs/EXPLAINER_plain_zh.md
M vllm-qaic/docs/MIGRATION_GPUModelRunner_old_vs_new.md
M vllm-qaic/docs/MIGRATION_zh.md
M vllm-qaic/docs/UnieAI_Quic_integrated.md
A vllm-qaic/vllm_qaic/compile_config.py
M vllm-qaic/vllm_qaic/model_loader.py
M vllm-qaic/vllm_qaic/model_runner.py
M vllm-qaic/vllm_qaic/platform.py
A vllm-qaic/vllm_qaic/qserve_model_runner.py
M vllm-qaic/vllm_qaic/session.py
M vllm-qaic/vllm_qaic/worker.py
```

### `f8a47278c` - fix(vllm-qaic): address plugin review

主要改動：

- 改進 Gate 2 QPC loading test。
- Gate 2 變得更嚴格：必須真的 run 一次才算 PASS。
- 用真實 session binding metadata 建立測試 inputs。
- Gate 2 會跳過 QAIC KV retained-state buffers。
- 修正 rejection sampler sentinel handling，改成從 vLLM rejection sampler import sentinel。
- 收緊 plugin config / platform wiring。

改動檔案：

```text
M vllm-qaic/scripts/port_from_fork.sh
M vllm-qaic/tests/gate2_load_qpc.py
M vllm-qaic/vllm_qaic/__init__.py
M vllm-qaic/vllm_qaic/model_runner.py
M vllm-qaic/vllm_qaic/platform.py
```

### `e4ef269db` - Rewrite MIGRATION_zh as a readable communication doc

主要改動：

- 把中文 migration doc 改寫成更容易閱讀、比較適合溝通的版本。
- 保留 migration 內容，但調整結構和文字。
- 小幅更新 GPUModelRunner migration 文件。

改動檔案：

```text
M vllm-qaic/docs/MIGRATION_GPUModelRunner_old_vs_new.md
M vllm-qaic/docs/MIGRATION_zh.md
```

### `1876ebbd4` - Rewrite explainer for CS-master level

主要改動：

- 改寫中文 explainer。
- 讓說明更具體，減少抽象描述。
- 釐清 A3 / A4 相關段落和目前實作狀態。

改動檔案：

```text
M vllm-qaic/docs/EXPLAINER_plain_zh.md
M vllm-qaic/docs/UnieAI_Quic_integrated.md
```

### `720079e15` - Reframe "what we did" as porting/upgrade + ngram

主要改動：

- 新增中文白話 explainer。
- 新增 input prep rebuild guide。
- 把整件事重新整理成三個面向：
  - porting QAIC support；
  - upgrade 到新版 vLLM；
  - 保留 UnieAI ngram speculative decoding。

改動檔案：

```text
A vllm-qaic/docs/EXPLAINER_plain_zh.md
A vllm-qaic/docs/REBUILD_input_prep_4B.md
M vllm-qaic/docs/UnieAI_Quic_integrated.md
```

### `285648127` - Port low-risk QAIC pieces

主要改動：

- 新增 `port_from_fork.sh`。
- port 或準備 port 較低風險的 QAIC pieces，例如 session、quantization、model-loader skeleton。
- 擴充 model-runner migration notes。
- 釐清 `mxint8` 和 `fp8` 的限制。

改動檔案：

```text
M vllm-qaic/docs/MIGRATION_GPUModelRunner_old_vs_new.md
M vllm-qaic/docs/MIGRATION_zh.md
A vllm-qaic/scripts/port_from_fork.sh
M vllm-qaic/vllm_qaic/model_loader.py
M vllm-qaic/vllm_qaic/model_runner.py
M vllm-qaic/vllm_qaic/quant.py
M vllm-qaic/vllm_qaic/session.py
```

### `e35f9f103` - Add Chinese migration work-list

主要改動：

- 新增完整 QAIC migration work-list 的中文版。

改動檔案：

```text
A vllm-qaic/docs/MIGRATION_zh.md
```

### `d16bda58b` - Expand migration doc to the complete change-set

主要改動：

- 擴充 GPUModelRunner migration doc。
- 把原本比較局部的 migration notes，整理成更完整的 change-set。

改動檔案：

```text
M vllm-qaic/docs/MIGRATION_GPUModelRunner_old_vs_new.md
```

### `a033b50e3` - Fill QaicModelRunner port

主要改動：

- 新增 plugin 目錄用的 `.gitignore`。
- 新增 Qualcomm technical notes。
- 新增第一版較完整的 `QaicModelRunner`。
- 新增 old-vs-new GPUModelRunner migration notes。
- 把 ngram speculative decoding helper logic port 到 QAIC model runner。

改動檔案：

```text
A vllm-qaic/.gitignore
A vllm-qaic/docs/MIGRATION_GPUModelRunner_old_vs_new.md
A vllm-qaic/docs/UnieAI_Quic_integrated.md
M vllm-qaic/vllm_qaic/model_runner.py
```

### `0b7c1f79b` - Add vllm-qaic out-of-tree plugin scaffold

主要改動：

- 建立初始 `vllm-qaic` package。
- 新增 package metadata 和 vLLM plugin entry points。
- 新增 QAIC platform plugin wiring。
- 新增初始 quantization、model-loader、model-runner、session、worker skeleton files。
- 新增 Gate 1 和 Gate 2 tests。
- 新增 README，說明 plugin 策略和 split-env plan。

改動檔案：

```text
A vllm-qaic/README.md
A vllm-qaic/pyproject.toml
A vllm-qaic/tests/gate1_import_qaicrt.py
A vllm-qaic/tests/gate2_load_qpc.py
A vllm-qaic/vllm_qaic/__init__.py
A vllm-qaic/vllm_qaic/model_loader.py
A vllm-qaic/vllm_qaic/model_runner.py
A vllm-qaic/vllm_qaic/platform.py
A vllm-qaic/vllm_qaic/quant.py
A vllm-qaic/vllm_qaic/session.py
A vllm-qaic/vllm_qaic/worker.py
```

## 目前狀態

這個 branch 已經不只是單純 scaffold。最新 commit `6f5a832cf` 已經把大量 QAIC runtime path port 進 vLLM 0.21 plugin 架構。

目前最大的風險不是 plugin entry point，而是和 vLLM 0.21 internals 的整合正確性，特別是：

- `GPUModelRunner` 相關行為；
- input preparation；
- cache management；
- speculative decoding；
- QAIC QPC runtime path。

建議驗證順序：

1. 在 target serve environment 安裝 `vllm-qaic`。
2. 跑 Gate 1，確認 `qaicrt` 可以 import。
3. 跑 Gate 2，確認 precompiled QPC 可以載入並實際執行。
4. 啟動 vLLM，確認 platform selection 會選到 QAIC。
5. 先測不開 speculative decoding 的基本 serving path。
6. 再打開 ngram speculative decoding 測完整路徑。
