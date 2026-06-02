# `roy/qaic-plugin` 分支 QAIC 相關改動總覽（中文）

> **適用分支：** `roy/qaic-plugin`（當前 HEAD：`6f5a832cf`）  
> **分叉點：** `ca4ea2172`（來自 `origin/roy/tool-use-stream`）  
> **改動範圍：** 全部集中在 `vllm-qaic/` 目錄，主 vLLM core 未修改  
> **整體統計：** 20 個文件變更，新增 6,548 行

---

## 一、背景與整體目標

這個 branch 的核心工作是：

**把原本 fork 進 vLLM core 的 Qualcomm Cloud AI 100（QAIC）後端，重構成一個可以獨立 `pip install` 的樹外（out-of-tree）vLLM plugin**，目標接入 **vLLM 0.21+**（V1 引擎），同時保留 **UnieAI 的 ngram 投機解碼優化**。

整件事可以拆成三個面向：

| 面向 | 說明 |
|---|---|
| **Porting（移植）** | 把 Qualcomm QAIC patch（原本綁在 vLLM v0.10.1 fork）搬進獨立插件 |
| **Upgrade（升級）** | 對齊 vLLM 0.21 的新 V1 引擎 API（`GPUModelRunner` 大改、`InputBatch` 資料模型全新） |
| **Plugin 化** | 改用 vLLM 官方擴展點（entry points、`additional_config`），不 patch core |

> **目前狀態：** runtime 主路徑已移植進 `vllm-qaic/` 並通過本地語法檢查，但尚未在 AIC 實體機器上完成驗證。

---

## 二、新增的 Plugin Package 結構

```
vllm-qaic/
├── pyproject.toml              ← 插件包定義與 entry points
├── README.md                   ← 完整操作指南（含 Gate 1/2 測試、部署步驟）
├── tests/
│   ├── gate1_import_qaicrt.py  ← Gate 1：確認 qaicrt 可 import（torch 版本牆測試）
│   └── gate2_load_qpc.py       ← Gate 2：確認 QPC 可載入並實際執行
├── scripts/
│   └── port_from_fork.sh       ← 從舊 fork 搬 code 的輔助腳本
├── docs/                       ← 各項說明文件（見第七節）
└── vllm_qaic/
    ├── __init__.py             ← 插件入口（register_platform / register）
    ├── platform.py             ← QaicPlatform：QAIC 平台定義
    ├── worker.py               ← QaicWorker：QAIC worker
    ├── model_runner.py         ← QaicModelRunner：模型執行器 + ngram
    ├── model_loader.py         ← QaicCausalLM + load_qaic_model：QPC 載入
    ├── session.py              ← DisAgg_QAICInferenceSession：qaicrt 推理 session
    ├── compile_config.py       ← QaicCompileConfig + _get_qaic_compile_config
    ├── qserve_model_runner.py  ← QServeModelRunner：QPC 執行路徑
    └── quant.py                ← QaicQuantConfig：mxfp6 量化設定
```

---

## 三、各 Commit 改動摘要（時間序）

### `0b7c1f79b` — 建立 vllm-qaic 初始插件骨架

第一個 commit，從零建起整個插件包。

**新增內容：**
- `pyproject.toml`：定義插件包，透過 `vllm.platform_plugins` 和 `vllm.general_plugins` 兩個 entry points 與 vLLM 連接
- `vllm_qaic/__init__.py`：插件入口，偵測 `qaicrt` 是否可 import，決定是否載入 QaicPlatform
- `vllm_qaic/platform.py`：`QaicPlatform` 初版（`PlatformEnum.OOT`）
- `vllm_qaic/quant.py`：mxfp6 量化設定（透過 `register_quantization_config` 注冊）
- `vllm_qaic/model_loader.py`、`model_runner.py`、`session.py`、`worker.py`：各模組初始骨架
- `tests/gate1_import_qaicrt.py`、`gate2_load_qpc.py`：GO/NO-GO 測試腳本
- `README.md`：插件策略說明、split-env 設計、PART 0–3 完整操作指南

---

### `a033b50e3` — 填充 QaicModelRunner + GPUModelRunner 新舊對照

**新增/修改：**
- `vllm_qaic/model_runner.py`：加入第一版較完整的 `QaicModelRunner`，把 ngram 投機解碼 helper 函數移植進來
- `docs/MIGRATION_GPUModelRunner_old_vs_new.md`（新建）：逐項比對 v0.10.1 vs 0.21 `GPUModelRunner` 的差異
- `docs/UnieAI_Quic_integrated.md`（新建）：UnieAI × Qualcomm 技術整合說明，逐行標注代碼歸屬

---

### `d16bda58b` — 把 migration 文件擴充成完整變動清單

**修改：**
- `docs/MIGRATION_GPUModelRunner_old_vs_new.md`：從局部 migration notes 擴充成涵蓋所有 45 個新增文件 + 40 個修改文件的完整清單，加入 5 種處理分類（插件化/搬運/改本體/丟棄 V0/可選）

---

### `e35f9f103` — 新增完整中文 migration 工單

**新增：**
- `docs/MIGRATION_zh.md`（新建）：QAIC migration 工單的中文版，逐文件說明處理方式與狀態

---

### `285648127` — 移植低風險 QAIC 模組 + mxint8/fp8 釐清

**修改/新增：**
- `vllm_qaic/session.py`：填充 `DisAgg_QAICInferenceSession`（QPC 推理 session 邏輯）
- `vllm_qaic/model_loader.py`：擴充 QPC path 檢查與模型載入邏輯
- `vllm_qaic/model_runner.py`：擴充 model runner migration 注釋
- `vllm_qaic/quant.py`：修正 `is_qaic()` → `device_type == "qaic"` 的適配
- `scripts/port_from_fork.sh`（新建）：從舊 QAIC fork 搬 code 的輔助腳本（自動改 import 路徑）
- `docs/MIGRATION_GPUModelRunner_old_vs_new.md`、`docs/MIGRATION_zh.md`：釐清 `mxint8` 與 `fp8` 的分層關係（mxint8 是編譯器開關，走 `--additional-config`；vLLM 層用 `fp8`，不需改 core）

---

### `720079e15` — 重新整理「我們做了什麼」

**新增：**
- `docs/EXPLAINER_plain_zh.md`（新建）：白話中文說明，面向資工碩士，涵蓋 A1–A4 + B 五個工作面向
- `docs/REBUILD_input_prep_4B.md`（新建）：§4.B 上機實操指南，一步步說明如何在 0.21 上重建「輸入準備」（input preparation）

---

### `1876ebbd4` — 改寫 Explainer，釐清 A3/A4 與當前狀態

**修改：**
- `docs/EXPLAINER_plain_zh.md`：讓說明更具體，釐清 A3（QEfficient/SDK 版本對接）和 A4（0.21+ 遷移）的差別，明確標示各項狀態
- `docs/UnieAI_Quic_integrated.md`：補充版本邊界說明

---

### `e4ef269db` — 把中文 migration 文件改寫成更易溝通的版本

**修改：**
- `docs/MIGRATION_zh.md`：全面改寫，加入「怎麼讀」引導、術語說明（逐行確認過、優雅降級、字段），以及當前 repo 的誠實狀態表
- `docs/MIGRATION_GPUModelRunner_old_vs_new.md`：小幅更新

---

### `6f5a832cf` — 移植 QAIC runtime 完整路徑進插件（最大 commit）

這是目前最重要的一個 commit，把插件從「說明文件 + 骨架」推進到「有完整 runtime 實作」的狀態。

**新增文件：**
- `vllm_qaic/compile_config.py`：QAIC 編譯設定組裝（`QaicCompileConfig`、`_get_qaic_compile_config`、`_clean_config`、`QaicCausalLM`、`get_hf_model` 等）
- `vllm_qaic/qserve_model_runner.py`：`QServeModelRunner`，處理 qserve/QPC execution path（從 fork 搬入，upstream vLLM 沒有）

**大幅擴充：**
- `vllm_qaic/model_loader.py`：從 placeholder 變成實際的 `QaicCausalLM` + `load_qaic_model()`，包含 prefill/decode 執行路徑、disaggregated serving、LoRA 支援、vocab size 對齊
- `vllm_qaic/session.py`：完整的 `DisAgg_QAICInferenceSession`（qaicrt Context/Queue/Program/ExecObj 管理、pipeline 模式、KV buffer slicing）
- `vllm_qaic/worker.py`：`QaicWorker` 完整實作（init_device、load_model、execute_model、KV cache 管理等）
- `vllm_qaic/model_runner.py`：對齊 0.21 `InputBatch` 的 `execute_model` 主流程 + ngram 相關方法
- `vllm_qaic/platform.py`：`check_and_update_config` 完整實作（worker_cls 設定、prefix cache 停用、block_size 對齊）

**文件更新：**
- `README.md`、`docs/EXPLAINER_plain_zh.md`、`docs/MIGRATION_GPUModelRunner_old_vs_new.md`、`docs/MIGRATION_zh.md`、`docs/UnieAI_Quic_integrated.md`：全面更新，反映已移植進插件的 runtime 路徑狀態

---

### `f8a47278c` — 修正插件 review 問題

**修改：**
- `tests/gate2_load_qpc.py`：強化 Gate 2，必須真的執行一次 on-device inference 才算 PASS；從 session binding metadata 自動推導 input shape/dtype；跳過 QAIC KV retained-state buffers
- `vllm_qaic/__init__.py`：收緊 config/platform wiring
- `vllm_qaic/model_runner.py`：修正 rejection sampler sentinel handling（改從 vLLM rejection sampler import sentinel）
- `vllm_qaic/platform.py`：收緊 config 處理
- `scripts/port_from_fork.sh`：更新腳本

---

## 四、各核心模組說明

### 4.1 插件入口（`__init__.py`）

兩個 entry points 讓 vLLM 自動發現並載入插件，不需 patch vLLM core：

| Entry Point 群組 | 函數 | 作用 |
|---|---|---|
| `vllm.platform_plugins` | `register_platform()` | vLLM 啟動時呼叫，回傳 `QaicPlatform` 的完整類名（或 `None` 若 qaicrt 不可用） |
| `vllm.general_plugins` | `register()` | vLLM 啟動時執行，用於注冊量化設定、KV connector 等 |

偵測邏輯：嘗試 `import qaicrt` 和 `import QAicApi_pb2`，若成功則認定 QAIC 可用。

---

### 4.2 QAIC 平台（`platform.py`）

`QaicPlatform` 繼承 vLLM 的 `Platform`，設定：

- `device_type = "qaic"`、`device_name = "cpu"`（host 側張量在 CPU）
- 支援的量化：mxfp6、mxfp4、awq、gptq、fp8、compressed-tensors
- 只支援 V1 引擎（`supports_v1` 回傳 `True`）
- 不支援 prefix caching（啟動時自動關閉）
- `block_size` 設為 `max_model_len`（每個請求只用一個 KV block）
- QAIC 設定透過 `--additional-config` 傳入（取代舊 fork 的 `--override-qaic-config`）
- `_normalize_qaic_config()` 統一處理 legacy 的 `override_qaic_config` 和 `device_group` 格式

---

### 4.3 QAIC Worker（`worker.py`）

`QaicWorker` 繼承 `WorkerBase`，主要實作：

- `init_device()`：初始化 qaicrt 設備，檢查 device_group 可用性，建立 `QaicModelRunner`
- `determine_available_memory()`：根據 `max_num_seqs` 計算 KV block 數（QAIC 不支援 paged attention）
- `initialize_cache()`：設定 KV cache（不支援 prefix caching）
- `execute_model()`：呼叫 model runner 執行，回傳 `ModelRunnerOutput`
- `sample_tokens()`：拋出例外（QAIC model runner 直接回傳完整結果）
- `_configure_thread_parallelism()`：限制 PyTorch 線程數，避免 CPU 過度競爭

---

### 4.4 QAIC 推理 Session（`session.py`）

`DisAgg_QAICInferenceSession`（即 `QAICInferenceSession`）封裝 qaicrt 的低階操作：

- 透過 qaicrt Context/Queue/Program/ExecObj 管理 QPC 載入與執行
- 支援多個 ExecObj（pipeline 模式，`stages` 參數）
- KV buffer slicing（`setDataWithSlices`）支援 batch-wise KV cache 管理
- 支援 disaggregated serving（prefill/decode 分離）
- 自動處理 KV retained-state buffer（prefill → decode 狀態延續）

---

### 4.5 模型載入（`model_loader.py`）

`QaicCausalLM` + `load_qaic_model()` 負責 QPC 的選擇與載入：

- 支援使用環境變數 `VLLM_QAIC_QPC_PATH` 指定預編譯 QPC
- 支援透過 QEfficient（`QEFFAutoModelForCausalLM`）動態編譯 QPC
- prefill（`_run_prefill`）：支援 chunked prefill，逐 chunk 送卡
- decode（`_run_decode`）：支援 2D 輸入（投機解碼時 `[num_decodes, N+1]`）
- 支援 LoRA（`SupportsLoRA`）
- 支援 disaggregated serving（prefill/decode 分離部署）
- `kv_cache_info()`：從 QPC binding 讀取 KV cache 形狀與 dtype

---

### 4.6 編譯設定（`compile_config.py`）

組裝 QEfficient compile API 需要的設定：

- `_clean_config()`：統一正規化各種 key 名稱（如 `mxfp6`/`mxfp6_matmul`/`mxfp6_en` → `mxfp6_matmul`）
- `QaicCompileConfig`：dataclass 封裝所有編譯參數（qpc_path、device_group、num_cores、num_logits_to_keep 等）
- `QAIC_DEVICE_CONFIG`：支援 default/target/draft/turbo 多種 speculative decode 設定，對應不同環境變數
- `get_hf_model()`：透過 QEfficient 下載並準備 HF 模型（含 LoRA、多模態、語音等不同 model type）

---

### 4.7 量化設定（`quant.py`）

透過 `register_quantization_config("mxfp6")` 把 `QaicQuantConfig` 注冊進 vLLM，不需 patch core：

- `override_quantization_method()`：在 QAIC 平台上攔截量化設定，讓 mxfp6 生效
- 平台判斷改用 `device_type == "qaic"`（舊 fork 用 `is_qaic()`，0.21 OOT platform 沒有此方法）

---

## 五、UnieAI ngram 投機解碼（原創代碼）

這部分是 UnieAI 在 V1+QAIC 路徑上**從零新增**的功能（約 250 行）：

| 功能 | 說明 |
|---|---|
| **解禁投機解碼** | 舊 Qualcomm 代碼在 V1 路徑直接拋出 `raise ValueError("not yet supported")`，UnieAI 改為只允許 `ngram` method |
| **2D decode 打包** | 把 decode batch 打包成 `[num_decodes, N+1]`，讓 target QPC 一次驗證「上個 token + 至多 N 個候選」 |
| **CPU 拒絕採樣** | 寫了 7 個純 CPU PyTorch 函數，實作同數學等價的接受規則（vLLM 內建版用 Triton GPU kernel，QAIC host 可能沒有 Triton） |
| **N+1 logits** | 修改 loader 讓 target QPC 輸出 `N+1` 行 logits 供驗證 |

**7 個 CPU 採樣 helper 函數：**

| 函數 | 用途 |
|---|---|
| `_qaic_rejection_sample` | 頂層：對每個請求做 greedy 或 random 驗證，組裝接受的 tokens |
| `_qaic_is_greedy_request` | 判斷該請求是 greedy 還是 random 採樣 |
| `_qaic_rejection_sample_greedy_req` | greedy 接受：逐個比對是否等於 top-1，遇不符即停 |
| `_qaic_rejection_sample_random_req` | random 接受：按機率接受，拒絕時 recover 一個 replacement token |
| `_qaic_target_probs_for_req` | 把 logits 轉成機率（套用 temperature、top-k、top-p） |
| `_qaic_apply_top_k_top_p` | 標準 top-k/top-p 過濾 |
| `_qaic_sample_from_probs` | 從機率分佈抽一個 token（Gumbel/指數噪聲） |

這 7 個函數**在 0.21 上可逐字搬運**，因為它們只依賴 `torch`、`SamplingMetadata` 和 `SpecDecodeMetadata` 的字段，而這些字段在 0.21 上都還在。

---

## 六、核心設計決策

### 6.1 Split Environment（torch 版本牆）

QEfficient 綁 torch **2.7**，vLLM 0.21 綁 torch **2.11**，兩者無法在同一 venv 共存：

```
COMPILE（離線一次）               SERVE（在線，本插件）
torch 2.7 + QEfficient 1.21   →   torch 2.11 + vLLM 0.21 + vllm-qaic
  model.compile() → QPC binary →   qaicrt 載入預編譯的 QPC
                                    （完全不 import QEfficient compile 路徑）
```

### 6.2 不改 vLLM Core

原本 v0.10.1 fork 修改了約 40 個 core 文件，本插件用以下方式避免：

| 舊 fork 的做法 | 本插件的替代方式 |
|---|---|
| 修改 `platforms/__init__.py` 加 QAIC 偵測 | 用 `vllm.platform_plugins` entry point |
| 修改 `arg_utils.py` 加 `--override-qaic-config` | 改用 `--additional-config`（標準機制） |
| 修改 `config/__init__.py` 加 QAIC config 字段 | 從 `additional_config` 讀取 |
| 修改量化注冊表 | 用 `register_quantization_config("mxfp6")` |
| 修改 `envs.py` 加 QAIC 環境變數 | 插件自己讀 `os.environ` |
| `current_platform.is_qaic()` | 改用 `current_platform.device_type == "qaic"` |

### 6.3 mxint8 不需改 Core

QAIC 的卡上 KV 格式是 mxint8（由 QEfficient 編譯時決定，透過 `--additional-config` 傳入），vLLM 層的 `--kv-cache-dtype` 用 `fp8` 就好，兩者是不同層的設定，不需要把 `mxint8` 加進 vLLM 的 `CacheDType` 白名單。

---

## 七、文件清單

`vllm-qaic/docs/` 目錄下現有文件：

| 文件 | 說明 |
|---|---|
| `MIGRATION_zh.md` | QAIC → vLLM 0.21+ 遷移工單（中文），逐文件說明處理方式 |
| `MIGRATION_GPUModelRunner_old_vs_new.md` | GPUModelRunner 新舊對照（v0.10.1 vs 0.21），含完整 45+40 文件清單 |
| `EXPLAINER_plain_zh.md` | 白話說明（資工碩士可讀），涵蓋 A1–A4 + B 五個工作面向 |
| `REBUILD_input_prep_4B.md` | §4.B 上機實操指南：如何在 0.21 上重建 input preparation |
| `UnieAI_Quic_integrated.md` | 技術整合說明（含代碼歸屬），面向 Qualcomm 工程師 / 法律審查 |
| `QAIC_COMMIT_SUMMARY.md` | Commit-by-commit 摘要（繁體中文） |
| **`BRANCH_CHANGES_ZH.md`（本文件）** | 分支改動總覽（繁體中文），整合所有面向 |

---

## 八、驗證狀態與待辦

| 項目 | 狀態 | 說明 |
|---|---|---|
| 改動盘點與遷移設計 | ✅ 已完成 | 本文件 + `MIGRATION_zh.md` + `MIGRATION_GPUModelRunner_old_vs_new.md` |
| 插件外殼（entry points、platform、quant） | 🟡 已寫未上機 | 本地語法檢查通過 |
| session / model_loader / worker / compile_config / qserve_model_runner | 🟡 已搬入未上機 | 已從 v1_ngram fork 搬進並改 import |
| execute_model 主流程（0.21 InputBatch） | 🟡 已寫未上機 | 已重建 input prep，接入 QAIC decode/prefill |
| ngram CPU rejection sampler（7 函數）+ 2D decode 打包 | 🟡 已接入未上機 | 邏輯正確，尚待 AIC 上機驗證 |
| torch 2.7/2.11 split 方案（Gate 1/2） | 🔬 待驗證 | GO/NO-GO 腳本已備，需 AIC host 執行 |
| vllm serve smoke test | ⬜ 尚未完成 | 需 AIC host + Gate 2 PASS 後進行 |
| ngram smoke test | ⬜ 尚未完成 | 需基礎路徑跑通後進行 |

**驗證建議順序：**
1. Gate 1（`tests/gate1_import_qaicrt.py`）：確認 qaicrt 在 torch 2.11 環境可 import
2. Gate 2（`tests/gate2_load_qpc.py`）：確認 QPC 可載入並真正執行一次
3. 確認 `current_platform` == `QaicPlatform`
4. 無投機解碼的 `vllm serve` smoke test
5. 開啟 ngram 投機解碼的完整測試

---

*整理人：AI 輔助整理，基於 `roy/qaic-plugin` branch 截至 `6f5a832cf` 的改動。*
