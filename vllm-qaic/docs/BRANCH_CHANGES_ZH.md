# `v1_vllm021` / `roy/qaic-plugin` QAIC 相關改動總覽（中文）

> **適用分支：** `v1_vllm021`（已合入最新 `origin/roy/qaic-plugin`）  
> **當前 HEAD：** `a38135ce8 Merge branch 'roy/qaic-plugin' of github.com:UnieAI/vllm into v1_vllm021`  
> **分叉點：** `ca4ea2172`（來自 `origin/roy/tool-use-stream`）  
> **改動範圍：** 全部集中在 `vllm-qaic/` 目錄，主 vLLM core 未修改  
> **整體統計：** 20 個文件變更，新增 6,548 行
>
> **Branch 關係說明：**  
> `v1_vllm021` 以 `6f5a832cf` 為共同基底，先加入本地 `68396e128 vllm021 update`，  
> 再合入 `origin/roy/qaic-plugin` 最新三個 QAIC runner 修正（`2759ca198`、`4632ccea7`、`4338e8675`）。  
> 因此目前 `v1_vllm021` 是 `roy/qaic-plugin` 最新內容 + 本地 vLLM 0.21 QAIC 相容修正。

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

> **目前狀態：** runtime 主路徑已移植進 `vllm-qaic/`，本地 `py_compile` 通過；  
> 使用 `.venv-vllm021`、`VLLM_QAIC_QPC_PATH=/workspace/weiminc/GPT2_Static_QPC` 的 vLLM offline smoke 已確認 **V1 engine 有啟動並完成 generate**。  
> 目前 smoke 輸出文字不具語意參考價值，原因是 GPT2 QPC 搭配 Qwen2.5 HF model/tokenizer，詳見第十節。

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

> 說明：`0b7c1f79b`、`d16bda58b`、`e35f9f103`、`720079e15`、`1876ebbd4`、`e4ef269db` 這幾個 commit 改動集中在文件（docs），代碼 diff 為空或只有新建文件，不另列 diff。有代碼 diff 的 commit 另行展示。

---

### `0b7c1f79b` — 建立 vllm-qaic 初始插件骨架

第一個 commit，從零建起整個插件包，全部為新建文件，無 diff。

**新建文件：**
```
vllm-qaic/README.md
vllm-qaic/pyproject.toml
vllm-qaic/tests/gate1_import_qaicrt.py
vllm-qaic/tests/gate2_load_qpc.py
vllm-qaic/vllm_qaic/__init__.py
vllm-qaic/vllm_qaic/model_loader.py   ← 骨架（raise NotImplementedError）
vllm-qaic/vllm_qaic/model_runner.py   ← 骨架（raise NotImplementedError）
vllm-qaic/vllm_qaic/platform.py
vllm-qaic/vllm_qaic/quant.py          ← 骨架（pass）
vllm-qaic/vllm_qaic/session.py        ← 骨架
vllm-qaic/vllm_qaic/worker.py         ← 骨架
```

---

### `a033b50e3` — 填充 QaicModelRunner：ngram helpers + 方法骨架

**改動文件：** `vllm_qaic/model_runner.py`、`docs/MIGRATION_GPUModelRunner_old_vs_new.md`（新建）、`docs/UnieAI_Quic_integrated.md`（新建）、`vllm-qaic/.gitignore`（新建）

**`model_runner.py` 代碼 diff（主要變化）：**

把原本整個文件的 `raise NotImplementedError(...)` 替換成實際實作：

```diff
-raise NotImplementedError(
-    "Port vllm/v1/worker/qaic_model_runner.py from the v1_ngram fork ..."
-)
+class QaicModelRunner(GPUModelRunner):
+    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
+        super().__init__(vllm_config, device)
+        assert device == torch.device("cpu"), "QAIC keeps host tensors on CPU."
+
+        # UnieAI ngram gating（逐字從 fork 移植）
+        self.speculative_model_type: Optional[str] = None
+        if self.speculative_config is not None:
+            if self.speculative_config.method != "ngram":
+                raise ValueError("Only ngram speculative decoding is supported...")
+            self.speculative_model_type = "target"
```

加入 UnieAI 的 7 個 ngram CPU rejection sampler helper 函數（完整移植）：
```diff
+    def _qaic_rejection_sample(self, ...)         # 頂層驗證邏輯
+    def _qaic_is_greedy_request(self, ...)        # greedy/random 判斷
+    def _qaic_rejection_sample_greedy_req(...)    # greedy 接受規則
+    def _qaic_rejection_sample_random_req(...)    # random 接受 + recover
+    def _qaic_target_probs_for_req(...)           # logits → 機率
+    def _qaic_apply_top_k_top_p(...)              # top-k/top-p 過濾
+    def _qaic_sample_from_probs(...)              # 抽樣
```

各 GPUModelRunner override 方法以 0.21 簽名寫入，body 標 `raise NotImplementedError` + TODO：
```diff
+    def execute_model(self, scheduler_output) -> Optional[ModelRunnerOutput]:
+        raise NotImplementedError("TODO: rebuild input-prep from InputBatch")
+
+    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
+        raise NotImplementedError("Port get_kv_cache_spec from the fork (low risk).")
+
+    def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
+        raise NotImplementedError("Port initialize_kv_cache...")
+
+    def load_model(self, load_dummy_weights: bool = False):
+        raise NotImplementedError("Port load_model -> vllm_qaic.model_loader.load_qaic_model.")
```

`PLACEHOLDER_TOKEN_ID` 暫時容錯取值：
```diff
+try:
+    from vllm.v1.spec_decode.utils import PLACEHOLDER_TOKEN_ID
+except Exception:
+    PLACEHOLDER_TOKEN_ID = -1   # 暫時用 -1，後續在 f8a47278c 修正
```

---

### `d16bda58b` — 擴充 migration 文件（純文件改動，無代碼 diff）

**改動文件：** `docs/MIGRATION_GPUModelRunner_old_vs_new.md`

---

### `e35f9f103` — 新增中文 migration 工單（純文件，無代碼 diff）

**新建文件：** `docs/MIGRATION_zh.md`

---

### `285648127` — 移植低風險 QAIC 模組

**改動文件：** `vllm_qaic/quant.py`、`vllm_qaic/model_runner.py`、`vllm_qaic/model_loader.py`、`vllm_qaic/session.py`、`scripts/port_from_fork.sh`（新建）、docs 文件

**`quant.py` 代碼 diff：**

從骨架 `pass` 替換成完整的 `QaicQuantConfig`：
```diff
-def register_qaic_quant() -> None:
-    # TODO(port): paste QaicQuantConfig here and register it
-    pass
+def register_qaic_quant() -> None:
+    """Idempotent (general_plugins may run per-process)."""
+    global _REGISTERED
+    if _REGISTERED:
+        return
+    register_quantization_config("mxfp6")(QaicQuantConfig)
+    _REGISTERED = True
+
+class QaicQuantConfig(QuantizationConfig):
+    """MxFP6 Quantization Config class for QAIC Backend."""
+    # 從 fork 的 qaic_quant.py 移植
+    # 關鍵適配：current_platform.is_qaic() → device_type == "qaic"
+    @classmethod
+    def override_quantization_method(cls, hf_quant_cfg, user_quant):
-        if current_platform.is_qaic():            # 舊 fork 寫法（0.21 OOT 沒有）
+        if getattr(current_platform, "device_type", None) == "qaic":  # 改用 device_type
            if quant_method in QAICQuantList and user_quant == "mxfp6":
                return user_quant
```

**`model_runner.py` 代碼 diff（`get_kv_cache_spec`、`initialize_kv_cache`、`load_model` 從 stub → 實作）：**

```diff
 def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
-    raise NotImplementedError("Port get_kv_cache_spec from the fork (low risk).")
+    block_size = self.cache_config.block_size
+    kv_cache_spec: dict[str, "KVCacheSpec"] = {}
+    n_layers = self.model_config.get_num_layers(self.parallel_config)
+    for i in range(n_layers):
+        layer_name = f"layer_{i}"
+        kv_cache_spec[layer_name] = FullAttentionSpec(
+            block_size=block_size,
+            num_kv_heads=self.num_kv_heads,
+            head_size=self.head_size,
+            dtype=self.kv_cache_dtype,
+            use_mla=False,
+        )
+    return kv_cache_spec

 def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
-    raise NotImplementedError("Port initialize_kv_cache; mind the new is_profiling arg.")
+    self.kv_cache_config = kv_cache_config
+    # KV cache 在卡上，host 不需分配記憶體

 def load_model(self, load_dummy_weights: bool = False):
-    raise NotImplementedError("Port load_model -> vllm_qaic.model_loader.load_qaic_model.")
+    logger.info("Starting to load model %s...", self.model_config.model)
+    t0 = time.perf_counter()
+    with set_current_vllm_config(self.vllm_config):
+        self.model = load_qaic_model(
+            self.vllm_config,
+            speculative_model_type=self.speculative_model_type,
+        )
+    logger.info("Model loading took %.6f seconds", time.perf_counter() - t0)
```

新增 imports：
```diff
+from vllm.config import VllmConfig, set_current_vllm_config
+from vllm.logger import init_logger
+from vllm.v1.kv_cache_interface import FullAttentionSpec
+from vllm_qaic.model_loader import load_qaic_model
+import time
```

**`model_loader.py` diff（placeholder 文字更新，非實作）：**

把 TODO 說明改成「執行 `port_from_fork.sh` 後此文件會被覆寫」，正式 import 由腳本填入。

---

### `720079e15` — 新增中文白話說明 + §4.B 上機指南（純文件，無代碼 diff）

**新建：** `docs/EXPLAINER_plain_zh.md`、`docs/REBUILD_input_prep_4B.md`

---

### `1876ebbd4` — 改寫 Explainer（純文件，無代碼 diff）

**修改：** `docs/EXPLAINER_plain_zh.md`、`docs/UnieAI_Quic_integrated.md`

---

### `e4ef269db` — 改寫中文 migration 文件（純文件，無代碼 diff）

**修改：** `docs/MIGRATION_zh.md`、`docs/MIGRATION_GPUModelRunner_old_vs_new.md`

---

### `6f5a832cf` — 移植 QAIC runtime 完整路徑（最大 commit）

**新增：** `vllm_qaic/compile_config.py`、`vllm_qaic/qserve_model_runner.py`（均從 fork 搬入，見第二節說明）

**大幅擴充：** `model_loader.py`（placeholder → 完整 `QaicCausalLM`）、`session.py`（完整 qaicrt session）、`worker.py`（完整 `QaicWorker`）、`model_runner.py`（`execute_model` 主流程）、`platform.py`（`check_and_update_config` 完整實作）

此 commit 的代碼量最大（+6000 行），主要是從 fork 的三個大型 Qualcomm-proprietary 文件搬入並改 import 路徑，詳見第四節各模組說明。

---

### `f8a47278c` — 修正插件 review 問題（代碼 bug fix）

**改動文件：** `vllm_qaic/__init__.py`、`vllm_qaic/model_runner.py`、`vllm_qaic/platform.py`、`tests/gate2_load_qpc.py`、`scripts/port_from_fork.sh`

**`__init__.py` diff — 修正 sys.path 重複加入：**
```diff
+    def _add_path(p: str) -> None:   # 避免重複加入
+        if p not in sys.path:
+            sys.path.append(p)

-    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
+    _add_path(f"/opt/qti-aic/dev/lib/{platform.machine()}")
-    sys.path.append("/opt/qti-aic/dev/python")
+    _add_path("/opt/qti-aic/dev/python")
```

**`model_runner.py` diff — 修正 sentinel import（關鍵 bug fix）：**

greedy sentinel 在 v0.10.1 是 `-1.0`，0.21 改成 `GREEDY_TEMPERATURE = 0`，直接用舊值會導致 greedy/random 路由錯誤：

```diff
-try:
-    from vllm.v1.spec_decode.utils import PLACEHOLDER_TOKEN_ID
-except Exception:
-    PLACEHOLDER_TOKEN_ID = -1   # 靜默 fallback（危險）
+# 改從 vLLM rejection sampler 的同一模組 import，避免 sentinel 不一致
+from vllm.v1.sample.rejection_sampler import (
+    GREEDY_TEMPERATURE,       # 0.21 上 = 0（不再是 -1.0）
+    PLACEHOLDER_TOKEN_ID,     # = -1，fail loudly if not found
+)
```

對應修正 greedy 判斷：
```diff
 def _qaic_is_greedy_request(self, sampling_metadata, req_idx):
-    return float(sampling_metadata.temperature[req_idx].item()) == -1.0
+    return float(sampling_metadata.temperature[req_idx].item()) == float(GREEDY_TEMPERATURE)

 def _qaic_target_probs_for_req(self, logits, sampling_metadata, req_idx):
     temperature = float(sampling_metadata.temperature[req_idx].item())
-    if temperature != -1.0:
+    if temperature != float(GREEDY_TEMPERATURE):
         logits.div_(temperature)
```

`assert` 移到 `super().__init__` 之前，讓錯誤信息更清晰：
```diff
 def __init__(self, vllm_config, device):
-    super().__init__(vllm_config, device)
-    assert device == torch.device("cpu"), "QAIC keeps host tensors on CPU."
+    assert device == torch.device("cpu"), "QAIC keeps host tensors on CPU."
+    super().__init__(vllm_config, device)   # 先 assert 再 super，fail fast
```

**`platform.py` diff — 補回 mxfp4 + 移除舊 TODO：**
```diff
 supported_quantization: list[str] = [
-    "mxfp6", "awq", "gptq", "fp8", "compressed-tensors",
+    "mxfp6", "mxfp4", "awq", "gptq", "fp8", "compressed-tensors",  # 補回 mxfp4
 ]

-    # Stash the cleaned config where the model loader will read it.
-    # TODO(port): bring over the fork's _clean_config() normalisation ...
+    # _clean_config() 在 compile_config.py（已 ported），不在 platform。
+    # platform 只負責 raw passthrough。
```

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

---

## 九、`v1_vllm021` 相對於 `roy/qaic-plugin` 的額外 Commit

`v1_vllm021` 在 `roy/qaic-plugin`（`6f5a832cf`）之上多了一個 commit：

```
68396e128  vllm021 update   (author: weimin023, 2026-06-02)
```

**改動統計：** 8 個文件，+950 行 / -31 行  
（其中 2 個是新增文件：`BRANCH_CHANGES_ZH.md`、`QAIC_COMMIT_SUMMARY.md`；實際代碼改動集中在 6 個 `vllm_qaic/*.py`）

---

### 9.1 `platform.py` — 修正 QaicPlatform 相容性

**`device_name` 從 `"qaic"` 改為 `"cpu"`**
```diff
-    device_name: str = "qaic"
+    device_name: str = "cpu"   # vLLM 用 device_name 建構 torch.device 字串，
                                 # qaic 不是 torch device type，改用 cpu 避免崩潰
```

**新增三個 Platform 方法**
```diff
+    def set_device(cls, device): return None       # QAIC 不需要 set torch device
+    def manual_seed_all(cls, seed): torch.manual_seed(seed)
+    def uses_host_device_handling(self) -> bool: return True
+    # uses_host_device_handling=True 告訴 vLLM 不要呼叫 torch.device("qaic")
```

**`check_and_update_config` 強化**
```diff
+    scheduler_config.async_scheduling = False      # 關閉 async scheduling
+    getattr(envs, "VLLM_USE_V1", True)             # 改用 getattr，避免 envs 沒有此屬性時崩潰
+    if not hasattr(model_config, "max_seq_len_to_capture"):
+        model_config.max_seq_len_to_capture = model_config.max_model_len   # 欄位不存在時補上預設值
```

---

### 9.2 `worker.py` — 修正 import 路徑與邊界處理

**修正 import 路徑**（對齊 vLLM 0.21 重構後的模組位置）
```diff
-from vllm.model_executor import set_random_seed
-from vllm.utils import cdiv
+from vllm.utils.math_utils import cdiv
+from vllm.utils.torch_utils import set_random_seed
```

**`init_cached_hf_modules` 改為容錯式呼叫**
```diff
-    from vllm.utils import init_cached_hf_modules
-    init_cached_hf_modules()
+    try:
+        from vllm.utils import init_cached_hf_modules
+    except ImportError:
+        init_cached_hf_modules = None
+    if init_cached_hf_modules is not None:
+        init_cached_hf_modules()
```

**其他修正**
```diff
-    self.device = self.device_config.device
+    self.device = self.device_config.device or torch.device("cpu")   # device 可能為 None

-    page_size = get_uniform_page_size(self.get_kv_cache_spec())
+    page_size = get_uniform_page_size(self.get_kv_cache_spec().values())   # 需傳 values()

-    ensure_kv_transfer_initialized(vllm_config)
+    if vllm_config.kv_transfer_config is not None:
+        ensure_kv_transfer_initialized(vllm_config, None)   # 加 None-guard + 新版 API 多一個參數
```

---

### 9.3 `model_runner.py` — 最多改動，新增三個關鍵 helper + 若干修正

**新增 `_postprocess_host_tensors()`**  
把 `GPUModelRunner.__init__` 分配的所有 GPU tensor buffer 替換成對應的 CPU tensor，
讓 QAIC host-side execution 不依賴 CUDA 記憶體：
```diff
+    def _postprocess_host_tensors(self) -> None:
+        # 把所有 CpuGpuBuffer.gpu 指向 .cpu
+        for value in vars(self).values():
+            if isinstance(value, CpuGpuBuffer):
+                value.gpu = value.cpu
+        # 把 input_batch 上的 *_cpu_tensor → 對應 device tensor
+        for key, value in vars(self.input_batch).items():
+            if key.endswith("_cpu_tensor") and isinstance(value, torch.Tensor):
+                replace_tensor(self.input_batch, key, key[:-11])
+        # block_table 的 CpuGpuBuffer 同樣處理
+        for block_table in self.input_batch.block_table.block_tables:
+            for value in vars(block_table).values():
+                if isinstance(value, CpuGpuBuffer):
+                    value.gpu = value.cpu
```

**新增 `_postprocess_cpu_kernels()`**  
用純 Python 迴圈替換 Triton `_compute_slot_mapping_kernel`，
讓 slot mapping 在沒有 CUDA/Triton 的 QAIC host 上也能運行：
```diff
+    def _postprocess_cpu_kernels(self) -> None:
+        # monkey-patch vllm.v1.worker.block_table 裡的 Triton kernel
+        # 改成純 Python 實作（同等語義，但跑在 CPU）
+        vllm.v1.worker.block_table._compute_slot_mapping_kernel = _PythonSlotMappingKernel()
```

**新增三個 stub override**
```diff
+    def _init_device_properties(self) -> None:
+        self.num_sms = 1    # GPUModelRunner 需要 SM count，QAIC 給個無害的值

+    def _sync_device(self) -> None:
+        return None          # qaicrt 呼叫是同步的，不需要額外 sync

+    def _to_list(self, sampled_token_ids) -> list[list[int]]:
+        if sampled_token_ids.device.type != "cpu":
+            sampled_token_ids = sampled_token_ids.cpu()
+        return sampled_token_ids.tolist()    # 確保 tensor 在 CPU 再轉 list
```

**`__init__` 中呼叫兩個新 helper**
```diff
     super().__init__(vllm_config, device)
+    self._postprocess_host_tensors()
+    self._postprocess_cpu_kernels()
```

**`max_seq_len_to_capture` 改為 `getattr` 容錯取值**
```diff
-    self.max_seq_len = self.model_config.max_seq_len_to_capture
+    self.max_seq_len = getattr(self.model_config, "max_seq_len_to_capture",
+                               self.model_config.max_model_len)
```

**`grammar_bitmask` 改為 `getattr` 容錯取值**
```diff
-    if scheduler_output.grammar_bitmask is not None:
+    if getattr(scheduler_output, "grammar_bitmask", None) is not None:
```

**新增 import**
```diff
+from vllm.v1.utils import CpuGpuBuffer
```

---

### 9.4 `model_loader.py` — bug fix + vocab size 對齊

**新增 `embed_input_ids` stub**（與 `compile_config.py` 同步）
```diff
+    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
+        raise NotImplementedError("QAIC executes token embeddings inside the compiled QPC...")
```

**KV buffer 初始化 bug fix：`skip_buffers` → `unskip_buffers`**
```diff
-    self.session.skip_buffers([x for x in self.session.input_names if x.startswith("past_")])
-    self.session.skip_buffers([x for x in self.session.output_names if x.endswith("_RetainedState")])
+    self.session.unskip_buffers([x for x in self.session.input_names if x.startswith("past_")])
+    self.session.unskip_buffers([x for x in self.session.output_names if x.endswith("_RetainedState")])
# 載入 QPC 後應「啟用」KV buffer，原來反向 skip 導致 KV cache 失效
```

**QPC vocab size 與 HF config 不一致時自動對齊**
```diff
+    if "logits" in self.session.binding_index_map:
+        qpc_vocab_size = logits_binding.dims[-1]
+        if qpc_vocab_size != self.vocab_size:
+            logger.warning("QPC logits vocab size differs from HF config...")
+            self.vocab_size = qpc_vocab_size          # 以 QPC 實際形狀為準
+            self.logits_processor = LogitsProcessor(self.vocab_size, ...)
```

**`ignore_batch_index=True` 時正確移除 `batch_index`**
```diff
+    self.decode_batch_inputs.pop("batch_index", None)
```

**QPC input_ids 寬度與 decode buffer 不一致時自動對齊**
```diff
+    qpc_input_width = input_ids_binding.dims[1]
+    if qpc_input_width != self.decode_batch_inputs["input_ids"].shape[1]:
+        self.decode_batch_inputs["input_ids"] = np.full((self.decode_bsz, qpc_input_width), -1, ...)
+        self.decode_batch_inputs["position_ids"] = np.full((self.decode_bsz, qpc_input_width), -1, ...)
```

---

### 9.5 `compile_config.py` — import 路徑修正 + DispatchPooler 相容性

**修正 import 路徑**（對齊 vLLM 0.21）
```diff
-from vllm.model_executor.layers.pooler import DispatchPooler, Pooler, PoolingType
-from vllm.model_executor.pooling_metadata import PoolingMetadata
-from vllm.sequence import IntermediateTensors, PoolerOutput
-from vllm.entrypoints.openai.serving_models import LoRAModulePath
-from peft import PeftConfig
+from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
+from vllm.v1.outputs import PoolerOutput
+from vllm.v1.pool.metadata import PoolingMetadata
+from vllm.entrypoints.openai.models.protocol import LoRAModulePath
# peft 移到使用時再 lazy import（避免非 LoRA 場景也需要安裝）
```

```diff
-from vllm.utils import cdiv
+from vllm.utils.math_utils import cdiv
```

**`DispatchPooler` 版本相容**
```diff
-    self._pooler = DispatchPooler({"encode": ..., "embed": ..., ...})
+    if hasattr(Pooler, "for_embed"):
+        self._pooler = DispatchPooler({"encode": ..., "embed": ..., ...})
+    else:
+        self._pooler = DispatchPooler({
+            **DispatchPooler.for_embedding(pooler_config).poolers_by_task,
+            **DispatchPooler.for_seq_cls(pooler_config).poolers_by_task,
+        })   # 新版 API 改用 for_embedding / for_seq_cls
```

**新增 `embed_input_ids` stub**
```diff
+    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
+        raise NotImplementedError("QAIC executes token embeddings inside the compiled QPC...")
```

**`max_seq_len_to_capture` 容錯取值**
```diff
-    "prefill_seq_len": vllm_config.model_config.max_seq_len_to_capture,
-    "ctx_len": vllm_config.scheduler_config.max_model_len,
+    max_seq_len_to_capture = getattr(model_config, "max_seq_len_to_capture", model_config.max_model_len)
+    "prefill_seq_len": max_seq_len_to_capture,
+    "ctx_len": vllm_config.model_config.max_model_len,   # 改從 model_config 取（非 scheduler_config）
```

**`peft` lazy import**
```diff
-    from peft import PeftConfig   # 原本在 top-level
+    # 移到 search_adapters_in_cache() 函數內部：
+    def search_adapters_in_cache(base_model_name):
+        from peft import PeftConfig   # lazy，只有用到 LoRA adapter 搜尋時才 import
```

---

### 9.6 `session.py` — 新增 alias

```diff
+QAICInferenceSession = DisAgg_QAICInferenceSession
# 補上 alias，讓 model_loader.py 的 import 不報錯
```

---

### 9.7 改動性質總結

| 文件 | 改動性質 |
|---|---|
| `platform.py` | API 相容修正（`device_name`、新 Platform 方法、`getattr` 容錯） |
| `worker.py` | import 路徑更新、邊界條件 bug fix（`device or cpu`、`kv_transfer_config` guard） |
| `model_runner.py` | **最核心**：新增 `_postprocess_host_tensors`（GPU→CPU buffer 替換）、`_postprocess_cpu_kernels`（Triton→Python slot mapping）、3 個 stub override |
| `model_loader.py` | Bug fix（`skip→unskip` KV buffer）、vocab size 對齊、input width 對齊 |
| `compile_config.py` | Import 路徑更新、`DispatchPooler` 版本相容、`max_seq_len_to_capture` 容錯 |
| `session.py` | 補 `QAICInferenceSession` alias |

這批改動的主題是：**讓 vllm 0.21 的 API 在沒有 CUDA/GPU 的 QAIC host 上能正常初始化**，  
核心突破是 `_postprocess_host_tensors` 和 `_postprocess_cpu_kernels` 兩個 post-init hook，
把 `GPUModelRunner` 預設分配的 GPU 資源全部替換成等效的 CPU 版本。

---

## 十、最新 `roy/qaic-plugin` 合入與 offline smoke 結果

2026-06-02 重新 fetch `origin/roy/qaic-plugin` 後，remote 最新 commit 為：

```
4338e8675  Keep QAIC row-count guards active
```

目前 `v1_vllm021` 已透過 merge commit 合入：

```
a38135ce8  Merge branch 'roy/qaic-plugin' of github.com:UnieAI/vllm into v1_vllm021
```

合入的 remote commits：

| Commit | 主題 | 評估 |
|---|---|---|
| `2759ca198` | Fix QAIC mixed decode prefill ordering | 值得合入。修正 QAIC runner 原本假設 decode request 一定位於 batch 前段的問題；新版改用 mask 分離 decode/prefill，再把 logits scatter 回 vLLM 預期順序。 |
| `4632ccea7` | Harden QAIC prompt logprob handling | 值得合入。`prompt_logprobs` 尚未支援時明確 `raise NotImplementedError`，避免走到錯誤或 silent corruption。 |
| `4338e8675` | Keep QAIC row-count guards active | 值得合入。保留 block id / output row-count guard，若排程 tokens 與 QPC 回傳列數不一致會直接報錯。 |

**實際改動範圍：**

```
vllm-qaic/vllm_qaic/model_runner.py | 152 ++++++++++++++++++++++++++++++------
```

合入後驗證：

```bash
.venv-vllm021/bin/python -m py_compile vllm-qaic/vllm_qaic/model_runner.py
```

vLLM offline smoke：

```bash
VLLM_USE_V1=1 \
HF_HUB_OFFLINE=1 \
VLLM_QAIC_QPC_PATH=/workspace/weiminc/GPT2_Static_QPC \
VLLM_CONFIG_ROOT=/tmp/vllm-config \
XDG_CACHE_HOME=/tmp/xdg-cache \
.venv-vllm021/bin/python -c "from vllm import LLM, SamplingParams; llm=LLM(model='/workspace/models/Qwen2.5-0.5B-Instruct', max_model_len=128, max_num_seqs=1, enforce_eager=True, trust_remote_code=True); out=llm.generate(['Hello'], SamplingParams(max_tokens=4, temperature=0)); print('OUTPUT:', out[0].outputs[0].text)"
```

關鍵 log：

```text
Initializing a V1 LLM engine (v0.20.1rc1.dev701+g6f5a832cf)
```

最後 smoke 可以完成 generate 並 shutdown：

```text
OUTPUT: ,ion    Un
```

### 10.1 為什麼目前 offline inference 輸出不正常

目前輸出 `,ion    Un` 不正常，根因不是 V1 engine 沒啟動，而是 **QPC 與 HF model/tokenizer 不匹配**。

現在 smoke 使用的是：

| 元件 | 實際值 |
|---|---|
| HF model / tokenizer | `/workspace/models/Qwen2.5-0.5B-Instruct` |
| QPC | `/workspace/weiminc/GPT2_Static_QPC` |
| QPC logits vocab | `50257`（GPT2 vocab） |
| Qwen2.5 vocab | `151936` |
| QPC input shape | static `[1, 128]` |

所以目前能證明的是：

1. `vllm-qaic` plugin 可被 vLLM 載入。
2. V1 engine 有啟動。
3. QAIC QPC session 可以跑 prefill/decode。
4. vLLM offline generate path 可以完成。

但不能拿這個輸出判斷模型品質，因為 GPT2 QPC 的 logits 被 Qwen tokenizer 解碼，token id / vocab 語意完全對不上。

### 10.2 怎麼讓 offline inference 輸出正常

要讓輸出正常，必須讓下列三者一致：

1. **HF model config**
2. **tokenizer**
3. **QPC**

可選做法有兩條：

| 路線 | 做法 | 適合情境 |
|---|---|---|
| A. 用 GPT2 HF model/tokenizer 搭現在的 GPT2 QPC | 準備本地 GPT2 HF 目錄，offline script 的 `model=` 改成 GPT2 model path，`max_model_len=128` | 最快驗證現有 `/workspace/weiminc/GPT2_Static_QPC` 是否能產生正常 GPT2 文字 |
| B. 重新 compile Qwen2.5 QPC | 用 `/workspace/models/Qwen2.5-0.5B-Instruct` 編出對應 QPC，vocab/shape/hidden config 都與 Qwen 一致 | 要正式驗證 Qwen2.5 offline inference |

目前 repo/機器上只看到：

```text
/workspace/models/Qwen2.5-0.5B-Instruct
/workspace/weiminc/GPT2_Static_QPC
```

因此現狀是「有 Qwen HF model，但 QPC 是 GPT2」。要讓輸出正常，下一步要嘛補一份 GPT2 HF model 到本地，要嘛 compile 一份 Qwen2.5 QPC。  
不建議再靠 plugin 端硬改 vocab 或 tokenizer 去「湊」，那只能讓 buffer shape 對齊，不能讓語意正確。

---

## 十一、GPUModelRunner / QaicModelRunner 繼承關係圖

### 11.1 ModelRunner 繼承鏈

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  vLLM 0.21 upstream                        vllm-qaic plugin                │
│  (vllm/v1/worker/)                         (vllm_qaic/)                    │
│                                                                             │
│  ┌──────────────────────┐                                                   │
│  │  LoRAModelRunnerMixin│ ◄──── mixin (LoRA add/remove/list/pin)            │
│  └──────────┬───────────┘                                                   │
│             │                                                               │
│  ┌──────────────────────────┐                                               │
│  │KVConnectorModelRunnerMixin│ ◄── mixin (KV transfer / disagg prefill)     │
│  └──────────┬───────────────┘                                               │
│             │                                                               │
│  ┌──────────────────────────┐                                               │
│  │ECConnectorModelRunnerMixin│ ◄── mixin (expert-cache connector)           │
│  └──────────┬───────────────┘                                               │
│             │                                                               │
│             │  (多重繼承)                                                    │
│             ▼                                                               │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │           GPUModelRunner                             │                  │
│  │  (vllm/v1/worker/gpu_model_runner.py, ~7000 lines)   │                  │
│  │                                                      │                  │
│  │  class GPUModelRunner(                               │                  │
│  │      LoRAModelRunnerMixin,                           │                  │
│  │      KVConnectorModelRunnerMixin,                    │                  │
│  │      ECConnectorModelRunnerMixin,                    │                  │
│  │  )                                                   │                  │
│  │                                                      │                  │
│  │  ・execute_model()     ← 0.21 新架構：可能先回傳 None  │                  │
│  │  ・sample_tokens()     ← 0.21 新增，獨立採樣步驟      │                  │
│  │  ・_prepare_inputs()   ← 內部 InputBatch 處理         │                  │
│  │  ・load_model()        ← (load_dummy_weights=False)   │                  │
│  │  ・get_kv_cache_spec() ← dict[str, KVCacheSpec]       │                  │
│  │  ・initialize_kv_cache(kv_cache_config, is_profiling) │                  │
│  └──────────────────────────┬───────────────────────────┘                  │
│                             │ 繼承                                          │
│                             ▼                                               │
│                  ┌────────────────────────────────────────────┐            │
│                  │           QaicModelRunner                  │            │
│                  │  (vllm_qaic/model_runner.py)               │            │
│                  │                                            │            │
│                  │  class QaicModelRunner(GPUModelRunner)     │            │
│                  │                                            │            │
│                  │  【override 方法】                          │            │
│                  │  ・__init__()          ← 移除 spec_type 參數│            │
│                  │  ・execute_model()     ← 同步完整路徑        │            │
│                  │  ・load_model()        ← 呼叫 load_qaic_model│           │
│                  │  ・get_kv_cache_spec() ← FullAttentionSpec  │            │
│                  │  ・initialize_kv_cache() ← 只存 config      │            │
│                  │  ・sample_tokens()     ← raise（不使用）     │            │
│                  │  ・_init_device_properties() ← num_sms=1    │            │
│                  │  ・_sync_device()      ← no-op              │            │
│                  │  ・_to_list()          ← 確保 CPU tensor    │            │
│                  │  ・_postprocess_host_tensors() ← GPU→CPU    │            │
│                  │  ・_postprocess_cpu_kernels()  ← Triton→Py  │            │
│                  │                                            │            │
│                  │  【UnieAI 新增方法（ngram）】               │            │
│                  │  ・_pack_decode_batch()                     │            │
│                  │  ・_qaic_rejection_sample()                 │            │
│                  │  ・_qaic_is_greedy_request()                │            │
│                  │  ・_qaic_rejection_sample_greedy_req()      │            │
│                  │  ・_qaic_rejection_sample_random_req()      │            │
│                  │  ・_qaic_target_probs_for_req()             │            │
│                  │  ・_qaic_apply_top_k_top_p()                │            │
│                  │  ・_qaic_sample_from_probs()                │            │
│                  └────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 11.2 Worker 繼承鏈

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  vLLM 0.21 upstream                        vllm-qaic plugin                │
│  (vllm/v1/worker/)                         (vllm_qaic/)                    │
│                                                                             │
│  ┌──────────────────────────────────────────────────┐                      │
│  │                WorkerBase                        │                      │
│  │  (vllm/v1/worker/worker_base.py)                 │                      │
│  │                                                  │                      │
│  │  ・init_device()                                  │                      │
│  │  ・load_model()                                   │                      │
│  │  ・execute_model()                                │                      │
│  │  ・determine_available_memory()                   │                      │
│  │  ・get_kv_cache_spec()                            │                      │
│  │  ・initialize_from_config()                       │                      │
│  │  ・compile_or_warm_up_model()                     │                      │
│  └──────────────┬───────────────────────────────────┘                      │
│                 │                                                           │
│        ┌────────┴────────────────────────────────┐                         │
│        │                                         │                         │
│        ▼                                         ▼                         │
│  ┌─────────────────────┐             ┌──────────────────────────────────┐  │
│  │  Worker (GPU)       │             │       QaicWorker                 │  │
│  │  (gpu_worker.py)    │             │  (vllm_qaic/worker.py)           │  │
│  │                     │             │                                  │  │
│  │  CUDA/GPU 路徑      │             │  class QaicWorker(WorkerBase)    │  │
│  │                     │             │                                  │  │
│  │                     │             │  ・init_device()  ← qaicrt 初始化 │  │
│  │                     │             │  ・load_model()   ← 呼叫 QaicMR  │  │
│  │                     │             │  ・execute_model() ← 呼叫 QaicMR │  │
│  │                     │             │  ・determine_available_memory()   │  │
│  │                     │             │    ← 依 max_num_seqs 計算         │  │
│  │                     │             │  ・sample_tokens() ← raise       │  │
│  │                     │             │                                  │  │
│  │                     │             │  【持有】                         │  │
│  │                     │             │  self.model_runner: QaicModelRunner  │
│  └─────────────────────┘             └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 11.3 Model（nn.Module）繼承鏈

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PyTorch / vLLM interfaces              vllm-qaic plugin                   │
│                                                                             │
│  ┌────────────────┐   ┌────────────────┐                                   │
│  │  torch.nn.Module│   │  SupportsLoRA  │ ← vllm Protocol (interfaces.py)  │
│  │  (PyTorch)     │   │  (Protocol)    │                                   │
│  └───────┬────────┘   └───────┬────────┘                                   │
│          │                    │                                             │
│          └────────┬───────────┘                                             │
│                   │ 多重繼承                                                 │
│                   ▼                                                         │
│        ┌──────────────────────────────────────────┐                        │
│        │       QaicCausalLM  (model_loader.py)    │                        │
│        │                                          │                        │
│        │  class QaicCausalLM(nn.Module, SupportsLoRA)                      │
│        │                                          │                        │
│        │  ・forward()   ← 呼叫 _run_prefill/decode │                        │
│        │  ・load_model() ← 建立 QAICInferenceSession                        │
│        │  ・_run_prefill()  ← chunked prefill on AIC│                       │
│        │  ・_run_decode()   ← 2D decode batch       │                       │
│        │  ・kv_cache_info() ← 從 QPC binding 讀 shape│                      │
│        │                                          │                        │
│        │  【持有】                                  │                        │
│        │  self.session: QAICInferenceSession       │                        │
│        └──────────────────────────────────────────┘                        │
│                                                                             │
│  ┌────────────────┐                                                         │
│  │  torch.nn.Module│                                                        │
│  └───────┬────────┘                                                         │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────┐                         │
│  │     QaicCausalLM  (compile_config.py)         │                         │
│  │  ※ 注意：同名但不同檔案，功能有部分重疊            │                         │
│  │                                               │                         │
│  │  class QaicCausalLM(nn.Module)                │                         │
│  │  ・forward()  ← 直接呼叫 self.model.run()      │                         │
│  │  ・sample()   ← 呼叫 Sampler                   │                         │
│  │  ・generate_proposals() ← turbo/medusa 投機     │                         │
│  │  ・process_logits()     ← target/turbo 分支    │                         │
│  └───────────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 11.4 整體元件協作關係

```
  vLLM Scheduler
       │
       │ SchedulerOutput
       ▼
  ┌─────────────────┐
  │   QaicWorker    │  ← WorkerBase
  │                 │
  │  execute_model()│
  └────────┬────────┘
           │ 委派
           ▼
  ┌────────────────────────────────┐
  │      QaicModelRunner           │  ← GPUModelRunner
  │                                │    (LoRAMixin, KVConnMixin, ECConnMixin)
  │  1. _prepare_inputs()          │
  │     └─ 從 InputBatch 組 numpy  │
  │  2. model.forward()            │
  │     └─ prefill / decode        │
  │  3. _qaic_rejection_sample()   │  ← UnieAI ngram（CPU 版 RejectionSampler）
  │  4. 回傳 ModelRunnerOutput     │
  └────────┬───────────────────────┘
           │ 呼叫
           ▼
  ┌──────────────────────────┐
  │     QaicCausalLM         │  ← nn.Module, SupportsLoRA
  │   (model_loader.py)      │
  │                          │
  │  _run_prefill()          │
  │  _run_decode()           │
  └────────┬─────────────────┘
           │ 呼叫
           ▼
  ┌──────────────────────────┐
  │  QAICInferenceSession    │  ← DisAgg_QAICInferenceSession
  │   (session.py)           │
  │                          │
  │  qaicrt.Context/Queue    │
  │  qaicrt.Program          │
  │  qaicrt.ExecObj          │
  └────────┬─────────────────┘
           │ on-device 執行
           ▼
  ┌──────────────────────────┐
  │  Qualcomm AI 100 (AIC)   │
  │  QPC（預編譯模型圖）       │
  │  KV cache（常駐卡上）      │
  └──────────────────────────┘
```

---

### 11.5 v0.10.1 vs 0.21 的關鍵差異（繼承角度）

| 面向 | v0.10.1（fork 基線） | vLLM 0.21（目標） |
|---|---|---|
| `GPUModelRunner` 基類大小 | ~2,000–3,000 行 | **7,368 行** |
| 基類 mixin | 無 | `LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin`、`ECConnectorModelRunnerMixin` |
| `execute_model` 架構 | 單一方法：prep → forward → sample → 回傳 | 可能先回傳 `None`，由 `sample_tokens()` 獨立採樣 |
| `__init__` 第三參數 | `speculative_model_type=None` | 已移除，內部自行推導 |
| `initialize_kv_cache` 簽名 | `(kv_cache_config)` | `(kv_cache_config, is_profiling=False)` |
| host input arrays | `self.positions_np`、`self.cu_num_tokens`、`self.num_decodes` 等 | **已全部移除**，改用 `InputBatch` |
| GPU buffer | 由基類在 `__init__` 分配 CUDA tensor | 同上，`QaicModelRunner` 需在 init 後用 `_postprocess_host_tensors()` 改回 CPU |
| Triton kernel | N/A（fork 本來就是 CPU host） | `_compute_slot_mapping_kernel` 為 Triton；`QaicModelRunner` 用 `_postprocess_cpu_kernels()` monkey-patch 成純 Python |

---

## 十二、單卡 Qwen2.5-7B offline smoke：目前結果

使用者確認目前 QAIC device group 只有 `0`，因此不能使用雙卡 QPC：

```text
/root/.cache/qeff_models/Qwen2ForCausalLM/Qwen2ForCausalLM-758cc60f519aa7d9/qpc-15e3af08f196f49b/qpc
```

該 QPC 是 `device_group=[0,1]` / `num_devices=2`，在單卡環境會報：

```text
Device 0 activate failed: Failed to access P2P device
Device 1 activate failed: Failed to access P2P device
RuntimeError: ExecObjFailed to create ExecObj
```

### 12.1 可用的單卡 QPC

已確認兩份單卡 QPC：

| QPC | compile config | smoke 結果 |
|---|---|---|
| `/root/.cache/qeff_models/Qwen2ForCausalLM/Qwen2ForCausalLM-758cc60f519aa7d9/qpc-1c76199bcb608b83/qpc` | `num_devices=1`、`full_batch_size=96`、`mxint8_kv_cache=True`、`kv_cache_dtype=fp8` | V1 engine 可啟動並完成 generate；中途曾因 KV buffer path 錯誤出現重複 token |
| `/root/.cache/qeff_models/Qwen2ForCausalLM/Qwen2ForCausalLM-758cc60f519aa7d9/qpc-30b1c39ccba6a4fc/qpc` | `num_devices=1`、`full_batch_size=96`、`mxint8_kv_cache=False`、`kv_cache_dtype=auto` | V1 engine 可啟動並完成 generate；修正後輸出正常 |

HF model / tokenizer 解析到：

```text
/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

### 12.2 本輪修補

為了讓 Python offline API 等價於原 fork CLI，本輪補了三個 0.21 bridge / runtime shape 問題：

| 檔案 | 修補 |
|---|---|
| `vllm-qaic/vllm_qaic/platform.py` | 從 `additional_config["max_seq_len_to_capture"]` 寫回 `model_config.max_seq_len_to_capture`，避免 `prefill_seq_len` 被錯算成 `max_model_len=2048` |
| `vllm-qaic/vllm_qaic/model_loader.py` | logits vocab size 優先從 QPC allowed shapes 取最後一維，避免把 `96 * vocab_size` 誤當 vocab size |
| `vllm-qaic/vllm_qaic/session.py` | `unskip_buffers()` 對 retained-state buffer 優先使用 allowed shapes 的最大維度，避免 output retained-state shape 停在 binding 預設 batch 1 |
| `vllm-qaic/vllm_qaic/model_loader.py` | decode output row selection 後續再修正為 active compact row；詳見 12.5 |
| `vllm-qaic/vllm_qaic/model_loader.py` | 非 disagg 模式對齊 fork：`past_*` input 與 `_RetainedState` output 必須 `skip_buffers()`，不能 `unskip_buffers()` |

這些修補後，以下條件已成立：

- V1 engine 啟動成功。
- `prefill_seq_len` 正確為 `32`。
- 單卡 QPC 可完成 `generate()`，process exit code 為 `0`。
- 之前的 logits shape / retained-state shape `setData` error 已消失。
- 單卡 fp16 KV QPC 的 Qwen chat prompt 輸出已恢復正常。

### 12.3 根因與修正結果

中途語義輸出曾不正常。使用 Qwen chat template：

```text
請用一句話介紹台北。
```

錯誤狀態下，單卡 fp16 KV QPC 輸出：

```text
台北台北台北台北台北台北台北台北台北

``2000000000000
```

根因是 0.21 port 的非 disagg path 和 fork 行為不一致：

- fork 的 `qaic_v1.py` 在非 disagg 模式會 `skip_buffers()` 掉所有 `past_*` input 和 `_RetainedState` output。
- 0.21 port 曾改成 `unskip_buffers()`，導致 runtime 走錯 KV retained-state buffer path，雖然 engine 能跑完，但 decode 產生重複 token。

修正後使用同一個 prompt、同一份 QPC：

```text
OUTPUT: 台北是台灣的首都，一個融合了現代與傳統、繁忙與寧靜的國際大都市。
```

結論：單卡 Qwen2.5-7B offline inference 在 0.21 port 上已可正常完成，且輸出語義正常。

### 12.4 code diff 摘要

本輪實際 code diff 集中在三個檔案。

#### `vllm-qaic/vllm_qaic/model_loader.py`

1. 非 disagg 模式改回 fork 的 KV buffer 行為。

```diff
-            self.session.unskip_buffers(
+            self.session.skip_buffers(
                 [x for x in self.session.input_names if x.startswith("past_")]
             )
-            self.session.unskip_buffers(
+            self.session.skip_buffers(
                 [x for x in self.session.output_names if x.endswith("_RetainedState")]
             )
```

這是修復重複 token 的關鍵。非 disagg QPC 使用 card-resident KV state，不應把 `past_*` input / `_RetainedState` output 當一般 runtime buffer 傳入傳出；舊 fork 也是 `skip_buffers()`。

2. logits vocab size 從 QPC allowed shapes 讀，而不是直接讀 binding flat dims。

```diff
-            logits_binding = self.session.bindings[
-                self.session.binding_index_map["logits"]]
-            qpc_vocab_size = logits_binding.dims[-1]
+            logits_shapes = self.session.get_bindings_shapes(["logits"]).get(
+                "logits", [])
+            qpc_vocab_size = logits_shapes[0][-1] if logits_shapes else (
+                self.session.bindings[
+                    self.session.binding_index_map["logits"]].dims[-1])
```

這避免把 `96 * vocab_size` 誤判成 vocab size，造成 logits buffer shape 變成 `1 x 1 x 14598144`。

3. decode output row selection 的中間修補（後續已被 12.5 取代）。

```diff
         outputs = self.session.run(self.decode_batch_inputs)
         logits: np.ndarray = outputs["logits"]
+        decode_rows = np.asarray(batch_indices[:num_decodes], dtype=np.int64)
         if decode_lengths is not None:
             return np.concatenate(
-                [logits[i, :num_tokens] for i, num_tokens in enumerate(decode_lengths)],
+                [
+                    logits[row, :num_tokens]
+                    for row, num_tokens in zip(decode_rows, decode_lengths)
+                ],
                 axis=0)
         if self.is_spec_decode_target_model:
-            return logits[:num_decodes, 0]
-        return logits[:num_decodes].squeeze(1)
+            return logits[decode_rows, 0]
+        return logits[decode_rows].squeeze(1)
```

這個版本後續被證明不完整：對 QPC decode output 而言，回傳 logits 的 row order 仍是 compact active decode rows；`batch_index` 是 QPC 內部更新 retained-state slot 用，不應用來索引回傳 logits。最終修正見 12.5。

#### `vllm-qaic/vllm_qaic/platform.py`

Python offline API 沒有 CLI 的 `--max-seq-len-to-capture` keyword，因此 QAIC plugin 需要從 `additional_config` 接住這個值。

```diff
         cache_config = vllm_config.cache_config
         model_config = vllm_config.model_config
+        max_seq_len_to_capture = qaic_cfg.get("max_seq_len_to_capture")
+        if max_seq_len_to_capture is not None:
+            model_config.max_seq_len_to_capture = int(max_seq_len_to_capture)
```

否則 `_get_qaic_compile_config()` 會把 `prefill_seq_len` 算成 `max_model_len=2048`，和 QPC 的 `seq_len=32` 不匹配。

#### `vllm-qaic/vllm_qaic/session.py`

`unskip_buffers()` 改成使用 QPC allowed shapes 的最大維度配置 retained-state buffer。

```diff
-            dims: List[int] = binding.dims
+            allowed_dims = [
+                allowed_shape[binding.index][1]
+                for allowed_shape in self.allowed_shapes
+                if binding.index < len(allowed_shape)
+            ]
+            dims: List[int] = max(
+                allowed_dims, key=lambda shape: int(np.prod(shape))
+            ) if allowed_dims else binding.dims
```

這是防守性修補：雖然非 disagg 正確路徑已經會 `skip_buffers()`，但 disagg / 其他 retained-state 用法仍可能需要 `unskip_buffers()`，此時 shape 應以 QPC allowed shapes 為準。

### 12.5 修復 sequential offline request 第二筆亂碼 / `!` 連發

#### 錯誤現象

在同一個 `LLM` instance 內連續執行多個 offline request 時，第一筆 request 輸出正常，但第二筆開始會出現亂碼或大量 `!`：

```text
REQ 1 OUTPUT: ' 成都是四川省的省會，是一座歷史文化與現代文明交融的城市。'
REQ 2 OUTPUT: ' �!!!!!!!!!!!!!!!!!!!!!!!'
REQ 3 OUTPUT: ' ______!!!!!!!!!!!!!!!!!!!!!!!'
REQ 4 OUTPUT: ' !!!!!!!!!!!!!!!!!!!!!!!'
```

此問題同時出現在：

- fp16 KV QPC：`qpc-30b1c39ccba6a4fc/qpc`
- fp8 / mxint8 KV QPC：`qpc-1c76199bcb608b83/qpc`
- offline sequential `llm.generate([prompt])` 多次呼叫
- online server 的多次 request 路徑

#### 排除過程

已確認不是以下原因：

- 不是 sampler 問題：argmax logits 與 sampled token 一致，錯誤已存在於 QPC logits。
- 不是 prompt packing 問題：debug 顯示第二筆進 QPC 前的 `input_ids` / `positions` 仍正確。
- 不是 vLLM block id / `batch_index` 重用問題：改成每個 request 使用新的 QAIC slot 後仍錯。
- 不是單純 `past_key/value` 沒有補零：prefill 明確餵 zero past inputs 後仍錯，且會引入額外 runtime 風險。

debug 期間看到第二筆 prefill 進 QPC 前資料是乾淨的：

```text
QAIC_DEBUG prefill req_ids=['1-8748b1d3']
batch_indices=[1]
token_counts=[7]
input_ids=[100792, 11622, 104670, 86077, 111748, 106756, 1773]
positions=[0, 1, 2, 3, 4, 5, 6]
```

#### 根因

0.21 port 在 `_run_decode()` 內把 QPC decode output 當成「以 `batch_index` 排列的 full batch logits」讀取：

```python
decode_rows = np.asarray(batch_indices[:num_decodes], dtype=np.int64)
return logits[decode_rows].squeeze(1)
```

但參考舊 `qserve_model_runner.py` 後確認，QPC decode output 的有效 logits row order 是 **compact active decode row order**，不是 `batch_index` order。

也就是說：

- `batch_index` 用來告訴 QPC 要更新哪個 retained-state slot。
- `outputs["logits"]` 回傳時，active requests 仍在 row `0..num_decodes-1`。
- 第二筆 request 的 `batch_index=1` 時，舊寫法會讀 `logits[1]`，等於讀到 inactive row，造成 `�` / `!` 連發。

另外 inactive decode rows 的 `batch_index` 也對齊舊 runner：不要填 `-1`，而是填入本輪未使用的合法 slot id；inactive rows 的 `input_ids` / `position_ids` 仍維持 `-1`。

#### 最終 code diff

`vllm-qaic/vllm_qaic/model_loader.py`

```diff
         if not self.ignore_batch_index:
-            self.decode_batch_inputs["batch_index"][:num_decodes,0] = batch_indices
+            self.decode_batch_inputs["batch_index"][:num_decodes,
+                                                     0] = batch_indices
             if num_decodes < self.decode_bsz:
-                self.decode_batch_inputs["batch_index"][num_decodes:] = -1
+                active_batch_indices = set(batch_indices[:num_decodes])
+                inactive_batch_indices = [
+                    idx for idx in range(self.decode_bsz)
+                    if idx not in active_batch_indices
+                ]
+                self.decode_batch_inputs["batch_index"][
+                    num_decodes:, 0] = inactive_batch_indices[:(
+                        self.decode_bsz - num_decodes)]

         outputs = self.session.run(self.decode_batch_inputs)
         logits: np.ndarray = outputs["logits"]
-        decode_rows = np.asarray(batch_indices[:num_decodes], dtype=np.int64)
+        decode_rows = np.arange(num_decodes, dtype=np.int64)
```

#### 驗證結果

fp16 KV QPC offline multiple request 通過：

```text
REQ 1 OUTPUT: ' 成都是四川省的省會，是一座歷史文化與現代文明交融的城市。'
REQ 2 OUTPUT: ' 台北是台灣的首都，一個融合現代與傳統、多元文化與美食的城市。'
REQ 3 OUTPUT: ' ________．____\nA. Paris\nB. London\nC. Berlin\nD. Rome\n答案:\nA'
REQ 4 OUTPUT: ' 人工智慧是指由電腦模擬和複製人類智慧和行為的技術。'
```

fp8 / mxint8 KV QPC offline multiple request 也通過：

```text
REQ 1 OUTPUT: ' 成都是四川省的省會，是一座歷史文化與現代文明交融的城市。'
REQ 2 OUTPUT: ' 台北是台灣的首都，一個融合現代與傳統、多元文化與美食的城市。'
REQ 3 OUTPUT: ' ________．____\nA. Paris\nB. London\nC. Berlin\nD. Rome\n答案:\nA'
REQ 4 OUTPUT: ' 人工智慧是指由電腦或其他機器模擬、延伸和拓展人類智能的一門技術。'
```

結論：sequential offline / online multi-request 亂碼問題的主因是 decode logits row selection 錯誤。修正後 fp16 與 fp8 QPC 的多筆 request 輸出均正常。
