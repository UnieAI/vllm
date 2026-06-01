# QAIC → vLLM 0.21+ 迁移:完整改动清单(中文版)

本文件是把 Qualcomm QAIC 后端(fork `UnieAI/vllm@v1_ngram`,基线 **v0.10.1**)
迁移到 **vLLM 0.21/0.22**、并重构成树外插件 `vllm-qaic` 的**完整工单**。

涵盖 Qualcomm "qaic patch"(`bd90d0d`)动过的**每一个文件**(45 个新增 + 40 个
修改的核心文件),加上 UnieAI 自己的改动,以及 `GPUModelRunner` 的新旧深度对照。

> 英文版见 `MIGRATION_GPUModelRunner_old_vs_new.md`,两者内容一致。
> **可信度:** 标 `⚠待确认` 的分类是根据「文件名 + diff 体积」推断、尚未逐行读过,
> 采用前请先核实;其余均有读过代码支撑。术语保留英文以免歧义。

## 图例(5 个桶)

| 标记 | 含义 | 去向 |
|---|---|---|
| 🟢 **PLUGIN** | 可用官方扩展点实现,无需改核心 | `vllm-qaic` 包 |
| 🔵 **PORT** | 必须搬过来的 QAIC 逻辑(继承/复制) | `vllm-qaic` 包 |
| 🟠 **CORE-PATCH** | 不可避免的极小核心补丁,或向上游提 PR | 一个薄补丁 / PR |
| ⚪ **DROP-V0** | 只有 V0 引擎用到;V1-only 部署不需要 | 不迁移 |
| ⚫ **OPTIONAL** | 独立功能(gpt-oss / 拆分式 / pooling / 多模态)——需要才做 | 按功能 |

---

## 1. 策略概览

fork 改了约 40 个核心文件。插件化 + 只跑 V1,会大幅缩减工作量:

- **约一半是 ⚪ DROP-V0**(V0 的 scheduler / block manager / sequence / engine /
  spec-decode 基础设施)。V1-only 部署直接丢弃。
- **其余大多是 🟢 PLUGIN**(platform、量化、KV connector、模型注册、CLI 参数 →
  改走 `--additional-config`)。
- **真正绕不开的核心补丁(🟠)只剩很短一张表** —— 见 §6。
- **真正的工程量是 🔵 PORT**,集中在 `QaicModelRunner`(§4)。

---

## 2. qaic patch 新增的文件(45)

### 2a. 搬进插件

| 文件 | 桶 | 备注 |
|---|---|---|
| `vllm/v1/worker/qaic_model_runner.py` | 🔵 PORT | → `vllm_qaic/model_runner.py`。**最难的一块**(§4)。 |
| `vllm/v1/worker/qaic_worker.py` | 🔵 PORT | → `vllm_qaic/worker.py`。继承 `WorkerBase`。 |
| `vllm/model_executor/model_loader/qaic_v1.py` | 🔵 PORT | → `vllm_qaic/model_loader.py`。QPC 加载/编译。 |
| `vllm/model_executor/model_loader/qaic.py` | 🔵 PORT | → `vllm_qaic/compile_config.py`。`_clean_config()`。 |
| `vllm/model_executor/model_loader/qaic_session_np.py` | 🔵 PORT | → `vllm_qaic/session.py`。**BSD-3-Clause**(其余文件是专有)。纯 numpy+qaicrt。 |
| `vllm/platforms/qaic.py` | 🟢→🔵 | → `vllm_qaic/platform.py`(已起草,OOT enum)。 |
| `vllm/model_executor/layers/quantization/qaic_quant.py` | 🟢 PLUGIN | → `vllm_qaic/quant.py`,用 `register_quantization_config`。 |
| `vllm/distributed/kv_transfer/kv_connector/qaic_connector.py` | ⚫ OPTIONAL | 仅拆分式 prefill/decode 需要。用 factory 注册。 |
| `vllm/model_executor/models/qaic_custom_mm_processor.py` | ⚫ OPTIONAL | 仅多模态模型需要。 |

### 2b. V0 的 QAIC 文件 —— V1-only 部署丢弃

| 文件 | 桶 |
|---|---|
| `vllm/worker/qaic_model_runner.py` | ⚪ DROP-V0 |
| `vllm/worker/qaic_worker.py` | ⚪ DROP-V0 |
| `vllm/worker/qaic_pooling_model_runner.py` | ⚪ DROP-V0(需要 pooling 则 ⚫) |
| `vllm/spec_decode/qaic_multi_step_worker.py` | ⚪ DROP-V0(这是 V0 的 draft-model 投机,不是 V1 ngram 路径) |
| `vllm/core/block/qaic_prefix_caching_block.py` | ⚪ DROP-V0 |

### 2c. patch 重新加回的 V0 spec-decode 基础设施 —— V1 丢弃

`vllm/spec_decode/*`(重加的:`spec_decode_worker.py`、`multi_step_worker.py`、
`ngram_worker.py`、`top1_proposer.py`、`batch_expansion.py`、`mqa_scorer.py`、
`draft_model_runner.py`、`target_model_runner.py`、`medusa_worker.py`、
`mlp_speculator_worker.py`、`smaller_tp_proposer_worker.py`、
`proposer_worker_base.py`、`interfaces.py`、`metrics.py`、`util.py`、
`__init__.py`),加上 `vllm/model_executor/layers/rejection_sampler.py`、
`spec_decode_base_sampler.py`、`typical_acceptance_sampler.py`,以及
`vllm/engine/output_processor/multi_step.py` —— 全部 **⚪ DROP-V0**。V1 的投机
解码在 `vllm/v1/spec_decode/`,由上游提供。

### 2d. 示例 —— 仅供参考(⚪/⚫,按需改)
`examples/offline_inference/qaic*.py`(11 个)。

---

## 3. 修改的核心文件(40)—— 完整分类

### 3a. 🟢 PLUGIN —— 用扩展点替代,不改核心

| 文件 | +行 | 原本干啥 | 插件替代 |
|---|---|---|---|
| `vllm/platforms/__init__.py` | 33 | qaic 检测 | `vllm.platform_plugins` 入口点 |
| `vllm/engine/arg_utils.py` | 74 | `--override-qaic-config`、`--device-group` | `--additional-config` 字典 |
| `vllm/config/__init__.py` | 126 | `override_qaic_config` / `device_group` 字段 + 校验 | 从 `additional_config` 读 |
| `vllm/model_executor/layers/quantization/__init__.py` | 9 | 注册 `mxfp6` | `register_quantization_config("mxfp6")` |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 34 | 注册 qaic connector | `KVConnectorFactory.register_connector()` |
| `vllm/model_executor/models/registry.py` | 10 | 注册 qaic 模型 | `general_plugins` 里 `ModelRegistry.register_model()` |
| `vllm/transformers_utils/configs/__init__.py` | 10 | 注册自定义 HF config | 在 `general_plugins` 注册 |
| `vllm/envs.py` | 8 | `VLLM_QAIC_*` 环境变量 | 插件自己读 `os.environ` |

### 3b. 🟠 CORE-PATCH —— 不可避免的小补丁(或上游 PR)。见 §6。

| 文件 | +行 | 为什么插件做不到 |
|---|---|---|
| `vllm/config/cache.py` | 12 | 往**封闭的 `CacheDType` Literal** 里加 `"mxint8"` + 校验。枚举无法从外部扩展。 |
| `vllm/platforms/interface.py` | 10 | 加 Platform 基类方法/枚举。⚠待确认:多数 0.21 已有(`is_kv_cache_dtype_supported`、`OOT`),可能已是空操作。 |
| `vllm/_custom_ops.py` | 8 | ⚠待确认:大概率是让 custom-op 在无 CUDA 时优雅降级。0.21 上可能已不需要。 |
| `vllm/transformers_utils/config.py` | 15 | ⚠待确认:qaic 模型的 HF config 加载钩子。可能改用 configs registry 即可。 |

### 3c. ⚪ DROP-V0 —— V1-only 部署不需要

| 文件 | +行 | 备注 |
|---|---|---|
| `vllm/core/scheduler.py` | 37 | V0 scheduler(V1 用 `vllm/v1/core`) |
| `vllm/core/block_manager.py` | 17 | V0 |
| `vllm/core/block/cpu_gpu_block_allocator.py` | 46 | V0 block allocator |
| `vllm/engine/llm_engine.py` | 54 | V0 engine |
| `vllm/engine/output_processor/interfaces.py` | 28 | V0 multi-step 输出 |
| `vllm/sequence.py` | 36 | V0 序列结构 |
| `vllm/worker/worker_base.py` | 4 | V0 worker base |
| `vllm/model_executor/layers/sampler.py` | 6 | V0 sampler 钩子 |
| `vllm/engine/metrics.py` / `metrics_types.py` | 87 / 14 | V0 指标(⚫ 想在 V1 上要 QAIC 指标,用 V1 logging 重做) |
| `vllm/config/scheduler.py` | 15 | ⚠待确认:看是否有 V1 相关的行 |

### 3d. ⚫ OPTIONAL —— 独立功能,需要才搬

| 文件 | +行 | 对应功能 |
|---|---|---|
| `vllm/entrypoints/openai/serving_chat.py` | 252 | ⚠ on-device sampling + gpt-oss/harmony。基础服务不需要。 |
| `vllm/entrypoints/openai/protocol.py` | 72 | ⚠ on-device sampling / gpt-oss 的 API 字段 |
| `vllm/entrypoints/harmony_utils.py` | 106 | gpt-oss harmony 格式 |
| `vllm/reasoning/gptoss_reasoning_parser.py` | 45 | gpt-oss reasoning |
| `vllm/entrypoints/openai/serving_pooling.py` | 104 | QAIC 上的 embeddings/pooling |
| `vllm/distributed/kv_transfer/kv_connector/base.py` | 142 | 拆分式 prefill/decode |
| `vllm/distributed/kv_transfer/kv_transfer_state.py` | 9 | 拆分式服务 |
| `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` | 22 | mxfp4 量化 |
| `vllm/model_executor/layers/quantization/gguf.py` | 29 | ⚠待确认 gguf 交互 |
| `vllm/model_executor/layers/quantization/bitsandbytes.py` | 10 | ⚠待确认 |
| `vllm/entrypoints/chat_utils.py` | 17 | ⚠ chat 模板(多模态?) |
| `vllm/entrypoints/openai/api_server.py` | 9 | ⚠ 小服务钩子 |
| `vllm/entrypoints/openai/tool_parsers/__init__.py` | 5 | 注册一个 tool parser |
| `vllm/model_executor/models/config.py` | 6 | ⚠ 模型 config 微调 |
| `vllm/env_override.py` | 10 | ⚠ 环境覆盖(可移到插件 `register()`) |
| `vllm/utils/__init__.py` | 10 | ⚠ 工具函数(需要的复制进包) |
| `setup.py` | 40 | 构建/打包 —— 不适用(插件有自己的 `pyproject.toml`) |

---

## 4. `GPUModelRunner`:v0.10.1(fork 基线) vs 0.21/0.22(目标)—— 深度对照

`QaicModelRunner` 继承 `GPUModelRunner`,后者从约 2–3k 行被重构到 7000+ 行。
这是核心 🔵 PORT 工作。源文件:
- 旧:fork `vllm/v1/worker/qaic_model_runner.py`(819 行)。
- 新:已安装的 `vllm/v1/worker/gpu_model_runner.py`。

### 4.A 类与构造函数

| | 旧 | 新 | 动作 |
|---|---|---|---|
| 基类 | `GPUModelRunner(...)` | `GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin)` | 继承没问题;注意有 mixin |
| `__init__` 签名 | `(vllm_config, device, speculative_model_type=None)` | `(vllm_config, device)` | **去掉第 3 个参数**,内部推导(已做) |

### 4.B 输入准备的数据模型 —— **最大断点**

| fork 用的属性 | 0.21 还有吗 | 0.21 替代 |
|---|---|---|
| `self.positions_np` | ❌ | `self.positions` / 由 `num_computed_tokens_cpu` 重算 |
| `self.cu_num_tokens` | ❌ | `_prepare_inputs` 内计算;`self.query_start_loc` |
| `self.num_decodes` | ❌ | 用 `reorder_batch_to_split_decodes_and_prefills` 重算 |
| `self.input_ids_cpu` | ❌ | `self.input_ids`(`CpuGpuBuffer`)+ `InputBatch.token_ids_cpu` |
| `self.batch_indices` | ❌ | `InputBatch.block_table[...]` |
| `self.arange_np` | ❌ | 本地重建 |
| input batch 字段 | 不同 | `InputBatch`(`gpu_input_batch.py`):`token_ids_cpu`、`num_tokens_no_spec`、`num_computed_tokens_cpu`、`block_table`、`sampling_metadata` |

**后果:** 重写 `_prepare_qaic_inputs()` / `_postprocess_tensors()`,改从
`scheduler_output` + `InputBatch` 取数据。时间主要花这里。

### 4.C 前向 + 采样 —— **架构性拆分**

| | 旧 | 新 | 动作 |
|---|---|---|---|
| `execute_model` | 一个方法:准备 → QPC 前向 → 采样 → `ModelRunnerOutput` | 可能返回 `None` 并暂存 `self.execute_model_state`;再调 `sample_tokens(grammar_output)` 采样 | QAIC 前向是同步、host 驱动 → 最简做法是在 `execute_model` 里走完整 QAIC 流程、直接返回 `ModelRunnerOutput`,绕过 `sample_tokens`。需确认基类允许。 |

### 4.D 投机解码

| | 旧 | 新 | 动作 |
|---|---|---|---|
| `_calc_spec_decode_metadata` | fork 自带,numpy | 基类方法 `gpu_model_runner.py:2698`,torch 张量 | 优先用基类;删 fork 副本 |
| `SpecDecodeMetadata` | 字段较少 | `draft_token_ids`、`num_draft_tokens`(list)、`cu_num_draft_tokens`(tensor)、`cu_num_sampled_tokens`、`target_logits_indices`、`bonus_logits_indices`、`logits_indices` | UnieAI 采样器用到的字段都还在 |
| `NgramProposer.propose` | `propose(context_tokens)` | `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)` | 更新调用 / 委托给基类 |
| `propose_draft_token_ids` | `(scheduler_output, sampled_token_ids)` | `(+hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata, common_attn_metadata, slot_mappings)` | 重写 override 或委托 `super()` |
| `RejectionSampler` | 基于 Triton(UnieAI 用 CPU 实现绕开) | `nn.Module.forward(metadata, draft_probs, logits, sampling_metadata)`(`v1/sample/rejection_sampler.py`) | **确认是否还依赖 GPU/Triton。** 还依赖 → 保留 UnieAI 的 CPU `_qaic_rejection_sample`。 |

### 4.E 低风险 override

| 方法 | 旧 → 新 | 动作 |
|---|---|---|
| `load_model` | `(*args, **kwargs)` → `(load_dummy_weights=False)` | 对齐签名;调 `load_qaic_model` |
| `initialize_kv_cache` | `(kv_cache_config)` → `(kv_cache_config, is_profiling=False)` | 加参数 |
| `get_kv_cache_spec` | `FullAttentionSpec` 字典 → 形状相同 | 低风险;核对字段名 |
| `_init_device_properties` / `_sync_device` | 空 stub | 基类仍调用就保留 |
| `_may_reorder_batch` | 设 `self.num_decodes` | 围绕已删字段改写 |

### 4.F 不变的部分

UnieAI 那 7 个 `_qaic_rejection_sample*` 函数是纯 `torch` + `sampling_metadata`
+ `SpecDecodeMetadata` → **逐字移植**(已在 `vllm_qaic/model_runner.py` 完成);
只需瞄一眼 `SpecDecodeMetadata` 字段名(0.21 上都还在)。

---

## 5. UnieAI 自己的改动(唯一非 Qualcomm、非上游的代码)

在 commit `909a809`。完整叙述见 `docs/UnieAI_Quic_integrated.md`。摘要:

| 文件(fork) | 改动 | 移植状态 |
|---|---|---|
| `vllm/v1/worker/qaic_model_runner.py` | +222:7 个 ngram CPU 拒绝采样函数 + 2D decode 打包 + ngram gate | ✅ 已移植进 `vllm_qaic/model_runner.py`(函数逐字 + `_pack_decode_batch`) |
| `vllm/model_executor/model_loader/qaic_v1.py` | +27:target QPC 输出 `N+1` logits | 在 `vllm_qaic/model_loader.py` 内移植 |
| `docs/qaic-v1-ngram-speculative.md` | 设计文档 | 参考 |

---

## 6. 🟠 必须改核心的短清单(其余都是 plugin/port/drop)

若想做到核心完全不 fork,只剩这几项要决策:

1. **`vllm/config/cache.py` —— 往 `CacheDType` 加 `"mxint8"`。** 选项:
   (a) 打一行核心补丁;(b) 用 `fp8`、不要 mxint8;(c) 上游 PR。
   插件目前只放 fp8(`platform.py::is_kv_cache_dtype_supported`)。
2. **`vllm/platforms/interface.py`** —— ⚠待确认是否还需要;0.21 大概率已是空操作。
3. **`vllm/_custom_ops.py`** —— ⚠待确认;0.21 上可能不需要。
4. **`vllm/transformers_utils/config.py`** —— ⚠待确认;可能用 configs registry
   (`general_plugins`)即可替代。

2–4 都是「确认一下、大概率不用」。**第 1 项是唯一真正的核心缺口。**

---

## 7. 建议顺序(降风险)

1. **先做 GO 测试**(README PART 0)—— torch 墙(gate 1/2)。NO-GO 就停。
2. 搭起插件外壳:🟢 platform + 量化 + 模型/配置注册;确认
   `current_platform == QaicPlatform`。
3. 🔵 先搬低风险:`session.py`、`model_loader.py`、`get_kv_cache_spec`、
   `load_model`、`initialize_kv_cache`。
4. 🔵 重写输入准备(§4.B)—— 先把**不带投机**的 prefill/decode 跑通。
5. 🔵 再开 ngram:把已就位的 `_pack_decode_batch` + `_qaic_rejection_sample`
   接进重写后的 `execute_model`;解决 §4.D 的 RejectionSampler。
6. 🟠 决定 `mxint8`(§6.1)。
7. ⚫ 按需加可选功能(pooling / 拆分式 / gpt-oss / 多模态)。
