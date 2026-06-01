# QAIC → vLLM 0.21+ 迁移工单(中文)

把我们现在跑在 vLLM **v0.10.1** 上的 QAIC 后端,搬到 **vLLM 0.21/0.22**,并重构成
树外插件 `vllm-qaic`。本文件逐个文件说明**迁移时该怎么处理**,外加最难那块
(`GPUModelRunner`)的新旧对照。目标是**能拿来沟通和分工的工单**。

---

## 怎么读这份文档(先看这段)

我们要把 v0.10.1 上「QAIC 相关的所有文件」一个个搬到 0.21。每个文件标一个
**处理方式**(该怎么办)和一个**状态**(做到哪了)。

**处理方式(5 类):**

| 标记 | 意思 | 大白话 |
|---|---|---|
| 🟢 **插件化** | 用 0.21 官方留的扩展接口接上,不用改 vLLM 本体 | 「有现成插座,插上即可」 |
| 🔵 **搬运** | 这块 QAIC 逻辑得搬进我们的包(复制/继承后改) | 「家具得自己搬过去」 |
| 🟠 **改本体** | 绕不开,得直接改 vLLM 核心文件(或给上游提 PR) | 「得在房东的墙上打个洞」 |
| ⚪ **丢弃(V0)** | 只有旧引擎 V0 用得到;我们只跑 V1,直接不要 | 「旧零件,扔」 |
| ⚫ **可选** | 是独立功能(gpt-oss / 拆分式 / embedding / 多模态),要用才搬 | 「选配件」 |

**状态:** ✅ 已完成 ・ 🟡 已写未上机验证 ・ 🔬 待验证(go/no-go)・ ⬜ 还没做(需上机)

**几个会反复出现的词,先解释:**

- **逐行确认过**:我真的把那个文件在出货版里的改动 diff 一行行读过了,不是看文件名猜的。
- **优雅降级 (graceful fallback)**:某个「GPU 专用的加速算子」在 QAIC 上没法用时,
  代码**自动改用一段纯 Python 的慢速等价实现**,不报错。所以这类文件在 QAIC 上
  「能跑,只是慢一点」。
- **字段**:Python 对象上的属性名(例如 `metadata.num_draft_tokens`)。我们说「字段还在」,
  意思是**0.21 的对象上这个属性名依然存在**,所以照搬的代码不会因为「找不到属性」而崩。

> 出处说明:下面表格里的文件都来自我们的 v0.10.1 fork。本文件只讲「**迁移时怎么处理**」,
> 不区分「谁写的字」——那部分(原始码出处)在 `UnieAI_Quic_integrated.md` 里专门讲。
> 这里所有迁移/整合/重构工作,都是我们的工作。

---

## 1. 总览:工作量其实没看起来那么大

v0.10.1 上 QAIC 改了约 40 个核心文件 + 新增约 45 个。但「重构成插件 + 只跑 V1」后:

- **约一半是 ⚪ 丢弃**(都是旧引擎 V0 的东西:V0 的调度器 / block 管理 / sequence / engine)。
- **大部分剩下的是 🟢 插件化**(平台、量化、KV connector、模型注册、CLI 参数 → 改走
  `--additional-config`)。
- **🟠 改本体的清单非常短**,而且复查后**很可能一项都不需要**(见 §6)。
- **真正的工程量是 🔵 搬运**,集中在一个文件 `GPUModelRunner`(见 §4)。

---

## 2. 新增的 QAIC 文件(约 45 个)怎么处理

### 2a. 搬进插件(🔵 搬运 / 🟢 插件化)

| 文件 | 处理 | 备注 |
|---|---|---|
| `vllm/v1/worker/qaic_model_runner.py` | 🔵 搬运 | → `vllm_qaic/model_runner.py`。**最难的一块**(§4)。 |
| `vllm/v1/worker/qaic_worker.py` | 🔵 搬运 | → `vllm_qaic/worker.py`。继承 `WorkerBase`。 |
| `vllm/model_executor/model_loader/qaic_v1.py` | 🔵 搬运 | → `vllm_qaic/model_loader.py`。QPC 加载/编译。 |
| `vllm/model_executor/model_loader/qaic.py` | 🔵 搬运 | → `vllm_qaic/compile_config.py`。编译参数整理。 |
| `vllm/model_executor/model_loader/qaic_session_np.py` | 🔵 搬运 | → `vllm_qaic/session.py`。纯 numpy+qaicrt,不依赖 torch。 |
| `vllm/platforms/qaic.py` | 🔵 搬运 | → `vllm_qaic/platform.py`(已起草,用 0.21 的树外平台机制)。 |
| `vllm/model_executor/layers/quantization/qaic_quant.py` | 🟢 插件化 | → `vllm_qaic/quant.py`,用 `register_quantization_config` 注册 mxfp6。 |
| `vllm/distributed/kv_transfer/kv_connector/qaic_connector.py` | ⚫ 可选 | 只有用「拆分式 prefill/decode」才需要。 |
| `vllm/model_executor/models/qaic_custom_mm_processor.py` | ⚫ 可选 | 只有跑多模态模型才需要。 |

### 2b. 旧引擎 V0 的 QAIC 文件 —— 我们只跑 V1,⚪ 丢弃

`vllm/worker/qaic_model_runner.py`、`vllm/worker/qaic_worker.py`、
`vllm/worker/qaic_pooling_model_runner.py`(要 embedding 才留)、
`vllm/spec_decode/qaic_multi_step_worker.py`(V0 的 draft-model 投机,不是我们的 ngram)、
`vllm/core/block/qaic_prefix_caching_block.py`。

### 2c. 出货版顺手带进来的一堆 V0 投机解码基础设施 —— 也 ⚪ 丢弃

`vllm/spec_decode/*` 整目录(`spec_decode_worker.py`、`multi_step_worker.py`、
`ngram_worker.py`、`top1_proposer.py` …)、`vllm/model_executor/layers/rejection_sampler.py`、
`spec_decode_base_sampler.py`、`typical_acceptance_sampler.py`、
`vllm/engine/output_processor/multi_step.py`。
**原因:** 0.21 的 V1 投机解码在 `vllm/v1/spec_decode/`,由上游提供,这些 V0 旧件用不上。

### 2d. 示例脚本 `examples/.../qaic*.py`(11 个)—— 参考用,按需改。

---

## 3. 修改过的核心文件(约 40 个)怎么处理

### 3a. 🟢 插件化 —— 用 0.21 的扩展接口接上,不改本体

| 文件 | 出货版改了什么 | 0.21 怎么接 |
|---|---|---|
| `vllm/platforms/__init__.py` | 加 QAIC 平台检测 | `vllm.platform_plugins` 入口点 |
| `vllm/engine/arg_utils.py` | 加 `--override-qaic-config`、`--device-group` | 改走通用的 `--additional-config` |
| `vllm/config/__init__.py` | 加 QAIC 配置字段 + 校验 | 从 `additional_config` 读 |
| `vllm/model_executor/layers/quantization/__init__.py` | 注册 mxfp6 | `register_quantization_config("mxfp6")` |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 注册 QAIC connector | `KVConnectorFactory.register_connector()` |
| `vllm/model_executor/models/registry.py` | 注册 QAIC 模型 | 启动时用 `general_plugins` 注册 |
| `vllm/transformers_utils/configs/__init__.py` | 注册自定义 HF config | 同上 |
| `vllm/envs.py` | 加 `VLLM_QAIC_*` 环境变量 | 插件自己读 `os.environ` |

### 3b. 那些 QAIC-specific 的零散改动,其实归到两个根因

复查发现,出货版里散落在很多文件的 QAIC 改动,**根子上只有两件事**。把这两件处理掉,
一大半文件就不用单独管了:

- **根因 1:「这是不是 QAIC 平台?」** —— 出货版到处用 `current_platform.is_qaic()`
  来判断。0.21 的树外平台没这个方法,统一改成 `current_platform.device_type == "qaic"`,
  **不用改本体**。
- **根因 2:「QAIC 没有 GPU 专用算子」** —— QAIC 主机是 CPU,跑不了 vLLM 那些
  GPU/CUDA 编译算子。出货版的做法是「检测到不支持就**优雅降级**到纯 Python」。
  只要让「检测算子是否可用」在 QAIC 上返回「不可用」,下面这些文件的降级分支就
  **自动生效**,不是单独的活:`mxfp4_utils.py`、`gguf.py`、`bitsandbytes.py`。

逐行确认过的清单:

| 文件 | 出货版改了什么(逐行确认过) | 处理 |
|---|---|---|
| `vllm/utils/__init__.py` | (1) 加 `mxint8` 这个 dtype 映射;(2) 让「算子可用吗」在 QAIC 上返回 False | 🟠 改本体(根因 2 的开关)—— 或在插件初始化里复制这逻辑 |
| `vllm/_custom_ops.py` | QAIC 上跳过 `import vllm._C`(没有 CUDA 算子) | 见 §6(很可能不用) |
| `vllm/platforms/interface.py` | 加 `QAIC` 枚举 + `is_qaic()` 方法 | 见 §6(不用,改用树外机制) |
| `vllm/env_override.py` | QAIC 上跳过 torch inductor 的线程设置 | ⚪ 你启动已带 `TORCH_COMPILE_DISABLE=1`,自然不触发 |
| `vllm/transformers_utils/config.py` | mllama 模型在 QAIC 上关掉 cross-attention + 2 个新模型 config | ⚫ 可选(mllama);新 config 大概率已上游 |

### 3c. ⚪ 丢弃(只有 V0 引擎用)

`vllm/core/scheduler.py`、`block_manager.py`、`block/cpu_gpu_block_allocator.py`、
`engine/llm_engine.py`、`engine/output_processor/interfaces.py`、`sequence.py`、
`worker/worker_base.py`、`model_executor/layers/sampler.py`、`engine/metrics*.py`。
**这些都是旧引擎 V0 的内部件,只跑 V1 就用不到。**

### 3d. ⚫ 可选功能(基础聊天用不到,要哪个功能才搬)

| 文件 | 出货版改了什么(逐行确认过) | 属于哪个功能 |
|---|---|---|
| `vllm/entrypoints/openai/serving_chat.py` (+252) | kimi_k2 的 tool-call id、harmony、`return_token_ids` 调试 | 工具调用 / gpt-oss / 调试 |
| `vllm/entrypoints/openai/protocol.py` (+72) | 加可选返回字段、计时、embedding bytes | 向后兼容字段,大概率已上游 |
| `vllm/entrypoints/openai/serving_pooling.py` (+104) | embedding 的多种编码格式 | embedding |
| `vllm/entrypoints/openai/api_server.py` (+9) | 处理 embedding bytes 返回 | embedding |
| `vllm/entrypoints/chat_utils.py` (+17) | kimi_k2 tool-call id 辅助函数 | 工具调用 |
| `vllm/entrypoints/openai/tool_parsers/__init__.py` (+5) | 注册一个 tool parser | 工具调用(🟢 可插件化注册) |
| `vllm/entrypoints/harmony_utils.py` (+106) | gpt-oss 函数调用 | gpt-oss |
| `vllm/reasoning/gptoss_reasoning_parser.py` (+45) | gpt-oss reasoning | gpt-oss |
| `vllm/model_executor/models/config.py` (+6) | 改 gpt-oss reasoning 后端名 | 与 QAIC 无关,大概率已上游,可忽略 |
| `kv_connector/base.py` (+142) / `kv_transfer_state.py` (+9) | 重加 V0 KV connector | 拆分式服务(不用就丢) |
| `quantization/utils/mxfp4_utils.py` / `gguf.py` / `bitsandbytes.py` | 算子优雅降级 | 由根因 2 自动触发 |
| `setup.py` (+40) | QAIC 构建检测 / SDK 版本 | 插件自带 `pyproject.toml`,不适用 |

---

## 4. 最难的一块:`GPUModelRunner` 新旧对照

我们的 QAIC model runner 是**继承** `GPUModelRunner` 的。这个基类从 v0.10.1 的约
2–3 千行,被 0.21 重构到 **7000+ 行**,内部数据结构大改。所以继承它的我们这个子类,
**每个重写的方法都得对着新基类重新对一遍**。这是整次迁移真正花时间的地方。

源文件:旧 = fork `vllm/v1/worker/qaic_model_runner.py`(819 行);新 = 已装的
`vllm/v1/worker/gpu_model_runner.py`。

### 4.A 构造函数
- 旧:`__init__(vllm_config, device, speculative_model_type=None)`
- 新:`__init__(vllm_config, device)`(去掉第 3 个参数,内部自己推导)—— 已处理。

### 4.B 输入准备 —— **最大断点(看 `REBUILD_input_prep_4B.md` 上机做)**

「输入准备」= 把调度器的决定(这步每个请求各算几个 token)翻译成卡能吃的 numpy 数组。
旧版靠 runner 上几个现成数组,**0.21 全删了**,数据搬进了 `InputBatch`:

| 旧版用的属性 | 0.21 还有吗 | 改从哪拿 |
|---|---|---|
| `self.positions_np` | ❌ 删了 | 由 `InputBatch.num_computed_tokens_cpu` 算 |
| `self.cu_num_tokens` | ❌ 删了 | 自己 cumsum |
| `self.num_decodes` | ❌ 删了 | 用 `reorder_batch_to_split_decodes_and_prefills` 重算 |
| `self.input_ids_cpu` | ❌ 删了 | `InputBatch.token_ids_cpu` |
| `self.batch_indices` | ❌ 删了 | `InputBatch.block_table` |

**这块只能上机边跑边对**,实操步骤见 `REBUILD_input_prep_4B.md`。

### 4.C 前向 + 采样 —— 架构被拆成两半
- 旧:`execute_model` 一个方法里走完「准备 → 卡上前向 → 采样」并返回结果。
- 新:`execute_model` 可能先返回 `None`、把状态存起来,再由 `sample_tokens()` 出结果。
- 做法:QAIC 前向是同步的,最简单就是在 `execute_model` 里走完整流程直接返回,
  绕过 `sample_tokens`(上机确认基类允许)。

### 4.D 投机解码相关方法

| 项 | 旧 → 新 | 动作 |
|---|---|---|
| `_calc_spec_decode_metadata` | 旧自带 → 基类已有(`gpu_model_runner.py:2698`) | 用基类的,删自己那份 |
| `SpecDecodeMetadata` 的字段 | 见下 | 我们 ngram 采样器用到的字段,0.21 上**都还在**(见 §4.F) |
| `NgramProposer.propose` | 参数变多 | 更新调用,或直接委托给基类 |
| `propose_draft_token_ids` | 参数变多 | 重写或委托 `super()` |
| `RejectionSampler` | 旧是 Triton(GPU)→ 新仍是 `nn.Module` | **要上机确认它是否还只能 GPU/Triton。** 还是的话,继续用我们的 CPU 版 |

### 4.E 低风险方法(签名小改,已搬好)
`load_model`、`get_kv_cache_spec`、`initialize_kv_cache` —— 都已按 0.21 新签名搬进
`vllm_qaic/model_runner.py`。

### 4.F 不用改、可以照搬的部分(以及到底是哪些「字段」)

我们的 ngram CPU 采样器(7 个函数)**只用到这些字段**,而它们在 0.21 上都还在,
所以这段代码能**逐字照搬**:

- 来自 `SpecDecodeMetadata`(投机解码的元数据对象):
  - `num_draft_tokens` —— 每个请求这步猜了几个候选 token
  - `max_spec_len` —— 一次最多猜几个
  - `cu_num_draft_tokens` —— 候选数的累加和(0.21 上变成了 tensor,我们代码本来就用 `.item()` 取值,不受影响)
  - `draft_token_ids` —— 候选 token 本身
- 来自 `sampling_metadata`(采样参数对象):
  - `temperature`、`top_k`、`top_p` —— 采样温度/截断参数
  - `generators` —— 每个请求的随机数发生器
  - `all_greedy` / `all_random` —— 整批是否全贪心/全随机

「这些字段都还在」= 上面这些**属性名在 0.21 的对应对象上依然存在**,所以照搬的代码
不会因为「找不到属性」而崩。这也是为什么这段能照搬、而 §4.B 的输入准备不行 ——
它当初是对着「稳定的采样接口」写的,没碰那些被删掉的 runner 内部属性。

---

## 5. ngram 投机解码优化(我们写的原创代码)

这是我们在 V1 上从 0 到 1 加的投机解码优化(原理见 `EXPLAINER_plain_zh.md` §5、
逐行权属见 `UnieAI_Quic_integrated.md` §7)。迁移状态:

| 涉及文件 | 内容 | 迁移状态 |
|---|---|---|
| `qaic_model_runner.py` | 7 个 CPU 拒绝采样函数 + 2D 打包 + ngram 开关 | ✅ 已逐字搬进 `vllm_qaic/model_runner.py` |
| `qaic_v1.py` | 让 target QPC 多输出 `N+1` 行打分 | 随 `model_loader.py` 一起搬 |
| `docs/qaic-v1-ngram-speculative.md` | 设计文档 | 参考 |

---

## 6. 「要不要改 vLLM 本体」的最终结论:很可能一项都不用

复查后,原本以为绕不开的几处 🟠,逐条看下来基本都能不改本体:

1. **`vllm/config/cache.py` 加 `mxint8` —— 不需要。**
   QAIC 的**卡上 KV 本来就是 mxint8**(由 QEfficient 编译时决定,走 `--additional-config`)。
   vLLM 层的 `--kv-cache-dtype` 填 `fp8` 还是 `fp16` 都**不影响**卡上的实际存储格式 ——
   它对 QAIC 来说只是个记账标签。所以**没必要**把 `mxint8` 塞进 vLLM 那张 `--kv-cache-dtype`
   的合法值清单里。(`--kv-cache-dtype` 是一张写死在源码里的字符串白名单,插件无法从外部
   往里加值;但既然不需要加,这点就无所谓了。)

2. **`vllm/platforms/interface.py` —— 不需要,归入丢弃。**
   这个文件是 vLLM「平台」的基类定义(规定每种硬件平台要实现哪些方法)。出货版在这里
   加了个 `QAIC` 枚举值和 `is_qaic()` 方法。但 0.21 提供了**树外平台机制**(我们用
   `device_type == "qaic"` 判断),所以这个改动在 0.21 上**没有作用**(我之前说的「空操作」
   就是这个意思:加了也不会被用到)。→ **不留在清单里,直接归到「丢弃」。**

3. **`vllm/_custom_ops.py` —— 大概率不需要。**
   这个文件负责加载 vLLM 的 CUDA 编译算子(`vllm._C`)。出货版让 QAIC 跳过它。
   在 0.21 上,我们的树外平台本来就不会去 build/加载 CUDA 算子,所以**多半用不上**;
   归到「上机确认,基本可丢」。

4. **`vllm/transformers_utils/config.py` —— 目前无法判断,需上机确认。**
   出货版在这里做了两件事:给 mllama 模型在 QAIC 上关掉 cross-attention(QAIC 不支持),
   以及加了两个新模型 config。是否需要,**取决于你们实际要跑哪些模型**(只跑 Qwen 这类
   普通文本模型就不需要;要跑 mllama 才需要)。所以**现在判不了,等确定模型清单 + 上机
   才能定**。

> **结论:** 「必须改 vLLM 本体」的清单,复查后**很可能为空**。第 1、2 项确定不用,
> 第 3 项基本不用,第 4 项取决于模型清单、需上机确认。

---

## 7. 建议顺序(降风险)

1. **先做 GO 测试**(README PART 0)—— torch 版本墙(gate 1/2)。过不了就停,别白做。
2. 搭插件外壳:🟢 平台 + 量化 + 模型/配置注册;确认 `current_platform` 是 QaicPlatform。
3. 🔵 先搬低风险:`session.py`、`model_loader.py`,以及已搬好的
   `load_model` / `get_kv_cache_spec` / `initialize_kv_cache`。
4. 🔵 重写输入准备(§4.B,照 `REBUILD_input_prep_4B.md`)——**先把不带投机的 prefill/decode
   跑通**。
5. 🔵 再开 ngram:把已就位的 2D 打包 + CPU 拒绝采样接进 `execute_model`;
   并确认 §4.D 的 RejectionSampler 是否还依赖 Triton。
6. ⚫ 按需加可选功能(embedding / 拆分式 / gpt-oss / 多模态)。
