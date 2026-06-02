# 我们做了什么(资工硕士可读版)

> 面向懂 LLM 推理基础(token、prefill/decode、KV cache、量化、采样)的读者。
> 更精确的权属/逐行清单见 `UnieAI_Quic_integrated.md`;迁移工单见 `MIGRATION_zh.md`;
> 最难一步的上机实操见 `REBUILD_input_prep_4B.md`。
>
> **版本提醒:**A1/A2/A3/B 说的是既有 v0.10.1 fork/生产线;A4 说的是正在做的
> vLLM 0.21+ 迁移。当前本地分支不是 exact `v0.21.0` tag;runtime 主路径已经搬进
> 插件并通过本地语法检查,但还没有完成 AIC 上机验证。

---

## 0. 一句话

> **(A)** 我们做了大量**移植升級工程**,把高通的 QAIC 后端持续整合到不同版本的开源
> vLLM(v0.10.1 基线 → V1 引擎产品化 → 现在迁移到 0.2x),并对接 QEfficient /
> Cloud AI SDK;**(B)** 在此之上加了一个 speculative decode 的优化项(ngram)。

---

## 1. 具体做了什么(先看这张总表)

| | 工作 | 具体内容(浓缩) | 状态 |
|---|---|---|---|
| **A1** | 整合到 v0.10.1 基线 | 把高通 QAIC patch 对齐到 upstream vLLM v0.10.1,依赖/接口对位、可 apply 可运行 | ✅ 生产在跑 |
| **A2** | V1 引擎适配 + 产品化 | 官方只示范 V0;我们把 QAIC 后端跑在 **V1 引擎**(`VLLM_USE_V1=1`)并产品化:V1 worker/model_runner、调度→QPC、CPU 采样接通 | ✅ 生产在跑 |
| **A3** | QEfficient / SDK 版本对接 | 对接 efficient-transformers **v1.21** + Cloud AI SDK **1.21** 的编译/运行链路(详见 §4) | ✅ 生产在跑 |
| **B**  | ngram 投机解码优化 | 在 V1 上从 0 到 1 加 ngram:解禁 + 2D 打包 + CPU 拒绝采样(详见 §5) | ✅ 生产在跑 |
| **A4** | 迁移到 vLLM 0.2x | 把整套搬到 0.21+(引擎被大改),重构为树外插件(详见 §6) | 🟡 进行中(状态分项见 §6) |

状态图例:✅ 已完成并在生产 ・ 🟡 已设计/已写但未上机验证 ・ 🔬 待验证(go/no-go)・ ⬜ 尚未做(需上机)

---

## 2. 背景(最小必要)

QAIC 后端的执行模型:**vLLM 留在 host(CPU)做调度、KV-block 管理、采样;真正的
Transformer 前向是预编译的 QPC,经 `qaicrt` 在 AIC100 卡上跑。** 两侧用 numpy 交换,
KV cache 常驻卡上(权重 mxfp6、卡上 KV mxint8 由编译器决定)。

这套和原生 GPU 路径最大的不同:**卡上不是 torch 在算**,torch 只在 host 做采样等轻量
操作。所以凡是「需要 GPU/Triton kernel」的 vLLM 组件,在 QAIC 上都得换成 CPU 实现 ——
这是 B(ngram)那块存在的根本原因。

因此版本关系要分清楚:

- host 上的 vLLM 版本负责调度、batching、KV-block 记账、采样和 API server。
- AI100 上执行的是 QEfficient + Cloud AI SDK 编译好的 QPC;卡上算子在编译期已经固定。
- host↔AI100 的接口是 `qaicrt` + numpy buffer。
- 所以把 host vLLM 从 0.10 升到 0.21,本质是 host 端软件/API 迁移,不要求因为
  vLLM 升级就改 AI100 firmware/kernel/SDK。
- 但这不是无条件兼容:Python ABI、`qaicrt` 路径、SDK runtime、QPC 格式和 buffer
  shape 仍必须匹配,需要 Gate 1/2 和 serve smoke test 证明。

---

## 3. speculative decode + ngram(精简版)

- **speculative decode**:用便宜的 proposer 先产出 K 个候选 token,再让 target 模型
  在**一次** forward 里对这 K 个位置打分、做接受/拒绝校验。接受的成批吐出 →
  提升 tokens/step,且输出分布与逐 token 解码**等价**。
- **ngram(prompt-lookup)**:proposer 不用 draft 模型,而是在该请求自身上下文里做
  最长后缀匹配,复用历史里相同前缀后跟过的 token 作为候选。零额外模型成本。

---

## 4. A3 讲清楚:QEfficient / Cloud AI SDK 对接到底对接了什么

QAIC 不能直接跑 PyTorch/HF 模型,必须先经 **QEfficient(efficient-transformers)**
把 HF 模型 **导出 ONNX → 编译成 QPC**,再由 **Cloud AI SDK** 提供的 `qaicrt` 运行时
加载执行。「对接 v1.21」不是改个版本号,而是把下面这条链路在该版本上打通、对齐:

| 对接点 | 具体做了什么 |
|---|---|
| **编译 API 对齐** | vLLM 的 QAIC loader 调用 QEfficient v1.21 的 `QEFFAutoModelForCausalLM.compile()` / `QAICInferenceSession`;这些 API 的签名/行为随版本变,要对齐调用方式 |
| **编译配置透传** | 把 vLLM 启动参数翻译成 QEfficient 编译参数:`num_cores`、`prefill_seq_len`、`ctx_len`、`mxfp6_matmul`(权重 mxfp6)、`mxint8_kv_cache`(卡上 KV mxint8)、`aic_enable_depth_first` 等 |
| **版本/ABI 对齐** | QEfficient v1.21 钉 **torch 2.7 / transformers 4.57.3**;qaicrt 的 `.so` 按特定 Python ABI 编译。三者(torch、transformers、Python)要同时满足,否则编译期或加载期报错 |
| **运行时加载** | v1.21 产出的 QPC,要能被 SDK 1.21 的 `qaicrt` 正确加载(Context/Queue/Program/ExecObj),并把 `past_*` / `*_RetainedState` 这些 KV buffer 留在卡上 |
| **量化路径** | mxfp6(权重)+ mxint8(KV)是 SDK/编译器特性,要从 vLLM 的 `--quantization` / KV 配置一路接到编译调用 |

> 一句话:A3 = **让「vLLM 的参数 → QEfficient v1.21 编译 → SDK 1.21 运行时执行」这条
> 链路端到端跑通,并解决三方版本/ABI 的相互约束**。

---

## 5. B 讲清楚:ngram 我们改了什么(在 V1 上)

| | 之前(高通出货) | 之后(我们) |
|---|---|---|
| 开关 | V1 路径直接 `raise "not yet supported"` | 解禁,限定 `method=="ngram"`,模型标为 spec **target** |
| decode 输入 | 每请求每步 1 个 token(1D) | 打包成 `[num_decodes, N+1]`,target QPC 一次校验「上个 token + 至多 N 个候选」 |
| 校验器 | V1 内置 RejectionSampler 用 **Triton kernel**(QAIC 的 CPU host 跑不了) | **自写 CPU 拒绝采样**(7 个函数),实现同样的接受规则(按概率接受、拒绝时 recover、greedy 快路径) |
| target 输出 | 末位 1 行 logits | 改 loader 让 QPC 输出 **N+1 行 logits** 供校验 |

代码量约 250 行,集中在 `qaic_model_runner.py` 一个文件 + 一份设计文档。这是
**UnieAI 原创代码**,也是 speculative decode 的**一个优化项**。

---

## 6. A4 讲清楚:迁移到 vLLM 0.2x —— 具体做了什么 + 每项状态

0.21 把 `GPUModelRunner` 从约 2–3k 行重构到 **7000+ 行**,内部数据模型大改。迁移内容:

| 子项 | 具体做了什么 | 状态 |
|---|---|---|
| **完整改动盘点** | 把高通 patch 动过的 45 新增 + 40 修改文件全部分类(插件化/移植/丢 V0/小核心补丁);见 `MIGRATION_zh.md` | ✅ 已完成 |
| **重构为树外插件** | 用 `vllm.platform_plugins` / `register_quantization_config` / `--additional-config` 等官方扩展点,避免 fork 核心;`vllm-qaic/` 已建 | 🟡 已写,未上机验证 |
| **ngram 采样器移植** | 7 个函数 + 2D 打包搬到 0.21,并接入 `execute_model` 主流程 | 🟡 已写,未上机验证 |
| **loader/kv_cache 方法** | `load_model` / `get_kv_cache_spec` / `initialize_kv_cache` 的外层 0.21 签名已对齐;`load_model` 已接到 ported loader | 🟡 已写,未上机验证 |
| **torch 版本墙** | QEfficient 钉 torch 2.7、vLLM 0.21 钉 2.11,冲突;方案=编译/服务分离;GO 测试脚本已备 | 🔬 **待验证**(go/no-go,需上机跑 gate1/2) |
| **大文件移植(session/loader/qserve/worker)** | `session.py`、`model_loader.py`、`compile_config.py`、`qserve_model_runner.py`、`worker.py` 已从 fork 搬进插件并改 import | 🟡 已写,未上机验证 |
| **输入准备重写(§4.B)** | 旧的 host 数组(`positions_np`/`cu_num_tokens`/`num_decodes`…)在 0.21 删了,已改用新的 `InputBatch` / `_prepare_inputs` 重建 | 🟡 已写,未上机验证 |
| **execute_model 串接** | 已把 0.21 input-prep、QAIC decode/prefill、2D 打包、CPU 拒绝采样接进主流程 | 🟡 已写,未上机验证 |
| **mxint8 决策** | 结论:不需要核心补丁——mxint8 是编译器开关走 `--additional-config`,vLLM 层用 fp8 | ✅ 已定 |

> **A4 的诚实状态:** 「盘点 + 设计 + 插件骨架 + runtime 主路径移植」已完成或已写好;
> 但**整套尚未在 AIC 机器上验证过**。关键卡点是:① torch 墙的 GO 测试(待验证,决定
> 可行性);② precompiled QPC load/run;③ `vllm serve` smoke。换言之:**A4 = 路已接上,
> 还没通车验收。**
>
> 进一步说,现在不能把 A4 描述成「vLLM 0.21.0 已跑通」。更准确的说法是:
> **v0.10.1 QAIC patch 的 vLLM 0.21+ 迁移已盘点并完成 runtime 主路径移植;
> 尚未完成 AIC 上机验证。**

---

## 7. 权属边界(代码出处 vs 工程劳动)

**① 代码出处**(谁写的字):
```
开源 vLLM(Apache-2.0)         ← 底层引擎
高通 QAIC 代码(标「机密专有」） ← 卡上底座(平台/加载/运行/量化)
UnieAI 原创代码                 ← ngram(~250 行 + 设计文档)
```

**② 工程劳动**(谁让它跑通/搬过来,= A1–A4 的工作主体):
> 把上面两层对齐到 v0.10.1 → 跑通并产品化 V1 引擎 → 对接 QEfficient v1.21 / SDK 1.21 →
> 迁移到 0.2x。这些**整合与升級的工程**是 UnieAI 做的。

> 关键:**代码出处 ≠ 工程劳动。** 多数文件的「字」来自高通/开源,但「让它在这些
> 版本/引擎/SDK 上跑通并升級」是 UnieAI 的工作;ngram 那 250 行是 UnieAI 自写、且写进
> 高通文件中(逻辑独立、物理交织,逐行见 `UnieAI_Quic_integrated.md` §7)。

---

## 8. 名词小抄(只列 QAIC 特有)

| 词 | 含义 |
|---|---|
| QPC | QEfficient 编译产出、可在 AIC100 上跑的模型二进制 |
| qaicrt | Cloud AI SDK 的运行时,加载/执行 QPC |
| QEfficient | quic/efficient-transformers:把 HF 模型导出 ONNX 并编译成 QPC 的工具链 |
| mxfp6 / mxint8 | QAIC 的权重量化(mxfp6)/ 卡上 KV 量化(mxint8,microscaling int8) |
| rejection sampler | 投机解码的接受/拒绝校验器;V1 内置版用 Triton,我们写了 CPU 版 |
