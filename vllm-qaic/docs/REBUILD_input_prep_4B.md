# §4.B 实操指南:在机器上重建「输入准备」(input prep)

这是整个 0.21 迁移里**唯一只能在 AIC 机器上、边跑边对**完成的硬骨头。本指南把它拆成
能一步步照做、每步可验证的小任务。

> 读这份前先懂背景:`EXPLAINER_plain_zh.md`(白话)、`MIGRATION_zh.md` §4(对照)。

---

## 0. 「输入准备」到底是什么(一句话)

vLLM 的调度器决定了「这一步,每个请求各要算几个 token」。**输入准备 = 把这个决定,翻译
成高通模型 `forward()` 能吃的几个 numpy 数组**(哪些 token、在第几个位置、用哪块 KV)。

fork 在 v0.10.1 上是靠 model runner 上几个 host 数组现成拿(`positions_np`、
`cu_num_tokens`、`num_decodes`、`input_ids_cpu`…)。**这些在 0.21 全删了**,数据搬进了
`InputBatch` + GPU buffer。所以要**自己从新地方把这些数组重建出来**。

---

## 1. 目标契约:高通 `model.forward()` 要什么

(来自 fork `qaic_v1.py::QaicCausalLM.forward` / `_run_prefill` / `_run_decode`)

`forward()` 对 **prefill** 和 **decode** 分开调用,各要这些 host numpy 数组:

| 数组 | 含义 | dtype |
|---|---|---|
| `input_ids` | 这一步要算的 token(prefill 是展平的;decode 带投机时是 2D) | int64 |
| `positions` | 每个 token 在序列里的位置 | int64 |
| `batch_indices` | 每个请求用卡上哪一块 KV(block id) | int64 |
| `is_prompt` | True=prefill, False=decode | bool |
| `prefill_cum_sum` | (仅 prefill)各请求 token 数的累加和 | int64 |
| `decode_lengths` | (仅 decode + 投机)每行真实 token 数 | int32 |

**你的任务 = 在 0.21 上把上面这些算出来。**

---

## 2. 数据源地图:0.21 里每样东西从哪拿

(已在仓库核对过字段名)

| 你要的 | 0.21 的来源 |
|---|---|
| 每个请求这步算几个 token | `scheduler_output.num_scheduled_tokens[req_id]`(dict) |
| 批次里的请求顺序 | `self.input_batch.req_ids`(list) |
| 某请求的 token 历史 | `self.input_batch.token_ids_cpu[req_idx]`(numpy `[max_reqs, max_len]`, int32) |
| 某请求已经算到第几个(positions 起点) | `self.input_batch.num_computed_tokens_cpu[req_idx]` |
| 某请求总 token 数(不含投机) | `self.input_batch.num_tokens_no_spec[req_idx]` |
| block id | `self.input_batch.block_table`(`MultiGroupBlockTable`),取 group 0 的 `.block_table.np` |
| decode/prefill 拆分 | `reorder_batch_to_split_decodes_and_prefills(...)`(`vllm/v1/attention/backends/utils.py:606`) |

> **QAIC 简化:** 你们 `block_size = max_model_len`(平台代码里设的),所以**每个请求只用
> 一块 KV**,`batch_indices` 基本就等于「该请求在批次里的槽位 / block 表第 0 列」。
> 上机后用一个 2 请求的例子打印确认这个对应关系。

---

## 3. 一步步重建(每步都有「做什么 / 骨架 / 怎么验」)

在 `vllm_qaic/model_runner.py` 里实现一个 `_prepare_qaic_inputs(scheduler_output)`,
替代 fork 的同名方法。建议先实现**不带投机**的版本(见 §4 增量计划)。

### 步骤 1 — 每个请求这步算几个 token(按批次顺序)

```python
req_ids = self.input_batch.req_ids                      # 批次顺序
num_sched = np.array(
    [scheduler_output.num_scheduled_tokens[r] for r in req_ids],
    dtype=np.int64)
total = int(num_sched.sum())
```
**验:** `total == scheduler_output.total_num_scheduled_tokens`(若该字段存在);
`num_sched.min() >= 1`。

### 步骤 2 — 累加和 `cu_num_tokens`

```python
cu_num_tokens = np.cumsum(num_sched)                    # [num_reqs]
```
**验:** `cu_num_tokens[-1] == total`;严格递增。

### 步骤 3 — decode / prefill 拆分(先 reorder,再数 decode 个数)

照搬 0.21 `GPUModelRunner` 自己怎么调 `reorder_batch_to_split_decodes_and_prefills`
(在 `gpu_model_runner.py` 里 grep 这个函数名,抄它的调用方式)。reorder 后 decode 请求
排在前面。

```python
# 不带投机:一个请求这步只算 1 个 token = decode;否则 = prefill
self.num_decodes = int((num_sched == 1).sum())          # 先用最朴素判据,后续按 reorder 结果校正
```
**验:** 用一个「1 个请求在 decode、1 个在 prefill」的场景,打印 `num_decodes` 是否 = 1。

### 步骤 4 — 展平 `input_ids` 和 `positions`

```python
input_ids = np.empty(total, dtype=np.int64)
positions = np.empty(total, dtype=np.int64)
cursor = 0
for i, r in enumerate(req_ids):
    idx = self.input_batch.req_id_to_index[r]
    n   = int(num_sched[i])
    start = int(self.input_batch.num_computed_tokens_cpu[idx])   # ← 关键:不是 0!
    input_ids[cursor:cursor+n] = self.input_batch.token_ids_cpu[idx, start:start+n]
    positions[cursor:cursor+n] = np.arange(start, start+n)
    cursor += n
```
**验:** `cursor == total`;dtype 是 int64;对一个已知 prompt,打印头几个 `input_ids`
和 tokenizer 编码比对。**chunked prefill 时 `start` 必须是 `num_computed_tokens`,
不能从 0 开始**——这是最常见的 bug。

### 步骤 5 — `batch_indices`(block id)

```python
block_np = self.input_batch.block_table.block_tables[0].block_table.np  # group 0；属性名上机确认
batch_indices = np.array(
    [block_np[self.input_batch.req_id_to_index[r], 0] for r in req_ids],
    dtype=np.int64)
```
**验:** `block_size == max_model_len` 时,每请求只有 1 块;打印确认 `batch_indices` 与
请求槽位的对应。**这步属性路径最容易变,务必上机 `print(dir(...))` 核对。**

### 步骤 6 — prefill 部分的 `prefill_cum_sum`

```python
prefill_input_ids = input_ids[decode_token_count:total]
prefill_positions = positions[decode_token_count:total]
prefill_cum_sum   = (cu_num_tokens[self.num_decodes:] - decode_token_count)
```
(`decode_token_count` 来自下一步的打包;不带投机时就是 `num_decodes`。)

### 步骤 7 — decode 部分 + 2D 打包(投机才需要)

直接调用**已经移植好**的 helper:
```python
decode_input_ids, decode_positions, decode_lengths, decode_token_count = \
    self._pack_decode_batch(input_ids, positions, self.num_decodes, cu_num_tokens)
```
不带投机时它会退化成 1D(原样切片),无需特殊处理。

---

## 4. 增量计划(别一次全开,会找不到 bug)

```
① 先关投机(不传 --speculative-config)
   只跑通 步骤 1–6 的 prefill + 1D decode
   → 目标:能正确生成文字

② greedy 对拍
   固定 prompt + temperature=0,把前 ~20 个输出 token
   和「同模型同 prompt 的参考结果」逐个比对(参考 = 你们现成能跑的 v0.10.1 部署,
   或 HF/CPU 跑同一模型)。greedy 下必须逐 token 完全一致。

③ 再开 ngram
   加 --speculative-config '{"method":"ngram",...}'
   接入 步骤 7 的 _pack_decode_batch + 已移植的 _qaic_rejection_sample
   → 目标:输出与 ② 一致(只是更快),accept 率 > 0

④ 解决 §4.D:确认 0.21 的 RejectionSampler 是否仍依赖 Triton
   依赖 → 保留我们的 CPU 版;不依赖 → 评估能否直接用上游
```

---

## 5. 易踩的坑(checklist)

- [ ] **int32 → int64**:`InputBatch` 是 int32,QPC 要 int64,记得 `.astype(np.int64)`。
- [ ] **positions 起点**:必须用 `num_computed_tokens_cpu`,不是 0(chunked prefill)。
- [ ] **请求顺序一致**:重建数组的请求顺序,要和你读 `sampling_metadata` 的顺序一致,
      否则采样参数错位。
- [ ] **block id 取法**:`block_table` 的属性路径跨版本会变,上机 `dir()` 确认。
- [ ] **decode/prefill 判据**:带投机时 decode 请求这步算的是 `1..N+1` 个 token,
      不能再用「==1」判 decode;以 reorder 结果为准。
- [ ] **num_decodes 来源**:0.21 没有这个字段了,是你自己算的,别又去引用 `self.num_decodes`
      以为基类有。

---

## 6. 完成的标准(Definition of Done)

1. 关投机:固定 greedy prompt 的前 20 token 与参考**逐个一致**。
2. 开 ngram:输出与第 1 条**一致**,且统计到的 accept 数 > 0(说明投机真的在跑)。
3. 多请求并发(`--max-num-seqs > 1`)+ 混合 prefill/decode 一步里,不崩、结果正确。

做到这 3 条,§4.B 就算搬完,整个 V1+QAIC on 0.21 的主路径就通了。
