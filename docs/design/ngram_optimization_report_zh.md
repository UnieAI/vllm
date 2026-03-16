# N-gram Proposer 优化报告

## 1. 目标

这次优化的目标不是改写 speculative decoding 的语义，而是降低 `ngram` proposer 在 CPU 路径上的固定开销，并补上可以离线重现的 benchmark。

重点是：

- 不依赖 PyTorch compiler。
- 不依赖 GPU。
- 不引入默认行为变化。
- 先优化 hot path，再补可重现的测试与 benchmark。

## 2. 更新内容

### 2.1 修正 Numba 线程上限错误

**原因**

原本 `num_numba_thread_available = min(1, cpu_count // 2)`，这会把可用线程数几乎永远压成 `0` 或 `1`，导致多线程 matcher 基本上没有机会展开。

**解决方案**

- 改为 `min(8, max(1, cpu_count // 2))`
- 再按 TP size 做保守缩放，并确保结果至少为 `1`

**影响**

- 在 CPU batch 稍大时，可以真正启用有限度的 Numba 并行
- 默认仍然保守，避免把整机线程全部吃满

### 2.2 只对有效 request 计算工作量

**原因**

原本 multi-thread threshold 用的是整个 batch 的 `num_tokens_no_spec` 总和，即使 batch 中很多 request 根本不会进入 ngram matcher，也会被算进去。

**解决方案**

- 改成只统计 `valid_ngram_requests` 的 token 数

**影响**

- thread 开关更接近真实工作量
- 避免 batch 很大但有效 request 很少时误开多线程

### 2.3 预先计算 `valid_ngram_requests`

**原因**

原本 proposer 每次都在 Python 里重新扫一次 request，runner 也没有复用这个结果。

**解决方案**

- 新增 `get_valid_ngram_requests`
- 引入 `NgramProposalInputs`
- 在 runner 侧先完成 filtering，再把 `valid_ngram_requests` 传入 proposer

**影响**

- 减少重复 filtering
- runner integration benchmark 也能单独量 input-prep 成本

### 2.4 把 matcher 输出从临时数组改成 `(start_position, draft_len)`

**原因**

旧版 `_find_longest_matched_ngram_and_propose_tokens` 直接返回一个新的 `np.ndarray` 切片，这会在热点路径制造额外的中间对象。

**解决方案**

- matcher 改为回传 `(start_position, draft_len)`
- 外层直接把结果 copy 到预分配的 `valid_ngram_draft` buffer

**影响**

- 少一次临时数组分配
- 更适合高频 batch 路径

### 2.5 降低 JIT warmup 成本

**原因**

旧版 warmup 会分配 `1024 x max_model_len` 的大数组，`max_model_len` 较大时会放大初始化成本和内存压力。

**解决方案**

- 改成更小的 warmup shape
- 只保留足够触发 Numba 编译的最小形状

**影响**

- 初始化更轻
- 不改变运行时语义

### 2.6 新增可选的 `prompt_lookup_window`

**原因**

长 context 下，ngram matcher 的主要成本来自扫描整段 history。默认扫全历史最安全，但也是最贵的。

**解决方案**

- 新增 `prompt_lookup_window`
- 默认关闭
- 开启后只扫描最近 N 个 token

**影响**

- 默认行为不变
- 开启时会用更小工作集换取更低延迟
- 可能减少 proposal / acceptance，但不会破坏最终正确性

### 2.7 benchmark 补齐

**原因**

原本 `benchmark_ngram_proposer.py` 既不适合离线 CPU 环境，也不方便拆出 filtering / matcher / integration overhead。

**解决方案**

- 改写 `benchmark_ngram_proposer.py`
- 新增 CPU-only 的 `benchmark_ngram_compare.py`
- 支持 old-vs-new 对照
- 输出 `avg / p50 / p95 / p99 / max`

## 3. 如何重现 benchmark

以下命令都可以在 CPU-only 环境下执行。

### 3.1 量当前 proposer 路径

```bash
PYTHONPATH=. python benchmarks/benchmark_ngram_proposer.py \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

### 3.2 量当前 proposer 路径 + 限制搜索窗口

```bash
PYTHONPATH=. python benchmarks/benchmark_ngram_proposer.py \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5 \
  --search-window 64
```

### 3.3 量 runner integration 开销

```bash
PYTHONPATH=. python benchmarks/benchmark_ngram_proposer.py \
  --batched \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

### 3.4 量 old-vs-new 对照

在当前优化 commit 上，对照前一个 commit：

```bash
PYTHONPATH=. python benchmarks/benchmark_ngram_compare.py \
  --baseline-ref HEAD~1 \
  --num-req 64 \
  --num-token 1024 \
  --min-ngram 3 \
  --max-ngram 5 \
  --num-spec-token 3 \
  --compare-search-window 256 \
  --iterations 80 \
  --warmup 8
```

如果在未提交状态下比较工作树与当前 `HEAD`，则可改成：

```bash
PYTHONPATH=. python benchmarks/benchmark_ngram_compare.py \
  --baseline-ref HEAD \
  --num-req 64 \
  --num-token 1024 \
  --min-ngram 3 \
  --max-ngram 5 \
  --num-spec-token 3 \
  --compare-search-window 256 \
  --iterations 80 \
  --warmup 8
```

## 4. 本次实测结果

### 4.1 当前版本内，full-history vs search-window

场景：

- `8 req x 128 tokens`
- `max_ngram=5`
- `50 iterations`

结果：

- proposer-only `Propose Avg`: `23.752 us -> 22.412 us`
- 改善约 `5.6%`

- batched integration `Runner Propose Avg`: `26.314 us -> 23.677 us`
- 改善约 `10.0%`

### 4.2 修改前 vs 修改后

场景：

- `64 req x 1024 tokens`
- `25%` request 的 `sampled_token_ids` 为空
- `min_ngram=3`
- `max_ngram=5`
- `k=3`

结果：

- 旧版 `propose avg`: `170.09 us`
- 新版 `propose avg`（full-history）: `127.08 us`
- 改善：`39.39 us`
- 改善幅度：`23.7%`

- 新版 `propose avg`（window=256）: `110.13 us`
- 相对旧版改善：`56.34 us`
- 改善幅度：`33.8%`

- 旧版 runner path avg: `161.50 us`
- 新版 runner path avg（full-history）: `127.70 us`
- 改善：`33.80 us`
- 改善幅度：`20.9%`

- 新版 runner path avg（window=256）: `109.58 us`
- 相对旧版改善：`51.92 us`
- 改善幅度：`32.2%`

## 5. 结论

这次优化可以分成两层看：

1. **不改默认行为时**
   - 仅靠 hot-path 优化与 integration cleanup
   - `propose()` 大约改善 `24%`
   - runner integration 大约改善 `21%`

2. **允许限制搜索窗口时**
   - 在长 context 下收益更明显
   - 这次样本中可达到 `33% ~ 34%`

默认情况下，`prompt_lookup_window` 是关闭的，所以功能风险较低；开启后更像是“用 proposal 质量换 latency”的可选开关，适合后续按 workload 调优。
