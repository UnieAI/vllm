# Paged Attention — 在 QAIC 機器上測試(同事 runbook,一步一步)

> 目標:確認我們的 paged attention 在真實 AI100 上 (1) 編得出來 (2) 答案正確 (3) 省記憶體/更快。
> 每一步都可以直接複製貼上。卡關就把該步的完整輸出貼回來。

---

## 步驟 0:你需要什麼
- 一台已裝好 Qualcomm AI SDK / `qaicrt` 的 AI100 機器。
- 終端機(terminal)操作。
- 確認 `uv` 有裝:`uv --version`;沒有就 `curl -LsSf https://astral.sh/uv/install.sh | sh`。

---

## 步驟 1:把兩個分支抓下來
```bash
# QEfficient(私有 repo,paged 的圖/算子在這)
git clone https://github.com/UnieAI/efficient-transformers-unieai.git
cd efficient-transformers-unieai && git checkout unieai/paged-kv && cd ..

# vLLM plugin(serve 用)
git clone https://github.com/UnieAI/vllm.git
cd vllm && git checkout roy/qaic-paged-kv && cd ..
```

## 步驟 2:裝環境
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ./efficient-transformers-unieai
uv pip install -e ./vllm/vllm-qaic
```

## 步驟 3(可選,~5 分鐘):先確認程式邏輯沒壞(這步不需要卡,任何機器都能跑)
```bash
cd efficient-transformers-unieai
for t in test_paged_kv_parity test_paged_cache_layer test_paged_qwen2_e2e \
         test_paged_onnx_export test_paged_qwen2_onnx test_paged_export_plumbing \
         test_paged_compile_spec bench_paged_vs_contiguous; do
  echo "=== $t ==="
  PYTHONPATH=$PWD python tests/customop/$t.py 2>&1 | grep -E "PASS|FAIL|saving"
done
cd ..
```
**預期**:每個都印 `PASS`;最後一個印 `saving 86.7% (7.5x smaller)`。
有任何 `FAIL` 就貼回來。

---

## 步驟 4 — 【第一關】編譯 paged QPC(最重要:AI100 編不編得出來)

建立檔案 `compile_paged.py`,貼進去:
```python
from QEfficient import QEFFAutoModelForCausalLM as M

m = M.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
onnx = m.export(paged_kv=True, paged_block_size=128)
qpc = m.compile(
    onnx_path=onnx, paged_kv=True,
    page_size=128, num_blocks=512,     # 512 個 block 的共享池
    ctx_len=8192, num_cores=16,
)
print("PAGED QPC OK:", qpc)
```
執行:
```bash
python compile_paged.py
```
- **印出 `PAGED QPC OK: ...` = 第一關過。**
- **如果報錯**(特別是跟 `GatherND` / `ScatterND` / shape 有關):**把完整錯誤訊息貼回來** —— 這正是我們最想知道的(AI100 編譯器能不能編 block-pool 的 gather)。

## 步驟 5:編一個「對照組」非-paged QPC(同模型,用來比對)
建立 `compile_baseline.py`:
```python
from QEfficient import QEFFAutoModelForCausalLM as M

m = M.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
qpc = m.compile(
    prefill_seq_len=128, ctx_len=8192,
    full_batch_size=16, num_cores=16,  # 一般(非 paged)設定
)
print("BASELINE QPC OK:", qpc)
```
```bash
python compile_baseline.py
```

---

## 步驟 6 — 【第二關】精度:paged 跟非-paged 答案要一樣

開兩個終端機視窗(或用 `tmux`)。

**視窗 A:啟動 paged 服務**
```bash
source .venv/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 \
  --additional-config '{"paged_kv":true,"page_size":128,"num_blocks":512,"num_cores":16}'
```
**視窗 B:啟動非-paged 服務**
```bash
source .venv/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8002 \
  --additional-config '{"num_cores":16}'
```
等兩邊都印出 server ready 後,**第三個視窗**送同一個 prompt(temperature=0 = 不隨機,答案才可比對):
```bash
PROMPT='Explain paged attention in two sentences.'
for PORT in 8001 8002; do
  echo "=== port $PORT ==="
  curl -s http://localhost:$PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen/Qwen2.5-7B-Instruct\",\"prompt\":\"$PROMPT\",\"max_tokens\":64,\"temperature\":0}" \
    | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
done
```
**預期**:8001(paged)和 8002(非-paged)印出的文字**一模一樣**(或極接近)。
多試幾個 prompt。**不一樣就貼回來。**

---

## 步驟 7 — 【第三關】吞吐 / 記憶體:paged 應該能塞更多並行、更快

paged 省記憶體 → 可以把 `max_num_seqs` 設更大。比較兩者能撐的最大並行 + tok/s。

**對 paged 服務(視窗 A,改用更大的並行重啟):**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --max-num-seqs 64 \
  --additional-config '{"paged_kv":true,"page_size":128,"num_blocks":512,"num_cores":16}'
```
**跑壓測(第三視窗):**
```bash
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8001 --max-concurrency 64 --num-prompts 200
```
記下印出的 **throughput (tok/s)、TTFT、TPOT**。
然後對非-paged 服務(8002)用它能撐的最大 `--max-num-seqs` 做同樣壓測,**對照**。
**預期**:paged 能設更大的 `max-num-seqs`、總 tok/s 更高(短序列工作負載更明顯)。

---

## 三關通過判準(回報用)
| 關 | 通過 = |
|---|---|
| 1 編譯 | `compile_paged.py` 印出 QPC 路徑,無錯 |
| 2 精度 | paged 與非-paged 同 prompt(temp=0)輸出一致 |
| 3 吞吐 | paged 能設更大 max-num-seqs、tok/s ≥ 非-paged,且 decode 沒明顯變慢 |

**任何一關卡住:把那一步的完整指令 + 完整輸出貼回來即可。**

> 名詞:`page_size`=每個 block 幾個 token;`num_blocks`=共享池有幾個 block(要 ≥ 同時服務的總 token 量 / page_size);三個地方的 `num_blocks`/`page_size` 必須一致(編譯、serve 的 additional-config)。
