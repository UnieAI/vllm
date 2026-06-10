# QAIC Paged KV Branch Update

## 本次操作

- 目前 repo 已從 `v1_vllm021` 切換到 `roy/qaic-paged-kv`。
- 本地 branch `roy/qaic-paged-kv` 追蹤遠端 `origin/roy/qaic-paged-kv`。
- 這次沒有另外 clone repo；原本主管訊息裡提到的 QEfficient repo 需要 GitHub private repo 權限，這邊先依照最新指示只切換目前 vLLM branch。

## 相對 `v1_vllm021` 的主要變更

本 branch 在 `v1_vllm021` 的 QAIC plugin 基礎上，加入 paged KV / Mooncake 相關的硬體驗證準備工作。

整體 diff 摘要：

- 20 個檔案變更
- 約 1,648 行新增
- 約 157 行刪除

主要新增文件：

- `MOONCAKE_QAIC_HANDOFF.md`
  - QAIC box-side handoff checklist。
  - 包含 paged QPC 編譯、serve 參數、精度驗證、吞吐測試、decode latency、Mooncake 後續接線事項。
- `QAIC_PAGED_ATTENTION_REPORT.md`
  - paged attention 實作報告。
  - 說明 QEfficient 與 vLLM-QAIC plugin 各自負責的部分、CPU verification 狀態、box-gated TODO。
- `vllm-qaic/QAIC_TPOT_DECODE_PRIORITY.md`
  - QAIC TPOT / decode-priority scheduler 測試與分析文件。

主要程式變更：

- `vllm-qaic/vllm_qaic/platform.py`
  - paged mode 下將 block size 設為 `page_size`。
  - 允許 prefix caching，用於後續 Mooncake KV reuse。
- `vllm-qaic/vllm_qaic/worker.py`
  - paged mode 下使用 `num_blocks` 作為 block pool 大小。
  - 不再沿用非 paged 的 `max_num_seqs + 1` 假設。
- `vllm-qaic/vllm_qaic/model_runner.py`
  - paged mode 下準備並餵入完整 block table。
  - 新增 KV transfer / staging arena 相關接線點。
- `vllm-qaic/vllm_qaic/model_loader.py`
  - 將 `block_table` 放進 prefill / decode 的 session input。
- `vllm-qaic/vllm_qaic/scheduler.py`
  - 新增 decode-priority scheduler 相關邏輯，用於高併發下改善 TPOT。
- `vllm-qaic/vllm_qaic/fused_decode_window.py`
  - 新增 decode-window / short-prefill admission 相關工具。
- `vllm-qaic/vllm_qaic/kv_connector/qaic_kv_staging.py`
  - 新增 QAIC host-side KV staging arena。
- `vllm-qaic/vllm_qaic/kv_connector/qaic_mooncake_store_connector.py`
  - 新增 Mooncake Store connector skeleton，後續接 QAIC card KV read/write。
- `vllm-qaic/tools/parse_qaic_prof.py`
  - 新增 QAIC profiling log parser。
- `vllm-qaic/tools/run_tpot_ab.sh`
  - 新增 TPOT A/B 測試腳本。

新增測試：

- `vllm-qaic/tests/test_qaic_scheduler.py`
- `vllm-qaic/tests/test_qaic_kv_staging.py`
- `vllm-qaic/tests/test_fused_decode_window_routing.py`
- `vllm-qaic/tests/test_parse_qaic_prof.py`

## 目前 branch 的重點契約

QAIC paged KV 測試時要特別注意：

- `compile(..., num_blocks=N)` 的 `N` 必須和 serve 時的 `--additional-config '{"num_blocks":N}'` 一致。
- `page_size` / `paged_block_size` 建議先照主管訊息使用 `128`。
- paged KV 需要保留 null block；pool sizing 要把 reserved padding block 算進去。
- `mxint8_kv_cache=True` 在 paged KV 下還沒完全驗證，若使用要確認 host staging arena dtype 和 card KV pool dtype 一致。
- `QaicModelRunner.initialize_kv_cache` 是後續把 staging arena 接到 KV transfer / Mooncake 的主要位置。

## 接下來 QAIC box 上要驗證

1. 編譯 paged QPC。
   - 確認 QEfficient 的 paged ONNX 可以被 AIC compiler 成功編成 QPC。
   - 最大風險是大 block pool 的 `GatherND` / `ScatterND` 是否能成功 lowering，且效能可接受。

2. 建立非 paged baseline QPC。
   - 同模型、同 `ctx_len=8192`。
   - 用來做精度與吞吐對照。

3. 精度驗證。
   - 同一批 prompt。
   - greedy decode，`temperature=0`。
   - paged output 與 baseline output 逐字比對。

4. 吞吐與 `max-num-seqs` 驗證。
   - paged / baseline 分別把 `--max-num-seqs` 往上加到 OOM 前。
   - 記錄最大可承受併發、tok/s、TTFT、ITL。

5. decode latency 驗證。
   - 比較 paged block-pool gather 和 baseline contiguous KV decode latency。
   - 如果 gather 太慢，可能抵消 paged KV 提升併發帶來的收益，這是 go / no-go 關鍵。

## 建議輸出表格

| 項目 | baseline | paged KV | 結論 |
|---|---:|---:|---|
| greedy output | TBD | TBD | 是否一致 |
| 最大 `max-num-seqs` | TBD | TBD | paged 是否更高 |
| tok/s | TBD | TBD | 提升倍數 |
| TTFT | TBD | TBD | 是否退化 |
| ITL / decode latency | TBD | TBD | gather 是否可接受 |
| OOM 臨界點 | TBD | TBD | 記憶體優勢 |

## 目前狀態

已完成：目前 vLLM repo 已切到 `roy/qaic-paged-kv`。

尚未完成：QEfficient `unieai/paged-kv` repo 尚未在本 workspace 內取得；需要 GitHub private repo 權限或在 QAIC host 上用可登入的 GitHub credential 拉取。
