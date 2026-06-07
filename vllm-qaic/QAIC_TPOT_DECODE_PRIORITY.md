# QAIC vLLM 0.21:高並行 TPOT 變差的原因、修正與測試說明

> 給團隊 / 同事的說明文件。可直接閱讀或轉發。

## 一、為什麼 TPOT 在高並行會爆掉(根因)

QAIC 上,「處理新請求(prefill)」和「逐字生成(decode)」是**兩張各自固定形狀的編譯圖,只能輪流跑、不能同時**。

而新版 vLLM(0.21)的排程器**預設會把 prefill 和 decode 塞進同一個 step**(chunked prefill,這是為 GPU 設計的優化——GPU 一次運算就能同時處理兩者)。但 QAIC 一次只能做一件事,所以一個混合 step 會「先跑 decode 圖,再跑 prefill 圖」,時間相加。

由於 **TPOT(每生成一個字的間隔)= 一個 step 的時間**,高並行時幾乎每個 step 都夾著一次很慢的 prefill,等於**每生成一個字都要先等一次 prefill**,TPOT 因此暴增(從數據反推:一次 prefill 約 1150ms,被每個 token 攤上)。

對照舊版 0.10:當時 prefill 和 decode 是**分開的 step**,所以 decode step 很乾淨(TPOT 低 ~256ms),代價是新請求等比較久(TTFT 高)。0.21 把這個取捨翻面了——TTFT 變好、但 TPOT 變爛。

**一句話:這不是卡上 kernel 的問題,是 vLLM 的 GPU 式混合排程,撞到 QAIC「兩張圖只能輪流跑」的硬體模型。**

## 二、我們做了哪些改動

### 1. phase-timing profiler(零行為改變)
開 `VLLM_QAIC_PROFILE=1` 後,每個 step 會 log 各階段耗時,並標記是否為混合步:
```
QAIC-PROF reqs=.. decodes=.. prefills=.. mixed=1 |
  pack=..ms decode_qpc=..ms prefill_qpc=..ms merge=..ms logits=..ms sample=..ms bookkeeping=..ms
```
用途:直接看出「混合 step 佔比」、以及 decode 圖 vs prefill 圖各自吃多少時間,確認上面的根因。

### 2. decode 優先排程器(decode-priority scheduler)
核心邏輯:**只要還有 decode 在進行,就先不要放新的 prefill 進來**,讓 decode step 保持乾淨(TPOT 低)。為了避免新請求餓死,設了放行條件:
- 連續延後達 N 步後,強制開一個 prefill step;
- decode 負載低於門檻時,放行 prefill;
- 放行時給足預算,讓 prefill 一次盡量做完(避免被切成很多小塊、每塊都全額重算)。

實作保守安全:**不清空等待佇列、不影響搶佔(preemption)**;預設自動啟用,可用環境變數關閉。

可調環境變數:
| 變數 | 預設 | 作用 |
|---|---|---|
| `QAIC_PREFILL_EVERY_N_STEPS` | 8 | 最多延後幾步就強制開一次 prefill(調小→TTFT 較好、TPOT 略差;調大→反之) |
| `QAIC_PREFILL_RESUME_FRAC` | 0.5 | decode 負載低於此比例就放行 prefill |
| `QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER` | (未設) | 設 `1` 完全關閉此排程器,退回原本行為(用來做對照) |

## 三、請協助測試的內容

在 QAIC 機器上,用相同模型,跑並行度 **16 / 32 / 64**,比較下面三種設定,量 **TTFT、TPOT、throughput**。

### (A) 對照組——關掉新排程器、開 profiler
```bash
QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER=1 VLLM_QAIC_PROFILE=1 \
  vllm serve <模型> --additional-config '{...}'
```
觀察 log 裡 `mixed=1` 的 step 佔比、以及 `prefill_qpc` 的毫秒數(預期:很多混合 step、prefill 很重)。

### (B) 實驗組——開新排程器(預設就會開)、開 profiler
```bash
VLLM_QAIC_PROFILE=1 vllm serve <模型> --additional-config '{...}'
```

### 請回報這四件事
1. `mixed=1` 的 step 佔比,B 是否明顯**下降**(decode step 變乾淨)。
2. **TPOT** 是否回到接近舊版 0.10 的水準(~250ms 量級)。
3. **TTFT** 退化幅度是否可接受。
4. 有沒有 **卡住 / 請求一直排不到(starvation)**。

### (C) 調參(若 TTFT 退化太多)
試著調 `QAIC_PREFILL_EVERY_N_STEPS`(例如 4 或 16)與 `QAIC_PREFILL_RESUME_FRAC`,找 TTFT 與 TPOT 的平衡點,回報哪組參數最好。

> **已知殘留**:若某次放行的 prefill 是長 prompt、沒在那一步跑完,它會留到下一步、仍可能和 decode 同步。請在 profiler log 裡特別看「強制放行 prefill 之後的幾個 step 是否還有 `mixed=1`」,順便回報。緩解方向:把 `max_num_batched_tokens` 設大到讓 prompt 在一個放行步內做完。

## 四、相關程式碼
- profiler:`vllm-qaic/vllm_qaic/model_runner.py`(`execute_model`,`VLLM_QAIC_PROFILE` 區段)
- 排程器:`vllm-qaic/vllm_qaic/scheduler.py`(`QaicDecodePriorityScheduler`)
- 掛載點:`vllm-qaic/vllm_qaic/platform.py`(`scheduler_cls` 設定)
- 政策單元測試:`vllm-qaic/tests/test_qaic_scheduler.py`
