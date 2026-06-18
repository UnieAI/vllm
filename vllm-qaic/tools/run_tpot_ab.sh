#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Decode-priority scheduler A/B on the QAIC box (single card).
# For each concurrency, serves the model twice — scheduler OFF (baseline) and ON —
# with VLLM_QAIC_PROFILE=1, runs a load benchmark, then summarizes the profiler logs
# (mixed-step ratio + pure-decode vs mixed step time ~= TPOT) via parse_qaic_prof.py.
#
# Adapt the two TODO blocks (serve args / bench command) to your box. Requires a
# running QAIC host with the QPC compiled for this model + additional-config.
#
# Usage:
#   MODEL=Qwen/Qwen2.5-7B-Instruct \
#   ADDITIONAL_CONFIG='{"num_cores":16}' \
#   CONCURRENCIES="16 32 64" \
#   ./run_tpot_ab.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL}
ADDITIONAL_CONFIG=${ADDITIONAL_CONFIG:-'{}'}
CONCURRENCIES=${CONCURRENCIES:-"16 32 64"}
NUM_PROMPTS=${NUM_PROMPTS:-200}
PORT=${PORT:-8000}
OUTDIR=${OUTDIR:-tpot_ab_out}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSE="${HERE}/parse_qaic_prof.py"
mkdir -p "$OUTDIR"

wait_ready() {  # poll the OpenAI models endpoint until the server is up
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "server did not become ready" >&2; return 1
}

run_one() {  # $1=label(on|off)  $2=disable_flag("1" disables the scheduler)
  local label="$1" disable="$2" c log bench
  for c in $CONCURRENCIES; do
    log="${OUTDIR}/${label}_c${c}.server.log"
    bench="${OUTDIR}/${label}_c${c}.bench.txt"
    echo ">>> [${label}] concurrency=${c}: serving..."
    # --- TODO(box): your serve command + flags (tensor-parallel, dtype, etc.) ---
    QAIC_DISABLE_DECODE_PRIORITY_SCHEDULER="${disable}" VLLM_QAIC_PROFILE=1 \
      vllm serve "$MODEL" --port "$PORT" --additional-config "$ADDITIONAL_CONFIG" \
      >"$log" 2>&1 &
    local srv=$!
    wait_ready
    # --- TODO(box): swap in the SAME bench client that produced the 0529/0602 table
    #     so the TTFT/TPOT/throughput numbers are comparable. ---
    vllm bench serve --model "$MODEL" --base-url "http://localhost:${PORT}" \
      --max-concurrency "$c" --num-prompts "$NUM_PROMPTS" | tee "$bench" || true
    kill "$srv" 2>/dev/null || true; wait "$srv" 2>/dev/null || true
  done
}

echo "=== A: scheduler OFF (baseline) ==="
run_one off 1
echo "=== B: scheduler ON ==="
run_one on ""

echo
echo "################  PROFILER SUMMARY (mixed ratio + step time ~= TPOT)  ################"
for c in $CONCURRENCIES; do
  echo "----- concurrency ${c} -----"
  python3 "$PARSE" "${OUTDIR}/off_c${c}.server.log" "${OUTDIR}/on_c${c}.server.log"
done
echo "Bench client TTFT/TPOT/throughput: see ${OUTDIR}/{off,on}_c*.bench.txt"
echo "GO if: ON drops mixed-ratio and mean step time toward the pure-decode value,"
echo "       client TPOT approaches the 0529 baseline, with acceptable TTFT regression,"
echo "       and no starvation/hang."
