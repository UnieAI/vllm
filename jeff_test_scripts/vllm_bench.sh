#!/bin/bash
# =============================================================================
# vllm_bench.sh — 使用 vllm bench serve 跑 philschmid/mt-bench benchmark
#
# Endpoint: /v1/completions
# Dataset:  philschmid/mt-bench (HF dataset, 80 prompts)
# Output:   1024 tokens
# Concurrency: 1, 8, 512 (sequential runs)
#
# Usage:
#   ./vllm_bench.sh [BASE_URL] [MODEL]
#
# Defaults:
#   BASE_URL=http://localhost:2167
#   MODEL=auto (auto-detected from server)
# =============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:2167}"
MODEL="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_BIN="${SCRIPT_DIR}/../.venv/bin/vllm"

# Fallback: try system vllm if .venv not available
if [ ! -f "${VLLM_BIN}" ]; then
    VLLM_BIN="$(which vllm 2>/dev/null || echo vllm)"
fi

echo "============================================"
echo "  vLLM Benchmark: philschmid/mt-bench"
echo "============================================"
echo "  Server:    ${BASE_URL}"
echo "  Endpoint:  /v1/completions"
echo "  Dataset:   philschmid/mt-bench"
echo "  Output:    1024 tokens"
echo "  Concurrency levels: 1, 8, 512"
echo "============================================"
echo ""

# Check server is reachable
echo "Checking server health..."
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: Server at ${BASE_URL} is not reachable."
    echo "Please start the server first."
    exit 1
fi
echo "Server is healthy."

# Auto-detect model if not specified
if [ -z "${MODEL}" ]; then
    MODEL=$(curl -sf "${BASE_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
    echo "Auto-detected model: ${MODEL}"
fi
echo ""

# Results directory
RESULTS_DIR="${SCRIPT_DIR}/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Run benchmarks at each concurrency level
for CONCURRENCY in 1 8 512; do
    echo "--------------------------------------------"
    echo "  Running: concurrency=${CONCURRENCY}"
    echo "--------------------------------------------"

    RESULT_FILE="${RESULTS_DIR}/bench_c${CONCURRENCY}.json"

    "${VLLM_BIN}" bench serve \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --endpoint "/v1/completions" \
        --dataset-name "hf" \
        --dataset-path "philschmid/mt-bench" \
        --hf-split "train" \
        --hf-output-len 1024 \
        --num-prompts 80 \
        --max-concurrency "${CONCURRENCY}" \
        --request-rate inf \
        --save-result \
        --result-dir "${RESULTS_DIR}" \
        --result-filename "bench_c${CONCURRENCY}.json" \
        2>&1 | tee "${RESULTS_DIR}/bench_c${CONCURRENCY}.log"

    echo ""
    echo "  Completed concurrency=${CONCURRENCY}"
    echo ""
done

echo "============================================"
echo "  All benchmarks complete!"
echo "  Results: ${RESULTS_DIR}"
echo "============================================"
echo ""

# Print summary
echo "Quick Summary:"
echo "--------------------------------------------"
for CONCURRENCY in 1 8 512; do
    LOG="${RESULTS_DIR}/bench_c${CONCURRENCY}.log"
    if [ -f "${LOG}" ]; then
        echo ""
        echo "--- Concurrency ${CONCURRENCY} ---"
        grep -E "Throughput|TTFT|ITL|latency" "${LOG}" 2>/dev/null || echo "  (see full log)"
    fi
done
