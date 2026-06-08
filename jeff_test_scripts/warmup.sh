#!/bin/bash
# =============================================================================
# warmup.sh — 使用 neural-bridge/rag-dataset-12000 (train split) 對 vLLM server
# 發送 ~9.6K requests，讓 Adaptive Prefix Warmup 學習前綴頻率分布
#
# Usage:
#   ./warmup.sh [BASE_URL] [CONCURRENCY] [MAX_REQUESTS]
#
# Defaults:
#   BASE_URL=http://localhost:2167
#   CONCURRENCY=32
#   MAX_REQUESTS=9600 (全部 train split)
# =============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:2167}"
CONCURRENCY="${2:-32}"
MAX_REQUESTS="${3:-9600}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WARMUP_PY="${SCRIPT_DIR}/warmup_runner.py"

echo "============================================"
echo "  Adaptive Serving Warmup"
echo "============================================"
echo "  Server:       ${BASE_URL}"
echo "  Dataset:      neural-bridge/rag-dataset-12000 (train)"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Max requests: ${MAX_REQUESTS}"
echo "  Output tokens: 128"
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
echo ""

# Run the warmup
python3 "${WARMUP_PY}" \
    --base-url "${BASE_URL}" \
    --concurrency "${CONCURRENCY}" \
    --max-requests "${MAX_REQUESTS}" \
    --max-tokens 128

echo ""
echo "Warmup complete."
