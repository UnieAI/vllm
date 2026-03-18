#!/usr/bin/env bash
set -euo pipefail

find_executable() {
  local candidate
  for candidate in "$@"; do
    [[ -n "${candidate}" ]] || continue
    if [[ "${candidate}" == */* ]]; then
      [[ -x "${candidate}" ]] && { echo "${candidate}"; return 0; }
    elif command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

CURRENT_CC="${CC:-}"
CURRENT_CXX="${CXX:-}"
CURRENT_CUDAHOSTCXX="${CUDAHOSTCXX:-}"

if [[ -n "${CURRENT_CC}" ]] && ! find_executable "${CURRENT_CC}" >/dev/null 2>&1; then
  unset CC
fi
if [[ -n "${CURRENT_CXX}" ]] && ! find_executable "${CURRENT_CXX}" >/dev/null 2>&1; then
  unset CXX
fi
if [[ -n "${CURRENT_CUDAHOSTCXX}" ]] && ! find_executable "${CURRENT_CUDAHOSTCXX}" >/dev/null 2>&1; then
  unset CUDAHOSTCXX
fi

CC_BIN="$(find_executable "${CC:-}" gcc gcc-12 gcc-11 cc || true)"
CXX_BIN="$(find_executable "${CXX:-}" g++ g++-12 g++-11 c++ || true)"

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
  echo "Error: Could not find a working C/C++ compiler for Triton JIT."
  exit 1
fi

export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
echo "Using compilers for runtime JIT: CC=${CC} CXX=${CXX} CUDAHOSTCXX=${CUDAHOSTCXX}"

ATTN_BACKEND="${VLLM_ATTN_BACKEND:-FLASH_ATTN}"

# Keep cache + AOT enabled by default for performance.
# If you need a one-off clean run: VLLM_CLEAN_CACHE_ON_START=1 bash run-ngram.sh
if [[ "${VLLM_CLEAN_CACHE_ON_START:-0}" == "1" ]]; then
  rm -rf ~/.cache/vllm/torch_compile_cache ~/.cache/vllm/torch_aot_compile
fi
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-0}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
echo "Using attention backend: ${ATTN_BACKEND} (VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE}, VLLM_USE_AOT_COMPILE=${VLLM_USE_AOT_COMPILE})"

vllm serve /modelz/Qwen/Qwen3-8B \
  --served-model-name test \
  --host 0.0.0.0 \
  --port 8976 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype float16 \
  --max-num-batched-tokens 16384 \
  --attention-backend "${ATTN_BACKEND}" \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"prompt_lookup_min":3,"prompt_lookup_max":8}' 2>&1 | tee run.log
