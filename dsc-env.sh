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
