#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

default_python_bin() {
    if command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
    fi

    printf '%s\n' "python"
}

PYTHON_BIN="${PYTHON_BIN:-$(default_python_bin)}"
INSTALL_LOG="${INSTALL_LOG:-install.log}"

append_cmake_arg() {
    local arg="$1"
    if [[ " ${CMAKE_ARGS:-} " != *" ${arg} "* ]]; then
        export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }${arg}"
    fi
}

resolve_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
        printf '%s\n' "${CUDA_HOME}"
        return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(command -v nvcc)"
        printf '%s\n' "$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
        return 0
    fi

    return 1
}

find_nvrtc() {
    if [[ -n "${CUDA_NVRTC_LIBRARY:-}" && -f "${CUDA_NVRTC_LIBRARY}" ]]; then
        printf '%s\n' "${CUDA_NVRTC_LIBRARY}"
        return 0
    fi

    local cuda_home="${1:-}"
    local candidates=()

    if [[ -n "${cuda_home}" ]]; then
        candidates+=(
            "${cuda_home}/lib64/libnvrtc.so"
            "${cuda_home}/targets/x86_64-linux/lib/libnvrtc.so"
            "${cuda_home}/lib64/libnvrtc.so."*
            "${cuda_home}/targets/x86_64-linux/lib/libnvrtc.so."*
        )
    fi

    candidates+=(
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so"
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so."*
        "/usr/local/cuda/lib64/libnvrtc.so"
        "/usr/local/cuda/lib64/libnvrtc.so."*
        "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so"
        "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so."*
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    if command -v ldconfig >/dev/null 2>&1; then
        local ldconfig_match
        ldconfig_match="$(ldconfig -p 2>/dev/null | awk '/libnvrtc\.so/ { print $NF; exit }')"
        if [[ -n "${ldconfig_match}" && -f "${ldconfig_match}" ]]; then
            printf '%s\n' "${ldconfig_match}"
            return 0
        fi
    fi

    return 1
}

find_cuda_include_dir() {
    if [[ -n "${CUDA_INCLUDE_DIR:-}" && -d "${CUDA_INCLUDE_DIR}" ]]; then
        if [[ -f "${CUDA_INCLUDE_DIR}/cusparse.h" || -f "${CUDA_INCLUDE_DIR}/cublas_v2.h" ]]; then
            printf '%s\n' "${CUDA_INCLUDE_DIR}"
            return 0
        fi
    fi

    local cuda_home="${1:-}"
    local candidates=()

    if [[ -n "${cuda_home}" ]]; then
        candidates+=(
            "${cuda_home}/include"
            "${cuda_home}/targets/x86_64-linux/include"
        )
    fi

    candidates+=(
        "/usr/local/cuda/include"
        "/usr/local/cuda/targets/x86_64-linux/include"
        "/usr/local/cuda-12/include"
        "/usr/local/cuda-12/targets/x86_64-linux/include"
        "/usr/local/cuda-12.9/include"
        "/usr/local/cuda-12.9/targets/x86_64-linux/include"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/cusparse.h" || -f "${candidate}/cublas_v2.h" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

CUDA_HOME_DETECTED=""
if CUDA_HOME_DETECTED="$(resolve_cuda_home)"; then
    export CUDA_HOME="${CUDA_HOME_DETECTED}"
    append_cmake_arg "-DCUDAToolkit_ROOT=${CUDA_HOME}"
fi

NVRTC_LIBRARY=""
if NVRTC_LIBRARY="$(find_nvrtc "${CUDA_HOME_DETECTED:-}")"; then
    export CUDA_NVRTC_LIBRARY="${NVRTC_LIBRARY}"
    append_cmake_arg "-DCUDA_nvrtc_LIBRARY=${NVRTC_LIBRARY}"
else
    echo "warning: libnvrtc.so was not found automatically; the install may still fail during CMake configure." >&2
fi

CUDA_INCLUDE_DIR_DETECTED=""
if CUDA_INCLUDE_DIR_DETECTED="$(find_cuda_include_dir "${CUDA_HOME_DETECTED:-}")"; then
    export CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR_DETECTED}"
    # Ensure nvcc and host compiler see CUDA headers like cusparse.h / cublas_v2.h.
    export CPATH="${CUDA_INCLUDE_DIR_DETECTED}${CPATH:+:${CPATH}}"
    export CPLUS_INCLUDE_PATH="${CUDA_INCLUDE_DIR_DETECTED}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
else
    if [[ "${VLLM_USE_PRECOMPILED:-}" =~ ^(1|true)$ || "${ALLOW_NO_CUDA_HEADERS:-0}" == "1" ]]; then
        echo "warning: CUDA headers (cusparse.h/cublas_v2.h) not found; build may fail without dev headers." >&2
    else
        echo "error: CUDA headers (cusparse.h/cublas_v2.h) not found." >&2
        echo "Install CUDA Toolkit dev headers or set CUDA_INCLUDE_DIR to a valid include path." >&2
        echo "If you cannot install headers, try: VLLM_USE_PRECOMPILED=1 ./scripts/install.sh" >&2
        exit 1
    fi
fi

# Reuse the already-installed torch/CUDA stack instead of creating an isolated
# build env that may lose local toolkit visibility.
export PIP_NO_BUILD_ISOLATION="${PIP_NO_BUILD_ISOLATION:-1}"

echo "Using ${PYTHON_BIN}: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
if [[ -n "${CUDA_HOME_DETECTED}" ]]; then
    echo "Using CUDA_HOME=${CUDA_HOME_DETECTED}"
fi
if [[ -n "${NVRTC_LIBRARY}" ]]; then
    echo "Using CUDA_NVRTC_LIBRARY=${NVRTC_LIBRARY}"
fi
if [[ -n "${CUDA_INCLUDE_DIR_DETECTED}" ]]; then
    echo "Using CUDA_INCLUDE_DIR=${CUDA_INCLUDE_DIR_DETECTED}"
fi
if [[ -n "${CMAKE_ARGS:-}" ]]; then
    echo "Using CMAKE_ARGS=${CMAKE_ARGS}"
fi

"${PYTHON_BIN}" -m pip install -e . "$@" 2>&1 | tee "${INSTALL_LOG}"
exit "${PIPESTATUS[0]}"
