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
BUILD_BOOTSTRAP_PACKAGES=(
    "cmake>=3.26.1"
    "ninja"
    "packaging>=24.2"
    "setuptools>=77.0.3,<81.0.0"
    "setuptools-scm>=8.0"
    "wheel"
    "jinja2"
)

append_cmake_arg() {
    local arg="$1"
    if [[ " ${CMAKE_ARGS:-} " != *" ${arg} "* ]]; then
        export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }${arg}"
    fi
}

prepend_env_path() {
    local var_name="$1"
    local path="$2"
    local current="${!var_name:-}"

    case ":${current}:" in
        *":${path}:"*) ;;
        *)
            export "${var_name}=${path}${current:+:${current}}"
            ;;
    esac
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

find_python_cuda_include_dirs() {
    "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
from pathlib import Path
import site
import sysconfig

roots = []
seen_roots = set()

def add_root(path_str):
    if not path_str:
        return
    path = Path(path_str)
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            roots.append(path)

for site_path in getattr(site, "getsitepackages", lambda: [])():
    add_root(site_path)

add_root(getattr(site, "getusersitepackages", lambda: None)())
add_root(sysconfig.get_path("purelib"))
add_root(sysconfig.get_path("platlib"))

seen_dirs = set()
for root in roots:
    for rel_path in (
        "nvidia/cuda_runtime/include",
        "nvidia/cublas/include",
        "nvidia/cusparse/include",
        "nvidia/cusolver/include",
        "nvidia/cuda_nvrtc/include",
    ):
        include_dir = root / rel_path
        if include_dir.is_dir():
            resolved = str(include_dir.resolve())
            if resolved not in seen_dirs:
                seen_dirs.add(resolved)
                print(resolved)
PY
}

find_cuda_include_dirs() {
    local cuda_home="${1:-}"
    local candidates=()
    local candidate
    local path_var
    local value
    local -A seen=()
    local -a found=()
    local -a include_dirs=()
    local has_cublas=0
    local has_cusparse=0
    local has_cusolver=0
    local has_nvrtc_headers=0

    if [[ -n "${CUDA_INCLUDE_DIR:-}" && -d "${CUDA_INCLUDE_DIR}" ]]; then
        candidates+=("${CUDA_INCLUDE_DIR}")
    fi

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
        "/usr/include"
        "/usr/include/x86_64-linux-gnu"
    )

    for path_var in CPATH CPLUS_INCLUDE_PATH; do
        value="${!path_var:-}"
        if [[ -n "${value}" ]]; then
            IFS=':' read -r -a include_dirs <<< "${value}"
            candidates+=("${include_dirs[@]}")
        fi
    done

    while IFS= read -r candidate; do
        if [[ -n "${candidate}" ]]; then
            candidates+=("${candidate}")
        fi
    done < <(find_python_cuda_include_dirs)

    for candidate in "${candidates[@]}"; do
        if [[ -z "${candidate}" || ! -d "${candidate}" ]]; then
            continue
        fi

        candidate="$(cd -- "${candidate}" && pwd)"
        if [[ -n "${seen[${candidate}]:-}" ]]; then
            continue
        fi

        if [[ -f "${candidate}/cublas_v2.h" || -f "${candidate}/cusparse.h" || -f "${candidate}/cusolverDn.h" || -f "${candidate}/cusolver_common.h" || -f "${candidate}/nvrtc.h" ]]; then
            seen["${candidate}"]=1
            found+=("${candidate}")
            if [[ -f "${candidate}/cublas_v2.h" ]]; then
                has_cublas=1
            fi
            if [[ -f "${candidate}/cusparse.h" ]]; then
                has_cusparse=1
            fi
            if [[ -f "${candidate}/cusolverDn.h" || -f "${candidate}/cusolver_common.h" ]]; then
                has_cusolver=1
            fi
            if [[ -f "${candidate}/nvrtc.h" ]]; then
                has_nvrtc_headers=1
            fi
        fi
    done

    if (( has_cublas && has_cusparse && has_cusolver && has_nvrtc_headers )); then
        printf '%s\n' "${found[@]}"
        return 0
    fi

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
CUDA_INCLUDE_DIRS_DETECTED=()
CUDA_INCLUDE_DIRS_RAW=""
if CUDA_INCLUDE_DIRS_RAW="$(find_cuda_include_dirs "${CUDA_HOME_DETECTED:-}")"; then
    mapfile -t CUDA_INCLUDE_DIRS_DETECTED <<< "${CUDA_INCLUDE_DIRS_RAW}"
    for include_dir in "${CUDA_INCLUDE_DIRS_DETECTED[@]}"; do
        if [[ -f "${include_dir}/cuda_runtime.h" || -f "${include_dir}/cuda.h" ]]; then
            CUDA_INCLUDE_DIR_DETECTED="${include_dir}"
            break
        fi
    done

    if [[ -z "${CUDA_INCLUDE_DIR_DETECTED}" ]]; then
        CUDA_INCLUDE_DIR_DETECTED="${CUDA_INCLUDE_DIRS_DETECTED[0]}"
    fi

    export CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR_DETECTED}"
    # Ensure nvcc and host compiler see CUDA headers even when Python wheels
    # install them across separate include directories.
    for include_dir in "${CUDA_INCLUDE_DIRS_DETECTED[@]}"; do
        prepend_env_path CPATH "${include_dir}"
        prepend_env_path CPLUS_INCLUDE_PATH "${include_dir}"
    done
else
    if [[ "${VLLM_USE_PRECOMPILED:-}" =~ ^(1|true)$ || "${ALLOW_NO_CUDA_HEADERS:-0}" == "1" ]]; then
        echo "warning: CUDA headers (cublas_v2.h/cusparse.h/cusolverDn.h/nvrtc.h) not found; build may fail without dev headers." >&2
    else
        echo "error: CUDA headers (cublas_v2.h/cusparse.h/cusolverDn.h/nvrtc.h) not found." >&2
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
if [[ ${#CUDA_INCLUDE_DIRS_DETECTED[@]} -gt 1 ]]; then
    echo "Using additional CUDA include dirs: ${CUDA_INCLUDE_DIRS_DETECTED[*]}"
fi
if [[ -n "${CMAKE_ARGS:-}" ]]; then
    echo "Using CMAKE_ARGS=${CMAKE_ARGS}"
fi

# When build isolation is disabled, setuptools.build_meta imports run inside the
# active environment, so metadata generation needs these backend packages
# preinstalled.
"${PYTHON_BIN}" -m pip install "${BUILD_BOOTSTRAP_PACKAGES[@]}" 2>&1 | tee "${INSTALL_LOG}"
bootstrap_status="${PIPESTATUS[0]}"
if [[ "${bootstrap_status}" -ne 0 ]]; then
    exit "${bootstrap_status}"
fi

"${PYTHON_BIN}" -m pip install --no-build-isolation -e . "$@" 2>&1 | tee "${INSTALL_LOG}"
exit "${PIPESTATUS[0]}"
