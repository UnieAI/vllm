#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

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

detect_cuda_mm() {
  local cuda_ver
  local cuda_root="${CUDA_HOME:-/usr/local/cuda}"
  if [[ -f "${cuda_root}/version.json" ]]; then
    cuda_ver="$(rg -No '"version"\s*:\s*"[0-9]+\.[0-9]+' "${cuda_root}/version.json" | head -n1 | rg -No '[0-9]+\.[0-9]+' || true)"
  fi
  if [[ -z "${cuda_ver:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_ver="$(nvcc --version | rg -No 'release\s+[0-9]+\.[0-9]+' | head -n1 | awk '{print $2}' || true)"
  fi
  if [[ -z "${cuda_ver:-}" ]]; then
    cuda_ver="12.4"
  fi
  echo "${cuda_ver}"
}

version_lt() {
  local a="$1"
  local b="$2"
  [[ "$(printf '%s\n' "${a}" "${b}" | sort -V | head -n1)" != "${b}" ]]
}

detect_nvcc_version() {
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | rg -No 'release\s+[0-9]+\.[0-9]+' | head -n1 | awk '{print $2}' || true
  fi
}

detect_gpu_compute_cap() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true
  fi
}

detect_all_gpu_compute_caps() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | tr -d '[:space:]' \
      | rg -N '^[0-9]+\.[0-9]+$' \
      | awk '!seen[$0]++' || true
  fi
}

choose_arch_list_for_old_nvcc() {
  local caps_raw="${1:-}"
  local out=""
  local cc mapped
  while IFS= read -r cc; do
    [[ -n "${cc}" ]] || continue
    mapped="${cc}"
    if [[ "${cc}" =~ ^12\. ]]; then
      mapped="9.0a+PTX"
    elif [[ "${cc}" =~ ^9\.0$ ]]; then
      mapped="9.0a+PTX"
    fi
    if [[ -z "${out}" ]]; then
      out="${mapped}"
    else
      case ";${out};" in
        *";${mapped};"*) ;;
        *) out="${out};${mapped}" ;;
      esac
    fi
  done <<< "${caps_raw}"

  if [[ -z "${out}" ]]; then
    out="9.0a+PTX"
  fi
  echo "${out}"
}

detect_torch_cuda_version() {
  python - <<'PY'
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    print("")
PY
}

choose_torch_cuda_arch_list() {
  local nvcc_ver="$1"
  local gpu_cc="$2"
  if [[ -n "${VLLM_TLLM_TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "${VLLM_TLLM_TORCH_CUDA_ARCH_LIST}"
    return 0
  fi
  if [[ "${gpu_cc}" =~ ^12\. ]]; then
    # CUDA < 12.8 cannot compile native sm_120; use SM90a PTX fallback.
    echo "9.0a+PTX"
    return 0
  fi
  if [[ "${gpu_cc}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "${gpu_cc}+PTX"
    return 0
  fi
  # Safe default for unknown GPUs on CUDA 12.x.
  echo "9.0+PTX"
}

first_torch_arch() {
  local archs="${1:-}"
  archs="${archs//,/;}"
  archs="${archs// /;}"
  local arch
  IFS=';' read -r -a _parts <<< "${archs}"
  for arch in "${_parts[@]}"; do
    [[ -n "${arch}" ]] && { echo "${arch}"; return 0; }
  done
  return 1
}

torch_arch_to_single_gencode_flag() {
  local arch_raw="${1:-}"
  local with_ptx=0
  local arch="${arch_raw}"
  if [[ "${arch}" == *+PTX ]]; then
    with_ptx=1
    arch="${arch%+PTX}"
  fi

  if [[ "${arch}" =~ ^([0-9]+)\.([0-9]+)([a-z]?)$ ]]; then
    local cc="${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
    if [[ "${with_ptx}" -eq 1 ]]; then
      echo "-gencode=arch=compute_${cc},code=[sm_${cc},compute_${cc}]"
    else
      echo "-gencode=arch=compute_${cc},code=sm_${cc}"
    fi
  fi
}

cuda_dev_libs_present() {
  local cuda_root="${1}"
  [[ -f "${cuda_root}/lib64/libnvrtc.so" || -f "${cuda_root}/targets/x86_64-linux/lib/libnvrtc.so" ]] || return 1
  [[ -f "${cuda_root}/lib64/libcublas.so" || -f "${cuda_root}/targets/x86_64-linux/lib/libcublas.so" ]] || return 1
  [[ -f "${cuda_root}/lib64/libcublasLt.so" || -f "${cuda_root}/targets/x86_64-linux/lib/libcublasLt.so" ]] || return 1
  return 0
}

ensure_cuda_dev_libs() {
  local cuda_root="${CUDA_HOME:-/usr/local/cuda}"
  local -a apt_prefix=()
  if cuda_dev_libs_present "${cuda_root}"; then
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: CUDA dev libraries missing (nvrtc/cublas/cublasLt) and apt-get is unavailable."
    return 1
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      apt_prefix=(sudo)
    else
      echo "Error: CUDA dev libraries are missing and script is not running as root."
      echo "Install matching CUDA dev packages, then rerun."
      return 1
    fi
  fi

  local mm dashed
  mm="$(detect_cuda_mm)"
  dashed="${mm/./-}"

  echo "CUDA dev libraries missing. Installing CUDA ${mm} development packages..."
  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y \
    "cuda-nvrtc-dev-${dashed}" \
    "libcublas-dev-${dashed}" \
    "cuda-cudart-dev-${dashed}" \
    "cuda-libraries-dev-${dashed}" || true

  if ! cuda_dev_libs_present "${cuda_root}"; then
    echo "Error: Required CUDA dev libraries still missing under ${cuda_root}."
    echo "Expected: libnvrtc.so, libcublas.so, libcublasLt.so"
    return 1
  fi
}

ensure_ninja() {
  local ninja_bin
  local -a apt_prefix=()
  ninja_bin="$(find_executable ninja || true)"
  if [[ -n "${ninja_bin}" ]]; then
    echo "${ninja_bin}"
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: Ninja is required for CMake build, but it was not found."
    return 1
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      apt_prefix=(sudo)
    else
      echo "Error: Ninja is missing and script is not running as root."
      return 1
    fi
  fi

  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y ninja-build

  ninja_bin="$(find_executable ninja || true)"
  if [[ -z "${ninja_bin}" ]]; then
    echo "Error: Failed to find ninja after installing ninja-build."
    return 1
  fi
  echo "${ninja_bin}"
}

clean_stale_cmake_cache() {
  local build_base="${VLLM_BUILD_BASE:-${REPO_DIR}/build}"
  local build_temp="${build_base}/temp"
  local build_lib="${build_base}/lib"
  local cache_file="${build_temp}/CMakeCache.txt"
  local ninja_file="${build_temp}/build.ninja"
  local cached_make_program
  local stale_reason=""

  if [[ "${VLLM_TLLM_FORCE_CLEAN_BUILD:-0}" == "1" ]]; then
    stale_reason="VLLM_TLLM_FORCE_CLEAN_BUILD=1"
  fi

  if [[ -z "${stale_reason}" && -f "${cache_file}" ]]; then
    cached_make_program="$(rg -No '^CMAKE_MAKE_PROGRAM:FILEPATH=.*' "${cache_file}" | cut -d= -f2- || true)"

    if [[ -n "${cached_make_program}" && ! -x "${cached_make_program}" ]]; then
      stale_reason="missing build tool: ${cached_make_program}"
    elif rg -Nq '/tmp/pip-build-env-' "${cache_file}"; then
      stale_reason="stale pip-build-env path in CMakeCache.txt"
    fi
  fi

  if [[ -z "${stale_reason}" && -f "${ninja_file}" ]]; then
    if rg -Nq '/tmp/pip-build-env-' "${ninja_file}"; then
      stale_reason="stale pip-build-env path in build.ninja"
    fi
  fi

  if [[ -n "${stale_reason}" ]]; then
    echo "Detected stale CMake build state (${stale_reason})."
    echo "Removing stale build cache at ${build_temp} and ${build_lib}"
    rm -rf "${build_temp}" "${build_lib}"
  fi
}

ensure_pybind11() {
  if python - <<'PY' >/dev/null 2>&1
import pybind11  # noqa: F401
PY
  then
    return 0
  fi

  echo "pybind11 not found in current Python env; installing it for TurboMind CMake..."
  python -m pip install --upgrade pybind11

  if ! python - <<'PY' >/dev/null 2>&1
import pybind11  # noqa: F401
PY
  then
    echo "Error: pybind11 install did not succeed."
    return 1
  fi
}

ensure_build_requirements() {
  if [[ -f "${REPO_DIR}/requirements/build.txt" ]]; then
    python -m pip install --upgrade -r "${REPO_DIR}/requirements/build.txt" pybind11
  else
    python -m pip install --upgrade pybind11
  fi
}

activate_cuda_root() {
  local cuda_root="$1"
  [[ -d "${cuda_root}" ]] || return 1
  [[ -x "${cuda_root}/bin/nvcc" ]] || return 1
  export CUDA_HOME="${cuda_root}"
  export CUDAToolkit_ROOT="${cuda_root}"
  export PATH="${cuda_root}/bin:${PATH}"
  export LD_LIBRARY_PATH="${cuda_root}/lib64:${cuda_root}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  return 0
}

ensure_blackwell_compatible_nvcc() {
  local gpu_cc="$1"
  local nvcc_ver="$2"
  local -a apt_prefix=()
  local nvcc128

  [[ "${gpu_cc}" =~ ^12\. ]] || return 0
  if [[ -n "${nvcc_ver}" ]] && ! version_lt "${nvcc_ver}" "12.8"; then
    return 0
  fi

  echo "Blackwell GPU detected with nvcc ${nvcc_ver:-unknown}; trying CUDA 12.8+ toolkit for build compatibility."

  if activate_cuda_root "/usr/local/cuda-12.8"; then
    return 0
  fi
  nvcc128="$(command -v nvcc-12.8 || true)"
  if [[ -n "${nvcc128}" ]]; then
    activate_cuda_root "$(cd "$(dirname "${nvcc128}")/.." && pwd)"
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: nvcc 12.8+ is required for Blackwell build, but apt-get is unavailable."
    return 1
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      apt_prefix=(sudo)
    else
      echo "Error: nvcc 12.8+ is required for Blackwell build, and script is not running as root."
      return 1
    fi
  fi

  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y \
    cuda-toolkit-12-8 \
    cuda-nvrtc-dev-12-8 \
    cuda-cudart-dev-12-8 \
    cuda-libraries-dev-12-8 \
    libcublas-dev-12-8 || true

  if ! activate_cuda_root "/usr/local/cuda-12.8"; then
    echo "Error: Failed to activate CUDA 12.8 toolkit for Blackwell build."
    return 1
  fi
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
  echo "Error: Could not find a working C/C++ compiler."
  echo "Install build tools (e.g. build-essential) and retry."
  exit 1
fi

export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

NVCC_VER="$(detect_nvcc_version)"
GPU_CC="$(detect_gpu_compute_cap)"
ALL_GPU_CCS="$(detect_all_gpu_compute_caps)"
TORCH_CUDA_VER="$(detect_torch_cuda_version)"
CUDA_GENCODE_FLAG=""
NINJA_BIN="$(ensure_ninja)"

ensure_blackwell_compatible_nvcc "${GPU_CC}" "${NVCC_VER}"
NVCC_VER="$(detect_nvcc_version)"
ensure_cuda_dev_libs
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX} -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
if [[ "${CMAKE_ARGS}" != *"CMAKE_MAKE_PROGRAM"* ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_MAKE_PROGRAM=${NINJA_BIN}"
fi
clean_stale_cmake_cache

if [[ "${GPU_CC}" =~ ^12\. ]]; then
  if [[ -z "${TORCH_CUDA_VER}" ]] || version_lt "${TORCH_CUDA_VER}" "13.0"; then
    cat <<EOF_WARN
Warning: Blackwell-class GPU detected (compute capability ${GPU_CC}), but torch.version.cuda='${TORCH_CUDA_VER:-unknown}'.
Fallback build will target 9.0a+PTX for compatibility with CUDA ${NVCC_VER:-unknown}.
For best compatibility, prefer cu130 wheels:
  python -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
EOF_WARN
  fi
fi

if [[ -n "${NVCC_VER}" ]] && version_lt "${NVCC_VER}" "12.8"; then
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    if [[ -n "${ALL_GPU_CCS}" ]]; then
      export TORCH_CUDA_ARCH_LIST="$(choose_arch_list_for_old_nvcc "${ALL_GPU_CCS}")"
    elif [[ "${GPU_CC}" =~ ^12\. ]]; then
      # Force a safe arch list for CUDA 12.4/12.5/12.6/12.7 on Blackwell hosts.
      export TORCH_CUDA_ARCH_LIST="9.0a+PTX"
    else
      export TORCH_CUDA_ARCH_LIST="$(choose_torch_cuda_arch_list "${NVCC_VER}" "${GPU_CC}")"
    fi
  fi
  echo "Detected NVCC ${NVCC_VER} with GPU compute capability '${GPU_CC:-unknown}'."
  if [[ -n "${ALL_GPU_CCS}" ]]; then
    echo "Detected all GPU compute capabilities: ${ALL_GPU_CCS//$'\n'/, }"
  fi
  echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} to avoid unsupported sm_120 build flags."
fi

# Keep gencode forcing opt-in. A single forced gencode can accidentally
# under-target the runtime GPU when hosts have multiple GPU types.
if [[ "${VLLM_TLLM_FORCE_CMAKE_GENCODE:-0}" == "1" ]] && [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  first_arch="$(first_torch_arch "${TORCH_CUDA_ARCH_LIST}" || true)"
  if [[ -n "${first_arch}" ]]; then
    CUDA_GENCODE_FLAG="$(torch_arch_to_single_gencode_flag "${first_arch}" || true)"
    if [[ -n "${CUDA_GENCODE_FLAG}" ]]; then
      export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CUDA_FLAGS=${CUDA_GENCODE_FLAG}"
    fi
  fi
fi

echo "Using compilers: CC=${CC} CXX=${CXX} CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "Using CUDA toolkit root: ${CUDAToolkit_ROOT}"
if [[ -n "${TORCH_CUDA_VER}" ]]; then
  echo "Detected torch.version.cuda=${TORCH_CUDA_VER}"
fi
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi
if [[ -n "${CUDA_GENCODE_FLAG}" ]]; then
  echo "Using forced CMAKE_CUDA_FLAGS=${CUDA_GENCODE_FLAG}"
else
  echo "Using TORCH_CUDA_ARCH_LIST only (no forced CMAKE_CUDA_FLAGS override)."
fi
echo "Using ninja executable: ${NINJA_BIN}"
ensure_build_requirements
ensure_pybind11
python -m pip install -e . --no-build-isolation 2>&1 | tee install.log
