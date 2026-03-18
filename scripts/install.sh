#!/usr/bin/env bash
set -euo pipefail

pick_compiler() {
  local kind="$1"
  shift
  local candidate
  local path
  for candidate in "$@"; do
    path="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "${path:-}" ]] && "$path" --version >/dev/null 2>&1; then
      echo "$path"
      return 0
    fi
  done
  echo "ERROR: no usable ${kind} compiler found" >&2
  return 1
}

configure_detected_arch() {
  local detected_cc="${1:-}"
  local detected_sm="${2:-}"
  local nvcc_version="${3:-}"
  local gpu_name="${4:-}"

  local gpu_name_lc=""
  gpu_name_lc="$(echo "${gpu_name}" | tr '[:upper:]' '[:lower:]')"

  # Name-based fallback for environments where compute capability query fails.
  if [[ -z "${detected_cc}" || -z "${detected_sm}" ]]; then
    if [[ "${gpu_name_lc}" == *"pro 6000"* || "${gpu_name_lc}" == *"blackwell"* ]]; then
      detected_cc="12.0"
      detected_sm="120"
      echo "GPU name fallback detected Blackwell/Pro6000: ${gpu_name} -> ${detected_cc} (sm_${detected_sm})."
    fi
  fi

  if [[ -z "${detected_cc}" || -z "${detected_sm}" ]]; then
    if [[ "${nvcc_version}" == "12.8" ]]; then
      # Prefer modern arch default on CUDA 12.8 when detection fails.
      detected_cc="12.0"
      detected_sm="120"
    else
      detected_cc="9.0"
      detected_sm="90"
    fi
    echo "GPU CC detection unavailable; fallback to ${detected_cc} (sm_${detected_sm})."
  fi

  DETECTED_CC="${detected_cc}"
  DETECTED_SM="${detected_sm}"
}

set_flashattn_arch_env() {
  local cc="$1"
  local sm="$2"
  export TORCH_CUDA_ARCH_LIST="${cc}+PTX"
  export FLASH_ATTN_CUDA_ARCHS="${sm}"
  export FLASH_ATTN_CUDA_ARCH_LIST="${cc}"
}

set_vllm_arch_env() {
  local cc="$1"
  local _sm="$2"
  export TORCH_CUDA_ARCH_LIST="${cc}+PTX"
  # Keep vLLM build arch aligned with the runtime GPU.
  # NOTE: CUDA path uses vllm.vllm_flash_attn, not upstream flash-attn package.
  unset CUDA_ARCH CUDAARCHS CMAKE_CUDA_ARCHITECTURES
}

echo "Before reset: CC=${CC-} CXX=${CXX-} CUDAHOSTCXX=${CUDAHOSTCXX-} CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER-} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-} CUDA_ARCH=${CUDA_ARCH-} CUDAARCHS=${CUDAARCHS-} CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES-}"

# Clear potentially stale compiler pins from the container environment.
unset CC CXX CUDAHOSTCXX CMAKE_CUDA_HOST_COMPILER

export CC="$(pick_compiler c 'gcc' 'gcc-13' 'gcc-12' 'gcc-11' 'cc')"
export CXX="$(pick_compiler c++ 'g++' 'g++-13' 'g++-12' 'g++-11' 'c++')"

# Force CUDA host compiler to the same available C++ compiler.
export CUDAHOSTCXX="$CXX"
export CMAKE_CUDA_HOST_COMPILER="$CXX"

gpu_cc=""
gpu_sm=""
nvcc_ver=""
gpu_name=""
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  gpu_cc_raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true)"
  if [[ "${gpu_cc_raw:-}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    gpu_cc="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    gpu_sm="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
  fi
  nvcc_ver="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
  echo "Detected GPU: ${gpu_name:-unknown}; compute capability: ${gpu_cc:-unknown} (sm_${gpu_sm:-unknown}), nvcc: ${nvcc_ver:-unknown}"
fi

if [[ -z "${gpu_cc}" ]] && command -v python >/dev/null 2>&1; then
  gpu_cc_raw="$(python - <<'PY'
import contextlib
with contextlib.suppress(Exception):
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}.{minor}")
PY
)"
  gpu_cc_raw="$(echo "${gpu_cc_raw}" | tr -d '[:space:]')"
  if [[ "${gpu_cc_raw:-}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    gpu_cc="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    gpu_sm="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    echo "Detected GPU compute capability from torch: ${gpu_cc} (sm_${gpu_sm})"
  fi
fi

configure_detected_arch "${gpu_cc}" "${gpu_sm}" "${nvcc_ver}" "${gpu_name}"
set_flashattn_arch_env "${DETECTED_CC}" "${DETECTED_SM}"

echo "Using compilers: CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX"
echo "FlashAttention arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-} FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS-}"
"$CC" --version
"$CXX" --version

# Remove generated build metadata to avoid stale CMake cache/arch settings.
rm -rf ~/.cache/vllm/torch_compile_cache ~/.cache/vllm/torch_aot_compile
rm -rf build
find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} +

pip install -U pip setuptools wheel ninja packaging setuptools_scm
pip uninstall -y flash-attn || true
export FLASH_ATTN_FORCE_BUILD=TRUE
pip install --no-build-isolation --no-cache-dir flash-attn 2>&1 | tee flash-attn.log

vllm_cc="${DETECTED_CC}"
vllm_sm="${DETECTED_SM}"
set_vllm_arch_env "${vllm_cc}" "${vllm_sm}"
echo "vLLM build arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-} (target sm_${vllm_sm})"

# Build vLLM extensions against the current environment's torch/CUDA stack.
pip install --no-build-isolation -e . 2>&1 | tee install.log
