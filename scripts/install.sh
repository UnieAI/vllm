#!/usr/bin/env bash
set -euo pipefail

echo "Before reset: CC=${CC-} CXX=${CXX-} CUDAHOSTCXX=${CUDAHOSTCXX-} CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER-} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-} CUDA_ARCH=${CUDA_ARCH-} CUDAARCHS=${CUDAARCHS-} CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES-}"

# Clear potentially stale compiler pins from the container environment.
unset CC CXX CUDAHOSTCXX CMAKE_CUDA_HOST_COMPILER

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

# Force CUDA host compiler to the same available C++ compiler.
export CUDAHOSTCXX="$CXX"
export CMAKE_CUDA_HOST_COMPILER="$CXX"

nvcc_ver=""
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  gpu_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
  nvcc_ver="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
  echo "Detected GPU compute capability: ${gpu_cc:-unknown}, nvcc: ${nvcc_ver:-unknown}"
fi

# Default source-build settings.
# On CUDA 12.8, building for sm100 can fail with ptxas "Illegal vector size: 8".
# Use 8.9+PTX there as a compatibility fallback unless user overrides.
if [[ "${nvcc_ver:-}" == "12.8" ]]; then
  if [[ -z "${TORCH_CUDA_ARCH_LIST-}" || "${TORCH_CUDA_ARCH_LIST}" == "10.0+PTX" || "${TORCH_CUDA_ARCH_LIST}" == "10.0" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.9+PTX"
  fi
  if [[ -z "${CUDA_ARCH-}" || "${CUDA_ARCH}" == "100" || "${CUDA_ARCH}" == "10.0" ]]; then
    export CUDA_ARCH="89"
  fi
else
  if [[ -z "${TORCH_CUDA_ARCH_LIST-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="10.0+PTX"
  fi
  if [[ -z "${CUDA_ARCH-}" ]]; then
    export CUDA_ARCH="100"
  fi
fi

if [[ -z "${CUDAARCHS-}" ]]; then
  export CUDAARCHS="$CUDA_ARCH"
fi
if [[ -z "${CMAKE_CUDA_ARCHITECTURES-}" ]]; then
  export CMAKE_CUDA_ARCHITECTURES="$CUDAARCHS"
fi

echo "Using: CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-} CUDA_ARCH=${CUDA_ARCH-} CUDAARCHS=${CUDAARCHS-} CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES-}"
"$CC" --version
"$CXX" --version

# Remove generated build metadata to avoid stale CMake cache/arch settings.
rm -rf ~/.cache/vllm/torch_compile_cache ~/.cache/vllm/torch_aot_compile
rm -rf build
find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} +

pip install -e . 2>&1 | tee install.log
