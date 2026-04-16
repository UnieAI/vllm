#!/bin/bash
set -e

echo "==============================================================================="
echo " UnieInfra vLLM Blackwell 'Universal' Build "
echo "==============================================================================="

cd /workspace

echo "→ Deep cleaning build artifacts..."
rm -rf build/ dist/ vllm.egg-info/ vllm/*.so .deps __pycache__ */__pycache__ *.pyc 2>/dev/null || true

echo "→ Installing build dependencies..."
python3 -m pip install --upgrade pip setuptools wheel packaging setuptools_scm jinja2 numpy sympy

echo "→ Installing system build tools..."
apt-get update && apt-get install -y --no-install-recommends cmake ninja-build

echo "→ Installing NVIDIA library wheels..."
python3 -m pip install --force-reinstall --no-deps \
    nvidia-cudnn-cu12 \
    nvidia-nccl-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-nvshmem-cu12 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128

echo "→ Installing PyTorch Nightly..."
python3 -m pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall --no-deps

echo "→ Installing vLLM common runtime dependencies..."
python3 -m pip install -r requirements/common.txt --no-deps || true

echo "→ Installing safety net packages..."
python3 -m pip install \
    attrs aiosignal async_timeout multidict yarl frozenlist aiohappyeyeballs propcache \
    httpx jiter sniffio jmespath huggingface_hub networkx \
    transformers tokenizers sentencepiece pillow opencv-python-headless \
    pydantic fastapi uvicorn triton pycountry

#TORCH_CUDA_ARCH_LIST --- not sure ths arch list is having an effect
echo "→ Building vLLM for Blackwell (Native + Feature-Specific + Family Targets)..."
MAX_JOBS=48 \
TORCH_CUDA_ARCH_LIST="10.0 10.0a 10.0f 10.1 10.1a 12.0 12.0a 12.0f 12.1 12.1a+PTX" \
VLLM_INSTALL_PNC=1 \
VLLM_DISABLE_MARLIN=1 \
python3 -m pip install -v --no-build-isolation --force-reinstall --no-deps .

echo "=== Build completed! ==="
cd /
python3 -c '
import torch, vllm
print("="*70)
print("vLLM version :", vllm.__version__)
print("Torch version:", torch.__version__)
print("CUDA Runtime :", torch.version.cuda)
print("Device Name  :", torch.cuda.get_device_name(0))
print("Archs Built  :", torch.cuda.get_arch_list())
print("="*70)
print("✅ Blackwell Native build verified.")
'
