#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="${ROOT_DIR}/rust"
INSTALL_LOG="${INSTALL_LOG:-install.log}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-roy/rs}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if [[ "${INSTALL_LOG}" != /* ]]; then
    INSTALL_LOG="${ROOT_DIR}/${INSTALL_LOG}"
fi

RUST_ONLY=0
SKIP_RUST_BOOTSTRAP=0
INSTALL_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./scripts/install.sh [install args...]
  ./scripts/install.sh --rust-only [maturin args...]

Installs the current checkout of vLLM on this branch.

Default mode:
  - uses the active virtualenv, or creates .venv with python -m venv
  - installs Rust if cargo is missing
  - installs Python build dependencies into the environment
  - runs: pip install -e . --no-build-isolation

Rust-only mode:
  - uses the active virtualenv, or creates .venv with python -m venv
  - installs Rust if cargo is missing
  - installs maturin into the environment if needed
  - runs: cd rust && maturin develop --release

Options:
  --rust-only            Build only the Rust extension via maturin.
  --skip-rust-bootstrap  Fail instead of installing Rust when cargo is missing.
  -h, --help             Show this help message.

Examples:
  ./scripts/install.sh
  VLLM_USE_PRECOMPILED=1 ./scripts/install.sh
  ./scripts/install.sh --rust-only
EOF
}

log() {
    printf '==> %s\n' "$*"
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

ensure_command() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

default_python_bin() {
    if command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
    fi

    return 1
}

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

ensure_venv() {
    local python_bin
    python_bin="${PYTHON_BIN:-$(default_python_bin || true)}"
    [[ -n "${python_bin}" ]] || die "python executable not found"

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        return 0
    fi

    if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
        log "Creating .venv with Python ${PYTHON_VERSION}"
        "${python_bin}" -m venv "${ROOT_DIR}/.venv"
    fi

    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate"
}

ensure_pip() {
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
    python -m pip --version >/dev/null 2>&1 || die "pip is not available in the active Python environment"
}

ensure_rust() {
    if command -v cargo >/dev/null 2>&1; then
        return 0
    fi

    if [[ -f "${HOME}/.cargo/env" ]]; then
        # shellcheck disable=SC1090
        source "${HOME}/.cargo/env"
    fi

    if command -v cargo >/dev/null 2>&1; then
        return 0
    fi

    if [[ "${SKIP_RUST_BOOTSTRAP}" == "1" ]]; then
        die "cargo not found; install Rust first or omit --skip-rust-bootstrap"
    fi

    ensure_command curl
    log "Installing Rust with rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    if [[ ! -f "${HOME}/.cargo/env" ]]; then
        die "Rust installation completed but ${HOME}/.cargo/env was not created"
    fi

    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
    command -v cargo >/dev/null 2>&1 || die "cargo still not found after rustup install"
}

ensure_python_packages() {
    if [[ $# -eq 0 ]]; then
        return 0
    fi

    python -m pip install "$@"
}

ensure_maturin() {
    if command -v maturin >/dev/null 2>&1; then
        return 0
    fi

    log "Installing maturin into the active environment"
    ensure_python_packages maturin
}

ensure_build_dependencies() {
    local build_requirements="${ROOT_DIR}/requirements/build.txt"
    local filtered_requirements=""

    [[ -f "${build_requirements}" ]] || die "missing build requirements file: ${build_requirements}"

    if python -c 'import torch' >/dev/null 2>&1; then
        filtered_requirements="$(mktemp)"
        grep -v '^torch==' "${build_requirements}" > "${filtered_requirements}"
        log "Installing Python build dependencies (reusing existing torch)"
        python -m pip install -r "${filtered_requirements}"
        rm -f "${filtered_requirements}"
        return 0
    fi

    log "Installing Python build dependencies"
    python -m pip install -r "${build_requirements}"
}

ensure_version_override() {
    if command -v git >/dev/null 2>&1; then
        return 0
    fi

    if [[ -n "${VLLM_VERSION_OVERRIDE:-}" ]]; then
        return 0
    fi

    export VLLM_VERSION_OVERRIDE="0.0.0.dev0"
    warn "git is not available; setting VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE} for local editable install"
}

prepare_cuda_env() {
    local cuda_home_detected=""
    local nvrtc_library=""
    local cuda_include_dir_detected=""
    local precompiled_is_set=0

    if [[ -n "${VLLM_USE_PRECOMPILED+x}" ]]; then
        precompiled_is_set=1
    fi

    if cuda_home_detected="$(resolve_cuda_home)"; then
        export CUDA_HOME="${cuda_home_detected}"
        append_cmake_arg "-DCUDAToolkit_ROOT=${CUDA_HOME}"
    fi

    if nvrtc_library="$(find_nvrtc "${cuda_home_detected:-}")"; then
        export CUDA_NVRTC_LIBRARY="${nvrtc_library}"
        append_cmake_arg "-DCUDA_nvrtc_LIBRARY=${nvrtc_library}"
    else
        warn "libnvrtc.so was not found automatically; the install may still fail during CMake configure"
    fi

    if cuda_include_dir_detected="$(find_cuda_include_dir "${cuda_home_detected:-}")"; then
        export CUDA_INCLUDE_DIR="${cuda_include_dir_detected}"
        export CPATH="${cuda_include_dir_detected}${CPATH:+:${CPATH}}"
        export CPLUS_INCLUDE_PATH="${cuda_include_dir_detected}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    else
        if [[ "${VLLM_USE_PRECOMPILED:-}" =~ ^(1|true)$ || "${ALLOW_NO_CUDA_HEADERS:-0}" == "1" ]]; then
            warn "CUDA headers (cusparse.h/cublas_v2.h) not found; build may fail without dev headers"
        elif [[ "${precompiled_is_set}" == "0" ]]; then
            export VLLM_USE_PRECOMPILED=1
            warn "CUDA headers (cusparse.h/cublas_v2.h) not found; enabling VLLM_USE_PRECOMPILED=1 automatically"
        else
            die "CUDA headers (cusparse.h/cublas_v2.h) not found. Install CUDA Toolkit dev headers, set CUDA_INCLUDE_DIR, or use VLLM_USE_PRECOMPILED=1"
        fi
    fi

    if [[ -n "${cuda_home_detected}" ]]; then
        log "Using CUDA_HOME=${cuda_home_detected}"
    fi
    if [[ -n "${nvrtc_library}" ]]; then
        log "Using CUDA_NVRTC_LIBRARY=${nvrtc_library}"
    fi
    if [[ -n "${cuda_include_dir_detected}" ]]; then
        log "Using CUDA_INCLUDE_DIR=${cuda_include_dir_detected}"
    fi
    if [[ -n "${CMAKE_ARGS:-}" ]]; then
        log "Using CMAKE_ARGS=${CMAKE_ARGS}"
    fi
    if [[ "${VLLM_USE_PRECOMPILED:-}" =~ ^(1|true)$ ]]; then
        log "Using VLLM_USE_PRECOMPILED=1"
    fi
}

run_and_log() {
    "$@" 2>&1 | tee "${INSTALL_LOG}"
    return "${PIPESTATUS[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rust-only)
            RUST_ONLY=1
            shift
            ;;
        --skip-rust-bootstrap)
            SKIP_RUST_BOOTSTRAP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            INSTALL_ARGS+=("$@")
            break
            ;;
        *)
            INSTALL_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "${ROOT_DIR}"

current_branch="$(git branch --show-current 2>/dev/null || true)"
if [[ -n "${current_branch}" && "${current_branch}" != "${EXPECTED_BRANCH}" ]]; then
    warn "expected branch ${EXPECTED_BRANCH}, current branch is ${current_branch}"
fi

ensure_venv
ensure_pip

if [[ "${RUST_ONLY}" == "1" || "${VLLM_SKIP_RUST:-0}" != "1" ]]; then
    ensure_rust
    ensure_maturin
fi

log "Using Python $(python -c 'import sys; print(sys.executable)')"

if [[ "${RUST_ONLY}" == "1" ]]; then
    log "Building the Rust extension with maturin"
    (
        cd "${RUST_DIR}"
        run_and_log maturin develop --release "${INSTALL_ARGS[@]}"
    )
    exit $?
fi

prepare_cuda_env
ensure_build_dependencies
ensure_version_override

log "Installing editable vLLM from ${current_branch:-current checkout}"
run_and_log pip install -e . --no-build-isolation "${INSTALL_ARGS[@]}"
