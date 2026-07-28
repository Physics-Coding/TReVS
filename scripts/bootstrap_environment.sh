#!/usr/bin/env bash
# Create one mutually exclusive runtime environment from the repository root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

usage() {
    cat <<'EOF'
Usage: bash scripts/bootstrap_environment.sh --family <llava-family|qwen> [--skip-flash-attn]

Creates the requested Conda environment, installs this extracted package from
its own root, and, for the LLaVA family, builds FlashAttention against the
installed PyTorch/CUDA toolchain without build isolation. For the LLaVA family,
CUDA_HOME (when set) or the default nvcc must match the pytorch-cuda major.minor
version declared in environment/llava_family.yml.
EOF
}

family=""
skip_flash_attn=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            family="$2"
            shift 2
            ;;
        --skip-flash-attn)
            skip_flash_attn=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "${CONDA_BIN}" ]] || { echo "conda is required." >&2; exit 2; }
[[ -n "${family}" ]] || { usage >&2; exit 2; }

validate_flash_attn_toolchain() {
    local environment_file="$1"
    local expected_cuda
    local nvcc_path
    local nvcc_cuda

    expected_cuda="$(sed -n 's/^[[:space:]]*-[[:space:]]*pytorch-cuda=\([0-9][0-9.]*\)[[:space:]]*$/\1/p' "${REPO_ROOT}/${environment_file}" | head -n 1)"
    [[ -n "${expected_cuda}" ]] || {
        echo "Cannot determine pytorch-cuda version from ${environment_file}." >&2
        exit 2
    }

    if [[ -n "${CUDA_HOME:-}" ]]; then
        nvcc_path="${CUDA_HOME%/}/bin/nvcc"
    else
        nvcc_path="$(command -v nvcc || true)"
    fi
    [[ -n "${nvcc_path}" && -x "${nvcc_path}" ]] || {
        echo "FlashAttention requires a CUDA ${expected_cuda} nvcc compiler. Set CUDA_HOME to a matching toolkit installation, or use --skip-flash-attn only for a non-paper-matched fallback." >&2
        exit 2
    }

    nvcc_cuda="$("${nvcc_path}" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
    [[ "${nvcc_cuda}" == "${expected_cuda}" ]] || {
        echo "FlashAttention toolchain mismatch: pytorch-cuda=${expected_cuda}, nvcc=${nvcc_cuda:-unknown} (${nvcc_path}). Set CUDA_HOME to a matching CUDA ${expected_cuda} toolkit before rerunning." >&2
        exit 2
    }
}

cd "${REPO_ROOT}"
case "${family}" in
    llava-family)
        environment_file="environment/llava_family.yml"
        environment_name="trevs-llava-family"
        extra="llava-family"
        ;;
    qwen)
        environment_file="environment/qwen.yml"
        environment_name="trevs-qwen"
        extra="qwen"
        ;;
    *)
        echo "Unsupported family: ${family}" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "${family}" == "llava-family" && "${skip_flash_attn}" -eq 0 ]]; then
    validate_flash_attn_toolchain "${environment_file}"
fi

"${CONDA_BIN}" env create -f "${environment_file}"
"${CONDA_BIN}" run -n "${environment_name}" python -m pip install --no-build-isolation -e ".[${extra}]"

if [[ "${family}" == "llava-family" && "${skip_flash_attn}" -eq 0 ]]; then
    "${CONDA_BIN}" run -n "${environment_name}" python -m pip install --no-build-isolation "flash-attn==2.3.3"
fi

"${CONDA_BIN}" run -n "${environment_name}" python -c 'import sys; print(sys.version)'
echo "Bootstrap complete: ${environment_name}"
