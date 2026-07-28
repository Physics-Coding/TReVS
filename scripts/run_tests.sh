#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LLAVA_FAMILY_ENV="${LLAVA_FAMILY_ENV:-${LLAVA_VIDEO_ENV:-trevs-llava-family}}"
QWEN_ENV="${QWEN_ENV:-trevs-qwen}"
PURE_PYTHON="${PURE_PYTHON:-python3}"

run_python() {
    local env_name="$1"
    shift
    if [[ -n "${CONDA_EXE:-}" ]]; then
        "${CONDA_EXE}" run -n "${env_name}" python "$@"
    elif command -v conda >/dev/null 2>&1; then
        conda run -n "${env_name}" python "$@"
    else
        echo "conda is required to run the mutually exclusive model test environments." >&2
        return 2
    fi
}

cd "${REPO_ROOT}"
"${PURE_PYTHON}" -m unittest discover -s tests -v
run_python "${LLAVA_FAMILY_ENV}" -m unittest discover -s evaluation/tests -v
run_python "${LLAVA_FAMILY_ENV}" -m unittest discover -s llava/tests -v
run_python "${QWEN_ENV}" -m unittest discover -s qwen/tests -v
run_python "${LLAVA_FAMILY_ENV}" -m unittest discover -s videollava/tests -v

while IFS= read -r -d '' script; do
    bash -n "${script}"
done < <(find scripts -type f -name '*.sh' -print0 | sort -z)

echo "All unit, synthetic, launcher, and shell-syntax tests passed."
