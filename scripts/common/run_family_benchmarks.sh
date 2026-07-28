#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {llava15|llava_next|qwen25vl}" >&2
  exit 2
fi
family="$1"
case "${family}" in
  llava15|llava_next|qwen25vl) ;;
  *) echo "Unsupported family: ${family}" >&2; exit 2 ;;
esac

if [[ "${TReVS_FAMILY:-}" != "${family}" ]]; then
  echo "This canonical launcher is configured by scripts/reproduce.sh." >&2
  echo "Expected TReVS_FAMILY=${family}; received ${TReVS_FAMILY:-<unset>}." >&2
  exit 2
fi

supported=(gqa textvqa mme mmbench mmbench_cn sqa pope vqav2)
requested="${TARGET_DATASET:-all}"
if [[ "${requested}" == "all" ]]; then
  datasets=("${supported[@]}")
else
  IFS=',' read -r -a datasets <<< "${requested}"
fi

declare -A seen=()
for dataset in "${datasets[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  [[ -n "${dataset}" ]] || { echo "Empty dataset name." >&2; exit 2; }
  valid=0
  for candidate in "${supported[@]}"; do
    [[ "${dataset}" == "${candidate}" ]] && valid=1
  done
  (( valid == 1 )) || { echo "Unsupported dataset: ${dataset}" >&2; exit 2; }
  [[ -z "${seen[${dataset}]:-}" ]] || { echo "Duplicate dataset: ${dataset}" >&2; exit 2; }
  seen["${dataset}"]=1
  bash "${SCRIPT_DIR}/run_image_benchmark.sh" "${family}" "${dataset}"
done
