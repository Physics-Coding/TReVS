#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/trevs_env.sh"

requested="${TARGET_DATASET:-all}"
if [[ "${requested}" == "all" ]]; then
  datasets=(tgif msvd msrvtt)
else
  IFS=',' read -r -a datasets <<< "${requested}"
fi

declare -A seen=()
for dataset in "${datasets[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  case "${dataset}" in
    tgif|msvd|msrvtt) ;;
    *) echo "Unsupported VideoQA dataset: ${dataset}" >&2; exit 2 ;;
  esac
  [[ -z "${seen[${dataset}]:-}" ]] || { echo "Duplicate dataset: ${dataset}" >&2; exit 2; }
  seen["${dataset}"]=1
  bash "${SCRIPT_DIR}/run_qa_${dataset}.sh"
  if [[ "${RUN_JUDGE}" == "1" ]]; then
    bash "${SCRIPT_DIR}/eval_qa_${dataset}.sh"
  fi
done
