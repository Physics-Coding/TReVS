#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_dense_videollava.sh run [all|tgif|msvd|msrvtt ...]
  run_dense_videollava.sh judge [all|tgif|msvd|msrvtt ...]

Examples:
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/videollava/run_dense_videollava.sh run all
  bash scripts/videollava/run_dense_videollava.sh judge msrvtt

Optional environment variables include MODEL_PATH, DATA_ROOT, RESULT_ROOT,
EXPERIMENT, CUDA_VISIBLE_DEVICES, MAX_SAMPLES, MAX_NEW_TOKENS, RANDOM_SEED,
PYTHON_BIN, OPENAI_API_KEY, OPENAI_BASE_URL, and VIDEO_QA_JUDGE_MODEL.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

action="${1,,}"
shift
case "${action}" in
  run|judge) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unsupported action: ${action}. Use run or judge." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ $# -eq 0 || ( $# -eq 1 && "${1,,}" == "all" ) ]]; then
  datasets=(tgif msvd msrvtt)
else
  datasets=()
  declare -A seen=()
  for requested_dataset in "$@"; do
    dataset="${requested_dataset,,}"
    case "${dataset}" in
      tgif|msvd|msrvtt)
        if [[ -z "${seen[${dataset}]:-}" ]]; then
          datasets+=("${dataset}")
          seen["${dataset}"]=1
        fi
        ;;
      all)
        echo "Dataset 'all' cannot be combined with individual datasets." >&2
        exit 2
        ;;
      *)
        echo "Unsupported dataset: ${requested_dataset}." >&2
        exit 2
        ;;
    esac
  done
fi

# Fix every method-specific variable before trevs_env.sh is sourced by this
# process and the per-dataset child scripts. METHOD=dense is the runtime gate
# that bypasses both the frame router and the LLM phase-pruning branch.
export VIDEOLLAVA_TREVS_PRESET=custom
export METHOD=dense
export TREVS_ROUTE_TOPK=0
export TREVS_ROUTE_FPS=0
export TREVS_PHASE_SCORING=priority_heads
export TREVS_USE_SINK_TOKEN=0
export PHASE_TRANSITION_LAYER=8
export PHASE_TRANSITION_N_KEEP=0
export DOUBLE_TRACK_USE_CONSISTENCY_REWARD=0
export TREVS_SEMANTIC_LAYER=none
export USE_FLASH_ATTN=0
export RUN_JUDGE=$([[ "${action}" == "judge" ]] && printf 1 || printf 0)
export EXPERIMENT="${EXPERIMENT:-videollava_dense_avg2048_f8_sdpa}"
export PYTHONDONTWRITEBYTECODE=1

source "${SCRIPT_DIR}/trevs_env.sh"
if [[ "${TREVS_ACTIVE}" != "0" || "${STAGE1_VISUAL_TOKENS}" != "2048" \
  || "${STAGE2_VISUAL_TOKENS}" != "2048" \
  || "${AVERAGE_VISUAL_TOKENS}" != "2048" ]]; then
  echo "Dense Video-LLaVA configuration did not resolve to 2048 visual tokens." >&2
  exit 1
fi

printf 'Mode: %s\nExperiment: %s\nDatasets: %s\n' \
  "${action}" "${EXPERIMENT}" "${datasets[*]}"

for dataset in "${datasets[@]}"; do
  if [[ "${action}" == "judge" ]]; then
    bash "${SCRIPT_DIR}/eval_qa_${dataset}.sh"
  else
    bash "${SCRIPT_DIR}/run_qa_${dataset}.sh"
  fi
done
