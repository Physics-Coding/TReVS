#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/trevs_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {tgif|msvd|msrvtt}" >&2
  exit 2
fi
dataset="${1,,}"
case "${dataset}" in
  tgif|msvd|msrvtt) ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set to run VideoQA judging." >&2
  exit 1
fi
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL must be set to run VideoQA judging." >&2
  exit 1
fi
if [[ -z "${VIDEO_QA_JUDGE_MODEL}" ]]; then
  echo "VIDEO_QA_JUDGE_MODEL must be set to run VideoQA judging." >&2
  exit 1
fi

prediction_file="${RUN_DIR}/predictions/${dataset}/predictions.jsonl"
if [[ ! -f "${prediction_file}" ]]; then
  echo "Prediction file does not exist: ${prediction_file}" >&2
  exit 1
fi

judge_model_lower="${VIDEO_QA_JUDGE_MODEL,,}"
if [[ "${judge_model_lower}" == *"gpt-4.1"* ]]; then
  judge_family="gpt-4.1"
elif [[ "${judge_model_lower}" == *"gpt-3.5"* ]]; then
  judge_family="gpt-3.5"
else
  judge_family="other"
fi
judge_slug="${VIDEO_QA_JUDGE_MODEL//\//_}"
judge_slug="${judge_slug// /_}"
judge_output_dir="${RUN_DIR}/metrics/${dataset}/judge/${judge_family}/${judge_slug}"
mkdir -p "${judge_output_dir}"
record_videollava_judge_model
{
  printf 'VIDEO_QA_JUDGE_MODEL=%q\n' "${VIDEO_QA_JUDGE_MODEL}"
  printf 'VIDEO_QA_JUDGE_FAMILY=%q\n' "${judge_family}"
  printf 'VIDEO_QA_JUDGE_PROMPT_VERSION=%q\n' "videollava_videoqa_correctness_v1"
  printf 'PREDICTION_FILE=%q\n' "${prediction_file}"
} > "${judge_output_dir}/judge_config.env"

"${PYTHON_BIN}" -m videollava.eval.video.eval_video_qa \
  --pred-path "${prediction_file}" \
  --output-dir "${judge_output_dir}" \
  --summary-path "${judge_output_dir}/summary.json" \
  --model "${VIDEO_QA_JUDGE_MODEL}" \
  --workers "${VIDEO_QA_JUDGE_WORKERS}" \
  --timeout "${VIDEO_QA_JUDGE_TIMEOUT}" \
  --max-retries "${VIDEO_QA_JUDGE_MAX_RETRIES}" \
  --retry-base-seconds "${VIDEO_QA_JUDGE_RETRY_BASE_SECONDS}" \
  --requests-per-minute "${VIDEO_QA_JUDGE_RPM}"
