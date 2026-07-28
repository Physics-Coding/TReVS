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
  tgif)
    dataset_name="TGIF_Zero_Shot_QA"
    video_subdir="mp4"
    ;;
  msvd)
    dataset_name="MSVD_Zero_Shot_QA"
    video_subdir="videos"
    ;;
  msrvtt)
    dataset_name="MSRVTT_Zero_Shot_QA"
    video_subdir="videos/all"
    ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    exit 2
    ;;
esac

if [[ -d "${DATA_ROOT}/${dataset_name}" ]]; then
  dataset_root="${DATA_ROOT}/${dataset_name}"
else
  dataset_root="${DATA_ROOT}/GPT_Zero_Shot_QA/${dataset_name}"
fi
video_dir="${dataset_root}/${video_subdir}"
question_file="${dataset_root}/test_q.json"
answer_file="${dataset_root}/test_a.json"

for required_path in "${MODEL_PATH}" "${video_dir}" "${question_file}" "${answer_file}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required Video-LLaVA path does not exist: ${required_path}" >&2
    exit 1
  fi
done

IFS=',' read -r -a raw_gpus <<< "${CUDA_VISIBLE_DEVICES}"
gpus=()
for gpu in "${raw_gpus[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  [[ -n "${gpu}" ]] && gpus+=("${gpu}")
done
if (( ${#gpus[@]} == 0 )); then
  echo "CUDA_VISIBLE_DEVICES must contain at least one physical GPU ID." >&2
  exit 1
fi
num_chunks="${#gpus[@]}"

write_videollava_run_config
dataset_output_dir="${RUN_DIR}/predictions/${dataset}"
chunk_dir="${dataset_output_dir}/chunks"
log_dir="${RUN_DIR}/logs/${dataset}"
mkdir -p "${chunk_dir}" "${log_dir}"
{
  printf 'DATASET=%q\n' "${dataset}"
  printf 'VIDEO_DIR=%q\n' "${video_dir}"
  printf 'QUESTION_FILE=%q\n' "${question_file}"
  printf 'ANSWER_FILE=%q\n' "${answer_file}"
  printf 'CHUNK_COUNT=%q\n' "${num_chunks}"
  printf 'PREDICTION_FILE=%q\n' "${dataset_output_dir}/predictions.jsonl"
} > "${dataset_output_dir}/run_config.env"

"${PYTHON_BIN}" -m videollava.eval.video.preflight \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --datasets "${dataset}" \
  --decode-samples \
  > "${log_dir}/preflight.json"

pids=()
for chunk_idx in "${!gpus[@]}"; do
  physical_gpu="${gpus[${chunk_idx}]}"
  chunk_output="${chunk_dir}/${num_chunks}_${chunk_idx}.jsonl"
  chunk_log="${log_dir}/chunk_${chunk_idx}.log"
  echo "Launching ${dataset} chunk ${chunk_idx}/${num_chunks} on physical GPU ${physical_gpu} as cuda:0"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${PYTHON_BIN}" \
    -m videollava.eval.video.run_inference_video_qa \
    --model-path "${MODEL_PATH}" \
    --video-dir "${video_dir}" \
    --gt-file-question "${question_file}" \
    --gt-file-answers "${answer_file}" \
    --output-file "${chunk_output}" \
    --num-chunks "${num_chunks}" \
    --chunk-idx "${chunk_idx}" \
    --max-samples "${MAX_SAMPLES}" \
    --device cuda:0 \
    --attn-implementation "${VIDEO_LLAVA_ATTN_IMPLEMENTATION}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${RANDOM_SEED}" \
    > "${chunk_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for chunk_idx in "${!pids[@]}"; do
  if ! wait "${pids[${chunk_idx}]}"; then
    echo "Chunk ${chunk_idx} failed; see ${log_dir}/chunk_${chunk_idx}.log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one ${dataset} chunk failed; merge was not attempted." >&2
  exit 1
fi

merged_file="${dataset_output_dir}/predictions.jsonl"
"${PYTHON_BIN}" -m videollava.eval.video.merge_chunks \
  --question-file "${question_file}" \
  --chunk-dir "${chunk_dir}" \
  --num-chunks "${num_chunks}" \
  --output-file "${merged_file}" \
  --max-samples "${MAX_SAMPLES}"

echo "VideoQA inference complete: ${merged_file}"
