#!/usr/bin/env bash

# Shared, source-only configuration for Video-LLaVA TReVS evaluation.
_VIDEO_LLAVA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_VIDEO_LLAVA_SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export MODEL_PATH="${MODEL_PATH:-}"
export DATA_ROOT="${DATA_ROOT:-}"
export RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/videollava}"

# Named values are target layer-weighted average visual-token budgets.
export VIDEOLLAVA_TREVS_PRESET="${VIDEOLLAVA_TREVS_PRESET:-960}"
case "${VIDEOLLAVA_TREVS_PRESET}" in
  960)
    METHOD=trevs
    TREVS_ROUTE_TOPK=180
    TREVS_ROUTE_FPS=60
    TREVS_PHASE_SCORING=priority_heads
    TREVS_USE_SINK_TOKEN=0
    PHASE_TRANSITION_LAYER=8
    PHASE_TRANSITION_N_KEEP=640
    ;;
  136)
    METHOD=trevs
    TREVS_ROUTE_TOPK=26
    TREVS_ROUTE_FPS=8
    TREVS_PHASE_SCORING=priority_heads
    TREVS_USE_SINK_TOKEN=0
    PHASE_TRANSITION_LAYER=8
    PHASE_TRANSITION_N_KEEP=90
    ;;
  dense)
    METHOD=dense
    TREVS_ROUTE_TOPK=0
    TREVS_ROUTE_FPS=0
    TREVS_PHASE_SCORING=priority_heads
    TREVS_USE_SINK_TOKEN=0
    PHASE_TRANSITION_LAYER=8
    PHASE_TRANSITION_N_KEEP=2048
    ;;
  custom)
    METHOD="${METHOD:-trevs}"
    TREVS_ROUTE_TOPK="${TREVS_ROUTE_TOPK:-96}"
    TREVS_ROUTE_FPS="${TREVS_ROUTE_FPS:-32}"
    TREVS_PHASE_SCORING="${TREVS_PHASE_SCORING:-priority_heads}"
    TREVS_USE_SINK_TOKEN="${TREVS_USE_SINK_TOKEN:-0}"
    PHASE_TRANSITION_LAYER="${PHASE_TRANSITION_LAYER:-8}"
    PHASE_TRANSITION_N_KEEP="${PHASE_TRANSITION_N_KEEP:-341}"
    ;;
  *)
    echo "Unsupported VIDEOLLAVA_TREVS_PRESET=${VIDEOLLAVA_TREVS_PRESET}; use 136, 960, dense, or custom." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export METHOD TREVS_ROUTE_TOPK TREVS_ROUTE_FPS TREVS_PHASE_SCORING
export TREVS_USE_SINK_TOKEN PHASE_TRANSITION_LAYER PHASE_TRANSITION_N_KEEP
export TREVS_TEXT_SCORE_MODE="${TREVS_TEXT_SCORE_MODE:-rms}"
export TREVS_VISUAL_TEMPERATURE="${TREVS_VISUAL_TEMPERATURE:-1.0}"
export TREVS_TEXT_TEMPERATURE="${TREVS_TEXT_TEMPERATURE:-1.0}"
export TREVS_SEMANTIC_LAYER="${TREVS_SEMANTIC_LAYER:-none}"
export DOUBLE_TRACK_USE_CONSISTENCY_REWARD="${DOUBLE_TRACK_USE_CONSISTENCY_REWARD:-1}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
export RANDOM_SEED="${RANDOM_SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
export RUN_JUDGE="${RUN_JUDGE:-0}"

export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export VIDEO_QA_JUDGE_MODEL="${VIDEO_QA_JUDGE_MODEL:-}"
export VIDEO_QA_JUDGE_WORKERS="${VIDEO_QA_JUDGE_WORKERS:-4}"
export VIDEO_QA_JUDGE_TIMEOUT="${VIDEO_QA_JUDGE_TIMEOUT:-60}"
export VIDEO_QA_JUDGE_MAX_RETRIES="${VIDEO_QA_JUDGE_MAX_RETRIES:-4}"
export VIDEO_QA_JUDGE_RETRY_BASE_SECONDS="${VIDEO_QA_JUDGE_RETRY_BASE_SECONDS:-2}"
export VIDEO_QA_JUDGE_RPM="${VIDEO_QA_JUDGE_RPM:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export LLM_NUM_LAYERS=32

for _boolean_name in \
  DOUBLE_TRACK_USE_CONSISTENCY_REWARD TREVS_USE_SINK_TOKEN USE_FLASH_ATTN RUN_JUDGE; do
  if [[ "${!_boolean_name}" != "0" && "${!_boolean_name}" != "1" ]]; then
    echo "${_boolean_name} must be 0 or 1, got ${!_boolean_name}." >&2
    return 2 2>/dev/null || exit 2
  fi
done

for _integer_name in \
  TREVS_ROUTE_TOPK TREVS_ROUTE_FPS PHASE_TRANSITION_LAYER PHASE_TRANSITION_N_KEEP; do
  if [[ ! "${!_integer_name}" =~ ^[0-9]+$ ]]; then
    echo "${_integer_name} must be a nonnegative integer, got ${!_integer_name}." >&2
    return 2 2>/dev/null || exit 2
  fi
done
if (( PHASE_TRANSITION_LAYER < 1 || PHASE_TRANSITION_LAYER >= LLM_NUM_LAYERS )); then
  echo "PHASE_TRANSITION_LAYER must be in [1, $((LLM_NUM_LAYERS - 1))], got ${PHASE_TRANSITION_LAYER}." >&2
  return 2 2>/dev/null || exit 2
fi

case "${METHOD,,}" in
  trevs)
    export TREVS_ACTIVE=1
    if (( TREVS_ROUTE_TOPK < 0 || TREVS_ROUTE_FPS < 0 )); then
      echo "TReVS per-frame budgets must be nonnegative." >&2
      return 2 2>/dev/null || exit 2
    fi
    if (( TREVS_ROUTE_TOPK + TREVS_ROUTE_FPS > 256 )); then
      echo "TREVS_ROUTE_TOPK + TREVS_ROUTE_FPS must not exceed 256." >&2
      return 2 2>/dev/null || exit 2
    fi
    export STAGE1_TOPK_VISUAL_TOKENS=$((8 * TREVS_ROUTE_TOPK))
    export STAGE1_FPS_VISUAL_TOKENS=$((8 * TREVS_ROUTE_FPS))
    export STAGE1_VISUAL_TOKENS=$((8 * (TREVS_ROUTE_TOPK + TREVS_ROUTE_FPS)))
    ;;
  dense)
    export TREVS_ACTIVE=0
    export STAGE1_TOPK_VISUAL_TOKENS=0
    export STAGE1_FPS_VISUAL_TOKENS=0
    export STAGE1_VISUAL_TOKENS=2048
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}; use trevs or dense." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

if [[ "${TREVS_ACTIVE}" == "0" ]]; then
  export STAGE2_VISUAL_TOKENS="${STAGE1_VISUAL_TOKENS}"
elif [[ "${TREVS_USE_SINK_TOKEN}" == "1" ]]; then
  export STAGE2_VISUAL_TOKENS=$((PHASE_TRANSITION_N_KEEP + 1))
else
  export STAGE2_VISUAL_TOKENS="${PHASE_TRANSITION_N_KEEP}"
fi

if [[ "${TREVS_ACTIVE}" == "1" ]] \
  && (( PHASE_TRANSITION_N_KEEP >= STAGE1_VISUAL_TOKENS )); then
  echo "PHASE_TRANSITION_N_KEEP must be smaller than the Stage-1 visual-token count (${STAGE1_VISUAL_TOKENS})." >&2
  return 2 2>/dev/null || exit 2
fi

export AVERAGE_VISUAL_TOKENS_NUMERATOR=$((
  PHASE_TRANSITION_LAYER * STAGE1_VISUAL_TOKENS
  + (LLM_NUM_LAYERS - PHASE_TRANSITION_LAYER) * STAGE2_VISUAL_TOKENS
))
_average_whole=$((AVERAGE_VISUAL_TOKENS_NUMERATOR / LLM_NUM_LAYERS))
_average_remainder=$((AVERAGE_VISUAL_TOKENS_NUMERATOR % LLM_NUM_LAYERS))
if (( _average_remainder == 0 )); then
  export AVERAGE_VISUAL_TOKENS="${_average_whole}"
else
  printf -v _average_fraction '%05d' $((_average_remainder * 3125))
  while [[ "${_average_fraction: -1}" == "0" ]]; do
    _average_fraction="${_average_fraction%0}"
  done
  export AVERAGE_VISUAL_TOKENS="${_average_whole}.${_average_fraction}"
fi

if [[ "${VIDEOLLAVA_TREVS_PRESET}" == "136" || "${VIDEOLLAVA_TREVS_PRESET}" == "960" ]]; then
  if [[ "${TREVS_PHASE_SCORING}" != "priority_heads" || "${TREVS_USE_SINK_TOKEN}" != "0" ]]; then
    echo "Preset ${VIDEOLLAVA_TREVS_PRESET} requires priority_heads scoring and no sink token." >&2
    return 2 2>/dev/null || exit 2
  fi
  case "${VIDEOLLAVA_TREVS_PRESET}" in
    960)
      if (( TREVS_ROUTE_TOPK != 180 || TREVS_ROUTE_FPS != 60 \
        || STAGE1_VISUAL_TOKENS != 1920 || STAGE2_VISUAL_TOKENS != 640 )); then
        echo "Preset 960 resolved to an unexpected routing budget." >&2
        return 2 2>/dev/null || exit 2
      fi
      ;;
    136)
      if (( TREVS_ROUTE_TOPK != 26 || TREVS_ROUTE_FPS != 8 \
        || STAGE1_VISUAL_TOKENS != 272 || STAGE2_VISUAL_TOKENS != 90 )); then
        echo "Preset 136 resolved to an unexpected routing budget." >&2
        return 2 2>/dev/null || exit 2
      fi
      ;;
  esac
  if [[ "${VIDEOLLAVA_TREVS_PRESET}" == "960" \
    && "${AVERAGE_VISUAL_TOKENS}" != "960" ]]; then
      echo "Preset 960 resolved to average ${AVERAGE_VISUAL_TOKENS}." >&2
      return 2 2>/dev/null || exit 2
  elif [[ "${VIDEOLLAVA_TREVS_PRESET}" == "136" \
    && "${AVERAGE_VISUAL_TOKENS}" != "135.5" ]]; then
      echo "Preset 136 resolved to average ${AVERAGE_VISUAL_TOKENS}." >&2
      return 2 2>/dev/null || exit 2
  fi
fi

if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
  export VIDEO_LLAVA_ATTN_IMPLEMENTATION="flash_attention_2"
else
  export VIDEO_LLAVA_ATTN_IMPLEMENTATION="sdpa"
fi

_vt="${TREVS_VISUAL_TEMPERATURE//./p}"
_tt="${TREVS_TEXT_TEMPERATURE//./p}"
_semantic="${TREVS_SEMANTIC_LAYER//-/_}"
_average_tag="${AVERAGE_VISUAL_TOKENS//./p}"
_default_experiment="${METHOD}_preset-${VIDEOLLAVA_TREVS_PRESET}_avg${_average_tag}_f8_k${TREVS_ROUTE_TOPK}_fps${TREVS_ROUTE_FPS}_text-${TREVS_TEXT_SCORE_MODE}_vt${_vt}_tt${_tt}_sem${_semantic}_cons${DOUBLE_TRACK_USE_CONSISTENCY_REWARD}_phase-${TREVS_PHASE_SCORING}_sink${TREVS_USE_SINK_TOKEN}_L${PHASE_TRANSITION_LAYER}_keep${PHASE_TRANSITION_N_KEEP}"
export EXPERIMENT="${EXPERIMENT:-${_default_experiment}}"
export RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/${EXPERIMENT}}"

write_videollava_run_config() {
  mkdir -p "${RUN_DIR}"
  local target="${RUN_DIR}/run_config.env"
  local temporary="${RUN_DIR}/.run_config.env.tmp.$$"
  local gpu_mapping=""
  local physical_gpu
  local logical_idx=0
  IFS=',' read -r -a _physical_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  for physical_gpu in "${_physical_gpus[@]}"; do
    physical_gpu="${physical_gpu//[[:space:]]/}"
    [[ -n "${physical_gpu}" ]] || continue
    gpu_mapping+="${gpu_mapping:+,}${physical_gpu}->worker${logical_idx}:cuda:0"
    logical_idx=$((logical_idx + 1))
  done

  {
    printf 'METHOD=%q\n' "${METHOD}"
    printf 'TREVS_ACTIVE=%q\n' "${TREVS_ACTIVE}"
    printf 'VIDEOLLAVA_TREVS_PRESET=%q\n' "${VIDEOLLAVA_TREVS_PRESET}"
    printf 'EXPERIMENT=%q\n' "${EXPERIMENT}"
    printf 'TREVS_ROUTE_TOPK=%q\n' "${TREVS_ROUTE_TOPK}"
    printf 'TREVS_ROUTE_FPS=%q\n' "${TREVS_ROUTE_FPS}"
    printf 'VIDEO_FRAMES=%q\n' "8"
    printf 'PATCHES_PER_FRAME=%q\n' "256"
    printf 'LLM_NUM_LAYERS=%q\n' "${LLM_NUM_LAYERS}"
    printf 'STAGE1_TOPK_VISUAL_TOKENS=%q\n' "${STAGE1_TOPK_VISUAL_TOKENS}"
    printf 'STAGE1_FPS_VISUAL_TOKENS=%q\n' "${STAGE1_FPS_VISUAL_TOKENS}"
    printf 'STAGE1_VISUAL_TOKENS=%q\n' "${STAGE1_VISUAL_TOKENS}"
    printf 'TREVS_TEXT_SCORE_MODE=%q\n' "${TREVS_TEXT_SCORE_MODE}"
    printf 'TREVS_VISUAL_TEMPERATURE=%q\n' "${TREVS_VISUAL_TEMPERATURE}"
    printf 'TREVS_TEXT_TEMPERATURE=%q\n' "${TREVS_TEXT_TEMPERATURE}"
    printf 'TREVS_SEMANTIC_LAYER=%q\n' "${TREVS_SEMANTIC_LAYER}"
    printf 'DOUBLE_TRACK_USE_CONSISTENCY_REWARD=%q\n' "${DOUBLE_TRACK_USE_CONSISTENCY_REWARD}"
    printf 'TREVS_PHASE_SCORING=%q\n' "${TREVS_PHASE_SCORING}"
    printf 'TREVS_USE_SINK_TOKEN=%q\n' "${TREVS_USE_SINK_TOKEN}"
    printf 'PHASE_TRANSITION_LAYER=%q\n' "${PHASE_TRANSITION_LAYER}"
    printf 'PHASE_TRANSITION_N_KEEP=%q\n' "${PHASE_TRANSITION_N_KEEP}"
    printf 'STAGE2_VISUAL_TOKENS=%q\n' "${STAGE2_VISUAL_TOKENS}"
    printf 'AVERAGE_VISUAL_TOKENS=%q\n' "${AVERAGE_VISUAL_TOKENS}"
    printf 'AVERAGE_VISUAL_TOKENS_NUMERATOR=%q\n' "${AVERAGE_VISUAL_TOKENS_NUMERATOR}"
    printf 'AVERAGE_VISUAL_TOKENS_DENOMINATOR=%q\n' "${LLM_NUM_LAYERS}"
    printf 'USE_FLASH_ATTN=%q\n' "${USE_FLASH_ATTN}"
    printf 'ATTN_IMPLEMENTATION=%q\n' "${VIDEO_LLAVA_ATTN_IMPLEMENTATION}"
    printf 'RANDOM_SEED=%q\n' "${RANDOM_SEED}"
    printf 'CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'GPU_LOGICAL_MAPPING=%q\n' "${gpu_mapping}"
    printf 'MAX_SAMPLES=%q\n' "${MAX_SAMPLES}"
    printf 'GENERATION_BATCH_SIZE=%q\n' "1"
    printf 'GENERATION_DO_SAMPLE=%q\n' "0"
    printf 'GENERATION_TEMPERATURE=%q\n' "0"
    printf 'GENERATION_NUM_BEAMS=%q\n' "1"
    printf 'GENERATION_MAX_NEW_TOKENS=%q\n' "${MAX_NEW_TOKENS}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'DATA_ROOT=%q\n' "${DATA_ROOT}"
    printf 'RESULT_ROOT=%q\n' "${RESULT_ROOT}"
    printf 'RUN_DIR=%q\n' "${RUN_DIR}"
    printf 'RUN_JUDGE=%q\n' "${RUN_JUDGE}"
    printf 'VIDEO_QA_JUDGE_MODEL=%q\n' "${VIDEO_QA_JUDGE_MODEL}"
    printf 'VIDEO_QA_JUDGE_PROMPT_VERSION=%q\n' "videollava_videoqa_correctness_v1"
  } > "${temporary}"

  if [[ -f "${target}" ]] && ! cmp -s "${target}" "${temporary}"; then
    local existing_core="${RUN_DIR}/.run_config.env.existing_core.$$"
    local requested_core="${RUN_DIR}/.run_config.env.requested_core.$$"
    sed -e '/^RUN_JUDGE=/d' -e '/^VIDEO_QA_JUDGE_MODEL=/d' \
      "${target}" > "${existing_core}"
    sed -e '/^RUN_JUDGE=/d' -e '/^VIDEO_QA_JUDGE_MODEL=/d' \
      "${temporary}" > "${requested_core}"
    if cmp -s "${existing_core}" "${requested_core}"; then
      rm -f "${temporary}" "${existing_core}" "${requested_core}"
      return 0
    fi
    rm -f "${existing_core}" "${requested_core}"
    rm -f "${temporary}"
    echo "Refusing to mix a different configuration into ${RUN_DIR}." >&2
    echo "Set a new EXPERIMENT name or remove the stale result directory." >&2
    return 1
  fi
  mv -f "${temporary}" "${target}"
}

record_videollava_judge_model() {
  local target="${RUN_DIR}/run_config.env"
  local temporary="${RUN_DIR}/.run_config.env.judge.tmp.$$"
  if [[ ! -f "${target}" ]]; then
    echo "Missing inference configuration: ${target}" >&2
    return 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    case "${line}" in
      RUN_JUDGE=*) printf 'RUN_JUDGE=%q\n' "1" ;;
      VIDEO_QA_JUDGE_MODEL=*)
        printf 'VIDEO_QA_JUDGE_MODEL=%q\n' "${VIDEO_QA_JUDGE_MODEL}"
        ;;
      *) printf '%s\n' "${line}" ;;
    esac
  done < "${target}" > "${temporary}"
  mv -f "${temporary}" "${target}"
}

unset _VIDEO_LLAVA_SCRIPT_DIR _boolean_name _integer_name
unset _average_whole _average_remainder _average_fraction _average_tag
unset _vt _tt _semantic _default_experiment
