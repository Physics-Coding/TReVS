#!/bin/bash

trevs_sanitize_tag() {
    local value="${1:-}"
    value="${value// /}"
    value="${value//,/_}"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value//./p}"
    echo "${value}"
}

trevs_configure_env() {
    METHOD="$(echo "${METHOD:-trevs}" | tr '[:upper:]' '[:lower:]')"
    case "${METHOD}" in
        trevs|dense) ;;
        *)
            echo "Unsupported METHOD=${METHOD}. Expected trevs or dense." >&2
            return 1
            ;;
    esac
    export METHOD

    if [[ "${METHOD}" == "dense" ]]; then
        TREVS_TEXT_SCORE_TAG=""
        TREVS_TEMPERATURE_TAG=""
        TREVS_ROUTE_TAG=""
        TREVS_SEMANTIC_LAYER_TAG=""
        TREVS_PHASE_SCORING_TAG=""
        TREVS_SINK_TOKEN_TAG=""
        TREVS_CONSISTENCY_REWARD_TAG=""
        TREVS_EXPERIMENT_TAG=""
        export TREVS_TEXT_SCORE_TAG TREVS_TEMPERATURE_TAG TREVS_ROUTE_TAG
        export TREVS_SEMANTIC_LAYER_TAG TREVS_PHASE_SCORING_TAG TREVS_SINK_TOKEN_TAG
        export TREVS_CONSISTENCY_REWARD_TAG TREVS_EXPERIMENT_TAG
        return 0
    fi

    if [[ "${METHOD}" == "trevs" ]]; then
        local use_flash_attn="${USE_FLASH_ATTN:-0}"
        use_flash_attn="${use_flash_attn,,}"
        case "${use_flash_attn}" in
            0|false|no|off) ;;
            *)
                echo "TReVS requires USE_FLASH_ATTN=0: the custom Qwen decoder uses a 4D additive attention mask and must run with SDPA, not FlashAttention2." >&2
                return 1
                ;;
        esac
    fi

    case "${TREVS_TEXT_SCORE_MODE}" in
        rms|max|mean) ;;
        *)
            echo "Unsupported TREVS_TEXT_SCORE_MODE=${TREVS_TEXT_SCORE_MODE}. Expected rms, max, or mean." >&2
            return 1
            ;;
    esac
    case "${TREVS_PHASE_SCORING}" in
        priority_heads) TREVS_PHASE_SCORING_TAG="_psph" ;;
        all_heads) TREVS_PHASE_SCORING_TAG="_psall" ;;
        *)
            echo "Unsupported TREVS_PHASE_SCORING=${TREVS_PHASE_SCORING}. Expected priority_heads or all_heads." >&2
            return 1
            ;;
    esac
    case "${TREVS_USE_SINK_TOKEN}" in
        1) TREVS_SINK_TOKEN_TAG="" ;;
        0) TREVS_SINK_TOKEN_TAG="_sink0" ;;
        *)
            echo "Unsupported TREVS_USE_SINK_TOKEN=${TREVS_USE_SINK_TOKEN}. Expected 1 or 0." >&2
            return 1
            ;;
    esac
    case "${TREVS_USE_CONSISTENCY_REWARD}" in
        1) TREVS_CONSISTENCY_REWARD_TAG="" ;;
        0) TREVS_CONSISTENCY_REWARD_TAG="_cr0" ;;
        *)
            echo "Unsupported TREVS_USE_CONSISTENCY_REWARD=${TREVS_USE_CONSISTENCY_REWARD}. Expected 1 or 0." >&2
            return 1
            ;;
    esac

    TREVS_TEXT_SCORE_TAG="_ts$(trevs_sanitize_tag "${TREVS_TEXT_SCORE_MODE}")"
    TREVS_TEMPERATURE_TAG="_tv$(trevs_sanitize_tag "${TREVS_VISUAL_TEMPERATURE}")_tt$(trevs_sanitize_tag "${TREVS_TEXT_TEMPERATURE}")"
    TREVS_ROUTE_TAG="_topk${TREVS_ROUTE_TOPK}_fps${TREVS_ROUTE_FPS}"
    TREVS_SEMANTIC_LAYER_TAG=""
    case "${TREVS_SEMANTIC_LAYER}" in
        ""|none|off|default|current) ;;
        *) TREVS_SEMANTIC_LAYER_TAG="_semL$(trevs_sanitize_tag "${TREVS_SEMANTIC_LAYER}")" ;;
    esac

    if [[ "${METHOD}" == "trevs" ]]; then
        TREVS_EXPERIMENT_TAG="${TREVS_ROUTE_TAG}${TREVS_TEXT_SCORE_TAG}${TREVS_TEMPERATURE_TAG}${TREVS_SEMANTIC_LAYER_TAG}${TREVS_PHASE_SCORING_TAG}${TREVS_SINK_TOKEN_TAG}${TREVS_CONSISTENCY_REWARD_TAG}_pt${PHASE_TRANSITION_LAYER}-${PHASE_TRANSITION_N_KEEP}"
    else
        TREVS_EXPERIMENT_TAG=""
    fi

    export TREVS_TEXT_SCORE_TAG TREVS_TEMPERATURE_TAG TREVS_ROUTE_TAG
    export TREVS_SEMANTIC_LAYER_TAG TREVS_PHASE_SCORING_TAG TREVS_SINK_TOKEN_TAG
    export TREVS_CONSISTENCY_REWARD_TAG TREVS_EXPERIMENT_TAG
}
