"""LLaMA decoder with Video-LLaVA TReVS lookahead pruning.

The implementation deliberately stays close to Transformers 4.37's LLaMA
interfaces while adding three invariants required by TReVS:

* block ``k`` scores block ``k - 1`` output before block ``k`` executes;
* pruned RoPE positions are preserved and tables use ``position_ids.max()+1``;
* each decoder layer builds its mask from that layer's own KV-cache length.

The last point permits layers before the transition to keep the full prompt
cache while later layers keep the shortened prompt cache.
"""

from typing import List, Optional, Tuple, Union

import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaSdpaAttention,
    _prepare_4d_causal_attention_mask_for_sdpa,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import is_flash_attn_2_available

from .score import TREVS_ENABLED
from .trevs_router import (
    apply_sink_token_pruning,
    get_trevs_phase_scoring,
    score_phase_attention,
    use_sink_token,
)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE after co-locating tensors for device-mapped inference."""
    target_device = query_states.device
    return hf_apply_rotary_pos_emb(
        query_states,
        key_states.to(target_device),
        cos.to(target_device),
        sin.to(target_device),
        position_ids.to(target_device),
        unsqueeze_dim=unsqueeze_dim,
    )


def compute_manual_attention_probs(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    num_key_value_groups: int = 1,
    causal_query_offset: Optional[int] = None,
) -> torch.Tensor:
    """Compute attention probabilities without updating the decoder cache."""
    if key_states.shape[1] != query_states.shape[1]:
        key_states = repeat_kv(key_states, num_key_value_groups)

    _, _, query_length, _ = query_states.shape
    key_length = key_states.shape[-2]
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        query_states.shape[-1]
    )
    mask_value = torch.finfo(scores.dtype).min

    if is_causal:
        if causal_query_offset is None:
            causal_query_offset = max(key_length - query_length, 0)
        query_positions = (
            torch.arange(query_length, device=query_states.device) + int(causal_query_offset)
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=query_states.device).unsqueeze(0)
        causal_mask = key_positions <= query_positions
        scores = scores.masked_fill(
            ~causal_mask.view(1, 1, query_length, key_length), mask_value
        )

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            key_padding_mask = attention_mask[:, None, None, :].to(
                device=query_states.device, dtype=torch.bool
            )
            scores = scores.masked_fill(~key_padding_mask, mask_value)
        elif attention_mask.dim() == 4:
            if attention_mask.shape[-2:] != (query_length, key_length):
                raise ValueError(
                    "Lookahead attention mask has incompatible shape "
                    f"{tuple(attention_mask.shape)} for Q/K lengths {(query_length, key_length)}."
                )
            scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)


@torch.no_grad()
def compute_lookahead_text_to_vision_attention(
    decoder_layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    text_start: int,
    text_end: int,
    vision_start: int,
    vision_length: int,
) -> torch.Tensor:
    """Use block k's normalized Q/K to score before running block k."""
    _, sequence_length, _ = hidden_states.shape
    text_start = int(text_start)
    text_end = int(text_end)
    vision_start = int(vision_start)
    vision_length = int(vision_length)
    if not (0 <= vision_start < vision_start + vision_length <= sequence_length):
        raise ValueError(
            f"Invalid lookahead vision span [{vision_start}, {vision_start + vision_length}) "
            f"for sequence length {sequence_length}."
        )
    if not (0 <= text_start < text_end <= sequence_length):
        raise ValueError(
            f"Invalid lookahead text span [{text_start}, {text_end}) "
            f"for sequence length {sequence_length}."
        )

    self_attn = decoder_layer.self_attn
    normalized_states = decoder_layer.input_layernorm(hidden_states)
    batch_size = normalized_states.shape[0]
    query_states = self_attn.q_proj(normalized_states).view(
        batch_size, sequence_length, self_attn.num_heads, self_attn.head_dim
    ).transpose(1, 2)
    key_states = self_attn.k_proj(normalized_states).view(
        batch_size, sequence_length, self_attn.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)

    rotary_sequence_length = int(position_ids.max().item()) + 1
    cos, sin = self_attn.rotary_emb(key_states, seq_len=rotary_sequence_length)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    query_states = query_states[:, :, text_start:text_end, :]

    lookahead_mask = attention_mask
    if lookahead_mask is not None and lookahead_mask.dim() == 4:
        lookahead_mask = lookahead_mask[:, :, text_start:text_end, :]
    needs_explicit_causal_mask = lookahead_mask is None or lookahead_mask.dim() == 2
    attention_probs = compute_manual_attention_probs(
        query_states,
        key_states,
        attention_mask=lookahead_mask,
        is_causal=self_attn.is_causal and needs_explicit_causal_mask,
        num_key_value_groups=self_attn.num_key_value_groups,
        causal_query_offset=text_start,
    )
    return attention_probs[:, :, :, vision_start : vision_start + vision_length]


def get_layer_cache_lengths(
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]
) -> List[int]:
    """Expose layer-local lengths for diagnostics and regression tests."""
    if past_key_values is None:
        return []
    if isinstance(past_key_values, Cache):
        return [int(past_key_values.get_seq_length(idx)) for idx in range(len(past_key_values))]
    return [int(layer_cache[0].shape[-2]) for layer_cache in past_key_values]


def _crop_2d_attention_mask(
    attention_mask: Optional[torch.Tensor], target_length: int
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected a 2D attention mask, got {tuple(attention_mask.shape)}.")
    if attention_mask.shape[-1] < target_length:
        raise ValueError(
            f"Attention mask length {attention_mask.shape[-1]} is shorter than required {target_length}."
        )
    if attention_mask.shape[-1] == target_length:
        return attention_mask
    # Formal VideoQA inference is batch-one and unpadded. Cropping is therefore
    # only a length adaptation for the short-cache layers.
    if not bool(torch.all(attention_mask != 0)):
        raise ValueError(
            "Heterogeneous TReVS cache lengths currently require an unpadded batch-one prompt."
        )
    return attention_mask[:, -target_length:]


class TrevsLlamaSdpaAttention(LlamaSdpaAttention):
    """SDPA attention with layer-local cache masks and preserved-position RoPE."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        del kwargs
        if output_attentions:
            raise ValueError("Video-LLaVA TReVS does not support output_attentions with SDPA.")

        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        key_value_length = key_states.shape[-2]
        if past_key_value is not None:
            key_value_length += past_key_value.get_usable_length(
                key_value_length, self.layer_idx
            )
        rotary_sequence_length = (
            int(position_ids.max().item()) + 1 if TREVS_ENABLED else key_value_length
        )
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_sequence_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"sin": sin, "cos": cos},
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None and attention_mask.shape != (
            batch_size,
            1,
            query_length,
            key_value_length,
        ):
            raise ValueError(
                "Attention mask should have shape "
                f"{(batch_size, 1, query_length, key_value_length)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and query_length > 1,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, self.hidden_size
        )
        return self.o_proj(attention_output), None, past_key_value


class TrevsLlamaFlashAttention2(LlamaFlashAttention2):
    """FlashAttention2 variant that preserves non-renumbered RoPE positions."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        del use_cache, kwargs
        if output_attentions:
            raise ValueError("Video-LLaVA TReVS does not support FlashAttention2 attentions.")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError("FlashAttention2 requires None or a 2D padding mask.")

        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        key_value_length = key_states.shape[-2]
        if past_key_value is not None:
            key_value_length += past_key_value.get_usable_length(
                key_value_length, self.layer_idx
            )
        rotary_sequence_length = (
            int(position_ids.max().item()) + 1 if TREVS_ENABLED else key_value_length
        )
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_sequence_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"sin": sin, "cos": cos},
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attention_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=self.attention_dropout if self.training else 0.0,
        )
        attention_output = attention_output.reshape(
            batch_size, query_length, self.hidden_size
        ).contiguous()
        return self.o_proj(attention_output), None, past_key_value


TREVS_ATTENTION_CLASSES = {
    "sdpa": TrevsLlamaSdpaAttention,
    "flash_attention_2": TrevsLlamaFlashAttention2,
}


class TrevsLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        implementation = config._attn_implementation
        if implementation not in TREVS_ATTENTION_CLASSES:
            raise ValueError(
                "Video-LLaVA TReVS supports only SDPA or FlashAttention2; "
                f"got {implementation!r}. Set USE_FLASH_ATTN=0 for the supported default."
            )
        self.hidden_size = config.hidden_size
        self.self_attn = TREVS_ATTENTION_CLASSES[implementation](config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, attention_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual.to(attention_output.device) + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class TrevsLlamaModel(LlamaModel):
    """LLaMA backbone with one global lookahead visual-token transition."""

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [TrevsLlamaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        if self._use_flash_attention_2:
            if not is_flash_attn_2_available():
                raise ImportError(
                    "FlashAttention2 was requested for Video-LLaVA TReVS but flash_attn is unavailable."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("FlashAttention2 requires an available CUDA device.")
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                raise RuntimeError(
                    "FlashAttention2 for Video-LLaVA requires an Ampere-or-newer GPU; "
                    f"detected compute capability {capability[0]}.{capability[1]}."
                )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.last_trevs_metrics = {}
        self.post_init()

    def _layer_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Cache],
    ) -> Optional[torch.Tensor]:
        query_length = hidden_states.shape[1]
        past_length = (
            int(past_key_values.get_seq_length(layer_idx))
            if past_key_values is not None
            else 0
        )
        target_length = past_length + query_length
        if self._use_flash_attention_2:
            mask_2d = _crop_2d_attention_mask(attention_mask, target_length)
            return mask_2d if mask_2d is not None and not bool(torch.all(mask_2d != 0)) else None
        mask_2d = _crop_2d_attention_mask(attention_mask, target_length)
        return _prepare_4d_causal_attention_mask_for_sdpa(
            mask_2d,
            (hidden_states.shape[0], query_length),
            hidden_states,
            past_length,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_shape: int = 1024,
        token_length_list: Optional[List[int]] = None,
        pre_prompt_length_list: Optional[List[int]] = None,
        logger=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        del logger
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if output_attentions:
            raise ValueError("Video-LLaVA TReVS inference does not support output_attentions=True.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Specify input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, sequence_length = inputs_embeds.shape[:2]
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        use_legacy_cache = False
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_layer0_length = (
            int(past_key_values.get_seq_length(0))
            if isinstance(past_key_values, Cache)
            else 0
        )
        if position_ids is None:
            position_ids = torch.arange(
                past_layer0_length,
                past_layer0_length + sequence_length,
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)

        token_length_list = list(token_length_list or [])
        pre_prompt_length_list = list(pre_prompt_length_list or [])
        is_prefill = bool(pre_prompt_length_list) and sequence_length != 1
        if TREVS_ENABLED and is_prefill and batch_size != 1:
            raise ValueError("Video-LLaVA TReVS formal inference requires batch size 1.")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Video-LLaVA TReVS expects a 2D input attention mask; "
                "per-layer 4D causal masks are built internally."
            )

        phase_transition_layer = int(os.getenv("PHASE_TRANSITION_LAYER", "8"))
        phase_transition_n_keep = int(os.getenv("PHASE_TRANSITION_N_KEEP", "341"))
        if phase_transition_n_keep < 0:
            raise ValueError("PHASE_TRANSITION_N_KEEP must be >= 0.")
        if TREVS_ENABLED and not (1 <= phase_transition_layer < len(self.layers)):
            raise ValueError(
                "PHASE_TRANSITION_LAYER uses a full-layer-count convention and must satisfy "
                f"1 <= value < {len(self.layers)}, got {phase_transition_layer}."
            )

        hidden_states = inputs_embeds
        active_attention_mask = attention_mask
        vision_start = int(pre_prompt_length_list[0]) if is_prefill else 0
        vision_length = int(image_shape)
        text_start = vision_start + vision_length
        phase_transition_done = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = past_key_values
        if is_prefill or not self.last_trevs_metrics:
            self.last_trevs_metrics = {
                "trevs_enabled": bool(TREVS_ENABLED),
                "phase_transition_applied": False,
            }

        for layer_idx, decoder_layer in enumerate(self.layers):
            if (
                TREVS_ENABLED
                and is_prefill
                and layer_idx == phase_transition_layer
                and not phase_transition_done
                and vision_length > 1
            ):
                layer_device = decoder_layer.input_layernorm.weight.device
                hidden_states = hidden_states.to(layer_device)
                position_ids = position_ids.to(layer_device)
                if active_attention_mask is not None:
                    active_attention_mask = active_attention_mask.to(layer_device)
                lookahead_mask = self._layer_attention_mask(
                    active_attention_mask, hidden_states, layer_idx, past_key_values
                )
                text_end = min(
                    int(token_length_list[0]) if token_length_list else hidden_states.shape[1],
                    hidden_states.shape[1],
                )
                attention_tv = compute_lookahead_text_to_vision_attention(
                    decoder_layer=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=lookahead_mask,
                    position_ids=position_ids,
                    text_start=text_start,
                    text_end=text_end,
                    vision_start=vision_start,
                    vision_length=vision_length,
                )
                actual_n_keep = min(max(phase_transition_n_keep, 0), vision_length - 1)
                phase_scoring = get_trevs_phase_scoring()
                sink_enabled = use_sink_token()
                idx_keep, idx_drop = score_phase_attention(
                    attention_tv, n_keep=actual_n_keep, mode=phase_scoring
                )
                hidden_states, active_attention_mask, position_ids = apply_sink_token_pruning(
                    hidden_states=hidden_states,
                    v_token_start=vision_start,
                    n_vis_current=vision_length,
                    idx_keep_local=idx_keep,
                    idx_drop_local=idx_drop,
                    attention_mask=active_attention_mask,
                    position_ids=position_ids,
                    use_sink=sink_enabled,
                )
                before_pruning = vision_length
                vision_length = actual_n_keep + int(sink_enabled and idx_drop.shape[1] > 0)
                text_start = vision_start + vision_length
                phase_transition_done = True
                self.last_trevs_metrics = {
                    "trevs_enabled": True,
                    "phase_transition_applied": True,
                    "n_vis_before_phase": before_pruning,
                    "n_vis_phase": vision_length,
                    "prefill_seq_len_phase": hidden_states.shape[1],
                    "trevs_phase_scoring": phase_scoring,
                    "trevs_use_sink_token": sink_enabled,
                    "trevs_phase_mode": "lookahead",
                    "trevs_phase_full_layers": phase_transition_layer,
                    "trevs_phase_scoring_layer_idx": layer_idx,
                    "trevs_phase_pruned_after_layer_idx": layer_idx - 1,
                }

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_mask = self._layer_attention_mask(
                active_attention_mask, hidden_states, layer_idx, past_key_values
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        self.last_cache_lengths = get_layer_cache_lengths(next_cache)
        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, next_cache, all_hidden_states, all_self_attentions)
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TrevsLlamaForCausalLM(LlamaForCausalLM):
    """Causal-LM head that propagates TReVS metadata through greedy decode."""

    model_class = TrevsLlamaModel

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = self.model_class(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_shape: int = 1024,
        token_length_list: Optional[List[int]] = None,
        pre_prompt_length_list: Optional[List[int]] = None,
        logger=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_shape=image_shape,
            token_length_list=token_length_list,
            pre_prompt_length_list=pre_prompt_length_list,
            logger=logger,
        )
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = torch.cat(
                [F.linear(hidden_states, weight) for weight in lm_head_slices], dim=-1
            )
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_shape: int = 1024,
        token_length_list: Optional[List[int]] = None,
        pre_prompt_length_list: Optional[List[int]] = None,
        logger=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs.update(
            {
                "image_shape": image_shape,
                "token_length_list": token_length_list or [],
                "pre_prompt_length_list": pre_prompt_length_list or [],
                "logger": logger,
            }
        )
        if past_key_values is not None and model_inputs.get("position_ids") is not None:
            current_inputs = model_inputs.get("input_ids")
            query_length = current_inputs.shape[1] if current_inputs is not None else 1
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -query_length:]
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            standardize_cache_format=standardize_cache_format,
        )
        position_ids = model_kwargs.get("position_ids")
        if position_ids is not None:
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, position_ids[:, -1:] + 1], dim=-1
            )
        return model_kwargs
