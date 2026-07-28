"""Video-LLaVA multimodal wrapper for the independent TReVS LLaMA core.

The classes in this module are instantiated directly by the Video-LLaVA
builder. They intentionally do not register with ``AutoConfig`` or
``AutoModelForCausalLM`` and therefore cannot alter existing LLaVA/Qwen model
registries.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers import LlamaConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from .modelling_sparse_llama import TrevsLlamaForCausalLM, TrevsLlamaModel


class VideoLlavaConfig(LlamaConfig):
    model_type = "videollava_trevs"


class VideoLlavaTReVSModel(LlavaMetaModel, TrevsLlamaModel):
    config_class = VideoLlavaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class VideoLlavaTReVSForCausalLM(TrevsLlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = VideoLlavaConfig
    model_class = VideoLlavaTReVSModel

    def get_model(self):
        return self.model

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
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        image_shape: int = 1024,
        token_length_list: Optional[List[int]] = None,
        pre_prompt_length_list: Optional[List[int]] = None,
        logger=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_shape,
                token_length_list,
                pre_prompt_length_list,
            ) = self.prepare_sparse_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_shape=image_shape,
            token_length_list=token_length_list,
            pre_prompt_length_list=pre_prompt_length_list,
            logger=logger,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        logger=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if inputs is None:
            raise ValueError("Video-LLaVA generation requires input token IDs.")
        if inputs.shape[0] != 1:
            raise ValueError("Video-LLaVA TReVS generation requires batch size 1.")
        if bool(kwargs.get("do_sample", False)):
            raise ValueError("Video-LLaVA TReVS supports greedy generation only (do_sample=False).")
        if int(kwargs.get("num_beams", 1)) != 1:
            raise ValueError("Video-LLaVA TReVS supports greedy generation only (num_beams=1).")
        kwargs.pop("do_sample", None)
        kwargs.pop("num_beams", None)
        if float(kwargs.get("temperature", 1.0)) == 0.0:
            kwargs.pop("temperature")
        if "inputs_embeds" in kwargs:
            raise ValueError("Pass token IDs and images; pre-built inputs_embeds are not supported here.")

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        image_shape = int(kwargs.pop("image_shape", 1024))
        token_length_list = kwargs.pop("token_length_list", None)
        pre_prompt_length_list = kwargs.pop("pre_prompt_length_list", None)
        if images is not None:
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_shape,
                token_length_list,
                pre_prompt_length_list,
            ) = self.prepare_sparse_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        context_limit = int(
            getattr(
                self.config,
                "tokenizer_model_max_length",
                getattr(self.config, "max_position_embeddings", 0),
            )
            or 0
        )
        max_new_tokens = int(kwargs.get("max_new_tokens", 0) or 0)
        if context_limit and max_new_tokens and inputs_embeds.shape[1] + max_new_tokens > context_limit:
            raise ValueError(
                "Video-LLaVA prompt plus requested generation exceeds the model context and "
                "cannot be truncated without invalidating TReVS positions: "
                f"{inputs_embeds.shape[1]} + {max_new_tokens} > {context_limit}."
            )

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_shape=image_shape,
            token_length_list=token_length_list or [],
            pre_prompt_length_list=pre_prompt_length_list or [],
            logger=logger,
            do_sample=False,
            num_beams=1,
            **kwargs,
        )


__all__ = [
    "VideoLlavaConfig",
    "VideoLlavaTReVSModel",
    "VideoLlavaTReVSForCausalLM",
]
