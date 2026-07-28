"""Small inference helpers shared by Video-LLaVA evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import StoppingCriteria

from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: Optional[str] = None,
):
    """Tokenize a prompt while replacing every ``<image>`` with a sentinel."""

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
    input_ids = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk_index, chunk in enumerate(prompt_chunks):
        if chunk_index:
            input_ids.append(image_token_index)
        input_ids.extend(chunk[offset:])

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")


def tokenizer_video_token(
    prompt: str,
    tokenizer,
    video_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: Optional[str] = None,
):
    """Alias documenting Video-LLaVA's shared image/video sentinel format."""

    return tokenizer_image_token(prompt, tokenizer, video_token_index, return_tensors)


def get_model_name_from_path(model_path: str) -> str:
    path = Path(model_path.rstrip("/"))
    if path.name.startswith("checkpoint-"):
        return f"{path.parent.name}_{path.name}"
    return path.name


class KeywordsStoppingCriteria(StoppingCriteria):
    """Stop once every sequence in a batch ends with one of the keywords."""

    def __init__(self, keywords, tokenizer, input_ids: torch.LongTensor):
        self.keywords = list(keywords)
        self.keyword_ids = []
        for keyword in self.keywords:
            ids = tokenizer(keyword).input_ids
            if len(ids) > 1 and ids[0] == tokenizer.bos_token_id:
                ids = ids[1:]
            self.keyword_ids.append(torch.tensor(ids, dtype=torch.long))
        self.max_keyword_len = max((ids.numel() for ids in self.keyword_ids), default=0)
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def _call_for_batch(self, output_ids: torch.LongTensor) -> bool:
        for keyword_ids in self.keyword_ids:
            keyword_ids = keyword_ids.to(output_ids.device)
            if keyword_ids.numel() and output_ids.shape[1] >= keyword_ids.numel():
                if torch.equal(output_ids[0, -keyword_ids.numel():], keyword_ids):
                    return True

        generated = output_ids.shape[1] - self.start_len
        offset = min(max(generated, 0), self.max_keyword_len)
        if offset == 0:
            return False
        text = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        return any(keyword in text for keyword in self.keywords)

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return all(self._call_for_batch(row.unsqueeze(0)) for row in output_ids)
