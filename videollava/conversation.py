"""Prompt templates used by the Video-LLaVA evaluation entrypoints."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: Tuple[str, str]
    messages: Sequence[Sequence[Optional[str]]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: Optional[str] = None
    version: str = "Unknown"

    def get_prompt(self) -> str:
        if self.sep_style == SeparatorStyle.SINGLE:
            prompt = self.system + self.sep
            for role, message in self.messages:
                prompt += f"{role}: {message}{self.sep}" if message else f"{role}:"
            return prompt

        if self.sep_style == SeparatorStyle.TWO:
            separators = (self.sep, self.sep2)
            prompt = self.system + separators[0]
            for index, (role, message) in enumerate(self.messages):
                if message:
                    prompt += f"{role}: {message}{separators[index % 2]}"
                else:
                    prompt += f"{role}:"
            return prompt

        if self.sep_style == SeparatorStyle.MPT:
            prompt = self.system + self.sep
            for role, message in self.messages:
                prompt += role + (message + self.sep if message else "")
            return prompt

        if self.sep_style == SeparatorStyle.LLAMA_2:
            prompt = ""
            for index, (role, message) in enumerate(self.messages):
                if index == 0:
                    if not message or role != self.roles[0]:
                        raise ValueError("The first Llama-2 message must come from the user")
                    message = f"<<SYS>>\n{self.system}\n<</SYS>>\n\n{message}"
                if not message:
                    continue
                if index % 2 == 0:
                    prompt += self.sep + f"[INST] {message} [/INST]"
                else:
                    prompt += f" {message} {self.sep2}"
            return prompt.removeprefix(self.sep)

        if self.sep_style == SeparatorStyle.PLAIN:
            separators = (self.sep, self.sep2)
            prompt = self.system
            for index, (_, message) in enumerate(self.messages):
                if message:
                    prompt += message + separators[index % 2]
            return prompt

        raise ValueError(f"Unsupported separator style: {self.sep_style}")

    def append_message(self, role: str, message: Optional[str]) -> None:
        if not isinstance(self.messages, list):
            self.messages = [list(item) for item in self.messages]
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[list(item) for item in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self) -> dict:
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [list(item) for item in self.messages],
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

conv_vicuna_v1 = Conversation(
    system=_VICUNA_SYSTEM,
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1 = Conversation(
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v0 = Conversation(
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
    sep2="\n",
)

conv_llama_2 = Conversation(
    system="You are a helpful, respectful and honest assistant.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system=(
        "<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. "
        "The assistant gives helpful and honest answers."
    ),
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_llava_v0,
    "v0": conv_llava_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llava_v1": conv_llava_v1,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llama_2": conv_llama_2,
    "llava_llama_2": conv_llama_2,
    "mpt": conv_mpt,
}
