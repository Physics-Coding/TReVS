"""Deterministic Decord video decoding and CLIP preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional
from transformers.image_processing_utils import BatchFeature


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def uniform_sample_indices(total_frames: int, num_frames: int = 8) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError(f"Video contains no decodable frames: {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)


def decode_video_decord(video_path: Union[str, Path], num_frames: int = 8) -> torch.Tensor:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    try:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        indices = uniform_sample_indices(len(reader), num_frames)
        decoded = reader.get_batch(indices.tolist())
        if hasattr(decoded, "asnumpy"):
            decoded = torch.from_numpy(decoded.asnumpy())
        elif not torch.is_tensor(decoded):
            decoded = torch.as_tensor(decoded)
    except Exception as error:
        raise RuntimeError(f"Failed to decode video {path}: {error}") from error

    if decoded.ndim != 4 or decoded.shape[-1] != 3:
        raise RuntimeError(
            f"Decord returned {tuple(decoded.shape)} for {path}; expected [T, H, W, 3]"
        )
    return decoded.permute(0, 3, 1, 2).contiguous()


class DeterministicVideoTransform:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Expected decoded frames [T, 3, H, W], got {tuple(frames.shape)}")
        frames = frames.to(dtype=torch.float32).div_(255.0)
        mean = frames.new_tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1)
        std = frames.new_tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        frames = transform_functional.resize(
            frames,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        frames = transform_functional.center_crop(frames, [self.image_size, self.image_size])
        return frames.permute(1, 0, 2, 3).contiguous()


def _vision_config(config):
    return getattr(config, "vision_config", config)


def get_video_transform(config) -> DeterministicVideoTransform:
    config = _vision_config(config)
    if getattr(config, "video_decode_backend", "decord") != "decord":
        raise ValueError("The isolated Video-LLaVA evaluator supports only Decord decoding")
    return DeterministicVideoTransform(image_size=config.image_size)


def load_and_transform_video(
    video_path,
    transform,
    video_decode_backend: str = "decord",
    clip_start_sec: float = 0.0,
    clip_end_sec=None,
    num_frames: int = 8,
) -> torch.Tensor:
    if video_decode_backend != "decord":
        raise ValueError("The isolated Video-LLaVA evaluator supports only Decord decoding")
    if clip_start_sec != 0.0 or clip_end_sec is not None:
        raise ValueError("VideoQA evaluation always samples uniformly from the complete video")
    return transform(decode_video_decord(video_path, num_frames=num_frames))


class LanguageBindVideoProcessor:
    """Processor compatible with the original ``preprocess(...)[pixel_values]`` API."""

    def __init__(self, config):
        self.config = _vision_config(config)
        if self.config.num_frames != 8 or self.config.image_size != 224:
            raise ValueError(
                "Video-LLaVA requires deterministic 8-frame, 224x224 preprocessing"
            )
        self.transform = get_video_transform(self.config)
        self.image_processor = load_and_transform_video
        self.num_frames = self.config.num_frames

    def __call__(self, images=None, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        del kwargs
        if images is None:
            raise ValueError("A video path must be provided")
        if return_tensors not in (None, "pt"):
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        paths: Sequence = images if isinstance(images, (list, tuple)) else [images]
        pixel_values = [
            self.image_processor(
                path,
                self.transform,
                video_decode_backend=self.config.video_decode_backend,
                num_frames=self.config.num_frames,
            )
            for path in paths
        ]
        return BatchFeature(data={"pixel_values": torch.stack(pixel_values)}, tensor_type="pt")

    def preprocess(self, images, return_tensors: str = "pt") -> BatchFeature:
        return self(images=images, return_tensors=return_tensors)


def build_default_video_processor(num_frames: int = 8) -> LanguageBindVideoProcessor:
    """Build the standalone processor used by data preflight and evaluation."""

    if num_frames != 8:
        raise ValueError(f"Video-LLaVA requires exactly 8 frames, got {num_frames}")
    from .configuration_video import build_official_video_config

    return LanguageBindVideoProcessor(build_official_video_config())
