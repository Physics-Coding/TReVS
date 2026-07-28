from typing import Any, Iterable
import os

from PIL import Image


def _get_fixed_image_size() -> tuple[int, int]:
    height = int(os.environ.get("QWEN_IMAGE_RESIZED_HEIGHT", "896"))
    width = int(os.environ.get("QWEN_IMAGE_RESIZED_WIDTH", "1120"))
    return height, width


def _load_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        path = image
        if path.startswith("file://"):
            path = path[len("file://") :]
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(image)!r}")


def _prepare_image(item: dict[str, Any]) -> Image.Image:
    height = int(item.get("resized_height") or os.environ.get("QWEN_IMAGE_RESIZED_HEIGHT", "896"))
    width = int(item.get("resized_width") or os.environ.get("QWEN_IMAGE_RESIZED_WIDTH", "1120"))
    image = _load_image(item.get("image"))
    return image.resize((width, height), Image.BICUBIC)


def _iter_content_items(messages: Iterable[dict[str, Any]]):
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, dict):
            content = [content]
        if isinstance(content, str):
            continue
        for item in content:
            if isinstance(item, dict):
                yield item


def process_vision_info(messages):
    """Minimal local fallback for qwen-vl-utils used by the eval scripts.

    The upstream helper returns image and video payload lists extracted from
    chat messages. Our eval scripts only need path/PIL image forwarding plus
    the occasional video field, so keep this implementation intentionally small.
    """
    image_inputs = []
    video_inputs = []
    for item in _iter_content_items(messages):
        item_type = item.get("type")
        if item_type == "image":
            image_inputs.append(_prepare_image(item))
        elif item_type == "video":
            video_inputs.append(item.get("video"))
    return image_inputs or None, video_inputs or None
