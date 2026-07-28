"""Constants shared by the Video-LLaVA inference path."""

IGNORE_INDEX = -100

# Video-LLaVA represents every video frame with the same placeholder used for
# an image.  Eight consecutive placeholders are expanded into eight frame
# feature blocks by the multimodal input assembler.
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = DEFAULT_IMAGE_PATCH_TOKEN
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"

NUM_VIDEO_FRAMES = 8
VIDEO_PATCHES_PER_FRAME = 256
