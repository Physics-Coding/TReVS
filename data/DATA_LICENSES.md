# Data, checkpoints, and redistribution boundary

No dataset, checkpoint, tokenizer archive, CLIP cache, benchmark image, video,
prediction, upload workbook, or judge response is included in this anonymous
artifact. This file records the required inputs and conservative redistribution
boundary; it is not legal advice. The terms attached to the exact release used
for a run always take precedence.

## Deliberate no-subset decision

The ZIP distributes no real representative data subset. This is intentional:
benchmark annotations and underlying media may have different licenses, and the
artifact does not claim to have verified redistribution permission for every
release and item. The small fixtures under the test tree are synthetic and
exercise only schemas/evaluators. They must never be described as benchmark
samples or evidence of benchmark accuracy.

A future real sample may be added only after documenting its exact source
release, split, copyright/attribution notice, redistribution permission, and
whether every included image/video is covered. Until then, users must obtain
their own authorized copies directly from the original distributors.

## Dataset inputs

| Resource | Required release/split in this artifact | Artifact role | Distributed in ZIP | User obligation |
|---|---|---|---|---|
| GQA | balanced test-dev questions and matching image access | submission generation; external official evaluation | No | Obtain the official release/evaluator and comply with GQA plus underlying image terms. |
| TextVQA | v0.5.1 validation annotations and authorized images | local M4C accuracy | No | Obtain annotations/images and comply with the release and image-source terms. |
| MME | official perception/cognition category layout | local official-compatible calculation | No | Obtain the official release and preserve its category structure and terms. |
| MMBench / MMBench-CN | declared DEV EN/CN TSV releases | submission workbook generation; external official evaluation | No | Obtain the declared TSV releases and follow their evaluation/usage terms. |
| ScienceQA | test image questions, splits, problems, and images | local image-only accuracy | No | Obtain test data/images and comply with the release and source-image terms. |
| POPE | adversarial, popular, and random questions/references plus COCO media | local three-subset F1 | No | Obtain questions/references and authorized COCO images; comply with both sets of terms. |
| VQAv2 | test/test-dev questions and COCO test2015 images | submission generation | No | Obtain the official VQA/COCO releases and submit through the official evaluator. |
| TGIF-QA | zero-shot VideoQA annotations and authorized videos | optional external text-judge evaluation | No | Obtain metadata/videos from authorized sources and comply with their terms. |
| MSVD-QA | zero-shot VideoQA annotations and authorized videos | optional external text-judge evaluation | No | Obtain metadata/videos from authorized sources and comply with their terms. |
| MSR-VTT-QA | zero-shot VideoQA annotations and authorized videos | optional external text-judge evaluation | No | Obtain metadata/videos from authorized sources and comply with their terms. |

The project/release names above are source identifiers, not anonymous download
links or a substitute for the original terms. The exact resolved paths, release
identifiers, split, question/annotation hashes, file counts, and media counts
belong in each run's `data_manifest.json`, not in the submitted source ZIP.

## Checkpoints and offline vision towers

| Required local resource | Distributed in ZIP | Required use/terms |
|---|---|---|
| `liuhaotian/llava-v1.5-7b` | No | User-obtained LLaVA checkpoint; comply with LLaVA and Vicuna/Llama terms. |
| `liuhaotian/llava-v1.6-vicuna-7b` | No | User-obtained LLaVA-NeXT checkpoint; comply with LLaVA and Vicuna/Llama terms. |
| Config-declared LLaVA CLIP vision tower | No | Required for both LLaVA families. Read `mm_vision_tower` or `vision_tower` from `MODEL_PATH/config.json` and make the exact CLIP config, image processor, and weights available offline. `openai/clip-vit-large-patch14-336` is common but not a replacement for the declared value. |
| `Qwen/Qwen2.5-VL-7B-Instruct` | No | User-obtained checkpoint; use the exact model-card revision recorded by the run. |
| `LanguageBind/Video-LLaVA-7B` | No | User-obtained checkpoint; comply with Video-LLaVA, LanguageBind, and backbone terms. |

The LLaVA loader deliberately calls the CLIP APIs with `local_files_only=True`.
Set `HF_HOME` before environment activation when an explicit cache location is
needed, then run the unified launcher with `--preflight`. Do not upload weights,
tokenizer files, cache snapshots, signed URLs, or authentication tokens with
the source package.

## Evaluator outputs and services

Predictions may reproduce benchmark questions, media identifiers, or reference
metadata. Therefore complete JSON/JSONL/XLSX predictions, judge responses, and
official submission files are run artifacts and are excluded from the anonymous
ZIP. Only synthetic fixtures and table-backed metric values in
`expected_results/paper_metrics.csv` are distributed.

The VideoQA text judge is optional and user-selected. It is not needed for
local model inference. Its endpoint, provider account, model revision, and
credentials are external run inputs that must not be placed in source, a command
argument, run metadata intended for submission, or the ZIP.

## Submission boundary

This archive is a reproducibility package, not a vehicle for data/model
redistribution. It contains no author-controlled data mirror or external
supplement link. Before upload, review the current conference and OpenReview
requirements, upload the code ZIP only in the designated code-and-data field,
and upload the Reproducibility Checklist separately.
