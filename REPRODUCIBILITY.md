# Reproducibility protocol

This document defines what can and cannot be concluded from each validation
step in the anonymous artifact. It is intentionally stricter than a single
"tests passed" statement.

## 0. Fresh-archive and submission boundary

Create exactly one mutually exclusive runtime environment from the extracted
source root:

```bash
bash scripts/bootstrap_environment.sh --family llava-family
# or, in a different environment:
bash scripts/bootstrap_environment.sh --family qwen
```

The environment recipes pin the intended interpreter, PyTorch, TorchVision,
CUDA runtime, and primary Python dependencies. They do not lock the NVIDIA
driver, host CUDA toolkit, compiler, GPU architecture, or local model cache.
For the LLaVA family, `flash-attn==2.3.3` is built after PyTorch with
`--no-build-isolation` and requires `nvcc`, a compatible CUDA toolkit, a C++
compiler, and `ninja`. Before creating the environment, the bootstrap requires
`CUDA_HOME/bin/nvcc` when `CUDA_HOME` is set (otherwise the default `nvcc`) to
report the same major.minor version as `pytorch-cuda=12.1` in the recipe. A
different host-default compiler, including CUDA 13, is rejected before it can
produce a mismatched FlashAttention build. Record the resolved versions and
attention backend in each run's `environment.json`; do not infer them from the
YAML filename.

No checkpoint, CLIP cache, real benchmark sample, image, video, complete
prediction, or judge response is distributed. The decision not to include a
real representative subset is deliberate: this archive does not claim a
license audit sufficient to redistribute benchmark annotations or media.
Synthetic fixtures validate schemas and evaluators only.

For LLaVA-1.5 and LLaVA-NeXT, a complete LLaVA checkpoint alone is not enough.
Read `mm_vision_tower` (or `vision_tower`) from the checkpoint's `config.json`.
The model loader calls `CLIPVisionConfig`, `CLIPImageProcessor`, and
`CLIPVisionModel` for that exact value with `local_files_only=True`; the
config-declared tower must already resolve from the active Hugging Face cache
or a complete local path. The current preflight checks the local configuration
resolution, while a real model load verifies the remaining processor/weights.
`openai/clip-vit-large-patch14-336` is common, but is never a substitute for
the actual checkpoint configuration. `--preflight` is the required early gate.

Optional tests that require a user-supplied checkpoint or real video may be
reported as skipped when their opt-in environment variables are absent. A skip
is not a failed contract, and it is not GPU, data, speed, or accuracy evidence.

## 1. Evidence levels

| Level | Required evidence | What it establishes | What it does not establish |
|---|---|---|---|
| Static | imports, shell syntax, anonymous scan, manifest | package and launcher consistency | numerical or GPU correctness |
| Unit/synthetic | deterministic tests and small fixtures | routing, budgets, masks, cache and evaluator contracts | real checkpoint quality |
| Dry-run | resolved family/preset/datasets/commands | launcher configuration | input availability or inference |
| Preflight | checkpoint/data/schema/hash/GPU/backend checks | local inputs satisfy the declared contract | generated answers or accuracy |
| GPU smoke | real checkpoint and a bounded sample | multimodal load/forward/generation path works | complete benchmark coverage |
| Full benchmark | complete unique IDs plus official evaluator | result for one declared run | generalization to another protocol |
| Paper reproduction | full benchmark within CSV tolerance | a reported paper row was reproduced | rows absent from the expected CSV |

Never describe a CPU test, synthetic fixture, processor check, dry-run, or
single-sample generation as a reproduced 7B benchmark.

## 2. Immutable inputs

For each formal run, record the following before model loading:

- model family, public checkpoint identifier, local revision and small-file
  hashes; do not hash multi-gigabyte weights repeatedly unless requested;
- data set name, release/split, question count, unique-ID count and SHA-256 of
  question/annotation files;
- image/video root, discovered media count, missing media count and a bounded
  list of failures;
- Python, PyTorch, TorchVision, Transformers, CUDA runtime, GPU model and
  attention backend;
- resolved Stage-1 and Stage-2 budgets, transition layer, scoring mode, sink,
  temperatures, input resolution/frame count and seed;
- the exact command and repository manifest hash.

Preflight must fail before expensive model loading on missing files, duplicate
IDs, unsupported preset overrides, incompatible attention backends or an
unexpected input token grid.

## 3. Recommended execution order

From a fresh extraction of the anonymous ZIP:

1. Verify `MANIFEST.sha256` if present.
2. Create exactly one environment through
   `scripts/bootstrap_environment.sh --family llava-family` or
   `scripts/bootstrap_environment.sh --family qwen`; never install both extras
   together.
3. Obtain checkpoints and data directly from the third-party distributors.
4. Reproduce the tree in `data/README.md` or pass the documented dataset
   paths through the unified launcher.
5. Run `bash scripts/run_tests.sh`.
6. Run every intended command once with `--dry-run`.
7. Run every intended command once with `--preflight`.
8. Run one real GPU smoke and inspect decoded output plus token-budget traces.
9. Run the complete benchmark without silently dropping failed samples.
10. Run the declared evaluator and compare structured metrics against
    `expected_results/paper_metrics.csv`.

The public launcher form is:

```bash
bash scripts/reproduce.sh \
  --family <llava15|llava_next|qwen25vl|videollava> \
  --preset <supported-preset> \
  --model-path /path/to/checkpoint \
  --data-root /path/to/eval_data \
  --output-root /path/to/outputs \
  --datasets <comma-separated-list> \
  --gpus <physical-NVIDIA-device-ID-list> \
  --seed 42
```

Add exactly one of `--dry-run` or `--preflight` for the corresponding gate.
Dry-run must not create an output directory. Preflight may emit a validation
report but must not create benchmark predictions. `--gpus` uses physical
NVIDIA IDs, such as `3` or `3,5`; preflight launches a separate process with
each requested ID as its only visible CUDA device before it accepts the run.

## 4. Paper preset contracts

### LLaVA-1.5

The three displayed average-token presets use transition layer 8 in a 32-layer
language model. Their Stage-1/Stage-2 schedules are:

| Preset | Stage-1 top-k | Stage-1 FPS | Stage-1 total | Stage-2 total | Exact weighted average |
|---:|---:|---:|---:|---:|---:|
| 32 | 48 | 16 | 64 | 21 | 31.75 |
| 64 | 96 | 32 | 128 | 42 | 63.50 |
| 128 | 192 | 64 | 256 | 85 | 127.75 |

The paper names are rounded display budgets. The exact averages must remain in
run metadata.

### LLaVA-NeXT

The supported display presets are 160, 320 and 640. The sparse path uses a
fixed 672x672 input represented as one base crop and four local crops. It does
not use variable `spatial_unpad` or append `image_newline`. These presets are
code-supported, but this artifact intentionally does not place unverified
LLaVA-NeXT scores in `paper_metrics.csv`.

### Qwen2.5-VL

The input is explicitly resized to 1120x896 and must resolve to
`image_grid_thw=[1,64,80]`: 5120 raw patches and 1280 merged visual tokens.
TReVS uses SDPA, M-RoPE-aware pruning, and per-layer cache/mask lengths.

| Preset | Stage-1 top-k | Stage-1 FPS | Stage-1 total | Stage-2 total | Weighted average |
|---:|---:|---:|---:|---:|---:|
| 142 | 204 | 68 | 272 | 90 | 142 |
| 284 | 408 | 136 | 544 | 180 | 284 |
| 426 | 612 | 204 | 816 | 270 | 426 |

The transition occurs before language block 8 of 28. `dense` must restore the
original Hugging Face forward functions; merely setting a method flag inside
patched forwards is not a valid dense baseline.

### Video-LLaVA

Each sample uses eight uniformly sampled frames and 256 patch candidates per
frame. Both named presets prune before block 8 of 32, use priority-head
Stage-2 scoring and disable the sink token.

| Preset | Per-frame top-k/FPS | Stage 1 | Stage 2 | Exact weighted average |
|---:|---|---:|---:|---:|
| 136 | 26/8 | 272 | 90 | 135.5 |
| 960 | 180/60 | 1920 | 640 | 960 |

Chunk notation such as `0/4` describes question-set partitioning, not frames.
Every child process must finish before ID-based merge. Duplicate and missing
question IDs are fatal. A previous non-normative RTX 4090 smoke loaded the
checkpoint in about 3 minutes 40 seconds and generated ten TGIF examples in
about 9 seconds after loading. This is not a full-dataset runtime claim and
must not be extrapolated across hardware.

## 5. Evaluator contracts

### GQA

The public artifact converts predictions to the official evaluator schema and
marks the result `submission_only` with evaluator `external_official`. Run the
third-party official GQA evaluator separately, then import its structured
accuracy for comparison. Broad log regexes can accidentally capture later
word-count diagnostics; do not infer accuracy from an ambiguous model log.

### ScienceQA

The paper column is image-only `IMG-Accuracy`. Do not substitute full-test
`Accuracy`. A standard test release has 4241 total questions and 2017
image-associated questions; preflight must record the actual input count.

### TextVQA

Use the M4C normalization/evaluator on TextVQA v0.5.1 validation annotations.
The expected validation sample count is 5000. Results from a different custom
normalizer are not comparable.

### POPE

Retain adversarial, popular and random predictions, including valid empty
answers, and report the arithmetic mean of their F1 values in percent.

### MME

Convert predictions to the official task-directory layout, then use the
bundled official-compatible calculation code. Report `Overall MME Total`, not
one category or a partial sum.

### MMBench and MMBench-CN

The public artifact generates the submission workbooks and marks them
`submission_only` with evaluator `external_official`; it does not bundle an
official scorer. Submit/evaluate the workbooks with the declared DEV EN/CN
protocol, then import the resulting structured `Overall` value. A converter's
direct `prediction == answer` printout is not automatically the paper metric.
Read the TSVs as logical CSV records, not physical lines: the previously
audited DEV-EN input had 4377 records and 4329 clean official rows, while the
DEV-CN input had 4329 records. The local release must be recorded rather than
assumed.

### VQAv2

This is `submission_only`: preserve every question ID, merge chunks exactly
once, and generate the official upload JSON. Do not claim a local test-dev
accuracy.

### VideoQA judge

The optional judge receives only question, reference answer and prediction.
It requests strict JSON with `pred` (`yes`/`no`) and integer `score` (0-5) at
temperature zero. Formal comparisons require the same provider model, model
revision, prompt version and judge parameters for every method. A 2xx response
without a valid Chat Completions JSON body is a failure.

Credentials are read only from `OPENAI_API_KEY`; endpoint and model are read
from `OPENAI_BASE_URL` and `VIDEO_QA_JUDGE_MODEL`. Never place a key in a
launcher, command argument, result config or ZIP.

## 6. Derived metrics

All relative-accuracy calculations use the displayed benchmark precision.

LLaVA-1.5 uses eight dense references in this order:

```text
GQA=62.0, ScienceQA-Image=69.5, TextVQA=58.2, POPE=85.9,
MME=1862, VQAv2=78.5, MMBench=64.7, MMBench-CN=58.3
```

```text
RelAcc = 100/8 * sum(TReVS_metric_i / dense_metric_i)
```

Each LLaVA-1.5 preset therefore has nine CSV records: eight benchmark metrics
plus one derived RelAcc record. RelAcc is not included again in its own
denominator. GQA, VQAv2, MMBench and MMBench-CN values originate from external
official evaluation even though their expected paper values are listed in the
CSV.

Qwen2.5-VL uses four dense references:

```text
GQA=59.7, TextVQA=76.7, MME=2324, MMBench=83.8
```

```text
RelAcc = 100/4 * sum(TReVS_metric_i / dense_metric_i)
```

The expected CSV contains one-decimal paper displays and tolerance 0.1. A
comparison tool must reject family, preset, split, metric or evaluator
mismatches instead of applying the tolerance to incomparable runs.

## 7. Seed and determinism

The launcher seeds Python `random`, NumPy, PyTorch CPU and all visible CUDA
devices with the declared seed. Formal inference uses greedy decoding unless a
benchmark contract says otherwise. GPU kernels and third-party decoders may
still have nondeterministic implementations; report any observed rerun range
rather than claiming bitwise determinism without evidence.

## 8. Failure policy

- Missing/corrupt media, CUDA errors, OOM, model load errors and schema errors
  are fatal; do not convert them to empty predictions.
- A valid model-generated empty answer remains in the evaluation denominator.
- Resume only when the existing run configuration matches exactly.
- Merge by stable question ID and reject duplicates, missing IDs and unexpected
  IDs before scoring.
- Keep raw evaluator output and structured metrics together in the run
  directory; do not write them into the data tree.
- Mark a paper row `reproduced` only after full coverage and evaluator success.

## 9. Final artifact audit

Before OpenReview upload, `scripts/build_anonymous_zip.sh` must check that the
allowlisted ZIP contains no:

- author names, affiliations, personal email addresses, user home paths,
  internal hosts or credentials;
- `.git`, `.codex`, `.agents`, environment caches, databases or logs;
- model weights, complete benchmark data, predictions or judge responses;
- paper/checklist source, office documents, archives or undeclared large files;
- absolute/symlink targets or archive path traversal.

The built-in scan cannot know every submission author's name or affiliation.
Before the final build, create an external UTF-8 file outside the repository
with one literal per line for every author-name variant, affiliation/laboratory,
personal domain, and private-host fragment. Do not store that list in source
control, the stage tree, or the ZIP. Pass it to all source/stage/extraction
scans through the builder:

```bash
bash scripts/build_anonymous_zip.sh \
  --identity-patterns-file /tmp/trevs-aaai-identity-patterns.txt
```

Extract the ZIP into a new temporary directory, verify its SHA-256 manifest,
run all unit tests and dry-runs there, and inspect the actual OpenReview upload
field size limit immediately before submission. Upload that ZIP only through
OpenReview's `Supplementary Code and Data Package (ZIP)` field. Upload the
Reproducibility Checklist through its own designated field, not in the ZIP.
The paper must be self-contained; reviewers are not required to run or inspect
the artifact, and an anonymous repository or other web page is not a substitute
for the uploaded package.
