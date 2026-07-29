# TReVS

TReVS is an inference and evaluation toolkit for efficient visual-token routing
in multimodal language models. It provides one reproducible command interface
for image and video evaluation while preserving the native model behavior of
each supported family.

## Highlights

- Four model families: LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL, and Video-LLaVA.
- Fixed, tested TReVS presets with explicit stage budgets and attention
  backends.
- A single `scripts/reproduce.sh` entry point for dry runs, preflight checks,
  inference, and evaluator dispatch.
- Structured run metadata, predictions, metrics, and official-format exports
  written outside the source tree.
- Unit, fixture, launcher, packaging, and import-isolation tests for the
  supported inference paths.

## Supported Models

| Model family | CLI family | TReVS presets | Runtime environment | Primary package |
|---|---|---|---|---|
| LLaVA-1.5 | `llava15` | `32`, `64`, `128` | `trevs-llava-family` | `llava/` |
| LLaVA-NeXT | `llava_next` | `160`, `320`, `640` | `trevs-llava-family` | `llava/` |
| Qwen2.5-VL | `qwen25vl` | `142`, `284`, `426`, `dense` | `trevs-qwen` | `qwen/` |
| Video-LLaVA | `videollava` | `136`, `960`, `dense`, `custom` | `trevs-llava-family` | `videollava/` |

LLaVA-1.5 and LLaVA-NeXT share the `llava` package but use different visual
input protocols and token budgets. Video-LLaVA is isolated from both `llava`
and `qwen` at runtime.

## Quick Start

### 1. Create an Environment

The LLaVA family and Qwen integrations require different dependency stacks.
Create only the environment required for the model family you plan to run.

```bash
# LLaVA-1.5, LLaVA-NeXT, and Video-LLaVA
bash scripts/bootstrap_environment.sh --family llava-family
conda activate trevs-llava-family

# Qwen2.5-VL
bash scripts/bootstrap_environment.sh --family qwen
conda activate trevs-qwen
```

The environment recipes are available in
[`environment/llava_family.yml`](environment/llava_family.yml) and
[`environment/qwen.yml`](environment/qwen.yml). The LLaVA-family environment
builds `flash-attn==2.3.3`; it requires a CUDA 12.1 toolkit, `nvcc`, a C++
compiler, and `ninja`.

### 2. Prepare Checkpoints and Data

Model checkpoints and benchmark releases are not bundled with this repository.
Obtain them from their original distributors and keep them outside the source
tree. Dataset layouts, expected files, licenses, and attribution are documented
in [`data/README.md`](data/README.md) and
[`data/DATA_LICENSES.md`](data/DATA_LICENSES.md).

For LLaVA checkpoints, make the vision tower declared by `config.json`
available locally before running preflight. The loader uses
`local_files_only=True` and does not download model components during
inference.

### 3. Validate the Setup

Run a dry run first. It resolves the preset and planned commands without
loading a checkpoint or creating outputs.

```bash
bash scripts/reproduce.sh \
  --family llava15 \
  --preset 64 \
  --model-path /path/to/models/llava-v1.5-7b \
  --data-root /path/to/eval_data \
  --output-root /path/to/outputs \
  --datasets gqa,textvqa,mme,mmbench \
  --gpus 0 \
  --seed 42 \
  --dry-run
```

Then validate local checkpoints, data schemas, dependencies, GPU visibility,
and attention-backend requirements:

```bash
bash scripts/reproduce.sh \
  --family llava15 \
  --preset 64 \
  --model-path /path/to/models/llava-v1.5-7b \
  --data-root /path/to/eval_data \
  --output-root /path/to/outputs \
  --datasets gqa,textvqa,mme,mmbench \
  --gpus 0 \
  --seed 42 \
  --preflight
```

### 4. Run Evaluation

Remove `--dry-run` or `--preflight` to execute inference and evaluation:

```bash
bash scripts/reproduce.sh \
  --family llava15 \
  --preset 64 \
  --model-path /path/to/models/llava-v1.5-7b \
  --data-root /path/to/eval_data \
  --output-root /path/to/outputs \
  --datasets gqa,textvqa,mme,mmbench \
  --gpus 0 \
  --seed 42
```

`--model-path` and `--data-root` are required. `--output-root` defaults to
`outputs/` under the repository root. Use physical CUDA IDs in `--gpus` from
an unmasked parent process; do not set `CUDA_VISIBLE_DEVICES` before invoking
the launcher.

## Presets

| Family | Visual input protocol | Presets | Routing notes |
|---|---|---|---|
| `llava15` | 336x336 image, 576 visual patch tokens | 32 / 64 / 128 | Phase transition at layer 8 |
| `llava_next` | 672x672, one base crop plus four local crops | 160 / 320 / 640 | Fixed five-crop protocol |
| `qwen25vl` | 1120x896 resize, 1280 merged visual tokens | 142 / 284 / 426 | TReVS requires SDPA |
| `videollava` | Eight 224x224 frames, 256 patches per frame | 136 / 960 | SDPA, priority heads, sink off |

The displayed Video-LLaVA `136` preset has an exact layer-weighted average of
135.5 visual tokens. The `custom` Video-LLaVA preset is intended for controlled
diagnostics and requires explicit stage-budget arguments.

`dense` uses the corresponding dense baseline path. For Qwen, dense preserves
the native Hugging Face forward path; TReVS presets use SDPA because the sparse
path relies on four-dimensional masks and heterogeneous cache lengths.

## Model Family Examples

```bash
# LLaVA-NeXT
bash scripts/reproduce.sh \
  --family llava_next --preset 320 \
  --model-path /path/to/models/llava-next \
  --data-root /path/to/eval_data \
  --output-root /path/to/outputs \
  --datasets gqa,textvqa,mme,mmbench \
  --gpus 0 --seed 42

# Qwen2.5-VL
bash scripts/reproduce.sh \
  --family qwen25vl --preset 284 \
  --model-path /path/to/models/Qwen2.5-VL-7B-Instruct \
  --data-root /path/to/eval_data \
  --output-root /path/to/outputs \
  --datasets gqa,textvqa,mme,mmbench \
  --gpus 0 --seed 42

# Video-LLaVA
bash scripts/reproduce.sh \
  --family videollava --preset 136 \
  --model-path /path/to/models/Video-LLaVA-7B \
  --data-root /path/to/eval_data/videollava \
  --output-root /path/to/outputs \
  --datasets tgif,msvd,msrvtt \
  --gpus 0 --seed 42
```

## Outputs and Evaluation

Each run is isolated under the output root:

```text
OUTPUT_ROOT/<family>/<preset>/<run_id>/
├── run_config.json
├── environment.json
├── data_manifest.json
├── logs/
├── predictions/
├── submissions/
└── metrics/
```

`run_config.json` records the resolved preset, stage budgets, generation
configuration, seed, input protocol, data split metadata, and invocation.
`environment.json` records Python, PyTorch, Transformers, CUDA, GPU, and
attention-backend details. Outputs are never written into `DATA_ROOT`.

| Benchmark | Evaluation path | Reported quantity |
|---|---|---|
| ScienceQA | Local evaluator | Image-only accuracy |
| TextVQA | Local M4C evaluator | Validation accuracy |
| POPE | Local evaluator | Mean F1 over three splits |
| MME | Local evaluator | Overall MME total |
| GQA | Official-format export | External official result |
| MMBench / MMBench-CN | Official-format workbook | External official result |
| VQAv2 | Official-format JSON | External official result |
| TGIF-QA / MSVD-QA / MSR-VTT-QA | Optional text judge | Judge accuracy or mean score |

Use evaluator-produced JSON or CSV outputs as the metric source. The legacy
`all_datasets_summary.json` file is not treated as a metric authority.

## Testing

Run the test suite after both runtime environments are available:

```bash
bash scripts/run_tests.sh
```

The suite covers evaluator fixtures, preset contracts, launcher dry runs,
shell syntax, package imports, model routing, cache and mask behavior, and
Video-LLaVA chunk/merge logic. Optional real-checkpoint and real-video tests
are enabled only when their documented environment variables are set.

Unit and fixture tests establish code contracts. They do not replace a
full-checkpoint, full-dataset GPU evaluation.

## Repository Layout

```text
configs/        Preset definitions and runtime contracts
data/           Dataset layouts, licenses, and provenance notes
evaluation/     Local evaluators and metric aggregation
environment/    Conda environment recipes
llava/          LLaVA-1.5 and LLaVA-NeXT inference paths
qwen/           Qwen2.5-VL integration and routing patches
scripts/        Environment setup, reproducibility, and benchmark launchers
videollava/     Isolated Video-LLaVA inference and evaluation paths
```

## License and Third-Party Components

This repository is distributed under the [Apache License 2.0](LICENSE).
Third-party source, models, and data remain subject to their original licenses
and terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`data/DATA_LICENSES.md`](data/DATA_LICENSES.md) for attribution and usage
information.
