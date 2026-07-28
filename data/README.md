# Evaluation data layout

The anonymous ZIP contains no benchmark samples, images, videos or model
weights. Obtain every resource from its official third-party distributor and
accept its terms before use. `DATA_ROOT` must point to the parent of the
following logical directories.

## No real representative subset

This archive deliberately includes no real benchmark sample, even a small one.
The retained tests use synthetic schema and evaluator fixtures only. The reason
is conservative redistribution: benchmark annotations, images, and videos can
have independent terms, and this artifact does not claim a release-by-release
rights audit authorizing their redistribution. A synthetic fixture establishes
only a code contract, never a benchmark score. Do not add a real sample to the
submission ZIP unless its exact release, redistribution permission, and
attribution requirements have been documented and independently verified.

## Model-side offline prerequisites

`MODEL_PATH` is a user-obtained model checkpoint and is not part of `DATA_ROOT`.
For LLaVA-1.5 and LLaVA-NeXT, the checkpoint's `config.json` must contain a
nonempty `mm_vision_tower` or `vision_tower`. The LLaVA loader resolves that
exact value with `local_files_only=True` for the CLIP configuration, image
processor, and vision-model weights. The current preflight verifies that the
config-declared tower configuration resolves locally; a subsequent real model
load still requires the complete processor and weight files. Consequently,
downloading only the LLaVA checkpoint is insufficient.

A common public LLaVA configuration names
`openai/clip-vit-large-patch14-336`, but the value in the active checkpoint is
the requirement. Before running `--preflight`, make the config-declared tower
complete in the active Hugging Face cache (controlled, if desired, by
`HF_HOME`) or provide the complete local path declared by the config. Do not
replace it with another CLIP tower, let inference attempt a network download, or
copy a cache into this ZIP. The preflight failure is intentional: obtain the
config-declared tower first, then rerun the gate.

```text
DATA_ROOT/
├── gqa/
│   ├── llava_gqa_testdev_balanced.jsonl
│   ├── images/                       # or a path resolved by preflight
│   └── testdev_balanced_questions.json
├── textvqa/
│   ├── llava_textvqa_val_v051_ocr.jsonl
│   ├── TextVQA_0.5.1_val.json
│   └── train_images/
├── mme/
│   ├── llava_mme.jsonl
│   └── MME_Benchmark/                # official category directories directly below
├── mmbench/
│   ├── mmbench_dev_20230712.tsv
│   └── mmbench_dev_cn_20231003.tsv
├── scienceqa/
│   ├── llava_test_CQM-A.json
│   ├── pid_splits.json
│   ├── problems.json
│   └── images/test/
├── pope/
│   ├── llava_pope_test.jsonl
│   └── coco/                         # coco_pope_{adversarial,popular,random}.json
├── coco/
│   └── val2014/                      # shared COCO validation images for POPE
├── vqav2/
│   ├── llava_vqav2_mscoco_test-dev2015.jsonl
│   ├── llava_vqav2_mscoco_test2015.jsonl
│   └── test2015/                     # COCO test2015 images
└── videollava/
    └── GPT_Zero_Shot_QA/             # direct layout is accepted as well
        ├── TGIF_Zero_Shot_QA/
        │   ├── mp4/
        │   ├── test_q.json
        │   └── test_a.json
        ├── MSVD_Zero_Shot_QA/
        │   ├── videos/
        │   ├── test_q.json
        │   └── test_a.json
        └── MSRVTT_Zero_Shot_QA/
            ├── videos/all/
            ├── test_q.json
            └── test_a.json
```

The resolved paths, not the illustrative tree above, are authoritative and are
stored in run metadata. Predictions and metrics always go under `OUTPUT_ROOT`,
never inside this data tree.

## Required preflight record

Third-party mirrors and converted LLaVA question files are not byte-identical,
so this repository does not publish misleading universal hashes for them.
Instead, preflight emits a JSON report to standard output containing:

| Field | Requirement |
|---|---|
| dataset/release/split | exact declared identifier |
| question/annotation files | relative path, bytes and SHA-256 |
| question records | total, unique IDs, duplicate IDs and malformed rows |
| media | discovered count, missing count and supported extensions |
| transformations | conversion script/version and output hash, if applicable |
| evaluator | module/version and required reference files |

Useful manual checks before preflight are:

```bash
find /path/to/eval_data -type f -printf '%P\t%s\n' | LC_ALL=C sort
sha256sum /path/to/eval_data/gqa/llava_gqa_testdev_balanced.jsonl
sha256sum /path/to/eval_data/textvqa/TextVQA_0.5.1_val.json
sha256sum /path/to/eval_data/mmbench/mmbench_dev_20230712.tsv
sha256sum /path/to/eval_data/mmbench/mmbench_dev_cn_20231003.tsv
```

Do not place the resulting inventory, absolute paths or hashes from a private
storage hierarchy into the submitted ZIP. Keep them in the corresponding run
directory, for example by redirecting preflight output to a file below
`OUTPUT_ROOT` that is not part of the submitted source archive.

## Dataset-specific validation

- GQA: require the balanced test-dev question IDs and
  `gqa/testdev_balanced_questions.json`; the artifact emits an external
  official-evaluator submission and does not score GQA locally.
- TextVQA: require 5000 validation annotations and use the M4C evaluator.
- MME: `mme/MME_Benchmark/` is the data root passed to the converter. Its
  immediate children are the official category directories; missing categories
  are fatal rather than a partial score.
- MMBench: keep both EN and CN TSVs under `mmbench/`. The artifact creates
  external-official submission workbooks and does not score them locally.
  Parse TSV with a CSV parser because quoted fields may contain newlines. The
  previously audited DEV-EN release has 4377 logical records and its external
  clean set has 4329; the DEV-CN release has 4329 logical records. Record the
  local release's logical and externally accepted counts rather than assuming
  a physical-line count.
- ScienceQA: `scienceqa/pid_splits.json` and `scienceqa/problems.json` are both
  required. Record total and image-question counts; the expected standard test
  release has 4241 total and 2017 image-associated questions.
- POPE: require all three annotation files under `pope/coco/` and the shared
  COCO validation images under `coco/val2014/`.
- VQAv2: require both converted question JSONL files and COCO test2015 images
  under `vqav2/test2015/`. Preserve every test-dev question ID; this artifact
  only creates a submission file.
- VideoQA: verify question/reference lengths and IDs, resolve every referenced
  video, decode at least one sample per dataset, and sample exactly eight
  frames for the formal protocol.

See `DATA_LICENSES.md` before downloading or redistributing any resource.
