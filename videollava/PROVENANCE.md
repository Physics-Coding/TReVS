# Video-LLaVA implementation provenance

`videollava/` is an isolated inference and evaluation package. It does not
import implementation code from this repository's `llava` or `qwen` packages at
runtime. The isolation prevents model registries, attention patches, caches,
and dependency versions from leaking between model families.

## Source record

This cleaned anonymous artifact does not include VCS history. The following
record names a source revision only where it is present in the retained
materials; it deliberately does not infer a revision from a checkpoint, file
timestamp, or directory name.

| Source lineage | Included package surface | License and revision record | Packaging/modification boundary |
|---|---|---|---|
| Video-LLaVA | `videollava/` outside the local LanguageBind subtree | Apache-2.0; source revision not recorded | Retained the checkpoint-compatible VideoQA inference/evaluation closure only: model loading, prompt assembly, generation, chunking, strict merge, optional text judging, fixtures, and tests. |
| LanguageBind | `videollava/model/multimodal_encoder/languagebind/**` | MIT; full text in [`../THIRD_PARTY_LICENSES/LANGUAGEBIND-MIT.txt`](../THIRD_PARTY_LICENSES/LANGUAGEBIND-MIT.txt); source revision not recorded | Retained the video configuration, deterministic processor, and spatiotemporal vision tower needed by the supported checkpoint format. |
| LLaVA/FastChat-compatible prompt and model interfaces | local conversation, multimodal, and LLaMA-compatible interfaces under `videollava/` | Apache-2.0; exact file-level source revision not recorded | Compatibility is implemented locally. This is not a runtime dependency on the repository's `llava` package. |
| SparseVLM/TReVS closure | `videollava/model/language_model/{modelling_sparse_llama.py,sparse_videollava_llama.py,trevs_router.py,score.py,utils.py}` | Apache-2.0; source revision not recorded | Retained the Video-LLaVA sparse inference closure required by the documented presets. |
| DUET-VLM runner-layout reference | No DUET-VLM source tree is distributed in this package | Recorded upstream license: Apache-2.0; recorded reference revision `cd69e489f39a2a51e3b278a470dc8dc32c00e0fb` | Reference-only influence on the isolated runner layout. It is not a claim that a DUET-VLM checkout or model is included. |

The complete package-level license map is in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). No public URL is
included here because this ZIP must not act as a pointer to an author-controlled
online supplement. Project names and recorded revisions are attribution data,
not download instructions.

## Kept surface

The artifact keeps only the VideoQA execution closure for TGIF-QA, MSVD-QA, and
MSR-VTT-QA:

- checkpoint-compatible LanguageBind video preprocessing and vision modules;
- eight-frame multimodal prompt assembly as one contiguous visual span;
- TReVS Stage-1 routing and Stage-2 physical token pruning;
- synchronized attention-mask, position-id, and KV-cache handling;
- deterministic question chunking and strict ID-based merging; and
- an optional OpenAI-compatible text judge that receives question, reference
  answer, and prediction, but no image or video.

Training, serving, image-only benchmarks, model weights, source videos,
historical predictions, and unrelated pruning methods are outside this package.
The absence of a source revision in the table above is an honest provenance
limitation, not permission to replace it with an unverified revision later.

## Reproducible presets

Both named TReVS presets use eight uniformly sampled frames, transition before
language block 8 of 32, `priority_heads` Stage-2 scoring, no sink token, greedy
generation, and batch size one.

| Preset | Top-k/frame | FPS/frame | Stage 1 | Stage 2 | Actual layer-weighted average |
|---|---:|---:|---:|---:|---:|
| `960` | 180 | 60 | 1920 | 640 | 960 |
| `136` | 26 | 8 | 272 | 90 | 135.5 |

The name `136` is the reported target budget; the exact uniform-frame schedule
has a 135.5-token layer-weighted average and is recorded as such in run
metadata. `dense` retains all 2048 visual tokens. `custom` is for diagnostics
and is not an additional paper result.

## Submission boundary

The Video-LLaVA package contains no author identity, personal path, private
endpoint, credential, checkpoint, benchmark media, or third-party result. A
real GPU smoke verifies a bounded execution path only; it is not evidence of a
complete VideoQA benchmark reproduction. Full results require user-obtained
data, a user-obtained checkpoint, complete coverage, and the declared evaluator.
