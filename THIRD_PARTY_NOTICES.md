# Third-party notices and provenance map

This anonymous artifact contains retained, adapted, and compatibility code from
public third-party projects. Attribution is a license obligation; it does not
identify the authors of this submission. To preserve double-blind review, this
notice names upstream projects and versions where they are recorded locally but
does not contain links to online code, model, data, or author-controlled
supplements.

The root [`LICENSE`](LICENSE) contains Apache License 2.0. It governs the
Apache-licensed components listed below and the anonymous integration work to
the extent permitted by the original licenses. Existing copyright and license
headers in source files remain authoritative. The two non-Apache texts that are
required by bundled source are included verbatim under
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/).

## How to read this map

- **Bundled path** is the path or glob present in this ZIP. A path not listed
  here is not evidence that a project is absent; ordinary runtime dependencies
  are installed separately and are discussed below.
- **Revision** is reported only when it is recorded in the cleaned artifact.
  `Not recorded` is deliberate: this archive has no VCS metadata and must not
  invent a source commit from file timestamps or checkpoint names.
- **Modification scope** describes the packaged surface, not an assertion that
  the listed upstream project authored every line in that glob.

## Bundled and adapted source

| Component/source lineage | Bundled path(s) | License text and recorded revision | Modification scope in this artifact |
|---|---|---|---|
| LLaVA inference and evaluation foundations | `llava/**`, except the M4C file and sparse closure listed separately | Apache-2.0 in [`LICENSE`](LICENSE); LLaVA source revision: not recorded | Retained only the LLaMA/Vicuna inference, multimodal assembly, evaluators, fixtures, and tests needed by LLaVA-1.5 and LLaVA-NeXT. Training, serving, and release tooling are not bundled. |
| SparseVLM/TReVS LLaVA closure | `llava/model/language_model/{modelling_sparse_llama.py,sparse_llava_llama.py,trevs_router.py,score.py,utils.py}` | Apache-2.0 in [`LICENSE`](LICENSE); SparseVLM source revision: not recorded | Sparse language-model integration and visual-token routing/pruning closure needed by the documented LLaVA presets. This notice does not claim that a cleaned source tree can reconstruct a more granular upstream commit map. |
| Qwen2.5-VL compatibility and TReVS closure | `qwen/**`, `qwen_vl_utils.py` | Apache-2.0 in [`LICENSE`](LICENSE); Qwen2.5-VL source revision: not recorded | Checkpoint-compatible processor helpers, Hugging Face forward integration, evaluator entrypoints, and TReVS runtime closure. The separately installed Transformers dependency is fixed to commit `a25b8efa0a3220da89493a72f57081aa5720291f` in `pyproject.toml`; that is not a Qwen source revision. |
| Hugging Face Transformers-compatible LLaMA/Qwen integration | `llava/model/language_model/modelling_sparse_llama.py`, `qwen/model/qwen2_5_vl_custom.py`, and their direct wrappers | Apache-2.0 in [`LICENSE`](LICENSE); installed version/commit is declared in `environment/` and `pyproject.toml` | Retained compatibility code for the two mutually exclusive runtime stacks. It is not a bundled copy of an installed Transformers distribution. |
| Video-LLaVA inference and evaluation foundations | `videollava/**`, except `videollava/model/multimodal_encoder/languagebind/**` | Apache-2.0 in [`LICENSE`](LICENSE); Video-LLaVA source revision: not recorded | Isolated VideoQA model loading, eight-frame assembly, inference, chunk merge, optional text judging, fixtures, and tests. No image-only benchmark, training, serving, model weights, or videos are included. |
| LanguageBind video tower | `videollava/model/multimodal_encoder/languagebind/**` | MIT in [`THIRD_PARTY_LICENSES/LANGUAGEBIND-MIT.txt`](THIRD_PARTY_LICENSES/LANGUAGEBIND-MIT.txt); LanguageBind source revision: not recorded | Checkpoint-compatible video configuration, processor, and spatiotemporal vision modules retained for local Video-LLaVA inference. |
| MMF/M4C TextVQA normalizer | `llava/eval/m4c_evaluator.py` | BSD-3-Clause in [`THIRD_PARTY_LICENSES/MMF-BSD-3-CLAUSE.txt`](THIRD_PARTY_LICENSES/MMF-BSD-3-CLAUSE.txt); source reference embedded in the file names commit `c46b3b3391275b4181567db80943473a89ab98ab` | Retained answer normalization and TextVQA accuracy implementation. `evaluation/textvqa.py` is the artifact-local strict wrapper and is not presented as a copy of MMF. |
| FastChat/Alpaca conversational API lineage | `llava/conversation.py` and conversation-compatible prompt utilities under `llava/` | Apache-2.0 in [`LICENSE`](LICENSE); exact file-level upstream revision: not recorded | Retained prompt-template compatibility surface only. The source tree does not carry enough provenance metadata to assert a narrower unmodified-file lineage. |
| DUET-VLM runner-layout reference | No DUET-VLM source glob is asserted as bundled; the reference influenced package isolation and runner layout | Recorded upstream license: Apache-2.0; recorded reference revision `cd69e489f39a2a51e3b278a470dc8dc32c00e0fb` | Reference-only provenance. Do not interpret this row as a claim that the artifact redistributes a DUET-VLM checkout. |

## Separately installed runtime dependencies

PyTorch, TorchVision, NumPy, Pillow, Accelerate, bitsandbytes, SentencePiece,
tokenizers, safetensors, einops, timm, scikit-learn, pandas, OpenPyXL, Requests,
HTTPX, PyAV, Decord, FlashAttention, and Hugging Face Transformers are installed
from their ordinary distributors. They are not vendored in the ZIP. Their
licenses apply independently. The intended versions are declared in
`environment/` and `pyproject.toml`; a Conda recipe is a reproducibility recipe,
not a redistribution of those projects.

## Separately obtained models and data

The artifact does not redistribute model checkpoints, tokenizer archives,
vision-tower caches, benchmark annotations, images, videos, prediction files,
or external judge responses. Users must obtain and comply with the exact terms
for:

- `liuhaotian/llava-v1.5-7b`;
- `liuhaotian/llava-v1.6-vicuna-7b` and its Vicuna/Llama base terms;
- the `mm_vision_tower` declared by each LLaVA checkpoint configuration;
- `Qwen/Qwen2.5-VL-7B-Instruct`;
- `LanguageBind/Video-LLaVA-7B` and its language/vision backbones; and
- every benchmark and underlying image/video source documented in
  [`data/DATA_LICENSES.md`](data/DATA_LICENSES.md).

The root Apache-2.0 license does not override any checkpoint, model-card,
dataset, media, or external-service terms.

## Anonymous-submission boundary

This source ZIP intentionally contains no author-controlled web supplement.
It contains no author name, affiliation, email address, personal path, access
token, or endpoint. The conference submission must use the designated OpenReview
code-and-data ZIP field; the Reproducibility Checklist is a separate submission
field and is not part of this archive. The main paper must remain self-contained
because reviewers are not required to run this artifact.

## Notice corrections

During double-blind review there is intentionally no author-identifying contact
address in this archive. A notice issue should be raised through the
conference's confidential submission discussion mechanism.
