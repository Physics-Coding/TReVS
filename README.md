# TReVS: AAAI-27 Anonymous Reproducibility Artifact

This is an anonymous, inference-and-evaluation-only artifact for TReVS. It
retains the four execution paths needed by the paper: LLaVA-1.5, LLaVA-NeXT
(LLaVA-1.6), Qwen2.5-VL, and Video-LLaVA. It does not contain training or
serving code, historical experiments, model weights, third-party data, complete
predictions, benchmark uploads, or private machine configuration.

This README intentionally contains no submission-author identity and no link to
an author-controlled web supplement. Project attribution and licensing are
recorded locally in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Chinese
implementation detail follows the English quick start.

## English Quick Start

### Scope and evidence

The artifact is designed to reproduce the documented inference/evaluation
protocol, not to redistribute third-party assets. Unit tests, synthetic
fixtures, dry-runs, preflight checks, and a bounded GPU smoke establish
different things; none alone proves a complete 7B benchmark or a paper row.
Read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before interpreting a result.

### Install exactly one environment

The LLaVA family (LLaVA-1.5, LLaVA-NeXT, Video-LLaVA) and Qwen must remain in
separate environments. From a fresh extraction, use the source-root-aware
bootstrap script:

```bash
bash scripts/bootstrap_environment.sh --family llava-family
conda activate trevs-llava-family

# In a separate environment, never alongside the LLaVA-family extra:
bash scripts/bootstrap_environment.sh --family qwen
conda activate trevs-qwen
```

`environment/llava_family.yml` and `environment/qwen.yml` pin the intended
Python, PyTorch, TorchVision, CUDA runtime, and core Python-stack versions, but
they are not a portable binary lock. Driver compatibility, the system CUDA
toolkit, compiler, GPU architecture, and the local Hugging Face cache remain
host inputs and must be recorded in `environment.json`. The LLaVA-family
bootstrap builds `flash-attn==2.3.3` with `--no-build-isolation`; it therefore
needs `nvcc`, a compatible CUDA toolkit, a C++ compiler, and `ninja`. The
bootstrap checks that `CUDA_HOME/bin/nvcc` (when `CUDA_HOME` is set), or the
default `nvcc`, has the same major.minor version as the recipe's
`pytorch-cuda=12.1`. Set `CUDA_HOME` to a matching CUDA 12.1 toolkit before
running it; a host-default CUDA 13 compiler is intentionally rejected. The
`--skip-flash-attn` switch is only a build fallback, not evidence that a
paper-matched LLaVA runtime was installed.

### Obtain inputs outside the ZIP

Obtain every checkpoint and benchmark release directly from its legitimate
third-party distributor and follow its terms. Keep `MODEL_PATH`, `DATA_ROOT`,
and `OUTPUT_ROOT` outside this source tree. No real representative benchmark
subset is distributed: the artifact does not claim a rights audit sufficient to
redistribute benchmark annotations, images, or videos. The only fixtures are
synthetic schema/evaluator fixtures and they cannot replace a benchmark.

LLaVA-1.5 and LLaVA-NeXT require a second local model dependency in addition to
the LLaVA checkpoint. The active checkpoint `config.json` declares
`mm_vision_tower` (or `vision_tower`), and the code loads that exact CLIP tower
with `local_files_only=True`. A commonly used LLaVA value is
`openai/clip-vit-large-patch14-336`, but the checkpoint configuration is
authoritative. Before preflight, ensure that the declared tower's configuration,
image processor, and weights resolve from the active local Hugging Face cache
or a complete local path; no download is attempted during inference.

### Validate before a full run

```bash
# Choose the matching activated environment.
bash scripts/run_tests.sh

bash scripts/reproduce.sh --family llava15 --preset 64 --model-path /path/to/llava-v1.5-7b --data-root /path/to/eval_data --output-root /path/to/outputs --datasets gqa,textvqa,mme,mmbench --gpus 0 --seed 42 --dry-run

# This checks that the config-declared offline vision tower can be resolved,
# together with the input schema and GPU contract, without predictions.
bash scripts/reproduce.sh --family llava15 --preset 64 --model-path /path/to/llava-v1.5-7b --data-root /path/to/eval_data --output-root /path/to/outputs --datasets gqa,textvqa,mme,mmbench --gpus 0 --seed 42 --preflight
```

Some tests intentionally skip optional checkpoint/video coverage when the
corresponding real checkpoint or media-root environment variable is unset. A
skip is neither a failure nor GPU evidence. A real GPU smoke, full data run,
and evaluator comparison must be reported separately.

### Final OpenReview boundary

Before upload, create a temporary UTF-8 deny-list outside the repository. Put
one literal per line for every author name variant, affiliation, laboratory,
personal domain, and private-host fragment. Never place that file in the ZIP.

```bash
bash scripts/build_anonymous_zip.sh --identity-patterns-file /tmp/trevs-aaai-identity-patterns.txt
```

The builder applies this external list to the source tree, staged allowlist
tree, and fresh extraction. Upload only the resulting ZIP through OpenReview's
`Supplementary Code and Data Package (ZIP)` field. Upload the Reproducibility
Checklist through its separate designated field. Re-check the live OpenReview
field size limit immediately before submission, keep the paper self-contained,
and do not replace the ZIP with an anonymous repository, model page, or other
web supplement.

## 1. 代码包内容

| 模型家族 | `--family` | 支持的 preset | 环境 | 主要代码 |
|---|---|---|---|---|
| LLaVA-1.5-7B | `llava15` | `32`, `64`, `128` | `trevs-llava-family` | `llava/` |
| LLaVA-NeXT / LLaVA-1.6 Vicuna-7B | `llava_next` | `160`, `320`, `640` | `trevs-llava-family` | `llava/` |
| Qwen2.5-VL-7B-Instruct | `qwen25vl` | `142`, `284`, `426`, `dense` | `trevs-qwen` | `qwen/`, `qwen_vl_utils.py` |
| Video-LLaVA-7B | `videollava` | `136`, `960`, `dense`, `custom` | `trevs-llava-family` | `videollava/` |

LLaVA-1.5 与 LLaVA-NeXT 共用同一个 `llava` Python 包，但使用不同的图像组织和
token 预算。Video-LLaVA 是独立包，不在运行时导入 `llava` 或 `qwen`。

代码包不包含以下内容：

- 任何 LLaVA、Vicuna、Qwen、CLIP、LanguageBind 或 Video-LLaVA checkpoint；
- GQA、TextVQA、MME、MMBench、ScienceQA、POPE、VQAv2 或 VideoQA 数据；
- 完整预测、评分上传文件、实验日志、论文源文件或 Reproducibility Checklist；
- API 密钥、代理配置或机器相关的绝对路径。

## 2. 系统要求

- Linux x86-64；
- NVIDIA GPU、驱动和 CUDA 12.1 兼容运行时；
- Conda/Mamba；
- Git（Qwen 环境按固定提交安装 Transformers）；
- 足够存放第三方 checkpoint、数据和输出的仓库外磁盘空间。

正式评测使用 batch size 1。显存、初始化时间和总运行时间随 GPU、数据集、
attention backend 和本地存储速度变化，不能把 CPU/unit 测试或单样本 smoke
当作完整 7B benchmark 结果。运行前请先执行 preflight，并在输出的
`run_config.json` 中保存实际 GPU、CUDA、依赖版本与数据摘要。

## 3. 安装：两个互斥环境

两套环境不能合并。LLaVA/Video-LLaVA 使用 Transformers 4.37.2；Qwen 的
runtime patch 依赖固定的 Transformers 5.x 开发提交。

### 3.1 LLaVA-1.5、LLaVA-NeXT 与 Video-LLaVA

```bash
bash scripts/bootstrap_environment.sh --family llava-family
conda activate trevs-llava-family
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

预期核心版本为 Python 3.10.19、PyTorch 2.1.2、TorchVision 0.16.2、
Transformers 4.37.2、NumPy 1.26.4。该环境还固定 `flash-attn==2.3.3`；
编译它需要可用的 CUDA toolkit、C++ 编译器和 `ninja`。Video-LLaVA 的正式默认
backend 是 SDPA；LLaVA 使用哪个 backend 必须以 `run_config.json` 的记录为准。
bootstrap 会在创建环境前检查 `CUDA_HOME/bin/nvcc`（未设置时使用默认 `nvcc`）的
major.minor 是否严格等于 recipe 的 `pytorch-cuda=12.1`。请将 `CUDA_HOME` 指向
匹配的 CUDA 12.1 toolkit；默认 CUDA 13 编译器会被明确拒绝，不能用于该 FlashAttention
构建。

### 3.2 Qwen2.5-VL

```bash
bash scripts/bootstrap_environment.sh --family qwen
conda activate trevs-qwen
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

预期核心版本为 Python 3.10.19、PyTorch 2.5.1、TorchVision 0.20.1、
Transformers 提交 `a25b8efa0a3220da89493a72f57081aa5720291f`、NumPy 2.2.6。
Qwen TReVS 必须使用 SDPA；preflight 会拒绝 FlashAttention2，因为该路径使用
4D additive mask 和不同层长度的 KV cache。

## 4. 第三方模型与数据

建议把模型和数据放在仓库之外，例如：

```text
/path/to/models/
├── llava-v1.5-7b/
├── llava-v1.6-vicuna-7b/
├── Qwen2.5-VL-7B-Instruct/
└── Video-LLaVA-7B/

/path/to/eval_data/
├── gqa/
├── textvqa/
├── mme/
├── mmbench/        # contains both EN and CN TSV files
├── scienceqa/
├── pope/
├── coco/           # COCO val2014 images used by POPE
├── vqav2/
└── videollava/
```

公开 checkpoint 标识、数据目录约定和本地完整性记录方法见
[`data/README.md`](data/README.md)。下载或使用前必须阅读各上游模型卡和数据条款，
见 [`data/DATA_LICENSES.md`](data/DATA_LICENSES.md)。本代码包不替用户授予第三方
权重、图像、视频或标注的使用权。

对于 LLaVA-1.5 与 LLaVA-NeXT，`MODEL_PATH/config.json` 声明的
`mm_vision_tower`（或 `vision_tower`）是第二个必需模型依赖。代码使用
`local_files_only=True` 加载该准确的 CLIP tower；通常可见的
`openai/clip-vit-large-patch14-336` 只是示例，不能覆盖 checkpoint 配置。请在
运行 preflight 前，使 config、image processor 和权重从当前环境的本地 Hugging
Face cache 或完整本地路径可用。完整说明见 [`data/README.md`](data/README.md)。

## 5. 统一运行入口

所有命令从仓库根目录执行。先用 `--dry-run` 检查参数展开；它不创建结果目录，
也不加载 checkpoint：

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

再执行 preflight。它检查路径、数据 schema、依赖、GPU/backend、样本计数和可读取
文件的 SHA-256，不运行完整推理：

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

移除 `--dry-run`/`--preflight` 后运行正式评测：

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

`--model-path` 与 `--data-root` 必填；`--output-root` 默认是仓库根目录下的
`outputs/`。`--datasets` 是逗号分隔的白名单，`--gpus` 是未被预先屏蔽的父进程中的
物理 NVIDIA device ID；不要先设置一个会改变可见编号的 `CUDA_VISIBLE_DEVICES`。
preflight 会将每个请求 ID 放进独立子进程并作为其中的 `cuda:0` 验证。统一入口只接受
上表中的 family/preset，拒绝历史方法别名和与 preset 冲突的底层预算覆盖。

### 5.1 四个家族示例

```bash
# LLaVA-1.5, paper preset 128
bash scripts/reproduce.sh --family llava15 --preset 128 \
  --model-path /path/to/models/llava-v1.5-7b \
  --data-root /path/to/eval_data --output-root /path/to/outputs \
  --datasets gqa,sqa,textvqa,pope,mme,mmbench,mmbench_cn,vqav2 \
  --gpus 0 --seed 42

# LLaVA-NeXT, code-supported preset 320
bash scripts/reproduce.sh --family llava_next --preset 320 \
  --model-path /path/to/models/llava-v1.6-vicuna-7b \
  --data-root /path/to/eval_data --output-root /path/to/outputs \
  --datasets gqa,sqa,textvqa,mme,mmbench \
  --gpus 0 --seed 42

# Qwen2.5-VL, paper preset 284; activate trevs-qwen first
bash scripts/reproduce.sh --family qwen25vl --preset 284 \
  --model-path /path/to/models/Qwen2.5-VL-7B-Instruct \
  --data-root /path/to/eval_data --output-root /path/to/outputs \
  --datasets gqa,textvqa,mme,mmbench \
  --gpus 0 --seed 42

# Video-LLaVA, target preset 136 (actual weighted average 135.5)
bash scripts/reproduce.sh --family videollava --preset 136 \
  --model-path /path/to/models/Video-LLaVA-7B \
  --data-root /path/to/eval_data/videollava \
  --output-root /path/to/outputs \
  --datasets tgif,msvd,msrvtt --gpus 0 --seed 42
```

Qwen 的 `dense` 是同一输入协议下的原生 Hugging Face accuracy baseline；它不是
自动成立的速度 baseline。Video-LLaVA 的 `custom` 只用于诊断，必须在结果中记录
全部底层预算，不能作为未声明的新论文 preset。

## 6. Preset 与输入协议

| Family | Preset | 原始视觉输入协议 | 关键约束 |
|---|---:|---|---|
| `llava15` | 32/64/128 | 576 个图像 patch token | 第 8 层转阶段；名称是层加权平均的四舍五入显示 |
| `llava_next` | 160/320/640 | 固定 672x672、1 个 base crop + 4 个 local crop | 稀疏路径不做 `spatial_unpad`，不添加 `image_newline` |
| `qwen25vl` | 142/284/426 | 固定 resize 1120x896；1280 个 merged visual token | 第 8 层转阶段；SDPA；M-RoPE 和分层 cache |
| `videollava` | 136/960 | 8 帧，每帧 256 个 patch | 第 8 层转阶段；priority heads；sink off |

Video-LLaVA `136` 的每帧 Stage-1 配置为 top-k 26 + FPS 8，合计 Stage 1
272；Stage 2 为 90；`(8*272 + 24*90)/32 = 135.5`。因此 136 是论文中的
目标/显示 preset，而不是把实际平均数伪写成 136。完整 provenance 见
[`videollava/PROVENANCE.md`](videollava/PROVENANCE.md)。

## 7. 输出与评价器

正式运行只写入：

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

`run_config.json` 应包含解析后的 preset、两阶段 token 数、转阶段层、temperature、
priority-head/sink 配置、随机种子、输入尺寸、数据 split/hash 和命令行。
`environment.json` 应包含 Python、PyTorch、Transformers、CUDA、GPU 与 attention
backend。预测和指标不写回 `DATA_ROOT`，密钥也不会写入任何配置。

评价状态分为：

| 数据集 | 状态 | 指标来源 |
|---|---|---|
| GQA | `submission_only` / `external_official` | 本包只生成官方 evaluator 输入，不在本地宣称 accuracy |
| ScienceQA | local evaluator | `IMG-Accuracy`，不是全题 `Accuracy` |
| TextVQA | local M4C evaluator | val accuracy |
| POPE | local evaluator | 三个子集 F1 的平均值 |
| MME | local official calculation | `Overall MME Total` |
| MMBench / MMBench-CN | `submission_only` / `external_official` | 本包只生成 XLSX；正式 `Overall` 来自外部官方评测 |
| VQAv2 | `submission_only` | 生成官方上传 JSON，不在本地宣称 accuracy |
| TGIF/MSVD/MSR-VTT | optional external text judge | 同一 judge、prompt 和版本下的准确率/均分 |

历史 `all_datasets_summary.json` 不是论文指标真值。ScienceQA、TextVQA、POPE
和 MME 应读取各自 evaluator 的结构化输出。GQA 与 MMBench 的公开 runner 只生成
提交文件；获得外部官方结果后才能与期望值比较。不能把 converter 打印的字符串
相等率当作官方 `Overall`。

## 8. 论文数值与验证边界

[`expected_results/paper_metrics.csv`](expected_results/paper_metrics.csv) 只登记已经
核验的 LLaVA-1.5 与 Qwen2.5-VL 主表 TReVS 数值。LLaVA-NeXT 和 Video-LLaVA
虽然保留实现、preset 和测试，但当前 CSV 不虚构尚未锁定的论文期望值。

CSV 中 `tolerance=0.1` 对应论文一位小数的显示精度，不表示不同 evaluator、split
或输入协议之间可以容忍 0.1 的差异。只有 checkpoint、输入、split、预处理、
evaluator 和 seed 全部一致时，才能执行数值比较。

更完整的复现顺序、相对准确率公式、测试层级和失败处理见
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 9. 测试

```bash
bash scripts/run_tests.sh
```

测试应覆盖 LLaVA、Qwen、Video-LLaVA 单元测试、preset 预算、评价器 fixture、
launcher dry-run 与 shell syntax。通过单元测试只证明代码契约，不证明真实 7B
GPU 推理、完整数据覆盖、速度、显存或论文 accuracy。

构建匿名 ZIP 前运行：

```bash
# `/tmp/trevs-aaai-identity-patterns.txt` must remain outside this repository.
bash scripts/build_anonymous_zip.sh \
  --identity-patterns-file /tmp/trevs-aaai-identity-patterns.txt
```

构建脚本使用显式 allowlist，执行匿名信息扫描、符号链接/路径穿越检查、文件大小
检查、SHA-256 manifest、全新目录解压和测试。该外部 pattern 文件每行填写一个
投稿作者姓名变体、单位/实验室、个人域名或内部主机片段，且绝不能放入 ZIP。代码 ZIP
只能通过 OpenReview 的 `Supplementary Code and Data Package (ZIP)` 字段上传；
Reproducibility Checklist 与论文 LaTeX 源文件必须通过相应的独立字段提交，不放进
代码 ZIP。上传前再次检查 OpenReview 实际字段大小限制，且主论文必须自洽，不能把
匿名仓库、模型页或其他网页当作补充材料替代品。

## 10. License

仓库代码按根目录 [`LICENSE`](LICENSE) 中的 Apache License 2.0 分发。第三方代码、
模型和数据仍受各自许可与使用条款约束；参见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 和
[`data/DATA_LICENSES.md`](data/DATA_LICENSES.md)。
