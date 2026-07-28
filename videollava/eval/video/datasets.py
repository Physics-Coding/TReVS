from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(frozen=True)
class VideoQADatasetSpec:
    key: str
    directory: str
    video_directory: str


DATASETS: Dict[str, VideoQADatasetSpec] = {
    "tgif": VideoQADatasetSpec("tgif", "TGIF_Zero_Shot_QA", "mp4"),
    "msvd": VideoQADatasetSpec("msvd", "MSVD_Zero_Shot_QA", "videos"),
    "msrvtt": VideoQADatasetSpec("msrvtt", "MSRVTT_Zero_Shot_QA", "videos/all"),
}


def get_dataset_paths(data_root: Path, dataset: str) -> Tuple[Path, Path, Path]:
    try:
        spec = DATASETS[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {sorted(DATASETS)}") from exc
    direct_root = data_root / spec.directory
    packaged_root = data_root / "GPT_Zero_Shot_QA" / spec.directory
    if direct_root.is_dir():
        dataset_root = direct_root
    elif packaged_root.is_dir():
        dataset_root = packaged_root
    else:
        # Keep the user-facing default rooted at ``eval`` useful even when a
        # partially copied package has not created the dataset directory yet.
        dataset_root = packaged_root if (data_root / "GPT_Zero_Shot_QA").is_dir() else direct_root
    return (
        dataset_root / spec.video_directory,
        dataset_root / "test_q.json",
        dataset_root / "test_a.json",
    )


def chunk_sequence(items: Sequence, num_chunks: int, chunk_idx: int) -> List:
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if not 0 <= chunk_idx < num_chunks:
        raise ValueError(f"chunk_idx must be in [0, {num_chunks}), got {chunk_idx}")
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, len(items))
    return list(items[start:end])


def resolve_video_path(video_dir: Path, video_name: str) -> Path:
    direct_path = video_dir / video_name
    if direct_path.is_file():
        return direct_path
    if direct_path.suffix:
        raise FileNotFoundError(f"Video not found: {direct_path}")
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No video found for {video_name!r} under {video_dir}; tried {VIDEO_EXTENSIONS}"
    )


def iter_jsonl(path: Path) -> Iterable[dict]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield value
