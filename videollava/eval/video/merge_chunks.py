import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from .datasets import iter_jsonl


REQUIRED_FIELDS = ("id", "question", "answer", "pred")


def load_questions(path: Path, max_samples: int = 0) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if max_samples > 0:
        questions = questions[:max_samples]
    return questions


def merge_chunk_files(
    question_file: Path,
    chunk_files: List[Path],
    output_file: Path,
    max_samples: int = 0,
) -> int:
    questions = load_questions(question_file, max_samples=max_samples)
    expected_ids = [str(item["question_id"]) for item in questions]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"Duplicate question_id values in {question_file}")

    records_by_id: Dict[str, dict] = {}
    for chunk_file in chunk_files:
        if not chunk_file.is_file():
            raise FileNotFoundError(f"Missing chunk output: {chunk_file}")
        for record in iter_jsonl(chunk_file):
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"Missing fields {missing} in {chunk_file}: {record}")
            question_id = str(record["id"])
            if question_id in records_by_id:
                raise ValueError(f"Duplicate prediction for question_id={question_id}")
            records_by_id[question_id] = record

    expected_set = set(expected_ids)
    observed_set = set(records_by_id)
    missing_ids = expected_set - observed_set
    unexpected_ids = observed_set - expected_set
    if missing_ids or unexpected_ids:
        raise ValueError(
            "Chunk merge is incomplete: "
            f"missing={len(missing_ids)}, unexpected={len(unexpected_ids)}; "
            f"missing_examples={sorted(missing_ids)[:5]}, "
            f"unexpected_examples={sorted(unexpected_ids)[:5]}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for question_id in expected_ids:
            handle.write(json.dumps(records_by_id[question_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_file)
    return len(expected_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly merge VideoQA chunk JSONL outputs.")
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_files = [args.chunk_dir / f"{args.num_chunks}_{idx}.jsonl" for idx in range(args.num_chunks)]
    count = merge_chunk_files(
        question_file=args.question_file,
        chunk_files=chunk_files,
        output_file=args.output_file,
        max_samples=args.max_samples,
    )
    print(f"Merged {count} predictions into {args.output_file}")


if __name__ == "__main__":
    main()
