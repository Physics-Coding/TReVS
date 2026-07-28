import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import requests

from .datasets import iter_jsonl


PROMPT_VERSION = "videollava_videoqa_correctness_v1"
SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative "
    "outputs for question-answer pairs. Compare the predicted answer with the correct answer "
    "and determine whether they match meaningfully. Focus on meaningful agreement, consider "
    "synonyms and paraphrases valid, and evaluate the prediction against the correct answer. "
    "Return only a strict JSON object with exactly two keys: "
    "{\"pred\": \"yes\" or \"no\", \"score\": an integer from 0 to 5}."
)


class RateLimiter:
    def __init__(self, requests_per_minute: float):
        self.interval = 0.0 if requests_per_minute <= 0 else 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if delay > 0:
            time.sleep(delay)


def build_user_prompt(record: dict) -> str:
    return (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {record['question']}\n"
        f"Correct Answer: {record['answer']}\n"
        f"Predicted Answer: {record['pred']}\n\n"
        "Provide only a yes/no judgment and a score, where score is an integer from 0 to 5 "
        "and 5 indicates the highest meaningful match. Return strict JSON in this form: "
        "{\"pred\": \"yes\", \"score\": 5}. Do not provide explanation or other text."
    )


def parse_judge_content(content: str) -> Dict[str, object]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge response does not contain a JSON object: {content!r}")
    payload = json.loads(stripped[start : end + 1])
    if set(payload) != {"pred", "score"}:
        raise ValueError(f"Judge response must contain exactly pred and score, got {sorted(payload)}")
    prediction = str(payload["pred"]).strip().lower()
    if prediction not in {"yes", "no"}:
        raise ValueError(f"Invalid judge pred value: {payload['pred']!r}")
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Judge score must be an integer from 0 to 5, got {score!r}")
    if isinstance(score, float) and not score.is_integer():
        raise ValueError(f"Judge score must be integral, got {score!r}")
    score = int(score)
    if not 0 <= score <= 5:
        raise ValueError(f"Judge score must be in [0, 5], got {score}")
    return {"pred": prediction, "score": score}


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def request_judgment(
    record: dict,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
    retry_base_seconds: float,
    rate_limiter: RateLimiter,
    post: Optional[Callable] = None,
) -> dict:
    post = requests.post if post is None else post
    request_body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            rate_limiter.wait()
            response = post(
                _chat_completions_url(base_url),
                headers=headers,
                json=request_body,
                timeout=timeout,
            )
            response.raise_for_status()
            response_payload = response.json()
            content = response_payload["choices"][0]["message"]["content"]
            parsed = parse_judge_content(content)
            return {
                "id": str(record["id"]),
                "status": "ok",
                "pred": parsed["pred"],
                "score": parsed["score"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "raw_response": content,
            }
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_base_seconds * (2**attempt))
    assert last_error is not None
    return {
        "id": str(record["id"]),
        "status": "failed",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "error_type": type(last_error).__name__,
        "error": str(last_error),
    }


def aggregate_judgments(judgments: Iterable[dict], total_predictions: int, model: str) -> dict:
    judgments = list(judgments)
    successful = [item for item in judgments if item.get("status") == "ok"]
    failed = [item for item in judgments if item.get("status") != "ok"]
    yes_count = sum(item["pred"] == "yes" for item in successful)
    no_count = sum(item["pred"] == "no" for item in successful)
    judged_count = yes_count + no_count
    score_sum = sum(int(item["score"]) for item in successful)
    return {
        "accuracy": yes_count / judged_count if judged_count else None,
        "average_score": score_sum / judged_count if judged_count else None,
        "yes_count": yes_count,
        "no_count": no_count,
        "successful_count": len(successful),
        "failed_count": len(failed),
        "total_predictions": total_predictions,
        "complete": len(successful) == total_predictions and not failed,
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
    }


def _load_prediction_records(path: Path) -> List[dict]:
    records = list(iter_jsonl(path))
    required = {"id", "question", "answer", "pred"}
    seen = set()
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Prediction record is missing {sorted(missing)}: {record}")
        question_id = str(record["id"])
        if question_id in seen:
            raise ValueError(f"Duplicate prediction ID: {question_id}")
        seen.add(question_id)
    return records


def _load_existing_judgments(
    path: Path,
    allowed_ids: set[str],
    expected_model: Optional[str] = None,
    expected_prompt_version: Optional[str] = None,
) -> Dict[str, dict]:
    if not path.exists():
        return {}
    existing: Dict[str, dict] = {}
    for record in iter_jsonl(path):
        question_id = str(record.get("id", ""))
        if question_id not in allowed_ids:
            raise ValueError(f"Judgment output contains unknown ID: {question_id}")
        if expected_model is not None and record.get("model") != expected_model:
            raise ValueError(
                f"Judgment output mixes model {record.get('model')!r} with {expected_model!r}."
            )
        if (
            expected_prompt_version is not None
            and record.get("prompt_version") != expected_prompt_version
        ):
            raise ValueError(
                "Judgment output mixes prompt versions: "
                f"{record.get('prompt_version')!r} != {expected_prompt_version!r}."
            )
        # A failed record is intentionally retried by a resumed run. If that
        # run is interrupted after appending the replacement, the JSONL can
        # contain the same ID more than once; the most recent result wins.
        existing[question_id] = record
    return existing


def _write_judgments_atomically(path: Path, records: Iterable[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_predictions(args: argparse.Namespace) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = (args.model or os.getenv("VIDEO_QA_JUDGE_MODEL", "")).strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set when GPT judging is enabled")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL must be set when GPT judging is enabled")
    if not model:
        raise ValueError("VIDEO_QA_JUDGE_MODEL or --model must be set when GPT judging is enabled")

    predictions = _load_prediction_records(args.pred_path)
    prediction_ids = {str(item["id"]) for item in predictions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    judgments_file = args.output_dir / "judgments.jsonl"
    existing = (
        _load_existing_judgments(
            judgments_file,
            prediction_ids,
            expected_model=model,
            expected_prompt_version=PROMPT_VERSION,
        )
        if args.resume
        else {}
    )
    if not args.resume:
        judgments_file.write_text("", encoding="utf-8")

    completed = {
        question_id: item
        for question_id, item in existing.items()
        if item.get("status") == "ok"
    }
    pending = [item for item in predictions if str(item["id"]) not in completed]
    rate_limiter = RateLimiter(args.requests_per_minute)

    new_results: List[dict] = []
    with judgments_file.open("a", encoding="utf-8") as output_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            batch_size = max(args.workers * 4, 1)
            for start in range(0, len(pending), batch_size):
                futures = [
                    executor.submit(
                        request_judgment,
                        record,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_base_seconds=args.retry_base_seconds,
                        rate_limiter=rate_limiter,
                    )
                    for record in pending[start : start + batch_size]
                ]
                for future in as_completed(futures):
                    result = future.result()
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    new_results.append(result)

    final_by_id = dict(existing)
    for result in new_results:
        final_by_id[str(result["id"])] = result
    _write_judgments_atomically(
        judgments_file,
        (final_by_id[str(record["id"])] for record in predictions),
    )
    summary = aggregate_judgments(
        final_by_id.values(),
        total_predictions=len(predictions),
        model=model,
    )
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-assisted VideoQA correctness evaluation.")
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=float(os.getenv("VIDEO_QA_JUDGE_RPM", "0")),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    return args


def main() -> None:
    summary = evaluate_predictions(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
