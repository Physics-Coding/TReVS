#!/usr/bin/env python3
"""Fail closed on identity, secret, binary, and artifact-package leaks."""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


MAX_SOURCE_BYTES = 2 * 1024 * 1024
BANNED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "answers",
    "answers_upload",
    "artifacts",
    "assests",
    "assets",
    "build",
    "checkpoints",
    "dist",
    "essay",
    "logs",
    "models",
    "outputs",
    "playground",
    "presentation",
    "raw_answers",
    "results",
    "reviews",
    "runs",
    "tmp",
    "venv",
}
BANNED_SUFFIXES = {
    ".7z",
    ".avi",
    ".bib",
    ".bin",
    ".ckpt",
    ".cls",
    ".db",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".log",
    ".mdb",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".onnx",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".pt",
    ".pth",
    ".pyo",
    ".pyc",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tex",
    ".tgz",
    ".tiff",
    ".ipynb",
    ".sty",
    ".wav",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?:/" + r"home/|/" + r"users/|/data\d*/" + r"wang|[a-z]:\\\\" + r"users\\\\)"
)
INTERNAL_NETWORK_PATTERN = re.compile(
    r"(?i)(?:https?://|ssh://)?(?:"
    + r"local"
    + r"host|(?:[a-z0-9-]+\.)+local|127\.0\.0\.1|0\.0\.0\.0"
    + r"|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
    + r"|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?::\d+)?(?=$|[^a-z0-9.-])"
)
TEXT_PATTERNS = (
    ("personal identifier", re.compile(r"(?i)wang[._ -]?jing|jing[._ -]?wang|\u738b\u9759")),
    ("private absolute path", PRIVATE_PATH_PATTERN),
    ("internal network address", INTERNAL_NETWORK_PATTERN),
    ("private email", re.compile(r"(?i)\b[A-Z0-9._%+-]+@(?!example\.(?:com|org)\b)[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    (
        "student identifier",
        re.compile(
            r"(?i)(?:student[ _-]?(?:id|number)|\u5b66\u53f7)\s*[:=_-]?\s*[A-Z]?\d{6,14}\b"
        ),
    ),
    ("OpenAI-style secret", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("Hugging Face secret", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("GitHub secret", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Bearer token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    (
        "literal credential assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\b\s*[:=]\s*[\"'](?!\$|<|your[-_ ]|not-a-real|dummy|example|test[-_])[A-Za-z0-9._~+/=-]{8,}[\"']"
        ),
    ),
)
PATH_PATTERNS = (
    ("personal identifier in path", TEXT_PATTERNS[0][1]),
    (
        "student identifier in path",
        re.compile(r"(?i)(?:student[ _-]?(?:id|number)|\u5b66\u53f7)[ _.-]*[A-Z]?\d{6,14}"),
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str
    line: int | None = None
    excerpt: str | None = None

    def render(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        if self.excerpt:
            return f"{location}: {self.reason}: {self.excerpt}"
        return f"{location}: {self.reason}"


class ScanConfigurationError(ValueError):
    """Raised when an external anonymity-scan input is unsafe or malformed."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_identity_patterns(path: Path, root: Path) -> tuple[str, ...]:
    """Load literal case-insensitive identity patterns from an external UTF-8 file."""

    resolved_root = root.resolve()
    if path.is_symlink():
        raise ScanConfigurationError("Identity patterns file must not be a symbolic link.")
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise ScanConfigurationError(f"Cannot resolve identity patterns file: {path}") from exc
    if _is_relative_to(resolved_path, resolved_root):
        raise ScanConfigurationError(
            "Identity patterns file must remain outside the scanned tree."
        )
    if not resolved_path.is_file():
        raise ScanConfigurationError(
            f"Identity patterns file must be an external regular file: {resolved_path}"
        )
    try:
        text = resolved_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScanConfigurationError(
            f"Identity patterns file must be UTF-8 text: {resolved_path}"
        ) from exc
    if "\x00" in text:
        raise ScanConfigurationError("Identity patterns file contains a NUL byte.")
    patterns = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not patterns:
        raise ScanConfigurationError("Identity patterns file contains no nonempty patterns.")
    return patterns


def compile_identity_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Treat every supplied identity line as a literal, not a regular expression."""

    return tuple(re.compile(re.escape(pattern), re.IGNORECASE) for pattern in patterns)


def relative_files(root: Path) -> Iterable[tuple[Path, Path]]:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(dirnames):
            candidate = directory_path / name
            if candidate.is_symlink():
                yield candidate, candidate.relative_to(root)
                dirnames.remove(name)
        for name in filenames:
            candidate = directory_path / name
            yield candidate, candidate.relative_to(root)


def directory_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for directory, dirnames, _ in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            candidate = directory_path / name
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                continue
            if name.lower() in BANNED_PARTS:
                findings.append(
                    Finding(relative.as_posix(), f"forbidden path component: {name.lower()}")
                )
    return findings


def path_findings(
    path: Path,
    relative: Path,
    max_bytes: int,
    identity_patterns: Sequence[re.Pattern[str]],
) -> list[Finding]:
    findings: list[Finding] = []
    rel = relative.as_posix()
    if path.is_symlink():
        findings.append(Finding(rel, "symbolic links are forbidden"))
        return findings
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        return [Finding(rel, f"cannot stat file: {exc}")]
    if not stat.S_ISREG(mode):
        return [Finding(rel, "non-regular filesystem entry is forbidden")]
    lower_parts = {part.lower() for part in relative.parts}
    banned = sorted(lower_parts.intersection(BANNED_PARTS))
    if banned:
        findings.append(Finding(rel, f"forbidden path component: {banned[0]}"))
    if path.name == ".DS_Store" or path.suffix.lower() in BANNED_SUFFIXES:
        findings.append(Finding(rel, f"forbidden artifact type: {path.suffix or path.name}"))
    for reason, pattern in PATH_PATTERNS:
        if pattern.search(rel):
            findings.append(Finding(rel, reason))
    if any(pattern.search(rel) for pattern in identity_patterns):
        findings.append(Finding(rel, "configured identity pattern in path"))
    size = path.stat().st_size
    if size > max_bytes:
        findings.append(Finding(rel, f"undeclared large file ({size} bytes > {max_bytes})"))
    return findings


def text_findings(
    path: Path,
    relative: Path,
    identity_patterns: Sequence[re.Pattern[str]],
) -> list[Finding]:
    rel = relative.as_posix()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [Finding(rel, f"cannot read file: {exc}")]
    if b"\x00" in raw:
        return [Finding(rel, "binary/NUL content is forbidden")]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [Finding(rel, f"non-UTF-8 content is forbidden ({exc})")]
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for reason, pattern in TEXT_PATTERNS:
            match = pattern.search(line)
            if match:
                excerpt = line.strip()
                if len(excerpt) > 180:
                    excerpt = excerpt[:177] + "..."
                findings.append(Finding(rel, reason, line_number, excerpt))
        if any(pattern.search(line) for pattern in identity_patterns):
            # Do not print configured author/affiliation terms into build logs.
            findings.append(Finding(rel, "configured identity pattern", line_number))
    return findings


def scan(
    root: Path,
    max_bytes: int = MAX_SOURCE_BYTES,
    identity_patterns_file: Path | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    if root.is_symlink() or not root.is_dir():
        return [Finding(str(root), "scan root must be a real directory")]
    identity_patterns = compile_identity_patterns(
        load_identity_patterns(identity_patterns_file, root) if identity_patterns_file else ()
    )
    findings.extend(directory_findings(root))
    for path, relative in relative_files(root):
        file_findings = path_findings(path, relative, max_bytes, identity_patterns)
        findings.extend(file_findings)
        if not file_findings and path.is_file():
            findings.extend(text_findings(path, relative, identity_patterns))
    return sorted(findings, key=lambda item: (item.path, item.line or 0, item.reason))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-bytes", type=int, default=MAX_SOURCE_BYTES)
    parser.add_argument(
        "--identity-patterns-file",
        type=Path,
        help=(
            "External UTF-8 file with one literal author, affiliation, or identity pattern per line. "
            "It must stay outside the scanned tree and is never packaged."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_file_bytes <= 0:
        parser.error("--max-file-bytes must be positive")
    try:
        findings = scan(
            args.root.resolve(),
            args.max_file_bytes,
            args.identity_patterns_file,
        )
    except ScanConfigurationError as exc:
        parser.error(str(exc))
    if findings:
        print(f"Anonymity/package scan failed with {len(findings)} finding(s):", file=sys.stderr)
        for finding in findings:
            print(f"  {finding.render()}", file=sys.stderr)
        return 1
    print(f"Anonymity/package scan passed: {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
