#!/usr/bin/env python3
"""Build and independently revalidate the anonymous AAAI code ZIP."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from scan_anonymity import ScanConfigurationError, load_identity_patterns


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = REPO_ROOT / "packaging" / "allowlist.txt"
PACKAGE_NAME = "trevs-aaai27-anonymous"
DEFAULT_MAX_ZIP_BYTES = 10 * 1024 * 1024
ATTESTATION_SCHEMA_VERSION = 1


class PackageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_allowlist(path: Path) -> list[tuple[str, bool]]:
    rules: list[tuple[str, bool]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        optional = value.startswith("?")
        if optional:
            value = value[1:]
        pure = PurePosixPath(value)
        if not value or pure.is_absolute() or ".." in pure.parts or "\\" in value:
            raise PackageError(f"Unsafe allowlist rule at {path}:{line_number}: {raw!r}")
        rules.append((value, optional))
    if not rules:
        raise PackageError(f"Allowlist is empty: {path}")
    return rules


def _has_symlink_parent(path: Path, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return root.is_symlink()


def materialize_allowlist(root: Path, allowlist: Path) -> list[Path]:
    root = root.resolve()
    selected: dict[str, Path] = {}
    for pattern, optional in parse_allowlist(allowlist):
        absolute_pattern = str(root / pattern)
        matches = [Path(value) for value in glob.glob(absolute_pattern, recursive=True)]
        matches = [path for path in matches if path.is_file() or path.is_symlink()]
        if not matches and not optional:
            raise PackageError(f"Allowlist rule matched no files: {pattern}")
        for path in matches:
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise PackageError(f"Allowlist escaped repository root: {path}") from exc
            if _has_symlink_parent(path, root):
                raise PackageError(f"Allowlisted symbolic link is forbidden: {relative}")
            if not path.is_file():
                raise PackageError(f"Allowlisted path is not a regular file: {relative}")
            rel_text = relative.as_posix()
            selected[rel_text] = path
    return [selected[key] for key in sorted(selected)]


def copy_selected(root: Path, destination: Path, selected: Iterable[Path]) -> None:
    root = root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for source in selected:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        executable = source.suffix == ".sh" or source.read_bytes()[:2] == b"#!"
        target.chmod(0o755 if executable else 0o644)


def manifest_lines(root: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    return lines


def write_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    manifest.write_text("\n".join(manifest_lines(root)) + "\n", encoding="utf-8")
    manifest.chmod(0o644)
    return manifest


def verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file() or manifest.is_symlink():
        raise PackageError("MANIFEST.sha256 is absent or unsafe.")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise PackageError(f"Malformed manifest line {line_number}.") from exc
        pure = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in expected
            or relative == "MANIFEST.sha256"
        ):
            raise PackageError(f"Unsafe manifest line {line_number}: {line!r}")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise PackageError(f"Manifest membership mismatch: missing={missing}, extra={extra}")
    mismatches = [name for name in expected if expected[name] != actual[name]]
    if mismatches:
        raise PackageError("Manifest checksum mismatch: " + ", ".join(mismatches))


def validation_attestation_path(stage: Path) -> Path:
    return stage.parent / f"{stage.name}.validated.json"


def write_validation_attestation(stage: Path) -> Path:
    verify_manifest(stage)
    manifest = stage / "MANIFEST.sha256"
    payload = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "validated_stage": str(stage.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "files": len(manifest.read_text(encoding="utf-8").splitlines()),
    }
    target = validation_attestation_path(stage)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.is_file():
            temporary.unlink()
    return target


def verify_validation_attestation(stage: Path) -> None:
    target = validation_attestation_path(stage)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"Validated-stage attestation is absent or malformed: {target}") from exc
    manifest = stage / "MANIFEST.sha256"
    expected = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "validated_stage": str(stage.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "files": len(manifest.read_text(encoding="utf-8").splitlines()),
    }
    if payload != expected:
        raise PackageError("Validated-stage attestation does not match the stage manifest.")


def run_checked(command: Sequence[str], cwd: Path, env: Mapping[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode:
        raise PackageError(f"Validation command failed ({result.returncode}): {' '.join(command)}")


def validate_shell_scripts(root: Path) -> None:
    for path in sorted(root.rglob("*.sh")):
        run_checked(["bash", "-n", str(path)], root)


def validate_dry_runs(root: Path, python: str) -> None:
    presets = {
        "llava15": ("32", "64", "128"),
        "llava_next": ("160", "320", "640"),
        "qwen25vl": ("142", "284", "426", "dense"),
        "videollava": ("136", "960", "dense"),
    }
    with tempfile.TemporaryDirectory(prefix="trevs-dry-run.") as temp_dir:
        output_root = Path(temp_dir) / "outputs"
        for family, family_presets in presets.items():
            for preset in family_presets:
                run_checked(
                    [
                        python,
                        "scripts/reproduce.py",
                        "--family",
                        family,
                        "--preset",
                        preset,
                        "--model-path",
                        "/tmp/trevs-model-placeholder",
                        "--data-root",
                        "/tmp/trevs-data-placeholder",
                        "--output-root",
                        str(output_root),
                        "--datasets",
                        root_dataset(family),
                        "--dry-run",
                    ],
                    root,
                )
        run_checked(
            [
                python,
                "scripts/reproduce.py",
                "--family",
                "videollava",
                "--preset",
                "custom",
                "--model-path",
                "/tmp/trevs-model-placeholder",
                "--data-root",
                "/tmp/trevs-data-placeholder",
                "--output-root",
                str(output_root),
                "--datasets",
                "tgif",
                "--stage1-topk",
                "26",
                "--stage1-fps",
                "8",
                "--stage2-keep",
                "90",
                "--dry-run",
            ],
            root,
        )
        if output_root.exists():
            raise PackageError("A dry-run created the output directory.")


def root_dataset(family: str) -> str:
    return "tgif" if family == "videollava" else "gqa"


def conda_python_command(environment_name: str, *arguments: str) -> list[str]:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise PackageError("conda is required for the locked Python 3.10 installation check.")
    return [conda, "run", "-n", environment_name, "python", *arguments]


def validate_wheel(
    root: Path,
    llava_family_environment: str,
    qwen_environment: str,
    env: Mapping[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="trevs-wheel-source.") as source_dir, tempfile.TemporaryDirectory(
        prefix="trevs-wheel-output."
    ) as wheel_dir, tempfile.TemporaryDirectory(prefix="trevs-wheel-install-llava.") as llava_install_dir, tempfile.TemporaryDirectory(
        prefix="trevs-wheel-install-qwen."
    ) as qwen_install_dir:
        source = Path(source_dir) / PACKAGE_NAME
        shutil.copytree(root, source)
        run_checked(
            conda_python_command(
                llava_family_environment,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                wheel_dir,
                ".",
            ),
            source,
            env,
        )
        wheels = list(Path(wheel_dir).glob("*.whl"))
        if len(wheels) != 1:
            raise PackageError(f"Expected one wheel from package check; found {len(wheels)}.")
        wheel = wheels[0]
        required_members = {
            "evaluation/aggregate_metrics.py",
            "llava/__init__.py",
            "qwen/eval/attention_backend.py",
            "qwen/model/trevs_router.py",
            "qwen_vl_utils.py",
            "videollava/__init__.py",
        }
        with zipfile.ZipFile(wheel, "r") as archive:
            members = set(archive.namelist())
        missing = sorted(required_members.difference(members))
        if missing:
            raise PackageError("Wheel is missing required runtime members: " + ", ".join(missing))

        install_environment = dict(env)
        install_environment.pop("PYTHONPATH", None)
        import_code = """
import importlib
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
for name in sys.argv[2:]:
    module = importlib.import_module(name)
    locations = []
    if getattr(module, "__file__", None):
        locations.append(Path(module.__file__).resolve())
    if getattr(module, "__path__", None):
        locations.extend(Path(value).resolve() for value in module.__path__)
    if not locations or any(target not in location.parents and location != target for location in locations):
        raise RuntimeError(f"{name} was not imported from the installed wheel target: {locations}")
""".strip()
        environments_and_imports = (
            (
                llava_family_environment,
                llava_install_dir,
                ("llava", "videollava", "evaluation", "qwen_vl_utils"),
            ),
            (
                qwen_environment,
                qwen_install_dir,
                ("qwen.eval.attention_backend", "qwen.model.trevs_router", "qwen_vl_utils"),
            ),
        )
        for environment_name, install_dir, imports in environments_and_imports:
            run_checked(
                conda_python_command(
                    environment_name,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    install_dir,
                    str(wheel),
                ),
                Path(install_dir),
                install_environment,
            )
            run_checked(
                conda_python_command(
                    environment_name,
                    "-I",
                    "-c",
                    import_code,
                    install_dir,
                    *imports,
                ),
                Path(install_dir),
                install_environment,
            )


def validation_environment(args: argparse.Namespace, root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root),
            "LLAVA_FAMILY_ENV": args.llava_family_env,
            # Preserve the legacy name only for third-party-compatible shell code.
            "LLAVA_VIDEO_ENV": args.llava_family_env,
            "QWEN_ENV": args.qwen_env,
            "PURE_PYTHON": args.python,
        }
    )
    return environment


def anonymity_scan_command(args: argparse.Namespace, root: Path) -> list[str]:
    command = [args.python, "scripts/scan_anonymity.py", str(root)]
    if args.identity_patterns_file:
        command.extend(["--identity-patterns-file", str(args.identity_patterns_file)])
    return command


def validate_source_tree(root: Path, args: argparse.Namespace) -> None:
    environment = validation_environment(args, root)
    run_checked(anonymity_scan_command(args, root), root, environment)


def validate_tree(root: Path, args: argparse.Namespace) -> None:
    verify_manifest(root)
    environment = validation_environment(args, root)
    run_checked(anonymity_scan_command(args, root), root, environment)
    validate_shell_scripts(root)
    validate_dry_runs(root, args.python)
    validate_wheel(root, args.llava_family_env, args.qwen_env, environment)
    if not args.skip_tests:
        run_checked(["bash", "scripts/run_tests.sh"], root, environment)
    verify_manifest(root)
    run_checked(anonymity_scan_command(args, root), root, environment)


def zip_tree(root: Path, destination: Path, package_name: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                name = f"{package_name}/{relative}"
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                executable = bool(path.stat().st_mode & stat.S_IXUSR)
                info.external_attr = (0o100755 if executable else 0o100644) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        temporary.replace(destination)
    finally:
        if temporary.is_file():
            temporary.unlink()


def validate_archive(destination: Path, package_name: str) -> None:
    prefix = PurePosixPath(package_name)
    with zipfile.ZipFile(destination, "r") as archive:
        seen: set[str] = set()
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if (
                "\\" in info.filename
                or pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
                or pure.parts[0] != prefix.name
                or info.filename in seen
                or (len(info.filename) >= 2 and info.filename[1] == ":")
            ):
                raise PackageError(f"Unsafe ZIP member: {info.filename!r}")
            seen.add(info.filename)
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) != stat.S_IFREG:
                raise PackageError(f"ZIP contains a non-regular member: {info.filename}")
            if mode not in {0o100644, 0o100755} or info.create_system != 3:
                raise PackageError(f"ZIP contains unexpected member attributes: {info.filename}")
        if not seen:
            raise PackageError("ZIP archive is empty.")
        bad = archive.testzip()
        if bad:
            raise PackageError(f"ZIP CRC validation failed: {bad}")


def write_source_manifest(root: Path, allowlist: Path) -> None:
    selected = materialize_allowlist(root, allowlist)
    selected_without_manifest = [path for path in selected if path.name != "MANIFEST.sha256"]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in selected_without_manifest
    ]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_tree(root: Path, allowlist: Path, stage: Path) -> None:
    selected = materialize_allowlist(root, allowlist)
    copy_selected(root, stage, selected)
    write_manifest(stage)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    result.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    result.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "TReVS_AAAI27_Anonymous.zip",
    )
    result.add_argument("--stage-dir", type=Path, help="Keep a validated allowlist tree here")
    result.add_argument("--stage-only", action="store_true", help="Validate a tree without creating a ZIP")
    result.add_argument("--write-source-manifest", action="store_true")
    result.add_argument(
        "--identity-patterns-file",
        type=Path,
        help=(
            "External UTF-8 author/affiliation deny list. It is applied to source, stage, and "
            "extracted-tree scans and must not be inside the repository."
        ),
    )
    result.add_argument("--skip-tests", action="store_true", help="Skip only environment-dependent test suites")
    result.add_argument(
        "--llava-family-env",
        default=os.environ.get(
            "LLAVA_FAMILY_ENV",
            os.environ.get("LLAVA_VIDEO_ENV", "trevs-llava-family"),
        ),
    )
    result.add_argument(
        "--llava-video-env",
        dest="llava_family_env",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    result.add_argument("--qwen-env", default=os.environ.get("QWEN_ENV", "trevs-qwen"))
    result.add_argument("--python", default=os.environ.get("PURE_PYTHON", sys.executable))
    result.add_argument("--package-name", default=PACKAGE_NAME)
    result.add_argument("--max-zip-bytes", type=int, default=DEFAULT_MAX_ZIP_BYTES)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.repo_root.resolve()
    allowlist = args.allowlist.resolve()
    try:
        if not root.is_dir() or not allowlist.is_file():
            raise PackageError("Repository root and allowlist must exist.")
        if args.identity_patterns_file:
            try:
                args.identity_patterns_file = args.identity_patterns_file.resolve(strict=True)
                load_identity_patterns(args.identity_patterns_file, root)
            except ScanConfigurationError as exc:
                raise PackageError(str(exc)) from exc
        if args.max_zip_bytes <= 0:
            raise PackageError("--max-zip-bytes must be positive.")
        destination = None
        requested_stage = args.stage_dir.resolve() if args.stage_dir else None
        if not args.stage_only:
            destination = args.output.resolve()
            if destination == root or is_relative_to(destination, root):
                raise PackageError("ZIP output must be outside the repository tree.")
            if requested_stage is not None and (
                destination == requested_stage
                or is_relative_to(destination, requested_stage)
                or destination == validation_attestation_path(requested_stage)
            ):
                raise PackageError("ZIP output must be separate from the validated stage and its attestation.")
        validate_source_tree(root, args)
        if args.write_source_manifest:
            write_source_manifest(root, allowlist)
        if args.stage_dir:
            stage = requested_stage
            assert stage is not None
            if stage.exists():
                raise PackageError(f"Stage directory already exists: {stage}")
            if is_relative_to(stage, root):
                raise PackageError("Persistent stage directory must be outside the repository.")
            attestation = validation_attestation_path(stage)
            if attestation.exists():
                raise PackageError(f"Stage attestation already exists: {attestation}")
            candidate_stage = stage.with_name(f".{stage.name}.candidate.{os.getpid()}")
            if candidate_stage.exists():
                raise PackageError(f"Candidate stage already exists: {candidate_stage}")
            published_stage = False
            try:
                stage_tree(root, allowlist, candidate_stage)
                validate_tree(candidate_stage, args)
                candidate_stage.replace(stage)
                published_stage = True
                attestation = write_validation_attestation(stage)
            except Exception:
                if candidate_stage.is_dir():
                    shutil.rmtree(candidate_stage)
                if published_stage and stage.is_dir():
                    shutil.rmtree(stage)
                if published_stage and attestation.is_file():
                    attestation.unlink()
                raise
            print(f"Validated allowlist tree: {stage}")
            print(f"Validation attestation: {attestation}")
            if args.stage_only:
                return 0
        else:
            if args.stage_only:
                raise PackageError("--stage-only requires --stage-dir.")
            temporary = tempfile.TemporaryDirectory(prefix="trevs-package-stage.")
            stage = Path(temporary.name) / args.package_name
            stage_tree(root, allowlist, stage)
            validate_tree(stage, args)
            write_validation_attestation(stage)
        if args.stage_only:
            return 0
        assert destination is not None
        candidate = destination.with_name(f".{destination.name}.candidate.{os.getpid()}")
        if candidate.exists():
            candidate.unlink()
        zip_tree(stage, candidate, args.package_name)
        validate_archive(candidate, args.package_name)
        if candidate.stat().st_size > args.max_zip_bytes:
            raise PackageError(
                f"ZIP exceeds target: {candidate.stat().st_size} > {args.max_zip_bytes} bytes"
            )
        with tempfile.TemporaryDirectory(prefix="trevs-package-extract.") as extracted_dir:
            with zipfile.ZipFile(candidate, "r") as archive:
                archive.extractall(extracted_dir)
            extracted = Path(extracted_dir) / args.package_name
            validate_tree(extracted, args)
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(destination)
        print(f"Anonymous ZIP: {destination}")
        print(f"Size: {destination.stat().st_size} bytes")
        print(f"SHA-256: {sha256_file(destination)}")
        return 0
    except (OSError, PackageError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        candidate_path = locals().get("candidate")
        if isinstance(candidate_path, Path) and candidate_path.is_file():
            candidate_path.unlink()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
