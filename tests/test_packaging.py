from __future__ import annotations

import contextlib
import io
import stat
import tempfile
import unittest
from pathlib import Path
import sys
import zipfile


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_anonymous_zip import (  # noqa: E402
    PackageError,
    anonymity_scan_command,
    is_relative_to,
    materialize_allowlist,
    validate_archive,
    validation_attestation_path,
    verify_validation_attestation,
    verify_manifest,
    write_validation_attestation,
    write_manifest,
)
from scan_anonymity import ScanConfigurationError, main as scan_main, scan  # noqa: E402


class AnonymityScanTests(unittest.TestCase):
    def test_clean_source_tree_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("Anonymous artifact\n", encoding="utf-8")
            self.assertEqual(scan(root), [])

    def test_identity_path_secret_and_checkpoint_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_path = "/" + "home/private-user/checkpoint"
            fake_secret = "sk" + "-" + "ABCDEFGHIJKLMNOPQRST"
            (root / "config.py").write_text(
                f'MODEL = "{private_path}"\nAPI_KEY = "{fake_secret}"\n',
                encoding="utf-8",
            )
            (root / "weights.safetensors").write_bytes(b"fixture")
            reasons = {finding.reason for finding in scan(root)}
        self.assertIn("private absolute path", reasons)
        self.assertIn("OpenAI-style secret", reasons)
        self.assertTrue(any(reason.startswith("forbidden artifact type") for reason in reasons))

    def test_identity_in_filename_student_id_and_presentation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity_name = "wang" + "jing_notes.py"
            student_name = "student_" + "id_" + "1234" + "5678.txt"
            (root / identity_name).write_text("value = 1\n", encoding="utf-8")
            (root / student_name).write_text("fixture\n", encoding="utf-8")
            (root / "slides.pptx").write_bytes(b"fixture")
            reasons = {finding.reason for finding in scan(root)}
        self.assertIn("personal identifier in path", reasons)
        self.assertIn("student identifier in path", reasons)
        self.assertTrue(any(reason.startswith("forbidden artifact type") for reason in reasons))

    def test_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("fixture\n", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(source)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            self.assertTrue(any("symbolic links" in item.reason for item in scan(root)))

    def test_empty_forbidden_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".codex").mkdir()
            findings = scan(root)
        self.assertTrue(
            any(
                item.path == ".codex" and item.reason == "forbidden path component: .codex"
                for item in findings
            )
        )

    def test_external_identity_patterns_are_literal_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            patterns = Path(external) / "identity-patterns.txt"
            (root / "README.md").write_text("Affiliation: Team [Alpha]\n", encoding="utf-8")
            patterns.write_text("team [alpha]\n", encoding="utf-8")

            findings = scan(root, identity_patterns_file=patterns)
            self.assertTrue(any(item.reason == "configured identity pattern" for item in findings))

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    scan_main([str(root), "--identity-patterns-file", str(patterns)]),
                    1,
                )

    def test_identity_patterns_file_must_remain_external(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patterns = root / "identity-patterns.txt"
            patterns.write_text("Example University\n", encoding="utf-8")
            with self.assertRaisesRegex(ScanConfigurationError, "outside the scanned tree"):
                scan(root, identity_patterns_file=patterns)

    def test_local_hosts_and_submission_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_host = "mock" + "." + "local"
            (root / "endpoint.py").write_text(
                f'URL = "http://{local_host}/v1"\n', encoding="utf-8"
            )
            for suffix in (".tex", ".bib", ".sty", ".cls", ".ipynb"):
                (root / f"submission{suffix}").write_text("fixture\n", encoding="utf-8")
            banned_directories = (
                "tmp",
                "essay",
                "assests",
                "raw_answers",
                "runs",
                "models",
                "checkpoints",
                ".venv",
                "build",
                "dist",
            )
            for name in banned_directories:
                directory_path = root / name
                directory_path.mkdir()
                (directory_path / "trace.txt").write_text("trace\n", encoding="utf-8")
            findings = scan(root)
            reasons = {item.reason for item in findings}
        self.assertIn("internal network address", reasons)
        self.assertEqual(
            sum(item.reason.startswith("forbidden artifact type") for item in findings),
            5,
        )
        for name in banned_directories:
            self.assertIn(f"forbidden path component: {name}", reasons)


class AllowlistAndManifestTests(unittest.TestCase):
    def test_required_optional_and_traversal_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep.txt").write_text("keep\n", encoding="utf-8")
            allowlist = root / "allowlist.txt"
            allowlist.write_text("keep.txt\n?missing.txt\n", encoding="utf-8")
            selected = materialize_allowlist(root, allowlist)
            self.assertEqual([path.name for path in selected], ["keep.txt"])
            allowlist.write_text("../escape\n", encoding="utf-8")
            with self.assertRaises(PackageError):
                materialize_allowlist(root, allowlist)

    def test_manifest_rejects_tampering_and_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("original\n", encoding="utf-8")
            write_manifest(root)
            verify_manifest(root)
            payload.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(PackageError, "checksum"):
                verify_manifest(root)

    def test_attestation_rejects_moved_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stage = base / "stage"
            stage.mkdir()
            (stage / "keep.py").write_text("kept = True\n", encoding="utf-8")
            write_manifest(stage)
            attestation = write_validation_attestation(stage)
            moved = base / "moved"
            stage.rename(moved)
            attestation.rename(validation_attestation_path(moved))
            with self.assertRaisesRegex(PackageError, "does not match"):
                verify_validation_attestation(moved)

    def test_relative_path_check_distinguishes_repository_children(self) -> None:
        root = Path("/workspace/repo")
        self.assertTrue(is_relative_to(root / "dist" / "artifact.zip", root))
        self.assertFalse(is_relative_to(Path("/tmp/artifact.zip"), root))

    def test_anonymity_scan_command_passes_external_pattern_file(self) -> None:
        args = type(
            "Args",
            (),
            {
                "python": "python3",
                "identity_patterns_file": Path("/tmp/aaai-identities.txt"),
            },
        )()
        self.assertEqual(
            anonymity_scan_command(args, Path("/tmp/source")),
            [
                "python3",
                "scripts/scan_anonymity.py",
                "/tmp/source",
                "--identity-patterns-file",
                "/tmp/aaai-identities.txt",
            ],
        )


class ArchiveValidationTests(unittest.TestCase):
    @staticmethod
    def _write_member(archive: zipfile.ZipFile, name: str, mode: int) -> None:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = mode << 16
        archive.writestr(info, b"fixture")

    def test_regular_zip_member_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "valid.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                self._write_member(archive, "trevs/package.py", stat.S_IFREG | 0o644)
            validate_archive(archive_path, "trevs")

    def test_windows_member_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "windows-path.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                self._write_member(archive, "trevs\\package.py", stat.S_IFREG | 0o644)
            with self.assertRaisesRegex(PackageError, "Unsafe ZIP member"):
                validate_archive(archive_path, "trevs")

    def test_non_regular_zip_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "directory-entry.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                self._write_member(archive, "trevs/directory", stat.S_IFDIR | 0o755)
            with self.assertRaisesRegex(PackageError, "non-regular"):
                validate_archive(archive_path, "trevs")


if __name__ == "__main__":
    unittest.main()
