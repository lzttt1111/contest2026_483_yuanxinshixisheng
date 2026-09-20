from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from src.nine_analysis.orchestrator import NineAnalysisOrchestrator


def test_staged_jpeg_gets_missing_eoi_on_disposable_copy_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.jpg"
        source.write_bytes(b"\xff\xd8payload")

        staged = NineAnalysisOrchestrator._copy_input(source, root / "sample")

        assert source.read_bytes() == b"\xff\xd8payload"
        assert staged.read_bytes() == b"\xff\xd8payload\xff\xd9"


def test_staging_retries_a_short_nfs_copy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.png"
        source.write_bytes(b"0123456789")
        real_copy = __import__("shutil").copyfile
        attempts = 0

        def flaky_copy(source_path: Path, target_path: Path) -> Path:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                Path(target_path).write_bytes(b"short")
                return Path(target_path)
            return Path(real_copy(source_path, target_path))

        with mock.patch("src.nine_analysis.orchestrator.shutil.copyfile", side_effect=flaky_copy):
            staged = NineAnalysisOrchestrator._copy_input(source, root / "sample")

        assert attempts == 2
        assert staged.read_bytes() == source.read_bytes()


def test_staging_uses_content_magic_for_mislabeled_png() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "mislabeled.jpg"
        payload = b"\x89PNG\r\n\x1a\n" + b"payload"
        source.write_bytes(payload)

        staged = NineAnalysisOrchestrator._copy_input(source, root / "sample")

        assert staged.suffix == ".png"
        assert staged.read_bytes() == payload
