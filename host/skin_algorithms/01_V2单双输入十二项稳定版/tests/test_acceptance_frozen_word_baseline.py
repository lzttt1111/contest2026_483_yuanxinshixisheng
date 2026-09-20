from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.acceptance.acceptance_io import AcceptanceInputError
from scripts.acceptance.formal_baseline import verify_formal_baseline


def test_delivery_frozen_word_baseline_manifest_and_six_docx_pass() -> None:
    # Given
    project_root = Path(__file__).resolve().parents[1]
    frozen_root = project_root.parent / "00_runtime_assets/frozen_formal_word_baseline"

    # When / Then
    verify_formal_baseline(project_root, frozen_root)


def test_frozen_word_baseline_rejects_missing_structured_payload(
    tmp_path: Path,
) -> None:
    # Given
    project_root = Path(__file__).resolve().parents[1]
    source = project_root.parent / "00_runtime_assets/frozen_formal_word_baseline"
    frozen_root = tmp_path / "frozen_formal_word_baseline"
    shutil.copytree(source, frozen_root)
    target = (
        frozen_root
        / "clinic28-25/reports/formal_evidence/AISIA_clinic28-25_正式结构化数据.json"
    )
    target.unlink()

    # When / Then
    with pytest.raises(AcceptanceInputError, match="file set|DOCX mismatch"):
        verify_formal_baseline(project_root, frozen_root)
