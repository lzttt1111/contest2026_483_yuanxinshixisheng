from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

import run


def test_review_profile_rejects_partial_selection_before_input_scan(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    stderr = io.StringIO()

    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
        run.main([
            "--input-dir",
            str(input_root),
            "--output-dir",
            str(output_root),
            "--algorithms",
            "vascular",
        ])

    assert "review正式输出只支持all或legacy-nine" in stderr.getvalue()


def test_consumer_profile_accepts_word_and_reaches_input_scan(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    stderr = io.StringIO()

    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
        run.main([
            "--input-dir",
            str(input_root),
            "--output-dir",
            str(output_root),
            "--capture-profile",
            "consumer",
            "--generate-medical-report",
            "--precount",
        ])

    assert "未找到符合条件的图片" in stderr.getvalue()
