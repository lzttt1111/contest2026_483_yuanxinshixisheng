from __future__ import annotations

from pathlib import Path

from docx import Document
import pytest

from src.aisia_medical_report.identity import (
    clean_report_number,
    resolve_subject_id,
    single_rgb_report_identity,
    subject_id_from_source,
)
from src.aisia_medical_report.report import render_doctor_docx, render_user_docx


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates/AISIA_面部多指标检测汇总模板_V2.docx"


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("01_clinic28-25_RGB_M.jpg", "clinic28-25"),
        ("clinic28-09_CP_M.png", "clinic28-09"),
        ("1000037_1.jpg", "1000037_1"),
        ("李四 正脸照片.jpeg", "李四 正脸照片"),
    ),
)
def test_subject_id_comes_from_original_source(source: str, expected: str) -> None:
    assert subject_id_from_source(source) == expected


def test_explicit_subject_wins_with_whitespace_normalization() -> None:
    assert resolve_subject_id("  门店 A   0007  ", source="ignored.jpg") == (
        "门店 A 0007"
    )


def test_direct_single_rgb_default_separates_subject_and_report_number() -> None:
    identity = single_rgb_report_identity(
        source_image="/external/02_clinic28-09_RGB_M.jpg",
        result_name="report-94f4057d7b5d5e04.jpg",
        subject_id=None,
        report_id=None,
    )

    assert identity.subject_id == "clinic28-09"
    assert identity.report_id == "AISIA-SINGLE-RGB-report-94f4057d7b5d5e04"
    assert ".jpg" not in identity.report_id
    assert "external" not in identity.report_id


def test_explicit_report_number_removes_path_and_image_extension() -> None:
    assert clean_report_number("/tmp/private/REPORT-0001.jpg") == "REPORT-0001"


def test_institution_alias_remains_unchanged() -> None:
    assert resolve_subject_id(None, source="clinic28-23") == "clinic28-23"


@pytest.mark.parametrize(
    ("renderer", "suffix"),
    ((render_user_docx, "user"), (render_doctor_docx, "doctor")),
)
def test_first_page_labels_report_and_subject_id_without_fake_name(
    tmp_path: Path,
    renderer,
    suffix: str,
) -> None:
    payload = {
        "报告信息": {"报告编号": "REPORT-UNIQUE-001", "报告日期": "2026-08-31"},
        "受检者信息": {"姓名或编号": "clinic28-25", "采集日期": "2026-08-31"},
        "总体摘要": {},
        "检测模块": [],
        "医学局限性": [],
    }
    target = renderer(payload, TEMPLATE, tmp_path / f"{suffix}.docx")
    rows = [[cell.text for cell in row.cells] for row in Document(target).tables[0].rows]

    assert rows[:2] == [
        ["报告编号", "REPORT-UNIQUE-001"],
        ["受检者编号", "clinic28-25"],
    ]
    assert not any(row[0] in {"受检者", "受检者姓名"} for row in rows)
