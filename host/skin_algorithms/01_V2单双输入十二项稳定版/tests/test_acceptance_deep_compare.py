from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from scripts.acceptance.deep_compare import build_deep_report, write_deep_report
from scripts.acceptance.deep_compare_inventory import compare_images
from scripts.acceptance.deep_compare_types import ComparisonRequest
from scripts.acceptance.deep_compare_word import inspect_docx
from scripts.acceptance.deep_compare_word_formal import semantic_signature


def test_compare_images_reports_dimensions_sha_phash_and_pixel_diff(tmp_path: Path) -> None:
    # Given
    baseline = tmp_path / "baseline.png"
    merged = tmp_path / "merged.png"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(baseline)
    Image.new("RGB", (8, 6), color=(11, 20, 30)).save(merged)

    # When
    result = compare_images(baseline, merged)

    # Then
    assert result["baseline_dimensions"] == [8, 6]
    assert result["merged_dimensions"] == [8, 6]
    assert len(result["baseline_sha256"]) == 64
    assert len(result["baseline_phash"]) == 16
    assert result["pixel_diff"]["changed_pixels"] == 48


def test_inspect_docx_reports_chapters_tables_bookmarks_media_hash_and_path_leak(
    tmp_path: Path,
) -> None:
    # Given
    docx = tmp_path / "report.docx"
    document_xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        '<w:r><w:t>第一章 /home/private</w:t></w:r></w:p>'
        '<w:bookmarkStart w:id="1" w:name="chapter_1"/><w:tbl/></w:body></w:document>'
    )
    with ZipFile(docx, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/media/image1.png", b"media")

    # When
    result = inspect_docx(docx)

    # Then
    assert result["chapter_count"] == 1
    assert result["table_count"] == 1
    assert result["bookmark_count"] == 1
    assert len(result["media"][0]["sha256"]) == 64
    assert result["path_leaks"]


def test_build_deep_report_emits_every_required_machine_and_human_section(
    tmp_path: Path,
) -> None:
    # Given
    roots = []
    for name in (
        "baseline",
        "merged",
        "baseline_run",
        "merged_run",
        "baseline_cloud",
        "merged_cloud",
        "formal_baseline",
    ):
        path = tmp_path / name
        path.mkdir()
        roots.append(path)
    request = ComparisonRequest(*roots)
    json_path = tmp_path / "deep_comparison.json"
    markdown_path = tmp_path / "deep_comparison.md"

    # When
    report = build_deep_report(request)
    write_deep_report(report, json_path, markdown_path)

    # Then
    assert set(report.sections) == {
        "tracked_files",
        "file_tree",
        "images",
        "json",
        "csv_xlsx",
        "worker_contracts",
        "baseline_manifest_staleness",
        "acne_v1_to_v2",
        "added_three",
        "word",
        "path_leaks",
    }
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == "aisia_deep_comparison_v1"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Worker contracts" in markdown
    assert "## Word" in markdown


def test_deep_compare_cli_returns_nonzero_for_empty_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    roots = []
    for name in (
        "baseline",
        "merged",
        "baseline_run",
        "merged_run",
        "baseline_cloud",
        "merged_cloud",
        "formal_baseline",
    ):
        path = tmp_path / name
        path.mkdir()
        roots.append(path)
    comparisons = tmp_path / "comparisons"
    comparisons.mkdir()
    from scripts.acceptance import deep_compare

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deep_compare.py",
            "--baseline-root", str(roots[0]),
            "--merged-root", str(roots[1]),
            "--baseline-run", str(roots[2]),
            "--merged-run", str(roots[3]),
            "--baseline-cloud", str(roots[4]),
            "--merged-cloud", str(roots[5]),
            "--formal-baseline-root", str(roots[6]),
            "--json-output", str(comparisons / "report.json"),
            "--markdown-output", str(comparisons / "report.md"),
        ],
    )

    # When
    exit_code = deep_compare.main()

    # Then
    assert exit_code != 0


def test_deep_compare_cli_rejects_nonempty_comparisons_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    roots = []
    for name in (
        "baseline",
        "merged",
        "baseline_run",
        "merged_run",
        "baseline_cloud",
        "merged_cloud",
        "formal_baseline",
    ):
        path = tmp_path / name
        path.mkdir()
        roots.append(path)
    comparisons = tmp_path / "comparisons"
    comparisons.mkdir()
    (comparisons / "stale.json").write_text("{}", encoding="utf-8")
    from scripts.acceptance import deep_compare

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deep_compare.py",
            "--baseline-root", str(roots[0]),
            "--merged-root", str(roots[1]),
            "--baseline-run", str(roots[2]),
            "--merged-run", str(roots[3]),
            "--baseline-cloud", str(roots[4]),
            "--merged-cloud", str(roots[5]),
            "--formal-baseline-root", str(roots[6]),
            "--json-output", str(comparisons / "report.json"),
            "--markdown-output", str(comparisons / "report.md"),
        ],
    )

    # When
    exit_code = deep_compare.main()

    # Then
    assert exit_code == 2
    assert not (comparisons / "report.json").exists()
    assert not (comparisons / "report.md").exists()


def test_word_semantic_signature_ignores_bound_media_but_not_scores() -> None:
    # Given
    baseline = {
        "受检者信息": {"姓名或编号": "baseline"},
        "标准采集图像": {"path": "old.jpg"},
        "检测模块": [
            {
                "模块编号": "01",
                "综合得分": 81,
                "结果图": ["old-result.jpg"],
                "医生结果分组": [
                    {
                        "title": "group",
                        "images": [{"path": "old.jpg", "caption": "APPROVED"}],
                    }
                ],
            }
        ],
        "日常建议": ["keep frozen"],
    }
    rebound = json.loads(json.dumps(baseline))
    rebound["受检者信息"]["姓名或编号"] = "current"
    rebound["标准采集图像"]["path"] = "new.jpg"
    rebound["检测模块"][0]["结果图"] = ["new-result.jpg"]
    rebound["检测模块"][0]["医生结果分组"][0]["images"] = [
        {"path": "new.jpg", "caption": "APPROVED"}
    ]

    # When
    rebound_signature = semantic_signature(rebound)

    # Then
    assert rebound_signature == semantic_signature(baseline)
    rebound["检测模块"][0]["医生结果分组"][0]["images"][0][
        "caption"
    ] = "INTERNAL DEBUG"
    assert semantic_signature(rebound) != semantic_signature(baseline)
    rebound["检测模块"][0]["医生结果分组"][0]["images"][0][
        "caption"
    ] = "APPROVED"
    rebound["检测模块"][0]["综合得分"] = 99
    assert semantic_signature(rebound) != semantic_signature(baseline)
