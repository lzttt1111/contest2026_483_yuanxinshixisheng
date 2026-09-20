from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src import aisia_medical_report
from test_twelve_delivery import (
    MISMATCH_SHA256,
    _configure_baseline,
    _configure_delivery,
    _write,
    _write_v3_result,
)


def _missing_media_sha(index) -> None:
    index["items"]["pores"]["images"][0].pop("sha256")


def _mismatched_media_sha(index) -> None:
    index["items"]["pores"]["images"][0]["sha256"] = MISMATCH_SHA256


def _extra_media_entry(index) -> None:
    index["items"]["pores"]["images"].append({
        "path": "十二项检测/pores/extra.jpg",
        "sha256": MISMATCH_SHA256,
    })


def _duplicate_media_entry(index) -> None:
    index["items"]["pores"]["images"].append(
        dict(index["items"]["pores"]["images"][0])
    )


def _failed_item(index) -> None:
    index["items"]["pores"]["状态"] = "failed"


@pytest.mark.parametrize(
    "mutate",
    (
        _missing_media_sha,
        _mismatched_media_sha,
        _extra_media_entry,
        _duplicate_media_entry,
        _failed_item,
    ),
)
def test_formal_delivery_rejects_untrusted_index_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    # Given
    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)
    index_path = result / "十二项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mutate(index)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    # When / Then
    with pytest.raises(RuntimeError):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert not (result / "正式报告").exists()
    assert "formal_reports" not in json.loads(index_path.read_text(encoding="utf-8"))


def test_formal_delivery_rejects_alias_when_source_sha_is_not_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    baseline = tmp_path / "baseline"
    _configure_delivery(monkeypatch, baseline, source)
    _configure_baseline(
        monkeypatch,
        baseline,
        hashlib.sha256(b"different-source").hexdigest(),
    )

    # When / Then
    with pytest.raises(RuntimeError, match="Clinic-Fast3"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert not (result / "正式报告").exists()


def test_formal_delivery_fails_closed_when_stale_docx_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)
    report_root = result / "正式报告"
    stale = Path(_write(report_root / "stale-third.docx", b"stale"))

    # When / Then
    with pytest.raises(RuntimeError, match="DOCX"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert list(report_root.glob("*.docx")) == [stale]
    index = json.loads((result / "十二项检测结果索引.json").read_text(encoding="utf-8"))
    assert "formal_reports" not in index


def test_formal_delivery_leaves_no_partial_output_when_rendering_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    from src.aisia_medical_report import twelve_delivery

    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)

    def fail_doctor(_payload, _template, _output) -> None:
        raise OSError("doctor render failed")

    monkeypatch.setattr(twelve_delivery, "render_doctor_docx", fail_doctor)

    # When / Then
    with pytest.raises(OSError, match="doctor render failed"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert not (result / "正式报告").exists()
    assert not list(result.glob(".aisia-formal-*"))
    index = json.loads((result / "十二项检测结果索引.json").read_text(encoding="utf-8"))
    assert "formal_reports" not in index


def test_formal_delivery_rejects_a_renderer_created_third_docx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    from src.aisia_medical_report import twelve_delivery

    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)

    def render_user_with_extra(_payload, _template, output) -> None:
        target = Path(output)
        target.write_bytes(b"user")
        (target.parent / "unexpected-third.docx").write_bytes(b"stale")

    monkeypatch.setattr(twelve_delivery, "render_user_docx", render_user_with_extra)

    # When / Then
    with pytest.raises(RuntimeError, match="数量或文件名"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert not (result / "正式报告").exists()
    assert not list(result.glob(".aisia-formal-*"))


def test_formal_delivery_rejects_distinct_paths_with_duplicate_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)
    index_path = result / "十二项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    pores = result / index["items"]["pores"]["主结果图"]
    spots = result / index["items"]["spots"]["主结果图"]
    spots.write_bytes(pores.read_bytes())
    index["items"]["spots"]["images"][0]["sha256"] = hashlib.sha256(
        spots.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    # When / Then
    with pytest.raises(RuntimeError, match="SHA256"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )


@pytest.mark.parametrize(
    ("item_id", "extra_name"),
    (("pores", "unexpected-second"), ("wrinkle", "unexpected-fourth")),
)
def test_formal_delivery_rejects_wrong_item_media_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item_id: str,
    extra_name: str,
) -> None:
    # Given
    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)
    index_path = result / "十二项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    extra = Path(_write(
        result / f"十二项检测/{item_id}/{extra_name}.jpg",
        extra_name.encode("ascii"),
    ))
    relative = extra.relative_to(result).as_posix()
    index["items"][item_id]["附加结果图"].append(relative)
    index["items"][item_id]["images"].append({
        "path": relative,
        "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
    })
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    # When / Then
    with pytest.raises(RuntimeError, match="媒体"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )


def test_formal_delivery_rolls_back_when_index_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    from src.aisia_medical_report import twelve_delivery

    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    _configure_delivery(monkeypatch, tmp_path / "baseline", source)
    index_path = result / "十二项检测结果索引.json"
    original_index = index_path.read_bytes()
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source_path, target_path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("index publish failed")
        real_replace(source_path, target_path)

    monkeypatch.setattr(twelve_delivery.os, "replace", fail_second_replace)

    # When / Then
    with pytest.raises(OSError, match="index publish failed"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert not (result / "正式报告").exists()
    assert index_path.read_bytes() == original_index
