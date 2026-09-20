from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import pytest

from src import aisia_medical_report


ITEM_IDS = (
    "redness",
    "spots",
    "brown",
    "texture",
    "pores",
    "uv_spots",
    "porphyrin",
    "wrinkle",
    "acne",
    "surface_gloss",
    "vascular",
    "contour_firmness",
)
MISMATCH_SHA256: Final = "0" * 64


def _write(path: Path, data: bytes = b"image") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.as_posix()


def _write_v3_result(root: Path) -> Path:
    items: dict[str, dict[str, str | list[str]]] = {}
    for item_id in ITEM_IDS:
        image = Path(
            _write(
                root / f"十二项检测/{item_id}/main.jpg",
                item_id.encode("ascii"),
            )
        )
        items[item_id] = {
            "状态": "success",
            "主结果图": image.relative_to(root).as_posix(),
            "附加结果图": [],
            "images": [
                {
                    "path": image.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            ],
        }
    extra_roles = {
        "redness": ("redness-extra",),
        "brown": ("brown-extra",),
        "uv_spots": ("uv-extra",),
        "porphyrin": ("porphyrin-extra",),
        "wrinkle": ("08", "09"),
    }
    for item_id, roles in extra_roles.items():
        extras = [
            Path(_write(
                root / f"十二项检测/{item_id}/{role}.jpg",
                f"{item_id}-{role}".encode("ascii"),
            ))
            for role in roles
        ]
        items[item_id]["附加结果图"] = [
            path.relative_to(root).as_posix() for path in extras
        ]
        items[item_id]["images"].extend(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in extras
        )
    (root / "十二项检测结果索引.json").write_text(
        json.dumps(
            {
                "schema_version": "single_rgb_twelve_index_v3",
                "status": "success",
                "items": items,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return Path(_write(root / "source.jpg", b"source"))


def _write_baseline(root: Path, *, source_sha256: str = "") -> None:
    payload = {
        "受检者信息": {"姓名或编号": "frozen-subject"},
        "报告信息": {"报告编号": "frozen-report"},
        "标准采集图像": {"sha256": source_sha256},
        "检测模块": [
            {
                "模块编号": f"{module_id:02d}",
                "结果图": [],
                "综合得分": 80 + module_id,
                "核心指标": [{"name": "frozen", "value": module_id}],
                "医生结果分组": [],
            }
            for module_id in range(1, 12)
        ],
    }
    path = (
        root
        / "clinic28-25/reports/formal_evidence"
        / "AISIA_clinic28-25_正式结构化数据.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _configure_delivery(
    monkeypatch: pytest.MonkeyPatch,
    baseline: Path,
    source: Path,
) -> None:
    _configure_baseline(
        monkeypatch,
        baseline,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    from src.aisia_medical_report import twelve_delivery

    monkeypatch.setattr(
        twelve_delivery,
        "render_user_docx",
        lambda _payload, _template, output: Path(output).write_bytes(b"user"),
    )
    monkeypatch.setattr(
        twelve_delivery,
        "render_doctor_docx",
        lambda _payload, _template, output: Path(output).write_bytes(b"doctor"),
    )


def _configure_baseline(
    monkeypatch: pytest.MonkeyPatch,
    baseline: Path,
    source_sha256: str,
) -> None:
    from src.aisia_medical_report import twelve_delivery

    _write_baseline(
        baseline,
        source_sha256=source_sha256,
    )
    baseline_json = (
        baseline
        / "clinic28-25/reports/formal_evidence"
        / "AISIA_clinic28-25_正式结构化数据.json"
    )
    monkeypatch.setattr(twelve_delivery, "_baseline_root", lambda: baseline)
    monkeypatch.setitem(
        twelve_delivery.BASELINE_PAYLOAD_SHA256,
        "clinic28-25",
        hashlib.sha256(baseline_json.read_bytes()).hexdigest(),
    )


def test_v3_delivery_rebinds_current_media_and_writes_exactly_two_docx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    result = tmp_path / "clinic28-25"
    source = _write_v3_result(result)
    baseline = tmp_path / "baseline"
    _configure_delivery(monkeypatch, baseline, source)

    # When
    deliver = aisia_medical_report.generate_dual_reports_from_twelve_result
    outputs = deliver(result, source_image=source, subject_id="fixture-subject")

    # Then
    report_root = result / "正式报告"
    assert sorted(path.name for path in report_root.glob("*.docx")) == [
        "AISIA_clinic28-25_医生详细版.docx",
        "AISIA_clinic28-25_用户精简版.docx",
    ]
    assert Path(outputs["user_docx"]).read_bytes() == b"user"
    assert Path(outputs["doctor_docx"]).read_bytes() == b"doctor"
    structured = json.loads(Path(outputs["structured_json"]).read_text(encoding="utf-8"))
    modules = {row["模块编号"]: row for row in structured["检测模块"]}
    assert modules["07"]["结果图"][0].startswith("evidence/07/")
    assert modules["07"]["结果图"][0].endswith("_main.jpg")
    assert modules["08"]["结果图"][0].startswith("evidence/08/")
    assert modules["08"]["结果图"][0].endswith("_08.jpg")
    assert modules["09"]["结果图"][0].startswith("evidence/09/")
    assert modules["09"]["结果图"][0].endswith("_09.jpg")
    assert [modules[f"{index:02d}"]["综合得分"] for index in range(1, 12)] == [
        80 + index for index in range(1, 12)
    ]
    assert structured["标准采集图像"]["path"].startswith("evidence/00/")
    assert str(tmp_path) not in json.dumps(structured, ensure_ascii=False)
    index = json.loads((result / "十二项检测结果索引.json").read_text(encoding="utf-8"))
    assert set(index["formal_reports"]) == {"user_docx", "doctor_docx"}


def test_formal_baseline_payload_fails_closed_when_frozen_hash_changes(
    tmp_path: Path,
) -> None:
    # Given
    from src.aisia_medical_report import twelve_delivery

    _write_baseline(tmp_path)
    baseline_json = (
        tmp_path
        / "clinic28-25/reports/formal_evidence"
        / "AISIA_clinic28-25_正式结构化数据.json"
    )
    baseline_json.write_text("{}\n", encoding="utf-8")

    # When / Then
    with pytest.raises(RuntimeError, match="SHA256"):
        twelve_delivery._baseline_json(tmp_path, "clinic28-25")


@pytest.mark.parametrize(
    "index",
    [
        {"状态": "success", "成功项目数": 9, "九项结果": {}},
        {
            "schema_version": "single_rgb_twelve_index_v3",
            "status": "partial_success",
            "items": {},
        },
    ],
    ids=("legacy-nine", "partial-twelve"),
)
def test_formal_delivery_skips_legacy_and_partial_results(
    tmp_path: Path,
    index: dict[str, str | int | dict[str, str]],
) -> None:
    # Given
    result = tmp_path / "sample"
    result.mkdir()
    source = result / "source.jpg"
    source.write_bytes(b"source")
    (result / "十二项检测结果索引.json").write_text(
        json.dumps(index, ensure_ascii=False),
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(RuntimeError, match="十二项全部成功"):
        aisia_medical_report.generate_dual_reports_from_twelve_result(
            result,
            source_image=source,
        )
    assert not (result / "正式报告").exists()
