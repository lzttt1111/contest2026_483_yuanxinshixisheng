from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts.acceptance.acceptance_types import LocalInspection
from scripts.acceptance.surface_evidence import inspect_local_results


ITEM_NAMES = (
    "01_红区",
    "02_可见斑点",
    "03_棕区",
    "04_纹理",
    "05_毛孔",
    "06_UV色斑",
    "07_卟啉",
    "08_皱纹",
    "09_痤疮",
    "10_油光",
    "11_血管样结构",
    "12_轮廓紧致度",
)


def _write_valid_sample(root: Path) -> tuple[Path, Path]:
    sample = root / "batch_000001" / "clinic28-25_RGB_M"
    items = {}
    compact_rows = ["项目ID,检测项目,指标,值"]
    for index, item_name in enumerate(ITEM_NAMES, start=1):
        item_root = sample / "十二项检测" / item_name
        item_root.mkdir(parents=True)
        image = item_root / "result.jpg"
        image.write_bytes(f"image-{index}".encode())
        compact = item_root / "简版.csv"
        compact.write_text(f"指标,值\nmetric,{index}\n", encoding="utf-8")
        medical = item_root / "医学V2.csv"
        medical.write_text(f"指标,值\nmedical-{index},{index}\n", encoding="utf-8")
        metrics = item_root / "量化.json"
        metrics.write_text(json.dumps({"metric": index}), encoding="utf-8")
        relative_image = image.relative_to(sample).as_posix()
        items[f"item_{index}"] = {
            "项目": item_name,
            "状态": "success",
            "主结果图": relative_image,
            "附加结果图": [],
            "images": [
                {
                    "path": relative_image,
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            ],
            "量化CSV": compact.relative_to(sample).as_posix(),
            "医学V2CSV": medical.relative_to(sample).as_posix(),
            "量化JSON": metrics.relative_to(sample).as_posix(),
        }
        compact_rows.append(f"item_{index},{item_name},metric,{index}")
    index_path = sample / "十二项检测结果索引.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "single_rgb_twelve_index_v3",
                "status": "success",
                "items": items,
                "formal_reports": {
                    "user_docx": "正式报告/user.docx",
                    "doctor_docx": "正式报告/doctor.docx",
                },
            }
        ),
        encoding="utf-8",
    )
    (sample / "十二项简版量化指标.csv").write_text(
        "\n".join(compact_rows) + "\n", encoding="utf-8"
    )
    module_scores = {}
    remaining = 137
    for index in range(11):
        count = 13 if index < 5 else 12
        remaining -= count
        module_scores[f"module_{index}"] = {
            "score_valid": True,
            "score": 50.0,
            "grade": "正常",
            "groups": {
                "group": {
                    "metrics": {
                        f"metric_{metric}": {"value": float(metric)}
                        for metric in range(count)
                    }
                }
            },
        }
    assert remaining == 0
    (sample / "十二项完整量化指标.json").write_text(
        json.dumps(
            {
                "schema_version": "single_rgb_twelve_complete_v2_20260728",
                "detector_results": {
                    f"item_{index}": {
                        "status": "success",
                        "public_metrics": {"metric": index},
                    }
                    for index in range(1, 13)
                },
                "scoring_results": {"module_scores": module_scores},
            }
        ),
        encoding="utf-8",
    )
    (sample / "运行回执.json").write_text(
        json.dumps(
            {
                "timing": {"total": 1.0},
                "input_signature": {"sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    report_root = sample / "正式报告"
    report_root.mkdir()
    (report_root / "user.docx").write_bytes(b"docx")
    (report_root / "doctor.docx").write_bytes(b"docx")
    return sample, index_path


def _inspect(root: Path) -> LocalInspection:
    return inspect_local_results(
        profile="merged",
        result_root=root,
        expected_cases=("clinic28-25",),
        started_ns=0,
        require_word=True,
    )


def test_merged_local_accepts_complete_dual_csv_json_media_and_word(tmp_path: Path) -> None:
    # Given
    _write_valid_sample(tmp_path)

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.discovered_cases == ("clinic28-25",)
    assert inspection.contract_errors == ()


def test_merged_local_rejects_xlsx_and_extra_root_file(tmp_path: Path) -> None:
    # Given
    sample, _index = _write_valid_sample(tmp_path)
    (sample / "forbidden.xlsx").write_bytes(b"xlsx")
    (sample / "debug.json").write_text("{}", encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == (
        "unexpected_root_files",
        "unexpected_xlsx",
    )


def test_merged_local_rejects_extra_item_debug_artifact(tmp_path: Path) -> None:
    # Given
    sample, _index = _write_valid_sample(tmp_path)
    debug = sample / "十二项检测" / "01_红区" / "internal_debug.json"
    debug.write_text("{}", encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == ("unexpected_item_files",)


def test_merged_local_rejects_untrusted_media_sha(tmp_path: Path) -> None:
    # Given
    _sample, index_path = _write_valid_sample(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["items"]["item_1"]["images"][0]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index), encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == ("invalid_item_media",)


def test_merged_local_rejects_header_only_medical_csv_and_root_csv_drift(
    tmp_path: Path,
) -> None:
    # Given
    sample, index_path = _write_valid_sample(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    medical = sample / index["items"]["item_1"]["医学V2CSV"]
    medical.write_text("指标,值\n", encoding="utf-8")
    (sample / "十二项简版量化指标.csv").write_text(
        "指标,值\nwrong,0\n", encoding="utf-8"
    )

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == (
        "header_only_medical_csv",
        "root_compact_csv_drift",
    )


def test_merged_local_rejects_absolute_word_and_media_paths(tmp_path: Path) -> None:
    # Given
    sample, index_path = _write_valid_sample(tmp_path)
    outside_media = tmp_path / "outside.jpg"
    outside_media.write_bytes(b"outside-media")
    outside_word = tmp_path / "outside.docx"
    outside_word.write_bytes(b"outside-word")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["items"]["item_1"]["主结果图"] = str(outside_media)
    index["items"]["item_1"]["images"] = [
        {
            "path": str(outside_media),
            "sha256": hashlib.sha256(outside_media.read_bytes()).hexdigest(),
        }
    ]
    index["formal_reports"]["user_docx"] = str(outside_word)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert "unsafe_artifact_path" in inspection.contract_errors


def test_merged_local_rejects_parent_escape_json_path(tmp_path: Path) -> None:
    # Given
    sample, index_path = _write_valid_sample(tmp_path)
    outside = sample.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["items"]["item_1"]["量化JSON"] = "../outside.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert "unsafe_artifact_path" in inspection.contract_errors


def test_merged_local_rejects_symlink_escape_csv_path(tmp_path: Path) -> None:
    # Given
    sample, index_path = _write_valid_sample(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("指标,值\noutside,1\n", encoding="utf-8")
    link = sample / "十二项检测" / "01_红区" / "escaped.csv"
    os.symlink(outside, link)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["items"]["item_1"]["医学V2CSV"] = link.relative_to(sample).as_posix()
    index_path.write_text(json.dumps(index), encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert "unsafe_artifact_path" in inspection.contract_errors


def test_merged_local_normalizes_malformed_index_as_inspection_failure(
    tmp_path: Path,
) -> None:
    # Given
    _sample, index_path = _write_valid_sample(tmp_path)
    index_path.write_text("{malformed", encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == ("inspection_error:ValidationError",)


def test_merged_local_rejects_missing_scores_and_empty_receipt(tmp_path: Path) -> None:
    # Given
    sample, _index = _write_valid_sample(tmp_path)
    complete_path = sample / "十二项完整量化指标.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["scoring_results"]["module_scores"]["module_0"]["score_valid"] = False
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    (sample / "运行回执.json").write_text("{}", encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert "score_contract" in inspection.contract_errors
    assert "receipt_contract" in inspection.contract_errors


def test_merged_local_rejects_public_json_drift_from_complete_truth(tmp_path: Path) -> None:
    # Given
    sample, index_path = _write_valid_sample(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    public_json = sample / index["items"]["item_1"]["量化JSON"]
    public_json.write_text(json.dumps({"metric": 999}), encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == (
        "compact_metric_drift",
        "public_metric_drift",
    )


def test_merged_local_rejects_compact_csv_drift_from_public_truth(tmp_path: Path) -> None:
    # Given
    sample, index_path = _write_valid_sample(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    compact = sample / index["items"]["item_1"]["量化CSV"]
    compact.write_text("指标,值\nmetric,999\n", encoding="utf-8")
    root = sample / "十二项简版量化指标.csv"
    content = root.read_text(encoding="utf-8")
    content = content.replace("item_1,01_红区,metric,1\n", "item_1,01_红区,metric,999\n")
    root.write_text(content, encoding="utf-8")

    # When
    inspection = _inspect(tmp_path)

    # Then
    assert inspection.contract_errors == ("compact_metric_drift",)
