from __future__ import annotations

import json
from pathlib import Path

import src.acne.worker as worker
from src.acne.clinical_quantification import (
    build_medical_v2_report,
    build_report,
    write_medical_v2_report,
)
from src.acne.medical_v2_schema import to_english_document


def test_worker_returns_quantification_inside_six_field_envelope(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    summary_path = raw_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    class FakePipeline:
        def process_single(self, input_path, algorithms):
            return {
                "status": "success",
                "output_dir": str(raw_dir),
                "summary_path": str(summary_path),
                "results": {},
            }

    quantification = {
        "疑似痤疮圈选数量": 4,
        "痤疮严重程度等级": {
            "等级": 2,
            "注释": "本次为2级（偏轻）。",
        },
    }

    monkeypatch.setattr(worker, "download_via_internal_api", lambda oss_key: b"test-image")
    monkeypatch.setattr(worker, "_get_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        worker,
        "build_display_outputs",
        lambda **kwargs: (
            {"summary_json": str(summary_path)},
            {"量化结果": quantification},
            {"device": "cpu"},
        ),
    )
    monkeypatch.setattr(
        worker,
        "_upload_result_group",
        lambda **kwargs: {"summary_json": "acne-detection-worker/v1/test/05_检测结果摘要.json"},
    )

    result = worker.analyze_image.run("test", "inputs/test.jpg")

    assert result["status"] == "success"
    assert set(result) == {
        "record_id", "status", "schema_version", "meta_data", "raw_result", "debug_info",
    }
    assert result["schema_version"] == "1"
    assert result["meta_data"] == {"name": "acne", "version": "1"}
    assert set(result["raw_result"]) == {
        "acne_summary", "acne_circles", "acne_raw_boxes",
        "acne_detections", "acne_skin_mask", "acne_forbidden_mask",
        "acne_standardized", "max_recall_heatmap",
        "diffuse_erythema_heatmap", "focal_candidate_heatmap",
        "max_recall_circles", "combined_candidate_circles",
        "max_recall_debug", "max_recall_json", "original_yolo_circles",
        "original_unsupervised_circles", "original_combined_circles",
        "acne_presence", "acne_count", "region_counts", "grading_status",
        "grading_reason", "detector_status", "detector_reason", "input_mode",
        "detection_scope", "量化结果",
    }
    assert result["raw_result"]["量化结果"] == quantification
    assert set(result["debug_info"]) == {
        "display_result", "algorithm_version", "elapsed_seconds",
    }
    assert result["debug_info"]["display_result"] == {
        "summary_json": "acne-detection-worker/v1/test/05_检测结果摘要.json"
    }


def test_medical_v2_additions_are_optional_and_fail_soft(tmp_path: Path, monkeypatch) -> None:
    report = build_report({})
    csv_path = tmp_path / "痤疮医学量化指标_V2.csv"
    json_path = tmp_path / "痤疮量化指标.json"
    json_path.write_text(
        json.dumps(
            {
                "旧字段": "保持",
                "medical_metrics_v2": to_english_document(
                    build_medical_v2_report(report)
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_medical_v2_report(report, csv_path, None)
    uploads: list[tuple[str, str]] = []
    monkeypatch.setattr(worker.settings, "enable_medical_metrics_v2", True)
    monkeypatch.setattr(
        worker,
        "upload_via_internal_api",
        lambda local_path, report_id, algo_name, file_name: (
            uploads.append((local_path, f"report/{report_id}/{algo_name}/{file_name}"))
            or f"report/{report_id}/{algo_name}/{file_name}"
        ),
    )
    additions = worker._build_medical_v2_additions(
        record_id="record-v2",
        output_dir=tmp_path,
    )
    assert set(additions) == {"medical_metrics_v2", "medical_report_csv_v2"}
    assert additions["medical_metrics_v2"]["scoring_status"] == "uncalibrated"
    assert uploads[0][1].endswith("/痤疮医学量化指标_V2.csv")

    json_path.unlink()
    assert worker._build_medical_v2_additions(
        record_id="record-v2",
        output_dir=tmp_path,
    ) == {}
