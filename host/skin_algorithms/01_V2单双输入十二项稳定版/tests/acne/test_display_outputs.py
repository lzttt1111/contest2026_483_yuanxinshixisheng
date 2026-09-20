from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.acne.display_outputs import DISPLAY_FILENAMES, build_display_outputs
from src.acne.image_io import write_image


def _build_fixture(tmp_path: Path, *, detection_payload: dict, grading_payload: dict | None = None) -> tuple[dict, dict, dict]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    write_image(raw_dir / "01_original.jpg", image)
    write_image(raw_dir / "02_standardized.jpg", image)
    write_image(raw_dir / "17_acne_candidate_circles.jpg", image)
    write_image(raw_dir / "24_original_yolo_circles.jpg", image)

    with (raw_dir / "18_detections.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["confidence", "region", "source"])
        writer.writeheader()
        writer.writerow({"confidence": "0.88", "region": "forehead", "source": "yolo"})

    grading = grading_payload or {
        "status": "ok",
        "predicted_count": 7,
        "severity_level": 2,
        "severity_probabilities": [0.1, 0.7, 0.15, 0.05],
    }
    (raw_dir / "13_grading_result.json").write_text(json.dumps(grading, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "elapsed_seconds": 1.23,
        "profile": {
            "name": "v3_balanced",
            "weights": "models/detector/acne_candidate_baseline_v3_n_768.pt",
            "imgsz": 768,
            "conf": 0.15,
            "device": "cpu",
        },
        "preprocess": {
            "input_mode": "full_face",
            "detection_scope": "global",
        },
        "detection": detection_payload,
        "grading": grading,
        "max_recall": {
            "status": "ok",
            "unsupervised_focal_count": 2,
            "diffuse_erythema_region_count": 0,
            "combined_count": 2,
        },
    }
    (raw_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    display_dir = tmp_path / "display"
    return build_display_outputs(
        raw_dir=raw_dir,
        input_image="sample.jpg",
        analysis_summary=summary,
        raw_result_files={
            "original_image": str(raw_dir / "01_original.jpg"),
            "acne_standardized": str(raw_dir / "02_standardized.jpg"),
            "acne_circles": str(raw_dir / "17_acne_candidate_circles.jpg"),
            "original_yolo_circles": str(raw_dir / "24_original_yolo_circles.jpg"),
            "acne_detections": str(raw_dir / "18_detections.csv"),
        },
        target_dir=display_dir,
    )


def test_build_display_outputs_creates_expected_display_bundle(tmp_path: Path) -> None:
    display_files, compact, metadata = _build_fixture(
        tmp_path,
        detection_payload={
            "status": "ok",
            "raw_count": 1,
            "count": 1,
            "region_counts": {"forehead": 1},
            "region_analysis": {"status": "ok", "scope": "global"},
            "empty_detections_are_valid": True,
            "detections": [
                {
                    "confidence": 0.88,
                    "region": "forehead",
                    "source": "yolo",
                }
            ],
        },
    )

    assert set(display_files) == set(DISPLAY_FILENAMES)
    for saved_path in display_files.values():
        assert Path(saved_path).exists()
    assert compact["detection"]["filtered_count"] == 1
    assert compact["detection"]["acne_presence"]["status"] == "detected"
    assert compact["量化结果"]["疑似痤疮圈选数量"] == 1
    assert compact["量化结果"]["痤疮严重程度等级"]["等级"] == 2
    assert metadata["profile_name"] == "v3_balanced"
    assert metadata["device"] == "cpu"
    assert metadata["acne_presence"]["message"] == "已检测到疑似痤疮"
    assert metadata["region_counts"] == [{"region": "forehead", "count": 1}]


def test_build_display_outputs_marks_not_detected_when_filtered_count_is_zero(tmp_path: Path) -> None:
    display_files, compact, metadata = _build_fixture(
        tmp_path,
        detection_payload={
            "status": "ok",
            "raw_count": 0,
            "count": 0,
            "region_counts": {},
            "region_analysis": {"status": "ok", "scope": "global"},
            "empty_detections_are_valid": True,
            "detections": [],
        },
    )

    assert display_files["summary_json"]
    assert compact["detection"]["acne_presence"]["status"] == "not_detected"
    assert compact["detection"]["acne_presence"]["message"] == "未检测到痤疮"
    assert compact["量化结果"]["疑似痤疮圈选数量"] == 0
    assert compact["量化结果"]["痤疮严重程度等级"]["等级"] == 2
    assert metadata["acne_presence"]["status"] == "not_detected"
    assert metadata["region_counts"] == []
    report_text = Path(display_files["user_report_txt"]).read_text(encoding="utf-8")
    assert "未检测到痤疮" in report_text


def test_build_display_outputs_marks_not_applicable_when_detection_is_skipped(tmp_path: Path) -> None:
    _, compact, metadata = _build_fixture(
        tmp_path,
        detection_payload={
            "status": "skipped",
            "reason": "local_detection_not_allowed",
            "raw_count": 0,
            "count": 0,
            "region_counts": {},
            "region_analysis": {"status": "skipped", "scope": "local"},
            "empty_detections_are_valid": True,
            "detections": [],
        },
        grading_payload={
            "status": "skipped",
            "reason": "input_not_full_face",
            "predicted_count": None,
            "severity_level": None,
            "severity_probabilities": [],
        },
    )

    assert compact["detection"]["acne_presence"]["status"] == "not_applicable"
    assert compact["detection"]["acne_presence"]["message"] != "未检测到痤疮"
    assert compact["量化结果"]["疑似痤疮圈选数量"] is None
    assert compact["量化结果"]["痤疮严重程度等级"]["等级"] is None
    assert metadata["acne_presence"]["status"] == "not_applicable"
    assert metadata["region_counts"] == []
