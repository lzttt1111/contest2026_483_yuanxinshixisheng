from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.acceptance.deep_compare_diff_policy import json_difference_policy
from scripts.acceptance.deep_compare_structured import compare_json_trees
from src.nine_analysis.final_output_artifacts import export_item_documents
from src.nine_analysis import zero_target_metrics


ZERO_PATH = (
    "$.overall_metrics.core_metrics.scope_and_morphology."
    "p50_candidate_box_area_px"
)


def test_zero_target_csv_changes_only_instance_distribution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "main.jpg"
    compact = source / "compact.csv"
    medical = source / "medical.csv"
    metrics = source / "metrics.json"
    image.write_bytes(b"image")
    compact.write_text("指标,单位,总计\n数量,个,0\n", encoding="utf-8-sig")
    medical.write_text(
        "检测范围,评估状态,辅助-特征数量（个）,辅助-P50单体面积（像素）,辅助-P90单体面积（像素）,辅助-P90强度（0～1）\n"
        "下巴,可评估,0,0.0,0.0,0.0\n"
        "额头,可评估,1,12.0,18.0,0.0\n",
        encoding="utf-8-sig",
    )
    metrics.write_text('{"总计": 0}', encoding="utf-8")

    exported = export_item_documents(
        "redness",
        {
            "主结果图": str(image),
            "量化CSV": str(compact),
            "医学V2CSV": str(medical),
            "量化JSON": str(metrics),
        },
        tmp_path / "output",
    )

    with exported.medical_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["辅助-P50单体面积（像素）"] == "不可评估"
    assert rows[0]["辅助-P90单体面积（像素）"] == "不可评估"
    assert rows[0]["辅助-P90强度（0～1）"] == "0.0"
    assert rows[1]["辅助-P50单体面积（像素）"] == "12.0"
    assert medical.read_text(encoding="utf-8-sig").splitlines()[1].endswith(
        ",0.0,0.0,0.0"
    )


def test_acne_public_json_marks_zero_candidate_distribution_unavailable(
    tmp_path: Path,
) -> None:
    image = tmp_path / "main.jpg"
    metrics = tmp_path / "summary.json"
    image.write_bytes(b"image")
    metrics.write_text(json.dumps({
        "detection": {
            "status": "ok",
            "input_mode": "full_face",
            "detection_scope": "global",
            "region_counts": {"chin": 0},
            "detections": [],
        },
        "preprocess": {"skin_pixels": 100000, "face_count": 1},
        "grading": {"status": "ok", "severity_level": 0},
    }), encoding="utf-8")

    exported = export_item_documents(
        "acne",
        {"主结果图": str(image), "量化JSON": str(metrics)},
        tmp_path / "output",
    )

    public = json.loads(exported.public_json.read_text(encoding="utf-8"))
    overall = public["overall_metrics"]
    assert overall["core_metrics"]["scope_and_morphology"][
        "p50_candidate_box_area_px"
    ] == "不可评估"
    assert overall["core_metrics"]["signal_intensity"][
        "p90_confidence"
    ] == "不可评估"
    assert overall["auxiliary_metrics"]["count_and_density"][
        "suspected_acne_candidate_count"
    ] == 0
    assert overall["auxiliary_metrics"]["auxiliary_statistics"][
        "primary_concentration_region"
    ] == "—"


def test_json_policy_requires_zero_target_proof() -> None:
    difference = {
        "key": ZERO_PATH,
        "baseline": {"path": ZERO_PATH, "type": "number", "value": 0.0},
        "merged": {"path": ZERO_PATH, "type": "string", "value": "不可评估"},
    }
    comparison = {
        "merged_path": "sample/痤疮量化指标.json",
        "zero_target_unavailable_paths": [ZERO_PATH],
        "differences": [difference],
    }

    allowed_errors, allowed = json_difference_policy([comparison])
    comparison["zero_target_unavailable_paths"] = []
    rejected_errors, _ = json_difference_policy([comparison])

    assert allowed is True
    assert allowed_errors == []
    assert rejected_errors == ["sample/痤疮量化指标.json:" + ZERO_PATH]


def test_json_comparison_records_zero_target_acne_proof(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    merged = tmp_path / "merged"
    baseline.mkdir()
    merged.mkdir()
    document = {
        "overall_metrics": {
            "core_metrics": {
                "scope_and_morphology": {"p50_candidate_box_area_px": 0.0}
            },
            "auxiliary_metrics": {
                "count_and_density": {"suspected_acne_candidate_count": 0}
            },
        },
        "region_metrics": [],
    }
    changed = json.loads(json.dumps(document))
    changed["overall_metrics"]["core_metrics"]["scope_and_morphology"][
        "p50_candidate_box_area_px"
    ] = "不可评估"
    for root, value in ((baseline, document), (merged, changed)):
        (root / "痤疮量化指标.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )

    comparison = compare_json_trees(baseline, merged)["comparisons"][0]

    assert ZERO_PATH in comparison["zero_target_unavailable_paths"]


def test_detector_truth_marks_zero_target_distributions_unavailable() -> None:
    normalizer = getattr(
        zero_target_metrics,
        "normalize_zero_target_detector_results",
        None,
    )
    assert callable(normalizer), "完整真源零目标语义归一器尚未实现"
    detector_results = {
        "surface_gloss": {
            "metrics": {
                "region_metrics": {
                    "forehead": {
                        "patch_count": 0,
                        "gloss_area_px": 0,
                        "p50_gloss_intensity": 0.0,
                        "p90_gloss_intensity": 0.0,
                    },
                    "nose": {
                        "patch_count": 1,
                        "p50_gloss_intensity": 0.4,
                        "p90_gloss_intensity": 0.7,
                    },
                }
            }
        },
        "vascular": {
            "metrics": {
                "vascular_count": 0,
                "vascular_total_length_px": 0.0,
                "p50_width_px": 0.0,
                "p90_width_px": 0.0,
                "p50_redness": 0.0,
                "p90_redness": 0.0,
                "region_distribution": {
                    "forehead": {
                        "count": 0,
                        "total_length_px": 0.0,
                        "p90_redness": 0.0,
                    }
                },
            }
        },
    }

    normalized = normalizer(detector_results)

    gloss = normalized["surface_gloss"]["metrics"]["region_metrics"]
    assert gloss["forehead"]["p50_gloss_intensity"] == "不可评估"
    assert gloss["forehead"]["p90_gloss_intensity"] == "不可评估"
    assert gloss["forehead"]["gloss_area_px"] == 0
    assert gloss["nose"]["p90_gloss_intensity"] == 0.7
    vascular = normalized["vascular"]["metrics"]
    assert vascular["p50_width_px"] == "不可评估"
    assert vascular["p90_redness"] == "不可评估"
    assert vascular["vascular_count"] == 0
    assert vascular["vascular_total_length_px"] == 0.0
    assert vascular["region_distribution"]["forehead"]["p90_redness"] == (
        "不可评估"
    )
