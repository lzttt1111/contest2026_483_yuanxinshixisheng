"""scoring_input builder 单测：头部字段、evidence 投影、诚实降级、紫区拆分。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.scoring_input import build_scoring_input, extract_evidence
from src.scoring_input.builder import FIELD_LIST_SHA256, load_pinned_field_list
from src.summary_scoring.assets import load_field_list
from src.summary_scoring.completeness import algorithm_missing

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores",
    "purple", "acne", "wrinkle", "surface_gloss", "vascular", "contour_firmness",
)
FAST_FIVE = {"redness", "spots", "brown", "texture", "pores"}


@pytest.fixture(scope="module")
def fixture_detector_results() -> dict[str, Any]:
    return json.loads((FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8"))[
        "detector_results"
    ]


def _runtime(algorithm: str, source: Mapping[str, Any]) -> dict[str, Any]:
    """把 fixture detector_results 拆成 worker 运行时形状。"""
    if algorithm in FAST_FIVE:
        metrics = dict(source[algorithm]["metrics"])
        medical = metrics.pop("medical_metrics_v2", None)
        return {"metrics": metrics, "medical_metrics_v2": medical}
    if algorithm == "purple":
        return {
            "full_metrics": {
                "uv_spots": source["uv_spots"]["metrics"]["uv_spots"],
                "porphyrin": source["porphyrin"]["metrics"]["porphyrin"],
            }
        }
    if algorithm in {"surface_gloss", "vascular", "contour_firmness"}:
        return {"full_metrics": source[algorithm]["metrics"]}
    if algorithm == "wrinkle":
        return {"summary": source["wrinkle"]["metrics"]}
    if algorithm == "acne":
        metrics = source["acne"]["metrics"]
        return {
            "summary": {
                "detection": metrics.get("detection"),
                "preprocess": metrics.get("preprocess"),
            }
        }
    raise AssertionError(algorithm)


def _build(algorithm: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    return build_scoring_input(
        algorithm_name=algorithm,
        detection_schema_version="9",
        detection_impl_version="3",
        report_id="report-1",
        detection_attempt_id="attempt-1",
        source_image_sha256="a" * 64,
        capture_profile="consumer",
        input_route="consumer_queue",
        runtime=runtime,
        quality={"quality_status": "PASS", "quality_flags": ["low_contrast"]},
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_builder_matches_frozen_field_list_projection(
    algorithm: str, fixture_detector_results: dict[str, Any]
) -> None:
    result = _build(algorithm, _runtime(algorithm, fixture_detector_results))

    assert result["schema_version"] == "scoring_input_v1"
    assert result["algorithm_name"] == algorithm
    assert result["detection_schema_version"] == "9"
    assert result["detection_impl_version"] == "3"
    assert result["report_id"] == "report-1"
    assert result["detection_attempt_id"] == "attempt-1"
    assert result["source_image_sha256"] == "a" * 64
    assert result["capture_profile"] == "consumer"
    assert result["input_route"] == "consumer_queue"
    assert result["evidence_field_list_sha256"] == FIELD_LIST_SHA256
    assert result["evidence_status"] == "present"
    assert result["missing_fields"] == []
    assert result["missing_reason"] is None
    assert result["quality"] == {
        "algorithm_quality_status": "PASS",
        "algorithm_quality_flags": ["low_contrast"],
        "input_quality_gate": None,
    }

    field_list = load_field_list()
    expected = extract_evidence(algorithm, fixture_detector_results, field_list)
    if algorithm in FAST_FIVE:
        from src.nine_analysis.metrics import extract_item
        expected[algorithm]["word_features"] = extract_item(
            algorithm, fixture_detector_results[algorithm]["metrics"])[1]
    assert result["evidence"] == expected
    assert not algorithm_missing(
        field_list, algorithm, result["evidence"], "consumer"
    )


def test_purple_splits_uv_spots_and_porphyrin(
    fixture_detector_results: dict[str, Any]
) -> None:
    result = _build("purple", _runtime("purple", fixture_detector_results))
    assert set(result["evidence"]) == {"uv_spots", "porphyrin"}
    assert result["evidence"]["uv_spots"]["metrics"]["uv_spots"]["总体指标"]
    assert result["evidence"]["porphyrin"]["metrics"]["porphyrin"]["总体指标"]


def test_acne_keeps_internal_fields_out_of_public_metrics(
    fixture_detector_results: dict[str, Any]
) -> None:
    result = _build("acne", _runtime("acne", fixture_detector_results))
    metrics = result["evidence"]["acne"]["metrics"]
    assert "detection" in metrics
    assert "preprocess" in metrics
    # scoring_input 是内部字段容器，不带任何公开 envelope 键。
    assert "raw_result" not in result
    assert "overlay" not in result


def test_missing_required_leaf_is_partial_not_zero(
    fixture_detector_results: dict[str, Any]
) -> None:
    runtime = _runtime("redness", fixture_detector_results)
    runtime["metrics"].pop("mean_redness")
    runtime["medical_metrics_v2"] = None
    result = _build("redness", runtime)

    assert result["evidence_status"] == "partial"
    assert "redness.metrics.mean_redness" in result["missing_fields"]
    assert result["missing_reason"] == "旁路失败"
    assert "mean_redness" not in result["evidence"]["redness"]["metrics"]


def test_medical_bypass_failure_marks_partial(
    fixture_detector_results: dict[str, Any]
) -> None:
    runtime = _runtime("pores", fixture_detector_results)
    runtime["medical_metrics_v2"] = None
    result = _build("pores", runtime)

    assert result["evidence_status"] == "partial"
    assert result["missing_reason"] == "旁路失败"
    assert any("medical_metrics_v2" in path for path in result["missing_fields"])


def test_absent_algorithm_is_missing_without_evidence() -> None:
    result = _build("purple", {"full_metrics": {}})
    assert result["evidence_status"] == "missing"
    assert result["evidence"] == {}
    assert result["missing_fields"] == []
    assert result["missing_reason"] == "算法证据缺失"


def test_wrinkle_flat_summary_including_2d_fallback_fields(
    fixture_detector_results: dict[str, Any]
) -> None:
    result = _build("wrinkle", _runtime("wrinkle", fixture_detector_results))
    metrics = result["evidence"]["wrinkle"]["metrics"]
    assert metrics["line_width"] == 1
    assert len(metrics["region_metrics"]) == 4
    assert metrics["stage2_recommended_ratio"] > 0


def test_field_list_sha_is_pinned() -> None:
    assert load_pinned_field_list().source_sha256 == FIELD_LIST_SHA256
