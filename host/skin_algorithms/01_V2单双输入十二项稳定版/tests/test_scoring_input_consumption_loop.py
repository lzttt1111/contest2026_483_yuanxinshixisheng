"""消费回路：builder 产物 → B2a 聚合入口，闭合 worker 与评分合同。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.scoring_input import build_scoring_input
from src.summary_scoring import score_report_from_evidence

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
CASES = ("clinic28-09_r1", "clinic28-23_r1", "clinic28-25_r1", "clinic28-25_r3")
ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores",
    "purple", "acne", "wrinkle", "surface_gloss", "vascular", "contour_firmness",
)
FAST_FIVE = {"redness", "spots", "brown", "texture", "pores"}
MODULES = (
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
)


def _runtime(algorithm: str, source: Mapping[str, Any]) -> dict[str, Any]:
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


def _evidence_from_builder(source: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        scoring_input = build_scoring_input(
            algorithm_name=algorithm,
            detection_schema_version="3",
            detection_impl_version="1",
            report_id="fixture-report",
            detection_attempt_id="fixture-attempt",
            source_image_sha256="f" * 64,
            capture_profile="consumer",
            input_route="consumer_queue",
            runtime=_runtime(algorithm, source),
        )
        assert scoring_input["evidence_status"] == "present", algorithm
        evidence.update(scoring_input["evidence"])
    return evidence


@pytest.mark.parametrize("case", CASES)
def test_builder_evidence_scores_equivalently_to_full_fixture(case: str) -> None:
    source = json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))[
        "detector_results"
    ]

    shared = score_report_from_evidence(
        _evidence_from_builder(source),
        capture_profile="consumer",
        input_route="consumer_queue",
        quality={"quality_status": "PASS"},
        input_quality_gate={"status": "PASS"},
    )
    reference = score_report_from_evidence(
        source,
        capture_profile="consumer",
        input_route="consumer_queue",
        quality={"quality_status": "PASS"},
        input_quality_gate={"status": "PASS"},
    )

    assert shared["completeness"]["complete"] is True
    assert set(shared["modules"]) == set(MODULES)
    for module_id in MODULES:
        for system in ("word_display", "production_proxy_v1"):
            actual = shared["modules"][module_id][system]
            expected = reference["modules"][module_id][system]
            assert (
                actual["score"], actual["grade"], actual["score_valid"],
            ) == (
                expected["score"], expected["grade"], expected["score_valid"],
            ), (case, module_id, system)
