"""scoring_input 体积守卫：单算法 < 256KB，11 算法合计 < 2MB。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.scoring_input import build_scoring_input

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores",
    "purple", "acne", "wrinkle", "surface_gloss", "vascular", "contour_firmness",
)
FAST_FIVE = {"redness", "spots", "brown", "texture", "pores"}
SINGLE_LIMIT_BYTES = 256 * 1024
TOTAL_LIMIT_BYTES = 2 * 1024 * 1024


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


@pytest.fixture(scope="module")
def sizes() -> dict[str, int]:
    source = json.loads((FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8"))[
        "detector_results"
    ]
    measured: dict[str, int] = {}
    for algorithm in ALGORITHMS:
        scoring_input = build_scoring_input(
            algorithm_name=algorithm,
            detection_schema_version="3",
            detection_impl_version="1",
            report_id="volume-report",
            detection_attempt_id="volume-attempt",
            source_image_sha256="a" * 64,
            capture_profile="consumer",
            input_route="consumer_queue",
            runtime=_runtime(algorithm, source),
        )
        assert scoring_input["evidence_status"] == "present"
        measured[algorithm] = len(
            json.dumps(scoring_input, ensure_ascii=False).encode("utf-8")
        )
    return measured


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_single_algorithm_scoring_input_under_limit(
    algorithm: str, sizes: dict[str, int]
) -> None:
    assert sizes[algorithm] < SINGLE_LIMIT_BYTES


def test_all_algorithm_scoring_inputs_under_total_limit(sizes: dict[str, int]) -> None:
    assert sum(sizes.values()) < TOTAL_LIMIT_BYTES
    assert len(sizes) == len(ALGORITHMS)
