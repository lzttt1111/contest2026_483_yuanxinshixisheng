"""Equivalence: shared entry vs the current local formal scoring chain.

Both sides reuse the same underlying functions; the entry must introduce zero
divergence for full-evidence fixtures (zero tolerance).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.aisia_medical_report.word_scoring import (
    apply_word_scores,
    load_word_acne_2d_reference,
    load_word_population_profile,
    load_word_wrinkle_2d_references,
)
from src.nine_analysis.metrics import extract_item
from src.nine_analysis.production_proxy_scoring import score_production_proxy
from src.nine_analysis.v2_proxy_projection import build_v2_proxy_modules
from src.nine_analysis.zero_target_metrics import normalize_zero_target_detector_results
from src.scoring_calibration.v011.scoring import score_observation
from src.summary_scoring import load_default_assets, score_report_from_evidence

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
CASES = ("clinic28-09_r1", "clinic28-23_r1", "clinic28-25_r1", "clinic28-25_r3")
MODULES = {
    "01": "pores", "02": "oil_tendency", "03": "pigmentation",
    "04": "diffuse_redness", "05": "vascular", "06": "acne_activity",
    "07": "dry_fine_lines", "08": "stable_wrinkles", "09": "structural_grooves",
    "10": "smoothness", "11": "contour_firmness",
}
DIMENSIONS = {
    "visible_pores": "01", "combined_pigmentation": "03",
    "diffuse_redness": "04", "surface_smoothness_decline": "10",
}
V011_DETECTORS = (
    "redness", "spots", "brown", "texture", "pores",
    "uv_spots", "porphyrin", "wrinkle", "acne",
)


def _legacy_grade(score: float) -> str:
    for maximum, label in ((20, "未见明显"), (40, "轻度"), (60, "中度"), (80, "较明显")):
        if score <= maximum:
            return label
    return "显著"


def _reference(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    detector_results = normalize_zero_target_detector_results(dict(evidence))
    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "calibration"
         / "metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )
    medical_modules, scoring_features = build_v2_proxy_modules(detector_results, registry)
    production = score_production_proxy(medical_modules)
    features = {
        detector: extract_item(
            detector, (detector_results.get(detector) or {}).get("metrics")
        )[1]
        for detector in V011_DETECTORS
        if (detector_results.get(detector) or {}).get("metrics") is not None
    }
    assets = load_default_assets()
    scoring = score_observation(
        {
            "status": "success",
            "input_quality_gate": {"status": "PASS"},
            "quality": {},
            "features": features,
        },
        dict(assets.v011_references),
    )
    payload = {
        "报告信息": {},
        "检测模块": [
            {"模块编号": module_id, "综合得分": None, "程度等级": None}
            for module_id in MODULES
        ],
    }
    rows = {row["dimension_id"]: row for row in scoring["formal_dimension_scores"]}
    for module in payload["检测模块"]:
        dimension = next(
            (key for key, value in DIMENSIONS.items() if value == module["模块编号"]),
            None,
        )
        row = rows.get(dimension) if dimension else None
        if row and row["status"] == "formal" and isinstance(row["score"], (int, float)):
            module["综合得分"] = round(float(row["score"]), 2)
            module["程度等级"] = _legacy_grade(float(row["score"]))
    apply_word_scores(
        payload,
        {
            "detector_results": detector_results,
            "scoring_features": scoring_features,
            "provenance": {"capture_profile": "consumer"},
        },
        load_word_population_profile(capture_profile="consumer"),
        load_word_wrinkle_2d_references(),
        load_word_acne_2d_reference(),
    )
    word = {
        str(module["模块编号"]): (
            None if not module.get("score_valid") else module.get("综合得分"),
            module.get("程度等级") if module.get("score_valid") else "不可评估",
            bool(module.get("score_valid")),
        )
        for module in payload["检测模块"]
    }
    proxy = {
        module_id: (
            production["module_scores"][scoring_id]["score"],
            production["module_scores"][scoring_id]["grade"],
            production["module_scores"][scoring_id]["score_valid"],
        )
        for module_id, scoring_id in MODULES.items()
    }
    return {"word_display": word, "production_proxy_v1": proxy}


@pytest.mark.parametrize("case", CASES)
def test_shared_entry_matches_local_chain(case: str) -> None:
    evidence = json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))[
        "detector_results"
    ]
    reference = _reference(evidence)
    shared = score_report_from_evidence(
        evidence,
        capture_profile="consumer",
        input_route="consumer_rgb",
        quality={"quality_status": "PASS"},
        input_quality_gate={"status": "PASS"},
    )
    for module_id in MODULES:
        for system in ("word_display", "production_proxy_v1"):
            expected = reference[system][module_id]
            actual = shared["modules"][module_id][system]
            assert (actual["score"], actual["grade"], actual["score_valid"]) == expected, (
                case, module_id, system, expected, actual
            )
