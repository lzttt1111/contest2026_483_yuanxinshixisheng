"""Word-display scoring chain reused verbatim from the existing report code.

The module builds the minimal ``complete``/``payload`` shapes that
``apply_word_scores`` consumes and never reimplements its numeric policy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.aisia_medical_report.word_scoring import apply_word_scores
from src.capture_profile import CaptureProfile
from src.nine_analysis.metrics import extract_item
from src.scoring_calibration.v011.scoring import score_observation

from .assets import Assets

MODULE_ORDER = (
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
)
DIMENSION_MODULE = {
    "visible_pores": "01",
    "combined_pigmentation": "03",
    "diffuse_redness": "04",
    "surface_smoothness_decline": "10",
}
V011_DETECTORS = (
    "redness", "spots", "brown", "texture", "pores",
    "uv_spots", "porphyrin", "wrinkle", "acne",
)
PROFILE_STATUS = {
    "历史参考分布评分": "v011_legacy",
    "二维可见纹路参考评分": "wrinkle_2d",
    "二维可见痤疮样特征参考评分": "acne_2d",
    "参考人群分布评分": "population_profile",
}


def _legacy_grade(score: float) -> str:
    if score <= 20:
        return "未见明显"
    if score <= 40:
        return "轻度"
    if score <= 60:
        return "中度"
    if score <= 80:
        return "较明显"
    return "显著"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def build_v011_features(detector_results: Mapping[str, Any], *, capture_profile: str = "consumer") -> dict[str, Any]:
    features: dict[str, Any] = {}
    for detector in V011_DETECTORS:
        node = detector_results.get(detector) or {}
        if isinstance(node.get("word_features"), dict):
            features[detector] = dict(node["word_features"])
            continue
        raw = node.get("metrics")
        if isinstance(node.get("public_metrics"), dict):
            from src.aisia_medical_report.legacy_word_features import local_legacy_features
            uv = raw.get("uv_spots", raw) if detector == "uv_spots" and isinstance(raw, dict) else None
            features[detector] = local_legacy_features(
                detector, node["public_metrics"], capture_profile=capture_profile, uv_document=uv)
            continue
        if isinstance(raw, dict):
            features[detector] = extract_item(detector, raw)[1]
    return features


def _apply_legacy_scores(payload: dict[str, Any], scoring: Mapping[str, Any]) -> None:
    rows = {
        row.get("dimension_id"): row
        for row in scoring.get("formal_dimension_scores", [])
        if isinstance(row, dict)
    }
    for module in payload["检测模块"]:
        dimension = next(
            (
                key for key, module_id in DIMENSION_MODULE.items()
                if module_id == module["模块编号"]
            ),
            None,
        )
        row = rows.get(dimension) if dimension else None
        score = (
            _finite(row.get("score"))
            if row and row.get("status") == "formal"
            else None
        )
        if score is not None:
            module["综合得分"] = round(score, 2)
            module["程度等级"] = _legacy_grade(score)
        else:
            module["综合得分"] = None
            module["程度等级"] = None


def _empty_payload() -> dict[str, Any]:
    return {
        "报告信息": {},
        "检测模块": [
            {"模块编号": module_id, "综合得分": None, "程度等级": None}
            for module_id in MODULE_ORDER
        ],
    }


def score_word_display(
    *,
    detector_results: Mapping[str, Any],
    scoring_features: Mapping[str, Any],
    capture_profile: str,
    input_quality_gate: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    assets: Assets,
) -> dict[str, Any]:
    profile = CaptureProfile(capture_profile)
    complete = {
        "detector_results": detector_results,
        "scoring_features": scoring_features,
        "provenance": {"capture_profile": capture_profile},
    }
    observation = {
        "status": "success",
        "input_quality_gate": input_quality_gate,
        "quality": quality or {},
        "features": build_v011_features(detector_results, capture_profile=capture_profile),
    }
    scoring = score_observation(
        observation,
        dict(assets.v011_references),
        scoring_profile_version=assets.v011_profile_version,
    )
    payload = _empty_payload()
    _apply_legacy_scores(payload, scoring)
    population = (
        assets.institution_profile
        if profile is CaptureProfile.INSTITUTION
        else assets.population_profile
    )
    wrinkle = (
        dict(assets.wrinkle_references)
        if assets.wrinkle_references is not None
        else None
    )
    apply_word_scores(
        payload,
        complete,
        population,
        wrinkle,
        assets.acne_reference,
    )
    return {
        "modules": _collect(payload),
        "v011_scoring": scoring,
    }


def _collect(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    for module in payload["检测模块"]:
        module_id = str(module["模块编号"])
        score = _finite(module.get("综合得分"))
        valid = bool(module.get("score_valid")) and score is not None
        collected[module_id] = {
            "score": score if valid else None,
            "grade": module.get("程度等级") if valid else "不可评估",
            "score_valid": valid,
            "score_status": (
                PROFILE_STATUS.get(str(module.get("评分Profile")), "unavailable")
                if valid
                else "unavailable"
            ),
        }
    return collected


__all__ = ["MODULE_ORDER", "build_v011_features", "score_word_display"]
