from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeAlias, TypedDict


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ProductionProxyScore(TypedDict):
    status: str
    scoring_profile_version: str
    score_status: str
    score_direction: str
    module_scores: dict[str, JsonValue]
    score_trace: dict[str, JsonValue]
    calibration_boundary: dict[str, JsonValue]


PROFILE_VERSION: Final = "production_proxy_v1"
FORMULA_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "calibration"
    / "v2_explicit_formula_registry_v1.json"
)
LINEAR_UNITS: Final = frozenset({"ratio_0_1", "relative_0_1"})
UNIT_SCALES: Final = {
    "count_per_100k_px": 100.0,
    "standardized_px2": 100.0,
    "delta_e_proxy": 20.0,
    "length_per_10k_px": 100.0,
    "standardized_px": 50.0,
    "count_per_10k_px": 10.0,
    "relative_2_5d": 1.0,
    "normalized_width": 1.0,
    "normalized_volume": 1.0,
    "normalized_volume_per_length": 1.0,
    "relative_curvature": 1.0,
    "normalized_volume_per_area": 1.0,
    "relative_slope": 1.0,
}
GRADE_THRESHOLDS: Final = (
    (20.0, "未见明显"),
    (40.0, "轻度"),
    (60.0, "中度"),
    (80.0, "较明显"),
    (100.0, "显著"),
)


def _clamp_score(value: float) -> float:
    return min(100.0, max(0.0, value))


def _metric_burden(metric: Mapping[str, JsonValue]) -> float | None:
    value = metric.get("value")
    if (
        metric.get("availability") == "unavailable"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    magnitude = abs(float(value))
    unit = str(metric["unit"])
    if unit in LINEAR_UNITS:
        normalized = _clamp_score(magnitude * 100.0)
    else:
        scale = UNIT_SCALES[unit]
        normalized = _clamp_score(100.0 * magnitude / (magnitude + scale))
    is_health_measure = metric.get("direction") == "higher_health"
    return 100.0 - normalized if is_health_measure else normalized


def _grade(score: float) -> str:
    return next(label for maximum, label in GRADE_THRESHOLDS if score <= maximum)


def _anchor_score(base_score: float, anchor_values: list[float]) -> float:
    lower = min(anchor_values)
    upper = max(anchor_values)
    if upper - lower <= 1e-12:
        return _clamp_score(base_score)
    return _clamp_score((base_score - lower) * 100.0 / (upper - lower))


def score_production_proxy(
    medical_v2_modules: Mapping[str, Mapping[str, JsonValue]],
) -> ProductionProxyScore:
    """Score the 11 V2 modules as bounded engineering proxies, never as norms."""
    formula_registry = json.loads(FORMULA_PATH.read_text(encoding="utf-8"))
    anchor_registry = formula_registry["module_score_anchors"]
    module_scores: dict[str, dict] = {}
    for module_id, module in medical_v2_modules.items():
        group_results: dict[str, dict] = {}
        weighted_total = 0.0
        available_weight = 0.0
        for group_id, group in module["groups"].items():
            metric_results: dict[str, dict] = {}
            weighted_metric_total = 0.0
            available_metric_weight = 0.0
            for metric_id, metric in group["metrics"].items():
                burden = _metric_burden(metric)
                if burden is None:
                    continue
                metric_weight = float(metric["metric_weight"])
                weighted_metric_total += burden * metric_weight
                available_metric_weight += metric_weight
                metric_results[metric_id] = {
                    "value": float(metric["value"]),
                    "unit": metric["unit"],
                    "direction": metric["direction"],
                    "burden_score": round(burden, 6),
                    "availability": metric["availability"],
                    "formula": metric.get("formula", "legacy_projection"),
                    "source_metrics": metric["source_metrics"],
                    "metric_weight": metric_weight,
                }
            if available_metric_weight <= 0.0:
                continue
            group_score = weighted_metric_total / available_metric_weight
            weight = float(group["weight"])
            contribution = group_score * weight
            weighted_total += contribution
            available_weight += weight
            group_results[group_id] = {
                "score": round(group_score, 6),
                "weight": weight,
                "base_score_contribution": round(contribution, 6),
                "metrics": metric_results,
            }
        base_score = weighted_total / available_weight if available_weight else 0.0
        if available_weight:
            for group_result in group_results.values():
                group_result["base_score_contribution"] = round(
                    float(group_result["base_score_contribution"]) / available_weight,
                    6,
                )
        anchors = anchor_registry[module_id]
        anchor_values = [float(value) for value in anchors["base_scores"].values()]
        score_valid = bool(module["completeness"]["complete"]) and available_weight > 0.0
        score = _anchor_score(base_score, anchor_values) if score_valid else None
        module_scores[module_id] = {
            "score": round(score, 4) if score is not None else None,
            "grade": _grade(score) if score is not None else "不可评估",
            "score_valid": score_valid,
            "base_engineering_score": round(base_score, 6) if score_valid else None,
            "available_group_weight": round(available_weight, 10),
            "groups": group_results,
            "contribution_basis": "deterministic_unit_mapping_before_anchor_normalization",
            "anchor_range": {
                "minimum": round(min(anchor_values), 6),
                "maximum": round(max(anchor_values), 6),
            },
        }
    return {
        "status": "uncalibrated",
        "scoring_profile_version": PROFILE_VERSION,
        "score_status": "production_proxy",
        "score_direction": "higher_is_more_visible_proxy_burden",
        "module_scores": module_scores,
        "score_trace": {
            "formula_version": formula_registry["formula_version"],
            "anchor_registry_sha256": hashlib.sha256(FORMULA_PATH.read_bytes()).hexdigest(),
            "anchor_ids": list(formula_registry["anchor_sources"]),
            "normalization": "three_anchor_min_max_then_clamp_0_100",
            "constant_anchor_fallback": "deterministic_unit_mapping",
        },
        "calibration_boundary": {
            "population_calibrated": False,
            "medical_threshold_validated": False,
            "reference_population_size": 0,
            "anchor_sample_count": len(formula_registry["anchor_sources"]),
            "statement": (
                "Scores and grades are single-RGB 2D/2.5D engineering proxies "
                "relative to three frozen acceptance images; they are not population "
                "norms, diagnoses, or physical 3D measurements."
            ),
        },
    }


def score_consumer_unavailable(
    medical_v2_modules: Mapping[str, Mapping[str, JsonValue]],
) -> ProductionProxyScore:
    """Legacy audit helper; the active local path uses production_proxy_v1."""
    module_scores: dict[str, JsonValue] = {
        module_id: {
            "score": None,
            "grade": "不可评估",
            "score_valid": False,
            "base_engineering_score": None,
            "available_group_weight": 0.0,
            "groups": {},
            "contribution_basis": "consumer_profile_not_calibrated",
            "anchor_range": None,
        }
        for module_id in medical_v2_modules
    }
    return {
        "status": "uncalibrated",
        "scoring_profile_version": "consumer_unavailable_v1",
        "score_status": "unavailable_consumer_profile",
        "score_direction": "higher_is_more_visible_proxy_burden",
        "module_scores": module_scores,
        "score_trace": {
            "normalization": "disabled_until_consumer_reference_distribution",
            "anchor_ids": [],
        },
        "calibration_boundary": {
            "population_calibrated": False,
            "medical_threshold_validated": False,
            "reference_population_size": 0,
            "anchor_sample_count": 0,
            "statement": (
                "Consumer RGB scores are unavailable until an independently "
                "validated consumer reference distribution is approved."
            ),
        },
    }


__all__ = ["score_consumer_unavailable", "score_production_proxy"]
