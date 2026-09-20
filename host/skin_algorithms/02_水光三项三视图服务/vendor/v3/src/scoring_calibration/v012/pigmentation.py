from __future__ import annotations

import hashlib
import json
from typing import Any

from src.scoring_calibration.v011.registry import REGISTRY
from src.scoring_calibration.v011.scoring import empirical_percentile, transform_value

from .explanation import build_score_explanation


UV_METRICS = (
    ("uv_spot_density", "uv_spots.UV样色素范围.density", "log1p", "个/10万有效皮肤像素"),
    ("uv_spot_area_ratio", "uv_spots.UV样色素范围.area_ratio", "identity", "比例"),
    ("uv_p90_intensity", "uv_spots.UV样色素强度.p90_intensity", "identity", "0-1"),
    ("uv_high_intensity_target_ratio", "uv_spots.UV样色素强度.high_intensity_ratio", "identity", "比例"),
)


def _get_path(document: dict[str, Any], path: str) -> Any:
    node: Any = document
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _score_metric(metric_id: str, raw: float, transform: str, references: dict[str, list[float]], unit: str) -> dict[str, Any]:
    transformed = transform_value(float(raw), transform)
    reference = list(references.get(metric_id) or [])
    if raw == 0:
        score = 0.0
        detail = {"reference_count": len(reference), "mapping": "zero_burden"}
    else:
        if metric_id in {"uv_spot_density", "uv_spot_area_ratio"}:
            reference = [value for value in reference if value > 0]
        score, detail = empirical_percentile(transformed, reference)
    return {
        "metric_id": metric_id,
        "raw_value": float(raw),
        "transformed_value": transformed,
        "score": score,
        "unit": unit,
        "transform": transform,
        "metric_weight": .25,
        **detail,
    }


def _v011_pigment_dimension(features: dict[str, Any], references: dict[str, list[float]]) -> dict[str, Any]:
    dimension = next(item for item in REGISTRY if item.id == "combined_pigmentation")
    groups: list[dict[str, Any]] = []
    for group in dimension.groups:
        if group.id == "uv_spots":
            metrics = []
            for metric_id, source, transform, unit in UV_METRICS:
                raw = _get_path(features, source)
                if not isinstance(raw, (int, float)):
                    raise ValueError(f"综合色素shadow缺少UV指标: {metric_id}")
                metrics.append(_score_metric(metric_id, float(raw), transform, references, unit))
            group_score = sum(row["score"] * row["metric_weight"] for row in metrics)
            groups.append({
                "group_id": group.id,
                "group_name": "UV下更明显的色斑负担",
                "group_weight": group.weight,
                "score": group_score,
                "status": "scored",
                "metrics": metrics,
                "weight_source": "engineering_default_equal_v1",
            })
            continue
        metric_rows = []
        for metric in group.metrics:
            raw = _get_path(features, metric.source)
            if not isinstance(raw, (int, float)):
                raise ValueError(f"综合色素shadow缺少指标: {metric.id}")
            transformed = transform_value(float(raw), metric.transform)
            reference = list(references.get(metric.id) or [])
            if metric.zero_is_no_burden and float(raw) == 0:
                score = 0.0
                detail = {"reference_count": len(reference), "mapping": "zero_burden"}
            else:
                if metric.zero_inflated:
                    reference = [value for value in reference if value > 0]
                score, detail = empirical_percentile(transformed, reference, metric.direction)
            metric_rows.append({
                "metric_id": metric.id,
                "raw_value": float(raw),
                "transformed_value": transformed,
                "score": score,
                "unit": metric.unit,
                "transform": metric.transform,
                "metric_weight": metric.weight,
                **detail,
            })
        groups.append({
            "group_id": group.id,
            "group_name": group.name,
            "group_weight": group.weight,
            "score": sum(row["score"] * row["metric_weight"] for row in metric_rows),
            "status": "scored",
            "metrics": metric_rows,
        })
    score = sum(row["score"] * row["group_weight"] for row in groups)
    return {
        "dimension_id": "combined_pigmentation",
        "dimension_name": "综合色素问题",
        "status": "shadow",
        "score_status": "shadow_v012",
        "score": score,
        "groups": groups,
        "reference_scope": "full_history_global_ecdf",
        "formal_formula": "Visible 40% + UV 25% + Brown 35%",
        "red_weight": 0.0,
        "uv_internal_weight_source": "engineering_default_equal_v1",
        "uv_doctor_target_gap": {
            "requested_metric": "uv_high_intensity_area_ratio",
            "current_available_metric": "uv_high_intensity_target_ratio",
            "status": "not_substituted",
        },
    }


def score_combined_pigmentation_shadow(features: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    row = _v011_pigment_dimension(features, profile.get("references") or {})
    row["score_explanation"] = build_score_explanation(row)
    stable = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    row["trace_sha256"] = hashlib.sha256(stable.encode()).hexdigest()
    return row
