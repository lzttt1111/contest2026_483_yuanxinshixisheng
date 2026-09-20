from __future__ import annotations

import bisect
import hashlib
import json
import math
from typing import Any

from .quality import evaluate_quality
from .registry import DISPLAY_ONLY_DIMENSIONS, REGISTRY, MetricSpec, registry_document


def transform_value(value: float, transform: str) -> float:
    if not math.isfinite(value):
        raise ValueError("指标值不是有限数")
    if transform == "identity":
        return value
    if transform == "log1p":
        if value < 0:
            raise ValueError("log1p指标不得为负数")
        return math.log1p(value)
    raise ValueError(f"未知变换: {transform}")


def empirical_percentile(value: float, reference: list[float], direction: str = "high") -> tuple[float, dict[str, Any]]:
    """完整经验CDF；并列值使用中位秩，不截断到0.5/99.5。"""
    if not reference:
        raise ValueError("参考分布为空")
    values = sorted(float(item) for item in reference if math.isfinite(float(item)))
    if not values:
        raise ValueError("参考分布没有有限值")
    left = bisect.bisect_left(values, value)
    right = bisect.bisect_right(values, value)
    midrank = (left + right) / 2.0
    percentile = 100.0 * midrank / len(values)
    if direction == "low":
        percentile = 100.0 - percentile
    return percentile, {
        "reference_count": len(values),
        "cdf_left_count": left,
        "cdf_right_count": right,
        "tie_count": right - left,
        "midrank": midrank,
    }


def _metric_score(metric: MetricSpec, raw_value: float, references: dict[str, list[float]]) -> tuple[float, dict[str, Any]]:
    transformed = transform_value(float(raw_value), metric.transform)
    reference = list(references.get(metric.id) or [])
    if metric.zero_is_no_burden and float(raw_value) == 0:
        return 0.0, {
            "raw_value": float(raw_value), "transformed_value": transformed,
            "score": 0.0, "mapping": "zero_burden", "reference_count": len(reference),
        }
    if metric.zero_inflated:
        reference = [item for item in reference if item > 0]
        mapping = "positive_only_empirical_cdf"
    else:
        mapping = "full_empirical_cdf"
    score, detail = empirical_percentile(transformed, reference, metric.direction)
    return score, {
        "raw_value": float(raw_value), "transformed_value": transformed,
        "score": score, "mapping": mapping, "direction": metric.direction,
        "transform": metric.transform, "unit": metric.unit, **detail,
    }


def _get_path(features: dict[str, Any], path: str) -> Any:
    node: Any = features
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def score_observation(
    observation: dict[str, Any],
    references: dict[str, list[float]],
    *,
    include_legacy_shadow: bool = False,
    normalization_profile_sha256: str | None = None,
    scoring_profile_version: str | None = None,
) -> dict[str, Any]:
    config = registry_document()
    gate = evaluate_quality(observation)
    features = observation.get("features") or observation.get("scoring_features") or {}
    dimensions: list[dict[str, Any]] = []
    formal_allowed = gate["status"] == "PASS"

    for dimension in REGISTRY:
        groups: list[dict[str, Any]] = []
        missing_groups: list[str] = []
        for group in dimension.groups:
            metric_rows: list[dict[str, Any]] = []
            missing_metrics: list[str] = []
            for metric in group.metrics:
                raw = _get_path(features, metric.source)
                count = _get_path(features, metric.source.split(".")[0] + ".instance_count")
                if metric.minimum_instances and count is not None and int(count) < metric.minimum_instances:
                    missing_metrics.append(metric.id)
                    continue
                if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    missing_metrics.append(metric.id)
                    continue
                try:
                    score, detail = _metric_score(metric, float(raw), references)
                except ValueError:
                    missing_metrics.append(metric.id)
                    continue
                metric_rows.append({
                    "metric_id": metric.id, "canonical_source": metric.source,
                    "metric_weight": metric.weight, **detail,
                })
            if missing_metrics:
                missing_groups.append(group.id)
                groups.append({
                    "group_id": group.id, "group_name": group.name, "group_weight": group.weight,
                    "status": "insufficient_evidence", "missing_metrics": missing_metrics,
                    "metrics": metric_rows, "score": None,
                })
                continue
            group_score = sum(row["score"] * row["metric_weight"] for row in metric_rows)
            groups.append({
                "group_id": group.id, "group_name": group.name, "group_weight": group.weight,
                "status": "scored", "missing_metrics": [], "metrics": metric_rows,
                "score": group_score,
            })
        if not formal_allowed:
            status = "shadow_only" if gate["status"] == "REVIEW" else "not_scored"
            dimension_score = None
        elif missing_groups:
            status = "insufficient_evidence"
            dimension_score = None
        else:
            status = "formal"
            dimension_score = sum(row["score"] * row["group_weight"] for row in groups)
        dimensions.append({
            "dimension_id": dimension.id, "dimension_name": dimension.name,
            "status": status, "score": dimension_score, "groups": groups,
            "missing_required_groups": missing_groups,
        })

    result: dict[str, Any] = {
        "scoring_profile_version": scoring_profile_version or config["scoring_profile_version"],
        "calibration_status": config["status"],
        "score_direction": config["score_direction"],
        "quality_gate": gate,
        "formal_dimension_scores": dimensions,
        "display_only_dimensions": list(DISPLAY_ONLY_DIMENSIONS),
        "overall_score": None,
        "overall_grade": None,
        "registry_sha256": config["registry_sha256"],
        "normalization_profile_sha256": normalization_profile_sha256,
    }
    if include_legacy_shadow:
        result["legacy_shadow_score"] = observation.get("legacy_score")
    stable = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    result["trace_sha256"] = hashlib.sha256(stable.encode()).hexdigest()
    return result
