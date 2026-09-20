from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any

from src.scoring_calibration.v011.registry import REGISTRY
from src.scoring_calibration.v011.scoring import transform_value


def features_from_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: value.get("评分输入", {})
        for key, value in payload["九项"].items()
    }


def _nested_value(features: dict[str, Any], path: str) -> Any:
    node: Any = features
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def prepare_references(references: dict[str, list[float]]) -> dict[str, tuple[float, ...]]:
    """Sort each full-history reference once for all candidate observations."""

    prepared: dict[str, tuple[float, ...]] = {}
    specs = {
        metric.id: metric
        for dimension in REGISTRY
        for group in dimension.groups
        for metric in group.metrics
    }
    for metric_id, spec in specs.items():
        values = (
            float(value)
            for value in references.get(metric_id, [])
            if math.isfinite(float(value))
        )
        if spec.zero_inflated:
            values = (value for value in values if value > 0)
        prepared[metric_id] = tuple(sorted(values))
    return prepared


def legacy_dimension_scores_from_features(
    features: dict[str, Any],
    references: dict[str, tuple[float, ...]],
) -> dict[str, float | None]:
    dimension_scores: dict[str, float | None] = {}
    for dimension in REGISTRY:
        group_scores: list[float] = []
        dimension_available = True
        for group in dimension.groups:
            metric_scores: list[float] = []
            for metric in group.metrics:
                raw = _nested_value(features, metric.source)
                count = _nested_value(
                    features,
                    metric.source.split(".")[0] + ".instance_count",
                )
                if metric.minimum_instances and count is not None and int(count) < metric.minimum_instances:
                    dimension_available = False
                    break
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    dimension_available = False
                    break
                raw_value = float(raw)
                if metric.zero_is_no_burden and raw_value == 0:
                    score = 0.0
                else:
                    values = references.get(metric.id, ())
                    if not values:
                        dimension_available = False
                        break
                    transformed = transform_value(raw_value, metric.transform)
                    left = bisect.bisect_left(values, transformed)
                    right = bisect.bisect_right(values, transformed)
                    score = 100.0 * (left + right) / 2.0 / len(values)
                    if metric.direction == "low":
                        score = 100.0 - score
                metric_scores.append(score * metric.weight)
            if not dimension_available:
                break
            group_scores.append(sum(metric_scores) * group.weight)
        dimension_scores[dimension.id] = (
            sum(group_scores) if dimension_available else None
        )
    return dimension_scores


def legacy_scores_from_features(
    features: dict[str, Any],
    references: dict[str, tuple[float, ...]],
) -> tuple[float, float, float, float] | None:
    scores = legacy_dimension_scores_from_features(features, references)
    ordered = [scores.get(dimension.id) for dimension in REGISTRY]
    if any(score is None for score in ordered):
        return None
    return tuple(float(score) for score in ordered)
