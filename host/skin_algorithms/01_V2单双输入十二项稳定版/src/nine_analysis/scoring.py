from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


QUANTILE_NODES = ("q005", "q10", "q25", "q50", "q75", "q90", "q95", "q995")
QUANTILE_PCTS = (0.5, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.5)


def _transform(value: float, profile: dict[str, Any]) -> float:
    return math.log1p(max(0.0, value)) if profile.get("transform") == "log1p" else value


def _percentile(value: float, profile: dict[str, Any]) -> float | None:
    if profile.get("usable") is False:
        return None
    value = _transform(value, profile)
    points = [
        (float(profile[key]), pct)
        for key, pct in zip(QUANTILE_NODES, QUANTILE_PCTS)
        if isinstance(profile.get(key), (int, float)) and math.isfinite(float(profile[key]))
    ]
    if len(points) < 2:
        return None
    points.sort()
    # 零方差指标没有排序能力，不能把所有受检者错误赋成同一百分位。
    if points[-1][0] - points[0][0] <= 1e-12:
        return None
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, p0), (x1, p1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            return p1 if x1 == x0 else p0 + (value - x0) * (p1 - p0) / (x1 - x0)
    return None


def score_features(
    features: dict[str, dict[str, dict[str, float]]],
    profile_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if profile.get("status") != "calibrated":
        return {"状态": "uncalibrated", "原因": profile.get("reason", "常模尚未达到标定样本量")}
    result: dict[str, Any] = {"状态": "candidate", "项目": {}}
    for item, groups in features.items():
        rule = registry.get("rules", {}).get(item, {})
        weights = {g["name"]: float(g["weight"]) for g in rule.get("groups", []) if g.get("available")}
        group_results: dict[str, Any] = {}
        weighted = weight_sum = 0.0
        for group, values in groups.items():
            percentiles = []
            for metric, value in values.items():
                metric_profile = profile.get("metrics", {}).get(item, {}).get(group, {}).get(metric)
                if metric_profile:
                    pct = _percentile(float(value), metric_profile)
                    if pct is not None:
                        percentiles.append({"指标": metric, "原始值": value, "负担百分位": round(pct, 4)})
            if not percentiles:
                continue
            group_score = sum(x["负担百分位"] for x in percentiles) / len(percentiles)
            weight = weights.get(group, 0.0)
            group_results[group] = {"候选分": round(group_score, 4), "指导权重": weight, "指标": percentiles}
            if weight > 0:
                weighted += group_score * weight
                weight_sum += weight
        item_score = weighted / weight_sum if weight_sum else None
        result["项目"][item] = {
            "项目候选分": round(item_score, 4) if item_score is not None else None,
            "分组": group_results,
            "说明": "AISIA参考人群相对负担候选分，不等同医学正常值或诊断。",
        }
    return result
