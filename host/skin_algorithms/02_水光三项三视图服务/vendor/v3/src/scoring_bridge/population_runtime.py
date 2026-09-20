from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.scoring_bridge.population_profile import (
    NumericFeatures,
    PopulationProfile,
    empirical_burden_percentile,
)


@dataclass(frozen=True, slots=True)
class PopulationModuleScore:
    module_id: str
    score: float | None
    grade: str
    score_valid: bool
    score_status: str


def _grade(score: float) -> str:
    if score <= 20.0:
        return "低"
    if score <= 40.0:
        return "偏低"
    if score <= 60.0:
        return "中"
    if score <= 80.0:
        return "偏高"
    return "高"


def score_population_modules(
    *,
    features: NumericFeatures,
    profile: PopulationProfile,
) -> Mapping[str, PopulationModuleScore]:
    results: dict[str, PopulationModuleScore] = {}
    for module_id, module in profile.modules.items():
        if module.status != "candidate":
            results[module_id] = PopulationModuleScore(
                module_id, None, "不可评估", False, "profile_blocked"
            )
            continue
        raw_score = 0.0
        missing = False
        for group in module.groups.values():
            group_score = 0.0
            for metric in group.metrics.values():
                if not metric.usable:
                    continue
                value = features.get(module_id, {}).get(group.group_id, {}).get(metric.spec.metric_id)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    missing = True
                    break
                group_score += empirical_burden_percentile(
                    float(value),
                    metric.reference,
                    direction=metric.spec.direction,
                    zero_inflated=metric.zero_inflated,
                ) * metric.effective_weight
            if missing:
                break
            raw_score += group_score * group.group_weight
        if missing:
            results[module_id] = PopulationModuleScore(
                module_id, None, "不可评估", False, "insufficient_evidence"
            )
            continue
        score = (
            0.0
            if raw_score == 0.0
            else empirical_burden_percentile(
                raw_score,
                module.raw_score_reference,
                direction="higher_burden",
                zero_inflated=False,
            )
        )
        results[module_id] = PopulationModuleScore(
            module_id=module_id,
            score=score,
            grade=_grade(score),
            score_valid=True,
            score_status="engineering_population_relative_burden",
        )
    return results


__all__ = ["PopulationModuleScore", "score_population_modules"]
