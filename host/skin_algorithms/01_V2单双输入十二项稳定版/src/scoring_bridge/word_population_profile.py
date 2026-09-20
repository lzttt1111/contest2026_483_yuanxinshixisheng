from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from src.scoring_bridge.population_profile import (
    NumericFeatures,
    PopulationGroupProfile,
    PopulationModuleProfile,
    PopulationProfile,
    empirical_burden_percentile,
)


MetricAllowlist = Mapping[str, Mapping[str, frozenset[str]]]


def _raw_score(
    features: NumericFeatures,
    groups: Mapping[str, PopulationGroupProfile],
) -> float | None:
    module_score = 0.0
    for group in groups.values():
        group_score = 0.0
        for metric in group.metrics.values():
            if not metric.usable:
                continue
            value = (
                features.get(metric.spec.module_id, {})
                .get(group.group_id, {})
                .get(metric.spec.metric_id)
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return None
            group_score += empirical_burden_percentile(
                float(value),
                metric.reference,
                direction=metric.spec.direction,
                zero_inflated=metric.zero_inflated,
            ) * metric.effective_weight
        module_score += group_score * group.group_weight
    return module_score


def _retained_groups(
    module: PopulationModuleProfile,
    *,
    allowed_metrics: Mapping[str, frozenset[str]] | None,
) -> tuple[dict[str, PopulationGroupProfile], float]:
    retained: dict[str, PopulationGroupProfile] = {}
    for group_id, group in module.groups.items():
        if group.status != "candidate":
            continue
        if allowed_metrics is not None and group_id not in allowed_metrics:
            continue
        allowed = allowed_metrics.get(group_id) if allowed_metrics is not None else None
        metrics = {
            metric_id: metric
            for metric_id, metric in group.metrics.items()
            if metric.usable and (allowed is None or metric_id in allowed)
        }
        effective_weight = sum(metric.effective_weight for metric in metrics.values())
        if effective_weight <= 0.0:
            continue
        retained[group_id] = replace(
            group,
            metrics={
                metric_id: replace(
                    metric,
                    effective_weight=metric.effective_weight / effective_weight,
                )
                for metric_id, metric in metrics.items()
            },
        )
    retained_weight = sum(group.group_weight for group in retained.values())
    if retained_weight <= 0.0:
        return {}, 0.0
    return {
        group_id: replace(
            group,
            group_weight=group.group_weight / retained_weight,
        )
        for group_id, group in retained.items()
    }, retained_weight


def build_word_population_profile(
    *,
    strict_profile: PopulationProfile,
    observations: Sequence[NumericFeatures],
    module_ids: Sequence[str],
    minimum_retained_group_weight: float = 0.6,
    metric_allowlist: MetricAllowlist | None = None,
    profile_id: str = "word_population_reference_1000",
) -> PopulationProfile:
    """Build a Word-only reference profile with weights frozen at build time."""

    if not 0.0 < minimum_retained_group_weight <= 1.0:
        raise ValueError("minimum retained group weight must be within (0, 1]")
    modules: dict[str, PopulationModuleProfile] = {}
    minimum_observations = math.ceil(len(observations) * 0.95)
    for module_id in module_ids:
        strict_module = strict_profile.modules[module_id]
        groups, retained_weight = _retained_groups(
            strict_module,
            allowed_metrics=(
                metric_allowlist.get(module_id)
                if metric_allowlist is not None
                else None
            ),
        )
        raw_scores = tuple(
            score
            for observation in observations
            if (score := _raw_score(observation, groups)) is not None
        ) if retained_weight >= minimum_retained_group_weight else ()
        score_values = np.asarray(raw_scores, dtype=np.float64)
        status = "candidate" if (
            len(raw_scores) >= minimum_observations
            and np.unique(score_values).size >= 20
        ) else "blocked"
        modules[module_id] = PopulationModuleProfile(
            module_id=module_id,
            status=status,
            groups=groups if status == "candidate" else {},
            raw_score_reference=(
                tuple(sorted(float(value) for value in raw_scores))
                if status == "candidate"
                else ()
            ),
        )
    return PopulationProfile(
        profile_id=profile_id,
        status=(
            "candidate"
            if all(module.status == "candidate" for module in modules.values())
            else "blocked"
        ),
        development_count=len(observations),
        modules=modules,
    )


__all__ = ["build_word_population_profile"]
