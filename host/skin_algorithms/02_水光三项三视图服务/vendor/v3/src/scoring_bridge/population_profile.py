from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np


NumericFeatures = Mapping[
    str,
    Mapping[str, Mapping[str, int | float | None]],
]


@dataclass(frozen=True, slots=True)
class PopulationMetricSpec:
    module_id: str
    group_id: str
    metric_id: str
    direction: str
    unit: str
    group_weight: float
    metric_weight: float


@dataclass(frozen=True, slots=True)
class PopulationMetricProfile:
    spec: PopulationMetricSpec
    usable: bool
    unusable_reason: str | None
    availability_rate: float
    finite_count: int
    unique_count: int
    positive_count: int
    zero_inflated: bool
    effective_weight: float
    reference: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PopulationGroupProfile:
    group_id: str
    group_weight: float
    retained_metric_weight_ratio: float
    status: str
    metrics: Mapping[str, PopulationMetricProfile]


@dataclass(frozen=True, slots=True)
class PopulationModuleProfile:
    module_id: str
    status: str
    groups: Mapping[str, PopulationGroupProfile]
    raw_score_reference: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PopulationProfile:
    profile_id: str
    status: str
    development_count: int
    modules: Mapping[str, PopulationModuleProfile]


def empirical_burden_percentile(
    value: float,
    reference: Sequence[float],
    *,
    direction: str,
    zero_inflated: bool,
) -> float:
    if zero_inflated and value == 0.0:
        return 0.0
    values = np.asarray(reference, dtype=np.float64)
    if values.size == 0:
        raise ValueError("population reference is empty")
    left = int(np.searchsorted(values, value, side="left"))
    right = int(np.searchsorted(values, value, side="right"))
    score = 100.0 * (left + right) / 2.0 / values.size
    return 100.0 - score if direction == "higher_health" else score


def _metric_values(
    observations: Sequence[NumericFeatures],
    spec: PopulationMetricSpec,
) -> tuple[float, ...]:
    values: list[float] = []
    for observation in observations:
        value = (
            observation.get(spec.module_id, {})
            .get(spec.group_id, {})
            .get(spec.metric_id)
        )
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            values.append(float(value))
    return tuple(values)


def _metric_profile(
    observations: Sequence[NumericFeatures],
    spec: PopulationMetricSpec,
) -> PopulationMetricProfile:
    values = _metric_values(observations, spec)
    data = np.asarray(values, dtype=np.float64)
    availability = len(values) / len(observations) if observations else 0.0
    unique_count = int(np.unique(data).size) if data.size else 0
    positive_count = int(np.count_nonzero(data > 0))
    zero_count = int(np.count_nonzero(data == 0))
    zero_inflated = zero_count > 0
    reason: str | None = None
    if availability < 0.98:
        reason = "insufficient_availability"
    elif len(values) < 950:
        reason = "insufficient_finite_count"
    elif unique_count < 20:
        reason = "insufficient_unique_values"
    elif zero_inflated and positive_count < 30:
        reason = "insufficient_positive_values"
    elif float(np.percentile(data, 95) - np.percentile(data, 5)) <= 1e-9:
        reason = "insufficient_robust_spread"
    reference = data[data > 0] if zero_inflated else data
    return PopulationMetricProfile(
        spec=spec,
        usable=reason is None,
        unusable_reason=reason,
        availability_rate=availability,
        finite_count=len(values),
        unique_count=unique_count,
        positive_count=positive_count,
        zero_inflated=zero_inflated,
        effective_weight=0.0,
        reference=tuple(sorted(float(value) for value in reference)),
    )


def _module_raw_score(
    features: NumericFeatures,
    groups: Mapping[str, PopulationGroupProfile],
) -> float | None:
    module_score = 0.0
    for group in groups.values():
        if group.status != "candidate":
            return None
        group_score = 0.0
        for metric in group.metrics.values():
            if not metric.usable:
                continue
            value = features.get(metric.spec.module_id, {}).get(group.group_id, {}).get(metric.spec.metric_id)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            group_score += empirical_burden_percentile(
                float(value),
                metric.reference,
                direction=metric.spec.direction,
                zero_inflated=metric.zero_inflated,
            ) * metric.effective_weight
        module_score += group_score * group.group_weight
    return module_score


def build_population_profile(
    *,
    observations: Sequence[NumericFeatures],
    metric_specs: Sequence[PopulationMetricSpec],
    module_ids: Sequence[str],
) -> PopulationProfile:
    metrics_by_group: dict[tuple[str, str], list[PopulationMetricProfile]] = defaultdict(list)
    group_weights: dict[tuple[str, str], float] = {}
    for spec in metric_specs:
        key = (spec.module_id, spec.group_id)
        metrics_by_group[key].append(_metric_profile(observations, spec))
        group_weights[key] = spec.group_weight
    modules: dict[str, PopulationModuleProfile] = {}
    for module_id in module_ids:
        groups: dict[str, PopulationGroupProfile] = {}
        for (metric_module, group_id), metric_rows in sorted(metrics_by_group.items()):
            if metric_module != module_id:
                continue
            total_weight = sum(row.spec.metric_weight for row in metric_rows)
            usable_weight = sum(
                row.spec.metric_weight for row in metric_rows if row.usable
            )
            retained_ratio = usable_weight / total_weight if total_weight else 0.0
            status = "candidate" if retained_ratio >= 0.70 else "blocked"
            metrics = {
                row.spec.metric_id: replace(
                    row,
                    effective_weight=(
                        row.spec.metric_weight / usable_weight
                        if row.usable and usable_weight
                        else 0.0
                    ),
                )
                for row in metric_rows
            }
            groups[group_id] = PopulationGroupProfile(
                group_id=group_id,
                group_weight=group_weights[(module_id, group_id)],
                retained_metric_weight_ratio=retained_ratio,
                status=status,
                metrics=metrics,
            )
        module_status = (
            "candidate"
            if groups and all(group.status == "candidate" for group in groups.values())
            else "blocked"
        )
        raw_scores = tuple(
            score
            for observation in observations
            if (score := _module_raw_score(observation, groups)) is not None
        ) if module_status == "candidate" else ()
        modules[module_id] = PopulationModuleProfile(
            module_id=module_id,
            status=module_status,
            groups=groups,
            raw_score_reference=tuple(sorted(raw_scores)),
        )
    return PopulationProfile(
        profile_id="population_ecdf_hybrid_v1_candidate",
        status=(
            "candidate"
            if modules and any(module.status == "candidate" for module in modules.values())
            else "blocked"
        ),
        development_count=len(observations),
        modules=modules,
    )


__all__ = [
    "NumericFeatures",
    "PopulationGroupProfile",
    "PopulationMetricProfile",
    "PopulationMetricSpec",
    "PopulationModuleProfile",
    "PopulationProfile",
    "build_population_profile",
    "empirical_burden_percentile",
]
