from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.scoring_bridge.compatibility_models import (
    CompatibilityModel,
    CompatibilityRow,
    predict_compatibility,
)
from src.scoring_bridge.diagnostics_v2 import (
    MappingDiagnosticsV2,
    PromotionGateDecision,
    evaluate_promotion_gate_v2,
    mapping_diagnostics_v2,
)
from src.scoring_bridge.population_diagnostics import (
    PopulationGateDecision,
    evaluate_population_module_gate,
)
from src.scoring_bridge.population_profile import NumericFeatures, PopulationProfile
from src.scoring_bridge.population_runtime import score_population_modules


@dataclass(frozen=True, slots=True)
class CompatibilityConfirmationResult:
    dimension_id: str
    status: str
    attempted_count: int
    evaluable_count: int
    attrition_count: int
    identity_diagnostics: MappingDiagnosticsV2
    candidate_diagnostics: MappingDiagnosticsV2
    decision: PromotionGateDecision


@dataclass(frozen=True, slots=True)
class PopulationConfirmationResult:
    module_id: str
    status: str
    attempted_count: int
    evaluable_count: int
    attrition_count: int
    scored_subject_ids: tuple[str, ...]
    scores: tuple[float, ...]
    direction_perturbation_passed: bool
    decision: PopulationGateDecision


def _attempted_subjects(subject_ids: Sequence[str]) -> frozenset[str]:
    attempted = frozenset(subject_ids)
    if len(attempted) != len(subject_ids):
        raise ValueError("confirmation attempted subjects must be unique")
    if not attempted:
        raise ValueError("confirmation attempted subjects cannot be empty")
    return attempted


def evaluate_compatibility_confirmation(
    *,
    rows_by_dimension: Mapping[str, Sequence[CompatibilityRow]],
    models_by_dimension: Mapping[str, CompatibilityModel],
    attempted_subject_ids: Sequence[str],
    bootstrap_repetitions: int = 2000,
    seed: int = 20260827,
) -> Mapping[str, CompatibilityConfirmationResult]:
    attempted = _attempted_subjects(attempted_subject_ids)
    if set(rows_by_dimension) != set(models_by_dimension):
        raise ValueError("confirmation dimensions and frozen models do not match")
    results: dict[str, CompatibilityConfirmationResult] = {}
    for dimension_id, rows in sorted(rows_by_dimension.items()):
        subject_ids = tuple(row.subject_id for row in rows)
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError(f"duplicate confirmation subject in {dimension_id}")
        if not set(subject_ids).issubset(attempted):
            raise ValueError(f"unexpected confirmation subject in {dimension_id}")
        target = tuple(row.legacy_score for row in rows)
        identity = tuple(row.current_score for row in rows)
        candidate = tuple(
            predict_compatibility(models_by_dimension[dimension_id], row)
            for row in rows
        )
        identity_diagnostics = mapping_diagnostics_v2(
            subject_ids=subject_ids,
            target=target,
            predicted=identity,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
        )
        candidate_diagnostics = mapping_diagnostics_v2(
            subject_ids=subject_ids,
            target=target,
            predicted=candidate,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
        )
        decision = evaluate_promotion_gate_v2(
            candidate=candidate_diagnostics,
            identity=identity_diagnostics,
        )
        results[dimension_id] = CompatibilityConfirmationResult(
            dimension_id=dimension_id,
            status="promoted" if decision.passed else "blocked",
            attempted_count=len(attempted),
            evaluable_count=len(rows),
            attrition_count=len(attempted) - len(rows),
            identity_diagnostics=identity_diagnostics,
            candidate_diagnostics=candidate_diagnostics,
            decision=decision,
        )
    return results


def _quantile(reference: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(reference, dtype=np.float64), percentile))


def _direction_test(profile: PopulationProfile, module_id: str) -> bool:
    module = profile.modules[module_id]
    if module.status != "candidate":
        return False
    baseline: dict[str, dict[str, dict[str, float]]] = {module_id: {}}
    for group in module.groups.values():
        baseline[module_id][group.group_id] = {
            metric.spec.metric_id: _quantile(metric.reference, 50.0)
            for metric in group.metrics.values()
            if metric.usable
        }
    for group in module.groups.values():
        for metric in group.metrics.values():
            if not metric.usable:
                continue
            favorable = {
                key: {group_id: dict(values) for group_id, values in value.items()}
                for key, value in baseline.items()
            }
            adverse = {
                key: {group_id: dict(values) for group_id, values in value.items()}
                for key, value in baseline.items()
            }
            low = 0.0 if metric.zero_inflated else _quantile(metric.reference, 25.0)
            high = _quantile(metric.reference, 75.0)
            if metric.spec.direction == "higher_health":
                favorable_value, adverse_value = high, low
            else:
                favorable_value, adverse_value = low, high
            favorable[module_id][group.group_id][metric.spec.metric_id] = favorable_value
            adverse[module_id][group.group_id][metric.spec.metric_id] = adverse_value
            favorable_score = score_population_modules(
                features=favorable, profile=profile
            )[module_id].score
            adverse_score = score_population_modules(
                features=adverse, profile=profile
            )[module_id].score
            if (
                favorable_score is None
                or adverse_score is None
                or adverse_score + 1e-9 < favorable_score
            ):
                return False
    return True


def evaluate_population_confirmation(
    *,
    profile: PopulationProfile,
    features_by_subject: Mapping[str, NumericFeatures],
    attempted_subject_ids: Sequence[str],
) -> Mapping[str, PopulationConfirmationResult]:
    attempted = _attempted_subjects(attempted_subject_ids)
    if not set(features_by_subject).issubset(attempted):
        raise ValueError("population confirmation contains unexpected subjects")
    scored = {
        subject_id: score_population_modules(features=features, profile=profile)
        for subject_id, features in features_by_subject.items()
    }
    results: dict[str, PopulationConfirmationResult] = {}
    for module_id in sorted(profile.modules):
        scored_rows = tuple(
            (subject_id, float(module.score))
            for subject_id in attempted_subject_ids
            if (modules := scored.get(subject_id)) is not None
            and (module := modules[module_id]).score_valid
            and module.score is not None
        )
        scored_subject_ids = tuple(subject_id for subject_id, _ in scored_rows)
        scores = tuple(score for _, score in scored_rows)
        direction_passed = _direction_test(profile, module_id)
        decision = evaluate_population_module_gate(
            scores=scores,
            attempted_count=len(attempted),
            direction_perturbation_passed=direction_passed,
        )
        results[module_id] = PopulationConfirmationResult(
            module_id=module_id,
            status="promoted" if decision.passed else "blocked",
            attempted_count=len(attempted),
            evaluable_count=len(scores),
            attrition_count=len(attempted) - len(scores),
            scored_subject_ids=scored_subject_ids,
            scores=scores,
            direction_perturbation_passed=direction_passed,
            decision=decision,
        )
    return results


__all__ = [
    "CompatibilityConfirmationResult",
    "PopulationConfirmationResult",
    "evaluate_compatibility_confirmation",
    "evaluate_population_confirmation",
]
