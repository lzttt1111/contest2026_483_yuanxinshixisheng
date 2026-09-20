from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np


GRADE_THRESHOLDS: Final = (20.0, 40.0, 60.0, 80.0)


@dataclass(frozen=True, slots=True)
class MappingDiagnosticsV2:
    count: int
    spearman: float
    pair_concordance: float
    mean_absolute_error: float
    absolute_error_p95: float
    grade_crossing_count: int
    multi_grade_crossing_count: int
    severe_comparable_pair_count: int
    severe_reversal_count: int
    severe_reversal_rate: float | None
    severe_reversal_rate_ci95_upper: float | None


@dataclass(frozen=True, slots=True)
class PromotionGateDecision:
    passed: bool
    failed_rules: tuple[str, ...]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    if left.size < 2 or np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pair_statistics(actual: np.ndarray, estimate: np.ndarray) -> tuple[int, int, float]:
    left, right = np.triu_indices(actual.size, 1)
    actual_delta = actual[right] - actual[left]
    predicted_delta = estimate[right] - estimate[left]
    comparable_mask = np.abs(actual_delta) > 1e-12
    comparable_delta = actual_delta[comparable_mask]
    comparable_predicted = predicted_delta[comparable_mask]
    concordance_values = np.where(
        comparable_delta * comparable_predicted > 0,
        1.0,
        np.where(comparable_predicted == 0, 0.5, 0.0),
    )
    concordance = (
        float(np.mean(concordance_values))
        if concordance_values.size
        else 1.0
    )
    actual_grades = np.searchsorted(GRADE_THRESHOLDS, actual, side="right")
    grade_delta = np.abs(actual_grades[right] - actual_grades[left])
    severe_mask = grade_delta >= 2
    severe_comparable = int(np.count_nonzero(severe_mask))
    severe_reversals = int(np.count_nonzero(
        severe_mask & (actual_delta * predicted_delta < 0)
    ))
    return severe_comparable, severe_reversals, concordance


def _bootstrap_upper(
    subject_ids: Sequence[str],
    actual: np.ndarray,
    estimate: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> float | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject_id in enumerate(subject_ids):
        grouped[subject_id].append(index)
    subjects = tuple(sorted(grouped))
    rng = np.random.default_rng(seed)
    rates: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.asarray(
            [index for subject in sampled for index in grouped[str(subject)]],
            dtype=np.int64,
        )
        comparable, reversals, _ = _pair_statistics(
            actual[indices],
            estimate[indices],
        )
        if comparable:
            rates.append(reversals / comparable)
    return float(np.percentile(rates, 95)) if rates else None


def mapping_diagnostics_v2(
    *,
    subject_ids: Sequence[str],
    target: Sequence[float],
    predicted: Sequence[float],
    bootstrap_repetitions: int = 2000,
    seed: int = 20260827,
) -> MappingDiagnosticsV2:
    actual = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if actual.shape != estimate.shape or actual.ndim != 1 or actual.size == 0:
        raise ValueError("diagnostics v2 requires equal non-empty vectors")
    if len(subject_ids) != actual.size:
        raise ValueError("subject id count does not match score count")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    errors = np.abs(actual - estimate)
    comparable, reversals, concordance = _pair_statistics(actual, estimate)
    actual_grades = np.searchsorted(GRADE_THRESHOLDS, actual, side="right")
    estimate_grades = np.searchsorted(GRADE_THRESHOLDS, estimate, side="right")
    grade_crossing = np.abs(actual_grades - estimate_grades)
    return MappingDiagnosticsV2(
        count=int(actual.size),
        spearman=_spearman(actual, estimate),
        pair_concordance=concordance,
        mean_absolute_error=float(np.mean(errors)),
        absolute_error_p95=float(np.percentile(errors, 95)),
        grade_crossing_count=int(np.count_nonzero(grade_crossing)),
        multi_grade_crossing_count=int(np.count_nonzero(grade_crossing >= 2)),
        severe_comparable_pair_count=comparable,
        severe_reversal_count=reversals,
        severe_reversal_rate=reversals / comparable if comparable else None,
        severe_reversal_rate_ci95_upper=_bootstrap_upper(
            subject_ids,
            actual,
            estimate,
            repetitions=bootstrap_repetitions,
            seed=seed,
        ) if comparable else None,
    )


def evaluate_promotion_gate_v2(
    *,
    candidate: MappingDiagnosticsV2,
    identity: MappingDiagnosticsV2,
) -> PromotionGateDecision:
    failed: list[str] = []
    checks = (
        (candidate.count >= 200, "count"),
        (candidate.mean_absolute_error <= 6.0, "mean_absolute_error"),
        (candidate.absolute_error_p95 <= 15.0, "absolute_error_p95"),
        (candidate.spearman >= 0.90, "spearman"),
        (
            candidate.severe_comparable_pair_count >= 1000,
            "severe_comparable_pair_count",
        ),
        (
            candidate.severe_reversal_rate_ci95_upper is not None
            and candidate.severe_reversal_rate_ci95_upper <= 0.01,
            "severe_reversal_rate_ci95_upper",
        ),
        (
            candidate.multi_grade_crossing_count == 0,
            "multi_grade_crossing_count",
        ),
        (
            candidate.grade_crossing_count <= identity.grade_crossing_count,
            "grade_crossing_count_vs_identity",
        ),
        (
            candidate.mean_absolute_error <= identity.mean_absolute_error,
            "mean_absolute_error_vs_identity",
        ),
        (
            candidate.absolute_error_p95 <= identity.absolute_error_p95,
            "absolute_error_p95_vs_identity",
        ),
        (
            candidate.spearman + 0.01 >= identity.spearman,
            "spearman_vs_identity",
        ),
        (
            candidate.pair_concordance + 0.01 >= identity.pair_concordance,
            "pair_concordance_vs_identity",
        ),
    )
    failed.extend(rule for passed, rule in checks if not passed)
    return PromotionGateDecision(passed=not failed, failed_rules=tuple(failed))


__all__ = [
    "MappingDiagnosticsV2",
    "PromotionGateDecision",
    "evaluate_promotion_gate_v2",
    "mapping_diagnostics_v2",
]
