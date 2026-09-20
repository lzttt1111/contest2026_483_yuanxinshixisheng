from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.scoring_bridge.compatibility_models import (
    CompatibilityMethod,
    CompatibilityModel,
    CompatibilityRow,
    fit_compatibility_model,
    predict_compatibility,
)
from src.scoring_bridge.diagnostics_v2 import (
    MappingDiagnosticsV2,
    mapping_diagnostics_v2,
)
from src.scoring_bridge.split_contract import FoldAssignment


REGULARIZATION_GRID = (0.1, 1.0, 10.0, 100.0)
METHOD_ORDER = {
    CompatibilityMethod.IDENTITY: 0,
    CompatibilityMethod.SCALAR_PAVA: 1,
    CompatibilityMethod.FIXED_GROUP_PAVA: 2,
    CompatibilityMethod.ADDITIVE: 3,
}


@dataclass(frozen=True, slots=True)
class SubjectPrediction:
    subject_id: str
    predicted: float
    target: float


@dataclass(frozen=True, slots=True)
class NestedCompatibilitySelection:
    method: CompatibilityMethod
    regularization: float | None
    model: CompatibilityModel
    identity_diagnostics: MappingDiagnosticsV2
    selected_diagnostics: MappingDiagnosticsV2
    outer_predictions: tuple[SubjectPrediction, ...]


def _diagnostics(
    predictions: Sequence[SubjectPrediction],
    *,
    seed: int,
) -> MappingDiagnosticsV2:
    return mapping_diagnostics_v2(
        subject_ids=tuple(row.subject_id for row in predictions),
        target=tuple(row.target for row in predictions),
        predicted=tuple(row.predicted for row in predictions),
        bootstrap_repetitions=1,
        seed=seed,
    )


def _selection_key(
    diagnostics: MappingDiagnosticsV2,
    method: CompatibilityMethod,
) -> tuple[float, ...]:
    return (
        float(diagnostics.multi_grade_crossing_count),
        diagnostics.severe_reversal_rate if diagnostics.severe_reversal_rate is not None else 1.0,
        diagnostics.mean_absolute_error,
        diagnostics.absolute_error_p95,
        float(METHOD_ORDER[method]),
    )


def _not_worse(
    candidate: MappingDiagnosticsV2,
    identity: MappingDiagnosticsV2,
) -> bool:
    return bool(
        candidate.multi_grade_crossing_count <= identity.multi_grade_crossing_count
        and candidate.grade_crossing_count <= identity.grade_crossing_count
        and candidate.mean_absolute_error <= identity.mean_absolute_error + 1e-9
        and candidate.absolute_error_p95 <= identity.absolute_error_p95 + 1e-9
        and candidate.spearman + 0.01 >= identity.spearman
        and candidate.pair_concordance + 0.01 >= identity.pair_concordance
    )


def cross_validated_predictions(
    method: CompatibilityMethod,
    rows: Sequence[CompatibilityRow],
    *,
    assignments: Mapping[str, FoldAssignment],
    group_weights: Mapping[str, float],
    regularization: float | None = None,
) -> tuple[SubjectPrediction, ...]:
    predictions: list[SubjectPrediction] = []
    for outer_fold in range(5):
        training = tuple(
            row for row in rows
            if assignments[row.subject_id].outer_fold != outer_fold
        )
        validation = tuple(
            row for row in rows
            if assignments[row.subject_id].outer_fold == outer_fold
        )
        model = fit_compatibility_model(
            method,
            training,
            group_weights=group_weights,
            regularization=regularization,
        )
        predictions.extend(
            SubjectPrediction(
                subject_id=row.subject_id,
                predicted=predict_compatibility(model, row),
                target=row.legacy_score,
            )
            for row in validation
        )
    return tuple(sorted(predictions, key=lambda row: row.subject_id))


def _inner_additive_regularization(
    rows: Sequence[CompatibilityRow],
    *,
    outer_fold: int,
    assignments: Mapping[str, FoldAssignment],
    group_weights: Mapping[str, float],
    seed: int,
) -> float:
    training = tuple(
        row for row in rows
        if assignments[row.subject_id].outer_fold != outer_fold
    )
    evaluated: list[tuple[tuple[float, ...], float]] = []
    for regularization in REGULARIZATION_GRID:
        predictions: list[SubjectPrediction] = []
        for inner_fold in range(4):
            inner_training = tuple(
                row for row in training
                if assignments[row.subject_id].inner_fold_by_outer[outer_fold] != inner_fold
            )
            inner_validation = tuple(
                row for row in training
                if assignments[row.subject_id].inner_fold_by_outer[outer_fold] == inner_fold
            )
            model = fit_compatibility_model(
                CompatibilityMethod.ADDITIVE,
                inner_training,
                group_weights=group_weights,
                regularization=regularization,
            )
            predictions.extend(
                SubjectPrediction(
                    row.subject_id,
                    predict_compatibility(model, row),
                    row.legacy_score,
                )
                for row in inner_validation
            )
        diagnostics = _diagnostics(predictions, seed=seed + outer_fold)
        evaluated.append((
            _selection_key(diagnostics, CompatibilityMethod.ADDITIVE),
            regularization,
        ))
    return min(evaluated, key=lambda item: (item[0], item[1]))[1]


def _nested_additive_predictions(
    rows: Sequence[CompatibilityRow],
    *,
    assignments: Mapping[str, FoldAssignment],
    group_weights: Mapping[str, float],
    seed: int,
) -> tuple[tuple[SubjectPrediction, ...], float]:
    predictions: list[SubjectPrediction] = []
    selected_regularizations: list[float] = []
    for outer_fold in range(5):
        regularization = _inner_additive_regularization(
            rows,
            outer_fold=outer_fold,
            assignments=assignments,
            group_weights=group_weights,
            seed=seed,
        )
        selected_regularizations.append(regularization)
        training = tuple(
            row for row in rows
            if assignments[row.subject_id].outer_fold != outer_fold
        )
        validation = tuple(
            row for row in rows
            if assignments[row.subject_id].outer_fold == outer_fold
        )
        model = fit_compatibility_model(
            CompatibilityMethod.ADDITIVE,
            training,
            group_weights=group_weights,
            regularization=regularization,
        )
        predictions.extend(
            SubjectPrediction(row.subject_id, predict_compatibility(model, row), row.legacy_score)
            for row in validation
        )
    counts = Counter(selected_regularizations)
    final_regularization = min(
        counts,
        key=lambda value: (-counts[value], value),
    )
    return tuple(sorted(predictions, key=lambda row: row.subject_id)), final_regularization


def select_nested_compatibility(
    rows: Sequence[CompatibilityRow],
    *,
    assignments: Mapping[str, FoldAssignment],
    group_weights: Mapping[str, float],
    seed: int,
) -> NestedCompatibilitySelection:
    candidates: list[
        tuple[CompatibilityMethod, float | None, tuple[SubjectPrediction, ...], MappingDiagnosticsV2]
    ] = []
    for method in (
        CompatibilityMethod.IDENTITY,
        CompatibilityMethod.SCALAR_PAVA,
        CompatibilityMethod.FIXED_GROUP_PAVA,
    ):
        predictions = cross_validated_predictions(
            method,
            rows,
            assignments=assignments,
            group_weights=group_weights,
        )
        candidates.append((method, None, predictions, _diagnostics(predictions, seed=seed)))
    additive_predictions, regularization = _nested_additive_predictions(
        rows,
        assignments=assignments,
        group_weights=group_weights,
        seed=seed,
    )
    candidates.append((
        CompatibilityMethod.ADDITIVE,
        regularization,
        additive_predictions,
        _diagnostics(additive_predictions, seed=seed),
    ))
    identity = next(row for row in candidates if row[0] is CompatibilityMethod.IDENTITY)
    eligible = [row for row in candidates if _not_worse(row[3], identity[3])]
    selected = min(eligible, key=lambda row: _selection_key(row[3], row[0]))
    model = fit_compatibility_model(
        selected[0],
        rows,
        group_weights=group_weights,
        regularization=selected[1],
    )
    return NestedCompatibilitySelection(
        method=selected[0],
        regularization=selected[1],
        model=model,
        identity_diagnostics=identity[3],
        selected_diagnostics=selected[3],
        outer_predictions=selected[2],
    )


__all__ = [
    "NestedCompatibilitySelection",
    "SubjectPrediction",
    "cross_validated_predictions",
    "select_nested_compatibility",
]
