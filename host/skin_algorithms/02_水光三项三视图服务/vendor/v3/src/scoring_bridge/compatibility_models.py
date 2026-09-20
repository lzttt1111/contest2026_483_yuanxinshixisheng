from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from typing_extensions import assert_never


class CompatibilityMethod(str, Enum):
    IDENTITY = "identity"
    SCALAR_PAVA = "scalar_pava"
    FIXED_GROUP_PAVA = "fixed_group_pava"
    ADDITIVE = "additive_compatibility"


@dataclass(frozen=True, slots=True)
class GroupScore:
    group_id: str
    score: float


@dataclass(frozen=True, slots=True)
class CompatibilityRow:
    subject_id: str
    current_score: float
    legacy_score: float
    current_groups: tuple[GroupScore, ...]
    legacy_groups: tuple[GroupScore, ...]


@dataclass(frozen=True, slots=True)
class PavaMapping:
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CompatibilityModel:
    method: CompatibilityMethod
    group_ids: tuple[str, ...]
    group_weights: tuple[float, ...]
    scalar_mapping: PavaMapping | None = None
    group_mappings: tuple[PavaMapping, ...] = ()
    intercept: float = 0.0
    regularization: float | None = None


@dataclass(frozen=True, slots=True)
class CompatibilityFitError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _fit_pava(source: Sequence[float], target: Sequence[float]) -> PavaMapping:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise CompatibilityFitError("PAVA requires equal vectors with at least two rows")
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    weights = np.bincount(inverse).astype(np.float64)
    means = np.bincount(inverse, weights=y) / weights
    blocks: list[list[float]] = [
        [float(value), float(mean), float(weight)]
        for value, mean, weight in zip(unique, means, weights, strict=True)
    ]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][1] <= blocks[index + 1][1]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[2] + right[2]
        blocks[index : index + 2] = [[
            (left[0] * left[2] + right[0] * right[2]) / weight,
            (left[1] * left[2] + right[1] * right[2]) / weight,
            weight,
        ]]
        index = max(0, index - 1)
    return PavaMapping(
        x=tuple(block[0] for block in blocks),
        y=tuple(float(np.clip(block[1], 0.0, 100.0)) for block in blocks),
    )


def _apply_pava(mapping: PavaMapping, value: float) -> float:
    return float(np.interp(
        float(value),
        np.asarray(mapping.x),
        np.asarray(mapping.y),
        left=mapping.y[0],
        right=mapping.y[-1],
    ))


def _group_vector(row: CompatibilityRow, group_ids: tuple[str, ...], *, legacy: bool) -> np.ndarray:
    source = row.legacy_groups if legacy else row.current_groups
    by_id = {group.group_id: group.score for group in source}
    if set(by_id) != set(group_ids):
        raise CompatibilityFitError("required group scores cannot be reconstructed")
    return np.asarray([by_id[group_id] for group_id in group_ids], dtype=np.float64)


def _validated_weights(
    group_weights: Mapping[str, float],
) -> tuple[tuple[str, ...], np.ndarray]:
    group_ids = tuple(sorted(group_weights))
    weights = np.asarray([group_weights[group_id] for group_id in group_ids], dtype=np.float64)
    if not group_ids or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise CompatibilityFitError("group weights must be nonnegative and sum to one")
    return group_ids, weights


def _fit_additive(
    rows: Sequence[CompatibilityRow],
    group_ids: tuple[str, ...],
    original_weights: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, float]:
    matrix = np.vstack([
        _group_vector(row, group_ids, legacy=False)
        for row in rows
    ])
    target = np.asarray([row.legacy_score for row in rows], dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        weights = parameters[:-1]
        predicted = np.clip(matrix @ weights + parameters[-1], 0.0, 100.0)
        residual = np.abs(predicted - target)
        huber = np.where(residual <= 10.0, 0.5 * residual**2, 10.0 * residual - 50.0)
        return float(np.mean(huber) + regularization * np.sum((weights - original_weights) ** 2))

    initial = np.concatenate((original_weights, np.asarray([0.0])))
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in group_ids) + ((None, None),),
        constraints=({"type": "eq", "fun": lambda values: float(values[:-1].sum() - 1.0)},),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise CompatibilityFitError(f"additive optimization failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def fit_compatibility_model(
    method: CompatibilityMethod,
    rows: Sequence[CompatibilityRow],
    *,
    group_weights: Mapping[str, float],
    regularization: float | None = None,
) -> CompatibilityModel:
    if len(rows) < 2:
        raise CompatibilityFitError("compatibility fitting requires at least two rows")
    group_ids, weights = _validated_weights(group_weights)
    match method:
        case CompatibilityMethod.IDENTITY:
            return CompatibilityModel(method, group_ids, tuple(weights))
        case CompatibilityMethod.SCALAR_PAVA:
            return CompatibilityModel(
                method,
                group_ids,
                tuple(weights),
                scalar_mapping=_fit_pava(
                    [row.current_score for row in rows],
                    [row.legacy_score for row in rows],
                ),
            )
        case CompatibilityMethod.FIXED_GROUP_PAVA:
            current = np.vstack([
                _group_vector(row, group_ids, legacy=False)
                for row in rows
            ])
            legacy = np.vstack([
                _group_vector(row, group_ids, legacy=True)
                for row in rows
            ])
            mappings = tuple(
                _fit_pava(current[:, index], legacy[:, index])
                for index in range(len(group_ids))
            )
            return CompatibilityModel(
                method,
                group_ids,
                tuple(weights),
                group_mappings=mappings,
            )
        case CompatibilityMethod.ADDITIVE:
            if regularization is None or regularization <= 0:
                raise CompatibilityFitError("additive regularization must be positive")
            fitted_weights, intercept = _fit_additive(
                rows,
                group_ids,
                weights,
                regularization,
            )
            return CompatibilityModel(
                method,
                group_ids,
                tuple(float(value) for value in fitted_weights),
                intercept=intercept,
                regularization=regularization,
            )
        case unreachable:
            assert_never(unreachable)


def predict_compatibility(model: CompatibilityModel, row: CompatibilityRow) -> float:
    match model.method:
        case CompatibilityMethod.IDENTITY:
            value = row.current_score
        case CompatibilityMethod.SCALAR_PAVA:
            if model.scalar_mapping is None:
                raise CompatibilityFitError("scalar PAVA mapping is missing")
            value = _apply_pava(model.scalar_mapping, row.current_score)
        case CompatibilityMethod.FIXED_GROUP_PAVA:
            current = _group_vector(row, model.group_ids, legacy=False)
            if len(model.group_mappings) != len(model.group_ids):
                raise CompatibilityFitError("group PAVA mappings are incomplete")
            mapped = np.asarray([
                _apply_pava(mapping, current[index])
                for index, mapping in enumerate(model.group_mappings)
            ])
            value = float(mapped @ np.asarray(model.group_weights))
        case CompatibilityMethod.ADDITIVE:
            current = _group_vector(row, model.group_ids, legacy=False)
            value = float(current @ np.asarray(model.group_weights) + model.intercept)
        case unreachable:
            assert_never(unreachable)
    return float(np.clip(value, 0.0, 100.0))


__all__ = [
    "CompatibilityFitError",
    "CompatibilityMethod",
    "CompatibilityModel",
    "CompatibilityRow",
    "GroupScore",
    "PavaMapping",
    "fit_compatibility_model",
    "predict_compatibility",
]
