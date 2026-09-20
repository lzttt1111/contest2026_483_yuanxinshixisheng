from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
GRADE_THRESHOLDS = (20.0, 40.0, 60.0, 80.0)


def fit_isotonic(source: Iterable[float], target: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(tuple(source), dtype=np.float64)
    y = np.asarray(tuple(target), dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("单调映射至少需要两个成对一维样本")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        raise ValueError("单调映射有效样本不足")
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
    return {
        "method": "weighted_pava_linear_interpolation_v1",
        "sample_count": int(x.size),
        "x": [round(block[0], 10) for block in blocks],
        "y": [round(float(np.clip(block[1], 0.0, 100.0)), 10) for block in blocks],
        "weights": [int(block[2]) for block in blocks],
    }


def apply_isotonic(mapping: dict[str, Any], value: float) -> float:
    x = np.asarray(mapping["x"], dtype=np.float64)
    y = np.asarray(mapping["y"], dtype=np.float64)
    if x.size == 1:
        return float(y[0])
    return float(np.interp(float(value), x, y, left=y[0], right=y[-1]))


def _rank(values: np.ndarray) -> np.ndarray:
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
    if left.size < 2:
        return 0.0
    left_rank, right_rank = _rank(left), _rank(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def mapping_diagnostics(target: Iterable[float], predicted: Iterable[float]) -> dict[str, Any]:
    actual = np.asarray(tuple(target), dtype=np.float64)
    estimate = np.asarray(tuple(predicted), dtype=np.float64)
    if actual.shape != estimate.shape or actual.ndim != 1 or actual.size == 0:
        raise ValueError("映射验收要求等长非空一维数组")
    errors = np.abs(actual - estimate)
    concordant = 0.0
    comparable = 0
    severe_reversals = 0
    actual_grades = np.searchsorted(GRADE_THRESHOLDS, actual, side="right")
    estimate_grades = np.searchsorted(GRADE_THRESHOLDS, estimate, side="right")
    for left in range(actual.size):
        for right in range(left + 1, actual.size):
            actual_delta = actual[right] - actual[left]
            if abs(actual_delta) <= 1e-12:
                continue
            predicted_delta = estimate[right] - estimate[left]
            comparable += 1
            concordant += 1.0 if actual_delta * predicted_delta > 0 else 0.5 if predicted_delta == 0 else 0.0
            if abs(actual_grades[right] - actual_grades[left]) >= 2 and actual_delta * predicted_delta < 0:
                severe_reversals += 1
    grade_crossings = np.abs(actual_grades - estimate_grades)
    return {
        "count": int(actual.size),
        "spearman": round(_spearman(actual, estimate), 8),
        "pair_concordance": round(concordant / comparable, 8) if comparable else 1.0,
        "mean_absolute_error": round(float(np.mean(errors)), 8),
        "absolute_error_p50": round(float(np.percentile(errors, 50)), 8),
        "absolute_error_p90": round(float(np.percentile(errors, 90)), 8),
        "absolute_error_p95": round(float(np.percentile(errors, 95)), 8),
        "grade_crossing_count": int(np.count_nonzero(grade_crossings)),
        "multi_grade_crossing_count": int(np.count_nonzero(grade_crossings >= 2)),
        "severe_reversal_count": severe_reversals,
    }


def _numeric_leaves(node: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(value, path)
    elif not isinstance(node, bool) and isinstance(node, (int, float)) and math.isfinite(float(node)):
        yield prefix, float(node)


def build_ecdf_profile(observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = tuple(observations)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation in rows:
        for module, payload in observation.items():
            for path, value in _numeric_leaves(payload):
                values[(str(module), path)].append(value)
    result: dict[str, Any] = {}
    total = len(rows)
    for (module, path), raw in sorted(values.items()):
        data = np.asarray(raw, dtype=np.float64)
        summary = {
            "count": int(data.size),
            "min": float(data.min()),
            **{f"p{quantile}": float(np.percentile(data, quantile)) for quantile in QUANTILES},
            "max": float(data.max()),
            "zero_rate": float(np.mean(data == 0.0)),
            "unevaluable_rate": float((total - data.size) / total) if total else 1.0,
        }
        result.setdefault(module, {})[path] = summary
    return result


__all__ = [
    "apply_isotonic",
    "build_ecdf_profile",
    "fit_isotonic",
    "mapping_diagnostics",
]
