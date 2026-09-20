from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


WRINKLE_2D_WEIGHTS = (0.45, 0.35, 0.20)
REGION_NAMES = {
    "stable_wrinkles": frozenset({"额头纹", "眉间纹", "左鱼尾纹", "右鱼尾纹"}),
    "structural_grooves": frozenset({"左法令纹", "右法令纹", "左木偶纹", "右木偶纹"}),
}


@dataclass(frozen=True, slots=True)
class Wrinkle2DMetrics:
    segment_count: float
    total_length_px: float
    max_segment_length_px: float


@dataclass(frozen=True, slots=True)
class Wrinkle2DReferences:
    segment_count: tuple[float, ...]
    total_length_px: tuple[float, ...]
    max_segment_length_px: tuple[float, ...]


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def extract_wrinkle_2d_metrics(
    wrinkle_metrics: Mapping[str, Any],
    module_id: str,
) -> Wrinkle2DMetrics:
    names = REGION_NAMES[module_id]
    segment_count = 0.0
    total_length = 0.0
    max_length = 0.0
    rows = wrinkle_metrics.get("region_metrics")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        region_name = row.get("analysis_region") or row.get("region_name")
        if region_name not in names:
            continue
        core = _dictionary(_dictionary(row.get("core_metrics")).get("scope_and_morphology"))
        auxiliary = _dictionary(row.get("auxiliary_metrics"))
        counts = _dictionary(auxiliary.get("count_and_density"))
        scope = _dictionary(auxiliary.get("scope_and_morphology"))
        segment_count += _number(
            counts.get("wrinkle_segment_count", row.get("segment_count"))
        )
        total_length += _number(
            core.get("total_wrinkle_length_px", row.get("wrinkle_pixels"))
        )
        max_length = max(max_length, _number(
            scope.get("max_segment_length_px", row.get("max_segment_length"))
        ))
    return Wrinkle2DMetrics(segment_count, total_length, max_length)


def _percentile(value: float, reference: Sequence[float]) -> float:
    if value <= 0.0:
        return 0.0
    values = np.asarray(sorted(item for item in reference if item > 0.0), dtype=np.float64)
    if values.size == 0:
        raise ValueError("wrinkle 2D reference is empty")
    left = int(np.searchsorted(values, value, side="left"))
    right = int(np.searchsorted(values, value, side="right"))
    return 100.0 * (left + right) / 2.0 / values.size


def score_wrinkle_2d(
    metrics: Wrinkle2DMetrics,
    references: Wrinkle2DReferences,
) -> float:
    """Score visible 2D burden using count, total length, and longest segment."""

    if metrics.segment_count <= 0.0:
        return 0.0
    percentiles = (
        _percentile(metrics.segment_count, references.segment_count),
        _percentile(metrics.total_length_px, references.total_length_px),
        _percentile(metrics.max_segment_length_px, references.max_segment_length_px),
    )
    return float(sum(
        percentile * weight
        for percentile, weight in zip(percentiles, WRINKLE_2D_WEIGHTS, strict=True)
    ))


def build_wrinkle_2d_references(
    rows: Sequence[Wrinkle2DMetrics],
) -> Wrinkle2DReferences:
    if len(rows) < 950:
        raise ValueError("wrinkle 2D reference count is insufficient")
    return Wrinkle2DReferences(
        segment_count=tuple(sorted(row.segment_count for row in rows if row.segment_count > 0)),
        total_length_px=tuple(sorted(row.total_length_px for row in rows if row.total_length_px > 0)),
        max_segment_length_px=tuple(sorted(row.max_segment_length_px for row in rows if row.max_segment_length_px > 0)),
    )


def wrinkle_2d_reference_document(
    references: Mapping[str, Wrinkle2DReferences],
) -> dict[str, Any]:
    return {
        "weights": {
            "segment_count": WRINKLE_2D_WEIGHTS[0],
            "total_length_px": WRINKLE_2D_WEIGHTS[1],
            "max_segment_length_px": WRINKLE_2D_WEIGHTS[2],
        },
        "modules": {
            module_id: asdict(reference)
            for module_id, reference in references.items()
        },
    }


def wrinkle_2d_references_from_document(
    document: Mapping[str, Any],
) -> dict[str, Wrinkle2DReferences]:
    modules = document.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("wrinkle 2D profile modules are missing")
    return {
        module_id: Wrinkle2DReferences(
            segment_count=tuple(float(value) for value in values["segment_count"]),
            total_length_px=tuple(float(value) for value in values["total_length_px"]),
            max_segment_length_px=tuple(float(value) for value in values["max_segment_length_px"]),
        )
        for module_id, values in modules.items()
    }


__all__ = [
    "Wrinkle2DMetrics",
    "Wrinkle2DReferences",
    "build_wrinkle_2d_references",
    "extract_wrinkle_2d_metrics",
    "score_wrinkle_2d",
    "wrinkle_2d_reference_document",
    "wrinkle_2d_references_from_document",
]
