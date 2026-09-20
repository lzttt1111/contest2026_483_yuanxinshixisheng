from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


ACNE_2D_WEIGHTS = (0.95, 0.02, 0.02, 0.01)


@dataclass(frozen=True, slots=True)
class Acne2DMetrics:
    candidate_count: float
    density_per_100k_px: float
    candidate_area_ratio: float
    p90_confidence: float


@dataclass(frozen=True, slots=True)
class Acne2DReferences:
    candidate_count: tuple[float, ...]
    density_per_100k_px: tuple[float, ...]
    candidate_area_ratio: tuple[float, ...]
    p90_confidence: tuple[float, ...]


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def extract_acne_2d_metrics(acne_result: Mapping[str, Any]) -> Acne2DMetrics:
    metrics = _dictionary(acne_result.get("metrics"))
    detection = _dictionary(metrics.get("detection"))
    detections = detection.get("detections")
    rows = detections if isinstance(detections, list) else []
    count = _number(detection.get("count"))
    public = _dictionary(acne_result.get("public_metrics"))
    overall = _dictionary(public.get("overall_metrics"))
    scope = _dictionary(_dictionary(overall.get("auxiliary_metrics")).get("analysis_scope"))
    valid_area = _number(scope.get("valid_skin_area_px"))
    total_area = sum(_number(_dictionary(row).get("box_area")) for row in rows)
    confidence = tuple(_number(_dictionary(row).get("confidence")) for row in rows)
    return Acne2DMetrics(
        candidate_count=count,
        density_per_100k_px=(count / valid_area * 100_000.0 if valid_area > 0 else 0.0),
        candidate_area_ratio=(total_area / valid_area if valid_area > 0 else 0.0),
        p90_confidence=(float(np.percentile(confidence, 90)) if confidence else 0.0),
    )


def _percentile(value: float, reference: Sequence[float]) -> float:
    if value <= 0.0:
        return 0.0
    values = np.asarray(sorted(item for item in reference if item > 0.0), dtype=np.float64)
    if not values.size:
        raise ValueError("acne 2D reference is empty")
    left = int(np.searchsorted(values, value, side="left"))
    right = int(np.searchsorted(values, value, side="right"))
    return 100.0 * (left + right) / 2.0 / values.size


def score_acne_2d(metrics: Acne2DMetrics, references: Acne2DReferences) -> float:
    if metrics.candidate_count <= 0.0:
        return 0.0
    count_burden = 100.0 * (1.0 - math.exp(-metrics.candidate_count / 5.0))
    evidence = (
        count_burden,
        _percentile(metrics.density_per_100k_px, references.density_per_100k_px),
        _percentile(metrics.candidate_area_ratio, references.candidate_area_ratio),
        _percentile(metrics.p90_confidence, references.p90_confidence),
    )
    return float(sum(
        value * weight
        for value, weight in zip(evidence, ACNE_2D_WEIGHTS, strict=True)
    ))


def build_acne_2d_references(rows: Sequence[Acne2DMetrics]) -> Acne2DReferences:
    if len(rows) < 950:
        raise ValueError("acne 2D reference count is insufficient")
    return Acne2DReferences(*(
        tuple(sorted(getattr(row, field) for row in rows if getattr(row, field) > 0))
        for field in ("candidate_count", "density_per_100k_px", "candidate_area_ratio", "p90_confidence")
    ))


def acne_2d_reference_document(reference: Acne2DReferences) -> dict[str, Any]:
    return {
        "weights": dict(zip(
            ("candidate_count", "density_per_100k_px", "candidate_area_ratio", "p90_confidence"),
            ACNE_2D_WEIGHTS,
            strict=True,
        )),
        "reference": asdict(reference),
    }


def acne_2d_reference_from_document(document: Mapping[str, Any]) -> Acne2DReferences:
    values = document.get("reference")
    if not isinstance(values, dict):
        raise ValueError("acne 2D profile reference is missing")
    return Acne2DReferences(*(
        tuple(float(value) for value in values[field])
        for field in ("candidate_count", "density_per_100k_px", "candidate_area_ratio", "p90_confidence")
    ))


__all__ = [
    "Acne2DMetrics", "Acne2DReferences", "acne_2d_reference_document",
    "acne_2d_reference_from_document", "build_acne_2d_references",
    "extract_acne_2d_metrics", "score_acne_2d",
]
