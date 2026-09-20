from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.scoring_bridge.compatibility_models import (
    CompatibilityRow,
    GroupScore,
    predict_compatibility,
)
from src.scoring_bridge.compatibility_serialization import (
    compatibility_model_from_document,
)


@dataclass(frozen=True, slots=True)
class HybridLegacyDimensionInput:
    dimension_id: str
    current_v011_score: float
    current_groups: tuple[GroupScore, ...]


@dataclass(frozen=True, slots=True)
class HybridLegacyDimensionScore:
    dimension_id: str
    score: float
    score_valid: bool
    score_source: str


def _fallback(value: HybridLegacyDimensionInput) -> HybridLegacyDimensionScore:
    return HybridLegacyDimensionScore(
        dimension_id=value.dimension_id,
        score=float(value.current_v011_score),
        score_valid=True,
        score_source="v011_fallback",
    )


def score_hybrid_legacy_dimensions(
    *,
    profile: Mapping[str, Any],
    dimensions: Mapping[str, HybridLegacyDimensionInput],
) -> Mapping[str, HybridLegacyDimensionScore]:
    if profile.get("status") != "promoted":
        raise ValueError("hybrid runtime requires a promoted profile")
    routes = profile.get("legacy_dimensions")
    if not isinstance(routes, dict):
        raise ValueError("hybrid legacy routes are invalid")
    results: dict[str, HybridLegacyDimensionScore] = {}
    for dimension_id, value in dimensions.items():
        if value.dimension_id != dimension_id:
            raise ValueError(f"hybrid dimension input key mismatch: {dimension_id}")
        route = routes.get(dimension_id)
        if not isinstance(route, dict) or route.get("status") != "promoted":
            results[dimension_id] = _fallback(value)
            continue
        if route.get("route") != "compatibility_v2":
            results[dimension_id] = _fallback(value)
            continue
        model_payload = route.get("model")
        if not isinstance(model_payload, dict):
            results[dimension_id] = _fallback(value)
            continue
        model = compatibility_model_from_document(model_payload)
        row = CompatibilityRow(
            subject_id="runtime",
            current_score=float(value.current_v011_score),
            legacy_score=float(value.current_v011_score),
            current_groups=value.current_groups,
            legacy_groups=value.current_groups,
        )
        results[dimension_id] = HybridLegacyDimensionScore(
            dimension_id=dimension_id,
            score=predict_compatibility(model, row),
            score_valid=True,
            score_source="compatibility_v2",
        )
    return results


__all__ = [
    "HybridLegacyDimensionInput",
    "HybridLegacyDimensionScore",
    "score_hybrid_legacy_dimensions",
]
