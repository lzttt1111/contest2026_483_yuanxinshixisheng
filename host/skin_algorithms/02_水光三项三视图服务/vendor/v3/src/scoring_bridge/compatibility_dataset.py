from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.compatibility_models import CompatibilityRow, GroupScore


class ScoreGroup(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    group_id: str
    status: str
    score: float | None


class ScoreDimension(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    dimension_id: str
    status: str
    score: float | None
    groups: tuple[ScoreGroup, ...]


class ScoreDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    formal_dimension_scores: tuple[ScoreDimension, ...]


def compatibility_rows_from_score_documents(
    *,
    subject_id: str,
    current: Mapping[str, object],
    legacy: Mapping[str, object],
) -> Mapping[str, CompatibilityRow]:
    current_document = ScoreDocument.model_validate(current)
    legacy_document = ScoreDocument.model_validate(legacy)
    legacy_by_id = {
        dimension.dimension_id: dimension
        for dimension in legacy_document.formal_dimension_scores
    }
    rows: dict[str, CompatibilityRow] = {}
    for current_dimension in current_document.formal_dimension_scores:
        legacy_dimension = legacy_by_id.get(current_dimension.dimension_id)
        if (
            legacy_dimension is None
            or current_dimension.status != "formal"
            or legacy_dimension.status != "formal"
            or current_dimension.score is None
            or legacy_dimension.score is None
        ):
            continue
        current_groups = tuple(
            GroupScore(group.group_id, group.score)
            for group in current_dimension.groups
            if group.status == "scored" and group.score is not None
        )
        legacy_groups = tuple(
            GroupScore(group.group_id, group.score)
            for group in legacy_dimension.groups
            if group.status == "scored" and group.score is not None
        )
        if {group.group_id for group in current_groups} != {
            group.group_id for group in legacy_groups
        }:
            continue
        rows[current_dimension.dimension_id] = CompatibilityRow(
            subject_id=subject_id,
            current_score=current_dimension.score,
            legacy_score=legacy_dimension.score,
            current_groups=current_groups,
            legacy_groups=legacy_groups,
        )
    return rows


__all__ = ["compatibility_rows_from_score_documents"]
