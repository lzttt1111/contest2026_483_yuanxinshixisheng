from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.compatibility_models import (
    CompatibilityMethod,
    CompatibilityModel,
    PavaMapping,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class PavaDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: tuple[float, ...]
    y: tuple[float, ...]


class ModelDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: CompatibilityMethod
    group_ids: tuple[str, ...]
    group_weights: tuple[float, ...]
    scalar_mapping: PavaDocument | None
    group_mappings: tuple[PavaDocument, ...]
    intercept: float
    regularization: float | None


def _pava_document(mapping: PavaMapping) -> PavaDocument:
    return PavaDocument(x=mapping.x, y=mapping.y)


def compatibility_model_document(
    model: CompatibilityModel,
) -> Mapping[str, JsonValue]:
    document = ModelDocument(
        method=model.method,
        group_ids=model.group_ids,
        group_weights=model.group_weights,
        scalar_mapping=(
            _pava_document(model.scalar_mapping)
            if model.scalar_mapping is not None
            else None
        ),
        group_mappings=tuple(
            _pava_document(mapping) for mapping in model.group_mappings
        ),
        intercept=model.intercept,
        regularization=model.regularization,
    )
    return document.model_dump(mode="json")


def compatibility_model_from_document(
    payload: Mapping[str, JsonValue],
) -> CompatibilityModel:
    document = ModelDocument.model_validate(payload)
    return CompatibilityModel(
        method=document.method,
        group_ids=document.group_ids,
        group_weights=document.group_weights,
        scalar_mapping=(
            PavaMapping(document.scalar_mapping.x, document.scalar_mapping.y)
            if document.scalar_mapping is not None
            else None
        ),
        group_mappings=tuple(
            PavaMapping(mapping.x, mapping.y)
            for mapping in document.group_mappings
        ),
        intercept=document.intercept,
        regularization=document.regularization,
    )


__all__ = [
    "compatibility_model_document",
    "compatibility_model_from_document",
]
