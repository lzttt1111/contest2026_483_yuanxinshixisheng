from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.population_profile import (
    PopulationGroupProfile,
    PopulationMetricProfile,
    PopulationMetricSpec,
    PopulationModuleProfile,
    PopulationProfile,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class MetricSpecDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module_id: str
    group_id: str
    metric_id: str
    direction: str
    unit: str
    group_weight: float
    metric_weight: float


class MetricProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: MetricSpecDocument
    usable: bool
    unusable_reason: str | None
    availability_rate: float
    finite_count: int
    unique_count: int
    positive_count: int
    zero_inflated: bool
    effective_weight: float
    reference: tuple[float, ...]


class GroupProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    group_weight: float
    retained_metric_weight_ratio: float
    status: str
    metrics: dict[str, MetricProfileDocument]


class ModuleProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module_id: str
    status: str
    groups: dict[str, GroupProfileDocument]
    raw_score_reference: tuple[float, ...]


class PopulationProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    status: str
    development_count: int
    modules: dict[str, ModuleProfileDocument]


def population_profile_document(
    profile: PopulationProfile,
) -> Mapping[str, JsonValue]:
    document = PopulationProfileDocument.model_validate(asdict(profile))
    return document.model_dump(mode="json")


def _metric(profile: MetricProfileDocument) -> PopulationMetricProfile:
    spec = PopulationMetricSpec(**profile.spec.model_dump())
    return PopulationMetricProfile(
        spec=spec,
        usable=profile.usable,
        unusable_reason=profile.unusable_reason,
        availability_rate=profile.availability_rate,
        finite_count=profile.finite_count,
        unique_count=profile.unique_count,
        positive_count=profile.positive_count,
        zero_inflated=profile.zero_inflated,
        effective_weight=profile.effective_weight,
        reference=profile.reference,
    )


def population_profile_from_document(
    payload: Mapping[str, JsonValue],
) -> PopulationProfile:
    document = PopulationProfileDocument.model_validate(payload)
    modules = {
        module_id: PopulationModuleProfile(
            module_id=module.module_id,
            status=module.status,
            groups={
                group_id: PopulationGroupProfile(
                    group_id=group.group_id,
                    group_weight=group.group_weight,
                    retained_metric_weight_ratio=group.retained_metric_weight_ratio,
                    status=group.status,
                    metrics={
                        metric_id: _metric(metric)
                        for metric_id, metric in group.metrics.items()
                    },
                )
                for group_id, group in module.groups.items()
            },
            raw_score_reference=module.raw_score_reference,
        )
        for module_id, module in document.modules.items()
    }
    return PopulationProfile(
        profile_id=document.profile_id,
        status=document.status,
        development_count=document.development_count,
        modules=modules,
    )


__all__ = [
    "population_profile_document",
    "population_profile_from_document",
]
