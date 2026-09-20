from __future__ import annotations

"""Typed producer plans and the single canonical twelve-item join."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, TypedDict

from typing_extensions import NotRequired

from src.detection_runtime.contracts import RuntimeRoute, TWELVE_DETECTION_ITEMS


class DetectionItem(TypedDict):
    项目: str
    状态: str
    主结果图: NotRequired[str | None]
    量化JSON: NotRequired[str | None]
    量化CSV: NotRequired[str | None]
    医学V2CSV: NotRequired[str | None]
    附加结果图: NotRequired[list[str]]


CanonicalItems: TypeAlias = dict[str, DetectionItem]
ProducedItems: TypeAlias = Mapping["ItemProducer", Mapping[str, DetectionItem]]


class ItemProducer(str, Enum):
    DERMAVISION = "dermavision"
    ACNE = "acne"
    WRINKLE = "wrinkle"
    CLINIC_MODALITY = "clinic_modality"


@dataclass(frozen=True, slots=True)
class DuplicateProducerError(Exception):
    item_id: str
    first: ItemProducer
    second: ItemProducer

    def __str__(self) -> str:
        return (
            f"detection item {self.item_id!r} has duplicate producers: "
            f"{self.first.value}, {self.second.value}"
        )


@dataclass(frozen=True, slots=True)
class IncompleteRoutePlanError(Exception):
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    def __str__(self) -> str:
        return (
            "route plan must own every canonical item exactly once: "
            f"missing={self.missing}, unexpected={self.unexpected}"
        )


@dataclass(frozen=True, slots=True)
class IncompleteProducerOutputError(Exception):
    missing: tuple[str, ...]
    failed: tuple[str, ...]

    def __str__(self) -> str:
        return (
            "canonical join requires every planned producer output to succeed: "
            f"missing={self.missing}, failed={self.failed}"
        )


@dataclass(frozen=True, slots=True)
class UnexpectedProducerError(Exception):
    item_id: str
    expected: ItemProducer
    actual: ItemProducer

    def __str__(self) -> str:
        return (
            f"detection item {self.item_id!r} expected producer "
            f"{self.expected.value}, received {self.actual.value}"
        )


@dataclass(frozen=True, slots=True)
class ProducerPlan:
    producer: ItemProducer
    item_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutePlan:
    route: RuntimeRoute
    dermavision_algorithms: tuple[str, ...]
    producers: tuple[ProducerPlan, ...]

    def __post_init__(self) -> None:
        owners: dict[str, ItemProducer] = {}
        for producer_plan in self.producers:
            for item_id in producer_plan.item_ids:
                first = owners.get(item_id)
                if first is not None:
                    raise DuplicateProducerError(
                        item_id=item_id,
                        first=first,
                        second=producer_plan.producer,
                    )
                owners[item_id] = producer_plan.producer
        expected = {item.item_id for item in TWELVE_DETECTION_ITEMS}
        actual = set(owners)
        if actual != expected:
            raise IncompleteRoutePlanError(
                missing=tuple(sorted(expected - actual)),
                unexpected=tuple(sorted(actual - expected)),
            )

    def items_for(self, producer: ItemProducer) -> tuple[str, ...]:
        for producer_plan in self.producers:
            if producer_plan.producer == producer:
                return producer_plan.item_ids
        return ()

    def item_producer(self, item_id: str) -> ItemProducer:
        for producer_plan in self.producers:
            if item_id in producer_plan.item_ids:
                return producer_plan.producer
        raise IncompleteRoutePlanError(missing=(item_id,), unexpected=())


@dataclass(frozen=True, slots=True)
class ClinicItemModalityContract:
    """Frozen pixel/geometry roles for one institution detection item."""

    item_id: str
    pixel_source_role: str
    geometry_mask_anchor_role: str
    auxiliary_pixel_source_roles: tuple[str, ...] = ()
    registration_required_roles: tuple[str, ...] = ()
    style_view_id: str | None = None
    fallback_allowed: bool = False


CLINIC_ITEM_MODALITY_CONTRACT = (
    ClinicItemModalityContract("redness", "CP_M", "RGB_M", registration_required_roles=("CP_M",), style_view_id="red_brown_cp_pixels_rgb_geometry"),
    ClinicItemModalityContract("spots", "RGB_M", "RGB_M"),
    ClinicItemModalityContract("brown", "CP_M", "RGB_M", registration_required_roles=("CP_M",), style_view_id="red_brown_cp_pixels_rgb_geometry"),
    ClinicItemModalityContract("texture", "RGB_M", "RGB_M"),
    ClinicItemModalityContract("pores", "RGB_M", "RGB_M"),
    ClinicItemModalityContract("uv_spots", "365_M", "365_M"),
    ClinicItemModalityContract("porphyrin", "365_M", "365_M"),
    ClinicItemModalityContract("wrinkle", "RGB_M", "RGB_M"),
    ClinicItemModalityContract("acne", "RGB_M", "RGB_M"),
    ClinicItemModalityContract("surface_gloss", "PP_M", "PP_M"),
    ClinicItemModalityContract(
        "vascular",
        "CP_M",
        "CP_M",
        auxiliary_pixel_source_roles=("RGB_M", "PP_M"),
        registration_required_roles=("CP_M", "PP_M"),
        style_view_id="vascular_cp_pixels_cp_geometry",
    ),
    ClinicItemModalityContract("contour_firmness", "RGB_M", "RGB_M"),
)


CONSUMER_ROUTE_PLAN = RoutePlan(
    route=RuntimeRoute.CONSUMER_RGB,
    dermavision_algorithms=("spots", "texture", "pores", "contour_firmness"),
    producers=(
        ProducerPlan(
            ItemProducer.DERMAVISION,
            (
                "spots",
                "texture",
                "pores",
                "contour_firmness",
            ),
        ),
        ProducerPlan(ItemProducer.WRINKLE, ("wrinkle",)),
        ProducerPlan(ItemProducer.ACNE, ("acne",)),
        ProducerPlan(
            ItemProducer.CLINIC_MODALITY,
            (
                "redness",
                "brown",
                "uv_spots",
                "porphyrin",
                "surface_gloss",
                "vascular",
            ),
        ),
    ),
)


CLINIC_ROUTE_PLAN = RoutePlan(
    route=RuntimeRoute.CLINIC_FOUR_LIGHT,
    dermavision_algorithms=("spots", "texture", "pores", "contour_firmness"),
    producers=(
        ProducerPlan(
            ItemProducer.DERMAVISION,
            ("spots", "texture", "pores", "contour_firmness"),
        ),
        ProducerPlan(ItemProducer.WRINKLE, ("wrinkle",)),
        ProducerPlan(ItemProducer.ACNE, ("acne",)),
        ProducerPlan(
            ItemProducer.CLINIC_MODALITY,
            (
                "redness",
                "brown",
                "uv_spots",
                "porphyrin",
                "surface_gloss",
                "vascular",
            ),
        ),
    ),
)


def canonical_join(plan: RoutePlan, produced: ProducedItems) -> CanonicalItems:
    """Fail closed, then join outputs once in the frozen public order."""

    available: dict[str, DetectionItem] = {}
    actual_owners: dict[str, ItemProducer] = {}
    for producer, items in produced.items():
        for item_id, item in items.items():
            previous = actual_owners.get(item_id)
            if previous is not None:
                raise DuplicateProducerError(item_id, previous, producer)
            expected = plan.item_producer(item_id)
            if expected != producer:
                raise UnexpectedProducerError(item_id, expected, producer)
            actual_owners[item_id] = producer
            available[item_id] = dict(item)

    expected_order = tuple(item.item_id for item in TWELVE_DETECTION_ITEMS)
    missing = tuple(item_id for item_id in expected_order if item_id not in available)
    failed = tuple(
        item_id
        for item_id in expected_order
        if item_id in available and available[item_id]["状态"] != "success"
    )
    if missing or failed:
        raise IncompleteProducerOutputError(
            missing=missing,
            failed=failed,
        )
    return {item_id: available[item_id] for item_id in expected_order}


__all__ = [
    "CLINIC_ROUTE_PLAN",
    "CONSUMER_ROUTE_PLAN",
    "CanonicalItems",
    "CLINIC_ITEM_MODALITY_CONTRACT",
    "ClinicItemModalityContract",
    "DetectionItem",
    "DuplicateProducerError",
    "IncompleteProducerOutputError",
    "IncompleteRoutePlanError",
    "ItemProducer",
    "ProducerPlan",
    "RoutePlan",
    "UnexpectedProducerError",
    "canonical_join",
]
