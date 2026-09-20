"""Consumer-only formal marker consolidation shared by pigment modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from typing_extensions import assert_never

from src.consumer_pigment.markers import consolidate_existing_marker_mask
from src.engines.visia_regions import VisiaRegionSet, assign_region


class ConsumerPigmentMarkerKind(str, Enum):
    BROWN = "brown"
    UV_SPOTS = "uv_spots"


class ConsumerBrownRecallPreset(str, Enum):
    CURRENT = "current"
    RECALL_1 = "recall_1"
    RECALL_2 = "recall_2"
    RECALL_3 = "recall_3"
    RECALL_4 = "recall_4"


@dataclass(frozen=True, slots=True)
class ConsumerMarkerPolicy:
    minimum_peak_score: float
    minimum_distance_px: int
    minimum_local_prominence_z: float
    minimum_marker_edge_gap_px: int = 0
    large_component_prominence_relief: float = 0.0
    area_priority_weight: float = 0.0
    marker_radius_scale_for_spacing: float = 0.55
    minimum_marker_radius_px: int = 2
    maximum_marker_radius_px: int = 7


@dataclass(frozen=True, slots=True)
class ConsumerMarkerInput:
    candidate_mask: np.ndarray
    score_map: np.ndarray
    valid_mask: np.ndarray
    exclusion_mask: np.ndarray
    regions: VisiaRegionSet


@dataclass(frozen=True, slots=True)
class FormalMarkerFinding:
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    source_area_px: int
    marker_area_px: int
    marker_radius_px: int
    peak_score: float
    confidence: float
    region: str


@dataclass(frozen=True, slots=True)
class FormalMarkerProjection:
    kind: ConsumerPigmentMarkerKind
    findings: tuple[FormalMarkerFinding, ...]
    marker_mask: np.ndarray
    rejected_low_score: int
    rejected_exclusion: int
    rejected_nms: int
    rejected_region: int


def policy_for_consumer_marker(
    kind: ConsumerPigmentMarkerKind,
) -> ConsumerMarkerPolicy:
    match kind:
        case ConsumerPigmentMarkerKind.BROWN:
            return ConsumerMarkerPolicy(0.50, 12, 8.0)
        case ConsumerPigmentMarkerKind.UV_SPOTS:
            return ConsumerMarkerPolicy(0.90, 24, 0.0)
        case unreachable:
            assert_never(unreachable)


def policy_for_consumer_brown_recall(
    preset: ConsumerBrownRecallPreset,
) -> ConsumerMarkerPolicy:
    if preset is ConsumerBrownRecallPreset.CURRENT:
        return ConsumerMarkerPolicy(0.50, 12, 8.0)
    prominence = {
        ConsumerBrownRecallPreset.RECALL_1: 6.5,
        ConsumerBrownRecallPreset.RECALL_2: 5.0,
        ConsumerBrownRecallPreset.RECALL_3: 3.5,
        ConsumerBrownRecallPreset.RECALL_4: 2.0,
    }[preset]
    return ConsumerMarkerPolicy(
        0.50,
        15,
        prominence,
        minimum_marker_edge_gap_px=0,
        large_component_prominence_relief=0.35,
        area_priority_weight=0.06,
        marker_radius_scale_for_spacing=0.55,
        minimum_marker_radius_px=3,
        maximum_marker_radius_px=5,
    )


def _confidence(kind: ConsumerPigmentMarkerKind, peak_score: float) -> float:
    match kind:
        case ConsumerPigmentMarkerKind.BROWN | ConsumerPigmentMarkerKind.UV_SPOTS:
            return float(np.clip(0.35 + 0.65 * peak_score, 0.0, 1.0))
        case unreachable:
            assert_never(unreachable)


def project_consumer_markers(
    marker_input: ConsumerMarkerInput,
    kind: ConsumerPigmentMarkerKind,
    *,
    policy: ConsumerMarkerPolicy | None = None,
) -> FormalMarkerProjection:
    """Produce the one formal mask consumed by images, metrics, JSON and CSV."""
    active_policy = policy or policy_for_consumer_marker(kind)
    consolidated = consolidate_existing_marker_mask(
        marker_input.candidate_mask,
        marker_input.score_map,
        marker_input.valid_mask,
        marker_input.exclusion_mask,
        minimum_peak_score=active_policy.minimum_peak_score,
        minimum_distance_px=active_policy.minimum_distance_px,
        minimum_local_prominence_z=active_policy.minimum_local_prominence_z,
        minimum_marker_edge_gap_px=active_policy.minimum_marker_edge_gap_px,
        large_component_prominence_relief=(
            active_policy.large_component_prominence_relief
        ),
        area_priority_weight=active_policy.area_priority_weight,
        marker_radius_scale=active_policy.marker_radius_scale_for_spacing,
    )
    valid = np.asarray(marker_input.valid_mask) > 0
    exclusion = np.asarray(marker_input.exclusion_mask) > 0
    output_mask = np.zeros(valid.shape, dtype=np.uint8)
    findings: list[FormalMarkerFinding] = []
    rejected_region = 0
    height, width = valid.shape
    for source in consolidated.findings:
        x, y = source.centroid
        match kind:
            case ConsumerPigmentMarkerKind.BROWN:
                marker_radius = int(
                    np.clip(
                        round(np.sqrt(source.source_area_px / np.pi)),
                        active_policy.minimum_marker_radius_px,
                        active_policy.maximum_marker_radius_px,
                    )
                )
            case ConsumerPigmentMarkerKind.UV_SPOTS:
                marker_radius = int(
                    np.clip(
                        round(0.85 * np.sqrt(source.source_area_px / np.pi)),
                        2,
                        12,
                    )
                )
            case unreachable:
                assert_never(unreachable)
        region = assign_region(x, y, marker_input.regions.regions)
        if region == "other":
            rejected_region += 1
            continue
        marker = np.zeros(valid.shape, dtype=np.uint8)
        cv2.circle(
            marker,
            (x, y),
            marker_radius,
            255,
            -1,
            cv2.LINE_AA,
        )
        marker[~valid | exclusion] = 0
        coordinates = np.argwhere(marker > 0)
        if not coordinates.size:
            rejected_region += 1
            continue
        y_min, x_min = np.min(coordinates, axis=0)
        y_max, x_max = np.max(coordinates, axis=0)
        marker_area = int(np.count_nonzero(marker))
        output_mask[marker > 0] = 255
        findings.append(
            FormalMarkerFinding(
                centroid=(x, y),
                bbox=(
                    int(x_min),
                    int(y_min),
                    int(x_max - x_min + 1),
                    int(y_max - y_min + 1),
                ),
                source_area_px=source.source_area_px,
                marker_area_px=marker_area,
                marker_radius_px=marker_radius,
                peak_score=source.peak_score,
                confidence=round(_confidence(kind, source.peak_score), 4),
                region=region,
            )
        )
    return FormalMarkerProjection(
        kind=kind,
        findings=tuple(findings),
        marker_mask=output_mask,
        rejected_low_score=consolidated.rejected_low_score,
        rejected_exclusion=consolidated.rejected_exclusion,
        rejected_nms=consolidated.rejected_nms,
        rejected_region=rejected_region,
    )


__all__ = [
    "ConsumerMarkerInput",
    "ConsumerMarkerPolicy",
    "ConsumerBrownRecallPreset",
    "ConsumerPigmentMarkerKind",
    "FormalMarkerFinding",
    "FormalMarkerProjection",
    "policy_for_consumer_marker",
    "policy_for_consumer_brown_recall",
    "project_consumer_markers",
]
