"""Consumer UV-spots and porphyrin finding presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import cv2
import numpy as np
from skimage.feature import peak_local_max
from typing_extensions import assert_never

from src.consumer_pigment.brown_presets import ConsumerFindingPreset
from src.engines.visia_regions import VisiaRegionSet, assign_region


class PurpleFindingKind(str, Enum):
    UV_SPOTS = "uv_spots"
    PORPHYRIN = "porphyrin"


@dataclass(frozen=True, slots=True)
class PurpleFindingPolicy:
    minimum_scale_votes: int
    score_percentile: float
    nms_distance: int
    minimum_score: float
    density_warning_per_100k: float


@dataclass(frozen=True, slots=True)
class PurpleFindingEvidence:
    kind: PurpleFindingKind
    score_map: np.ndarray
    scale_votes: np.ndarray
    valid_mask: np.ndarray
    support_mask: np.ndarray
    suppression_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class PurpleFinding:
    centroid: tuple[int, int]
    support_area_px: int
    marker_radius_px: int
    confidence: float
    region: str


@dataclass(frozen=True, slots=True)
class PurplePresetResult:
    kind: PurpleFindingKind
    preset: ConsumerFindingPreset
    findings: tuple[PurpleFinding, ...]
    marker_mask: np.ndarray
    rejected_support: int
    rejected_suppression: int
    rejected_region: int
    density_per_100k: float
    density_status: Literal["PASS", "WARNING"]


def policy_for_purple_preset(
    preset: ConsumerFindingPreset,
) -> PurpleFindingPolicy:
    match preset:
        case ConsumerFindingPreset.SOFT:
            return PurpleFindingPolicy(4, 90.0, 18, 0.86, 100.0)
        case ConsumerFindingPreset.BALANCED:
            return PurpleFindingPolicy(3, 75.0, 12, 0.78, 200.0)
        case ConsumerFindingPreset.ENHANCED:
            return PurpleFindingPolicy(2, 60.0, 8, 0.70, 350.0)
        case unreachable:
            assert_never(unreachable)


def select_purple_findings(
    evidence: PurpleFindingEvidence,
    regions: VisiaRegionSet,
    preset: ConsumerFindingPreset,
) -> PurplePresetResult:
    """Select compact UV or porphyrin peaks with explicit suppression reasons."""
    policy = policy_for_purple_preset(preset)
    score = np.asarray(evidence.score_map, dtype=np.float32)
    votes = np.asarray(evidence.scale_votes, dtype=np.uint8)
    valid = np.asarray(evidence.valid_mask) > 0
    support = np.asarray(evidence.support_mask) > 0
    suppression = np.asarray(evidence.suppression_mask) > 0
    peak_domain = valid & (votes >= policy.minimum_scale_votes)
    scores = score[peak_domain]
    percentile = (
        float(np.percentile(scores, policy.score_percentile))
        if scores.size
        else 1.0
    )
    threshold = max(policy.minimum_score, percentile)
    coordinates = peak_local_max(
        score,
        min_distance=policy.nms_distance,
        threshold_abs=threshold,
        exclude_border=False,
        labels=peak_domain.astype(np.uint8),
    )
    marker_mask = np.zeros(score.shape, dtype=np.uint8)
    findings: list[PurpleFinding] = []
    rejected_support = 0
    rejected_suppression = 0
    rejected_region = 0
    height, width = score.shape
    yy, xx = np.ogrid[:height, :width]
    for y_raw, x_raw in coordinates:
        x, y = int(x_raw), int(y_raw)
        if suppression[y, x]:
            rejected_suppression += 1
            continue
        match evidence.kind:
            case PurpleFindingKind.UV_SPOTS:
                if not support[y, x]:
                    rejected_support += 1
                    continue
            case PurpleFindingKind.PORPHYRIN:
                pass
            case unreachable:
                assert_never(unreachable)
        peak_score = float(score[y, x])
        support_radius = max(6, policy.nms_distance // 2)
        local_circle = (
            np.square(xx - x) + np.square(yy - y)
            <= support_radius * support_radius
        )
        local_support = (
            local_circle
            & peak_domain
            & ~suppression
            & (score >= max(policy.minimum_score * 0.82, peak_score * 0.72))
        )
        support_area = max(int(np.count_nonzero(local_support)), 1)
        radius = int(np.clip(round(0.55 * np.sqrt(support_area / np.pi)), 2, 7))
        marker = np.zeros_like(marker_mask)
        cv2.circle(marker, (x, y), radius, 255, -1, cv2.LINE_AA)
        marker = cv2.bitwise_and(marker, evidence.valid_mask)
        marker[suppression] = 0
        region = assign_region(x, y, regions.regions)
        if region == "other" or not np.count_nonzero(marker):
            rejected_region += 1
            continue
        marker_mask[marker > 0] = 255
        findings.append(
            PurpleFinding(
                centroid=(x, y),
                support_area_px=support_area,
                marker_radius_px=radius,
                confidence=round(peak_score, 6),
                region=region,
            )
        )
    valid_area = max(int(np.count_nonzero(valid)), 1)
    density = len(findings) * 100_000.0 / valid_area
    status: Literal["PASS", "WARNING"] = (
        "WARNING" if density > policy.density_warning_per_100k else "PASS"
    )
    return PurplePresetResult(
        kind=evidence.kind,
        preset=preset,
        findings=tuple(findings),
        marker_mask=marker_mask,
        rejected_support=rejected_support,
        rejected_suppression=rejected_suppression,
        rejected_region=rejected_region,
        density_per_100k=round(density, 6),
        density_status=status,
    )


__all__ = [
    "ConsumerFindingPreset",
    "PurpleFinding",
    "PurpleFindingEvidence",
    "PurpleFindingKind",
    "PurpleFindingPolicy",
    "PurplePresetResult",
    "policy_for_purple_preset",
    "select_purple_findings",
]
