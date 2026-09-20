"""High-precision consumer brown finding selection presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import cv2
import numpy as np
from skimage.feature import peak_local_max
from typing_extensions import assert_never

from src.engines.visia_regions import VisiaRegionSet, assign_region


class ConsumerFindingPreset(str, Enum):
    """Engineering-only candidate selection variants."""

    SOFT = "soft"
    BALANCED = "balanced"
    ENHANCED = "enhanced"


@dataclass(frozen=True, slots=True)
class BrownFindingPolicy:
    minimum_scale_votes: int
    score_percentile: float
    nms_distance: int
    minimum_score: float
    density_warning_per_100k: float


@dataclass(frozen=True, slots=True)
class BrownFindingEvidence:
    score_map: np.ndarray
    scale_votes: np.ndarray
    valid_mask: np.ndarray
    specular_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class BrownFinding:
    centroid: tuple[int, int]
    support_area_px: int
    marker_radius_px: int
    confidence: float
    region: str


@dataclass(frozen=True, slots=True)
class BrownPresetResult:
    preset: ConsumerFindingPreset
    findings: tuple[BrownFinding, ...]
    marker_mask: np.ndarray
    rejected_specular: int
    rejected_region: int
    density_per_100k: float
    density_status: Literal["PASS", "WARNING"]


def policy_for_brown_preset(
    preset: ConsumerFindingPreset,
) -> BrownFindingPolicy:
    match preset:
        case ConsumerFindingPreset.SOFT:
            return BrownFindingPolicy(4, 90.0, 18, 0.80, 80.0)
        case ConsumerFindingPreset.BALANCED:
            return BrownFindingPolicy(3, 75.0, 12, 0.68, 150.0)
        case ConsumerFindingPreset.ENHANCED:
            return BrownFindingPolicy(2, 60.0, 8, 0.56, 250.0)
        case unreachable:
            assert_never(unreachable)


def select_brown_findings(
    evidence: BrownFindingEvidence,
    regions: VisiaRegionSet,
    preset: ConsumerFindingPreset,
) -> BrownPresetResult:
    """Select separated peaks without imposing a fixed feature count."""
    policy = policy_for_brown_preset(preset)
    score = np.asarray(evidence.score_map, dtype=np.float32)
    votes = np.asarray(evidence.scale_votes, dtype=np.uint8)
    valid = np.asarray(evidence.valid_mask) > 0
    specular = np.asarray(evidence.specular_mask) > 0
    peak_domain = valid & (votes >= policy.minimum_scale_votes)
    eligible = peak_domain & ~specular
    eligible_scores = score[peak_domain]
    percentile = (
        float(np.percentile(eligible_scores, policy.score_percentile))
        if eligible_scores.size
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
    findings: list[BrownFinding] = []
    rejected_specular = 0
    rejected_region = 0
    height, width = score.shape
    yy, xx = np.ogrid[:height, :width]
    for y_raw, x_raw in coordinates:
        x, y = int(x_raw), int(y_raw)
        if specular[y, x]:
            rejected_specular += 1
            continue
        peak_score = float(score[y, x])
        support_radius = max(6, policy.nms_distance // 2)
        local_circle = (
            np.square(xx - x) + np.square(yy - y)
            <= support_radius * support_radius
        )
        support = (
            local_circle
            & eligible
            & (score >= max(policy.minimum_score * 0.82, peak_score * 0.72))
        )
        support_area = max(int(np.count_nonzero(support)), 1)
        marker_radius = int(
            np.clip(round(0.55 * np.sqrt(support_area / np.pi)), 2, 7)
        )
        marker = np.zeros_like(marker_mask)
        cv2.circle(marker, (x, y), marker_radius, 255, -1, cv2.LINE_AA)
        marker = cv2.bitwise_and(marker, evidence.valid_mask)
        marker[specular] = 0
        region = assign_region(x, y, regions.regions)
        if region == "other" or not np.count_nonzero(marker):
            rejected_region += 1
            continue
        marker_mask[marker > 0] = 255
        findings.append(
            BrownFinding(
                centroid=(x, y),
                support_area_px=support_area,
                marker_radius_px=marker_radius,
                confidence=round(peak_score, 6),
                region=region,
            )
        )
    valid_area = max(int(np.count_nonzero(valid)), 1)
    density = len(findings) * 100_000.0 / valid_area
    status: Literal["PASS", "WARNING"] = (
        "WARNING" if density > policy.density_warning_per_100k else "PASS"
    )
    return BrownPresetResult(
        preset=preset,
        findings=tuple(findings),
        marker_mask=marker_mask,
        rejected_specular=rejected_specular,
        rejected_region=rejected_region,
        density_per_100k=round(density, 6),
        density_status=status,
    )


__all__ = [
    "BrownFinding",
    "BrownFindingEvidence",
    "BrownFindingPolicy",
    "BrownPresetResult",
    "ConsumerFindingPreset",
    "policy_for_brown_preset",
    "select_brown_findings",
]
