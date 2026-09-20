from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import cv2
import numpy as np


FOREHEAD_OCCLUSION_RATIO: Final = 0.80
LOCAL_LIGHTING_BLOCK_RANGE_REJECT: Final = 125.0
LOCAL_LIGHTING_SEVERE_RANGE: Final = 140.0
LOCAL_LIGHTING_SEVERE_GRADIENT_P99: Final = 3.10
LOCAL_LIGHTING_SEVERE_BRIGHT_RATIO: Final = 0.05


@dataclass(frozen=True, slots=True)
class ConsumerQualityDecision:
    score: float
    status: str
    flags: tuple[str, ...]
    forehead_occluded: bool
    local_lighting_block_range: float
    local_lighting_gradient_p99: float
    local_lighting_bright_ratio: float


def _local_block_median_range(image: np.ndarray, skin_mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = np.asarray(skin_mask) > 0
    block_size = max(32, min(image.shape[:2]) // 16)
    minimum_pixels = int(block_size * block_size * 0.30)
    medians: list[float] = []
    for top in range(0, image.shape[0], block_size):
        for left in range(0, image.shape[1], block_size):
            block_valid = valid[top : top + block_size, left : left + block_size]
            values = gray[top : top + block_size, left : left + block_size][
                block_valid
            ]
            if values.size >= minimum_pixels:
                medians.append(float(np.median(values)))
    return max(medians) - min(medians) if medians else 0.0


def _local_lighting_support(
    image: np.ndarray,
    skin_mask: np.ndarray,
) -> tuple[float, float]:
    valid = np.asarray(skin_mask) > 0
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 100:
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    weights = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 8.0)
    weighted = cv2.GaussianBlur(gray * valid, (0, 0), 8.0)
    illumination = weighted / np.maximum(weights, 1e-6)
    gradient_x = cv2.Sobel(illumination, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(illumination, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    interior = cv2.erode(
        valid.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ) > 0
    gradients = np.hypot(gradient_x, gradient_y)[interior]
    gradient_p99 = float(np.percentile(gradients, 99)) if gradients.size else 0.0
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    bright_ratio = float(np.count_nonzero(valid & (value >= 230)) / valid_count)
    return gradient_p99, bright_ratio


def evaluate_consumer_quality(
    analysis_image: np.ndarray,
    skin_mask: np.ndarray,
    debug_masks: Mapping[str, np.ndarray],
    base_score: float,
    base_status: str,
    base_flags: list[str],
) -> ConsumerQualityDecision:
    flags = list(dict.fromkeys(base_flags))
    forehead = np.asarray(
        debug_masks.get("forehead_completion", np.zeros_like(skin_mask))
    ) > 0
    hair = np.asarray(
        debug_masks.get("hair_mask", np.zeros_like(skin_mask))
    ) > 0
    forehead_pixels = int(np.count_nonzero(forehead))
    occlusion_ratio = (
        float(np.count_nonzero(forehead & hair) / forehead_pixels)
        if forehead_pixels
        else 0.0
    )
    forehead_occluded = occlusion_ratio >= FOREHEAD_OCCLUSION_RATIO
    if forehead_occluded and "FOREHEAD_OCCLUDED" not in flags:
        flags.append("FOREHEAD_OCCLUDED")
    lighting_range = _local_block_median_range(analysis_image, skin_mask)
    gradient_p99, bright_ratio = _local_lighting_support(
        analysis_image,
        skin_mask,
    )
    lighting_variation = lighting_range > LOCAL_LIGHTING_BLOCK_RANGE_REJECT
    severe_lighting = bool(
        lighting_range >= LOCAL_LIGHTING_SEVERE_RANGE
        and gradient_p99 >= LOCAL_LIGHTING_SEVERE_GRADIENT_P99
        and bright_ratio >= LOCAL_LIGHTING_SEVERE_BRIGHT_RATIO
    )
    if severe_lighting and "SEVERE_LOCAL_LIGHTING" not in flags:
        flags.append("SEVERE_LOCAL_LIGHTING")
    elif lighting_variation and "LOCAL_LIGHTING_VARIATION" not in flags:
        flags.append("LOCAL_LIGHTING_VARIATION")
    if severe_lighting:
        return ConsumerQualityDecision(
            score=min(float(base_score), 59.0),
            status="REJECT",
            flags=tuple(flags),
            forehead_occluded=forehead_occluded,
            local_lighting_block_range=round(lighting_range, 4),
            local_lighting_gradient_p99=round(gradient_p99, 4),
            local_lighting_bright_ratio=round(bright_ratio, 6),
        )
    if (forehead_occluded or lighting_variation) and base_status == "PASS":
        return ConsumerQualityDecision(
            score=min(float(base_score), 74.0),
            status="WARNING",
            flags=tuple(flags),
            forehead_occluded=forehead_occluded,
            local_lighting_block_range=round(lighting_range, 4),
            local_lighting_gradient_p99=round(gradient_p99, 4),
            local_lighting_bright_ratio=round(bright_ratio, 6),
        )
    return ConsumerQualityDecision(
        score=float(base_score),
        status=base_status,
        flags=tuple(flags),
        forehead_occluded=forehead_occluded,
        local_lighting_block_range=round(lighting_range, 4),
        local_lighting_gradient_p99=round(gradient_p99, 4),
        local_lighting_bright_ratio=round(bright_ratio, 6),
    )


__all__ = [
    "ConsumerQualityDecision",
    "FOREHEAD_OCCLUSION_RATIO",
    "LOCAL_LIGHTING_BLOCK_RANGE_REJECT",
    "LOCAL_LIGHTING_SEVERE_RANGE",
    "LOCAL_LIGHTING_SEVERE_GRADIENT_P99",
    "LOCAL_LIGHTING_SEVERE_BRIGHT_RATIO",
    "evaluate_consumer_quality",
]
