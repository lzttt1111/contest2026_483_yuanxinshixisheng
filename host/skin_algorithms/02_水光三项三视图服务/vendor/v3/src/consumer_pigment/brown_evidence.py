"""Build consumer brown multi-scale evidence without selecting findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from src.consumer_pigment.brown_presets import BrownFindingEvidence
from src.consumer_pigment.specular import build_specular_evidence
from src.engines.visia_regions import (
    VisiaRegionSet,
    regional_positive_z,
    robust_unit_map,
)
from src.utils.gpu_backend import get_cuda_backend


@dataclass(frozen=True, slots=True)
class ConsumerBrownEvidenceConfig:
    local_sigmas: tuple[float, ...] = (8.0, 16.0, 32.0, 48.0)
    scale_z_threshold: float = 1.65
    dark_weight: float = 0.44
    yellow_weight: float = 0.28
    red_brown_weight: float = 0.16
    rgb_brown_weight: float = 0.12
    safe_mask_erode_kernel: int = 15


DEFAULT_CONFIG: Final = ConsumerBrownEvidenceConfig()


def build_brown_finding_evidence(
    image: np.ndarray,
    regions: VisiaRegionSet,
    config: ConsumerBrownEvidenceConfig = DEFAULT_CONFIG,
) -> BrownFindingEvidence:
    """Compute one continuous score and independent cross-scale vote map."""
    analysis_mask = np.asarray(regions.analysis_mask, dtype=np.uint8)
    safe_mask = cv2.erode(
        analysis_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.safe_mask_erode_kernel, config.safe_mask_erode_kernel),
        ),
    )
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0] / 255.0
    redness = (lab[:, :, 1] - 128.0) / 127.0
    yellowness = (lab[:, :, 2] - 128.0) / 127.0
    rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= 255.0
    total = np.maximum(np.sum(rgb, axis=2), 1e-5)
    rgb_brown = rgb[:, :, 0] / total - rgb[:, :, 2] / total
    backgrounds = get_cuda_backend().masked_gaussian_batch(
        (lightness, redness, yellowness, rgb_brown),
        analysis_mask,
        config.local_sigmas,
    )
    z_maps: list[np.ndarray] = []
    for sigma in config.local_sigmas:
        bg_l, bg_a, bg_b, bg_rgb = backgrounds[float(sigma)]
        raw = (
            config.dark_weight * np.maximum(bg_l - lightness, 0.0) / 0.035
            + config.yellow_weight
            * np.maximum(yellowness - bg_b, 0.0)
            / 0.025
            + config.red_brown_weight
            * np.maximum(redness - bg_a, 0.0)
            / 0.025
            + config.rgb_brown_weight
            * np.maximum(rgb_brown - bg_rgb, 0.0)
            / 0.018
        )
        raw[analysis_mask == 0] = 0.0
        z_maps.append(regional_positive_z(raw, regions, sigma_floor=0.12))
    stack = np.stack(z_maps, axis=0)
    votes = np.sum(
        stack >= config.scale_z_threshold,
        axis=0,
    ).astype(np.uint8)
    combined = 0.65 * np.max(stack, axis=0) + 0.35 * np.mean(stack, axis=0)
    score = robust_unit_map(combined, analysis_mask, 2.0, 99.0)
    specular = build_specular_evidence(image, safe_mask)
    return BrownFindingEvidence(
        score_map=score,
        scale_votes=votes,
        valid_mask=safe_mask,
        specular_mask=specular.mask,
    )


__all__ = [
    "ConsumerBrownEvidenceConfig",
    "build_brown_finding_evidence",
]
