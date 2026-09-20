from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_width: int = 512
    min_height: int = 512
    blur_laplacian_var_warn: float = 55.0
    dark_mean_warn: float = 55.0
    bright_mean_warn: float = 205.0
    overexposed_ratio_warn: float = 0.08
    underexposed_ratio_warn: float = 0.08


def assess_image_quality(image_bgr: np.ndarray, config: QualityConfig | None = None) -> dict:
    config = config or QualityConfig()
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luma = float(gray.mean())
    overexposed_ratio = float((gray >= 245).mean())
    underexposed_ratio = float((gray <= 10).mean())

    warnings: list[str] = []
    if width < config.min_width or height < config.min_height:
        warnings.append("resolution_below_recommended")
    if blur_var < config.blur_laplacian_var_warn:
        warnings.append("possible_blur")
    if mean_luma < config.dark_mean_warn:
        warnings.append("possible_underexposure")
    if mean_luma > config.bright_mean_warn:
        warnings.append("possible_overexposure")
    if overexposed_ratio > config.overexposed_ratio_warn:
        warnings.append("large_overexposed_area")
    if underexposed_ratio > config.underexposed_ratio_warn:
        warnings.append("large_underexposed_area")

    return {
        "status": "ok" if not warnings else "warning",
        "width": int(width),
        "height": int(height),
        "blur_laplacian_var": blur_var,
        "mean_luma": mean_luma,
        "overexposed_ratio": overexposed_ratio,
        "underexposed_ratio": underexposed_ratio,
        "warnings": warnings,
        "beauty_filter": {
            "status": "not_assessed",
            "reason": "cannot reliably infer beauty filtering from one image",
        },
    }
