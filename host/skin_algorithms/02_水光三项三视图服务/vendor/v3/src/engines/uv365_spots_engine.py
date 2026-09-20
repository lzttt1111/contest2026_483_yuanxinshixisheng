# -*- coding: utf-8 -*-
"""Independent real-365nm UV spot contract for the clinic route.

The legacy :mod:`uv_spots_engine` remains the RGB/UV-like proxy implementation.
This wrapper accepts only a real ``365_M`` frame, keeps the actual 365 display
base, and subtracts follicular orange-fluorescence support before extracting
dark absorption-enhanced candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.engines.uv_spots_engine import UVSpotsAnalyzer, UVSpotsConfig, UVSpotsResult
from src.engines.visia_regions import REGION_LABELS, REGION_ORDER


@dataclass(frozen=True)
class UV365SpotsConfig:
    algorithm_version: str = "UV365Spots-V1"
    input_semantics: str = "REAL_365_M"
    porphyrin_exclusion_dilate_px: int = 1


@dataclass
class UV365SpotsResult:
    overlay: np.ndarray
    instance_mask: np.ndarray
    score_map: np.ndarray
    instances: list[dict[str, Any]]
    region_distribution: dict[str, dict[str, Any]]
    porphyrin_exclusion_mask: np.ndarray
    algorithm_version: str
    input_semantics: str
    measurement_support_mask: np.ndarray | None = None


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


class UV365SpotsAnalyzer:
    """Detect dark UV absorption candidates on a real aligned 365nm frame."""

    def __init__(
        self,
        config: UV365SpotsConfig | None = None,
        extraction_config: UVSpotsConfig | None = None,
    ) -> None:
        self.config = config or UV365SpotsConfig()
        self._extractor = UVSpotsAnalyzer(extraction_config or UVSpotsConfig())

    def detect(
        self,
        uv365_image: np.ndarray,
        regions,
        landmarks: np.ndarray,
        *,
        porphyrin_mask: np.ndarray,
        display_base: np.ndarray | None = None,
        instance_exclusion_mask: np.ndarray | None = None,
    ) -> UV365SpotsResult:
        if not isinstance(uv365_image, np.ndarray) or uv365_image.ndim != 3:
            raise ValueError("UV365SpotsAnalyzer requires a decoded real 365_M image")
        if uv365_image.shape[:2] != np.asarray(porphyrin_mask).shape[:2]:
            raise ValueError("UV365/porphyrin mask shape mismatch")
        kernel_size = max(1, int(self.config.porphyrin_exclusion_dilate_px) * 2 + 1)
        porphyrin_exclusion = cv2.dilate(
            _binary(porphyrin_mask),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        exclusion = porphyrin_exclusion
        if instance_exclusion_mask is not None:
            exclusion = cv2.bitwise_or(exclusion, _binary(instance_exclusion_mask))
        result: UVSpotsResult = self._extractor.detect(
            uv365_image,
            regions,
            landmarks,
            display_base=uv365_image if display_base is None else display_base,
            instance_exclusion_mask=exclusion,
        )
        # The legacy extractor rejects peaks inside ``exclusion`` but its
        # small circular display marker can still cross the one-pixel expanded
        # porphyrin boundary.  The real-365 contract therefore subtracts the
        # exclusion from the final support as well, then recomputes every
        # dependent instance/region value from the cleaned mask.
        clean_mask = cv2.bitwise_and(
            _binary(result.instance_mask),
            cv2.bitwise_not(porphyrin_exclusion),
        )
        clean_instances = [
            item
            for item in result.instances
            if (
                0 <= int(round(float(item["centroid"][1]))) < clean_mask.shape[0]
                and 0 <= int(round(float(item["centroid"][0]))) < clean_mask.shape[1]
                and porphyrin_exclusion[
                    int(round(float(item["centroid"][1]))),
                    int(round(float(item["centroid"][0]))),
                ] == 0
            )
        ]
        distribution: dict[str, dict[str, Any]] = {}
        for name in REGION_ORDER:
            region_mask = np.asarray(regions.regions[name]) > 0
            area = int(np.count_nonzero((clean_mask > 0) & region_mask))
            valid_area = int(np.count_nonzero(region_mask))
            local = [item for item in clean_instances if item.get("region") == name]
            values = np.asarray(result.score_map)[region_mask]
            distribution[name] = {
                "label": REGION_LABELS[name],
                "count": len(local),
                "area": area,
                "area_ratio": round(area / max(valid_area, 1), 8),
                "mean_score": round(float(np.mean(values)) if values.size else 0.0, 6),
            }
        clean_overlay = np.asarray(result.overlay).copy()
        display = uv365_image if display_base is None else display_base
        clean_overlay[porphyrin_exclusion > 0] = display[porphyrin_exclusion > 0]
        overlap = cv2.bitwise_and(clean_mask, porphyrin_exclusion)
        if np.any(overlap):
            raise RuntimeError("UV365 spot output overlaps porphyrin exclusion")
        return UV365SpotsResult(
            overlay=clean_overlay,
            instance_mask=clean_mask,
            score_map=result.score_map,
            instances=clean_instances,
            region_distribution=distribution,
            porphyrin_exclusion_mask=porphyrin_exclusion,
            algorithm_version=self.config.algorithm_version,
            input_semantics=self.config.input_semantics,
            measurement_support_mask=(
                cv2.bitwise_and(_binary(result.measurement_support_mask), cv2.bitwise_not(exclusion))
                if result.measurement_support_mask is not None else None
            ),
        )


__all__ = ["UV365SpotsAnalyzer", "UV365SpotsConfig", "UV365SpotsResult"]
