from __future__ import annotations

"""Draw our findings on an otherwise unchanged vendor RED/BROWN canvas."""

import numpy as np

from src.consumer_pigment.markers import (
    BROWN_STYLE_MARKER,
    render_compact_marker_mask,
)
from src.engines.brown_engine import Config as BrownConfig
from src.engines.rbx_engine import Config as RedConfig
from src.engines.visia_regions import build_visia_regions, draw_region_boundaries
from src.preprocess.image_preprocessor import PreprocessResultV2


def _marker(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = np.asarray(base)
    marker = np.asarray(mask)
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("vendor canvas must be a BGR image")
    if marker.shape[:2] != canvas.shape[:2]:
        raise ValueError("vendor marker mask shape mismatch")
    if marker.ndim == 3:
        return np.any(marker > 0, axis=2)
    return marker > 0


def _with_boundary(preprocessed: PreprocessResultV2, image: np.ndarray) -> np.ndarray:
    regions = build_visia_regions(
        preprocessed.analysis_image,
        preprocessed.skin_mask,
        preprocessed.landmarks,
        preprocessed.quality_flags,
        include_chin=True,
        mode="full",
    )
    return draw_region_boundaries(
        image,
        regions.display_regions,
        RedConfig.RED_FEATURE_ZONE_COLOR,
        thickness=RedConfig.RED_FEATURE_ZONE_THICKNESS,
        partial_face=regions.partial_face,
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )


def render_vendor_red_result(
    preprocessed: PreprocessResultV2,
    vendor_base: np.ndarray,
    marker_mask: np.ndarray,
) -> np.ndarray:
    pixels = _marker(vendor_base, marker_mask)
    output = np.asarray(vendor_base).copy()
    output[pixels] = RedConfig.RED_FEATURE_MARKER_COLOR
    return _with_boundary(preprocessed, output)


def render_vendor_brown_result(
    preprocessed: PreprocessResultV2,
    vendor_base: np.ndarray,
    marker_mask: np.ndarray,
) -> np.ndarray:
    pixels = _marker(vendor_base, marker_mask)
    output = np.asarray(vendor_base).copy()
    if np.any(pixels):
        colour = np.asarray(BrownConfig.INSTANCE_COLOR, dtype=np.float32)
        output[pixels] = np.clip(
            output[pixels].astype(np.float32) * 0.15 + colour * 0.85,
            0,
            255,
        ).astype(np.uint8)
    return _with_boundary(preprocessed, output)


def render_consumer_brown_result(
    preprocessed: PreprocessResultV2,
    consumer_base: np.ndarray,
    marker_mask: np.ndarray,
) -> np.ndarray:
    output = render_compact_marker_mask(
        consumer_base,
        marker_mask,
        BROWN_STYLE_MARKER,
    )
    return _with_boundary(preprocessed, output)


__all__ = [
    "render_consumer_brown_result",
    "render_vendor_brown_result",
    "render_vendor_red_result",
]
