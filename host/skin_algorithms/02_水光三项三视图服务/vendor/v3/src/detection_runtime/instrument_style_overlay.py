from __future__ import annotations

"""Project accepted red/brown findings onto the untouched vendor canvas."""

import numpy as np

from src.consumer_pigment.markers import (
    BROWN_STYLE_MARKER,
    render_compact_marker_mask,
)
from src.engines.rbx_engine import Config as RedConfig
from src.engines.visia_regions import build_visia_regions, draw_region_boundaries
from src.preprocess.image_preprocessor import PreprocessResultV2


def _regions(preprocessed: PreprocessResultV2):
    return build_visia_regions(
        preprocessed.analysis_image,
        preprocessed.skin_mask,
        preprocessed.landmarks,
        preprocessed.quality_flags,
        include_chin=True,
        mode="full",
    )


def _validate(base: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
    canvas = np.asarray(base)
    marker = np.asarray(marker_mask)
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("instrument-style public canvas must be BGR")
    if marker.shape[:2] != canvas.shape[:2]:
        raise ValueError("instrument-style marker mask shape mismatch")
    return (marker > 0).astype(np.uint8) * 255


def _with_boundary(preprocessed: PreprocessResultV2, image: np.ndarray) -> np.ndarray:
    regions = _regions(preprocessed)
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
    *,
    include_boundary: bool,
) -> np.ndarray:
    marker = _validate(vendor_base, marker_mask)
    output = np.asarray(vendor_base).copy()
    output[marker > 0] = RedConfig.RED_FEATURE_MARKER_COLOR
    return _with_boundary(preprocessed, output) if include_boundary else output


def render_vendor_brown_result(
    preprocessed: PreprocessResultV2,
    vendor_base: np.ndarray,
    instance_mask: np.ndarray,
    *,
    include_boundary: bool,
) -> np.ndarray:
    marker = _validate(vendor_base, instance_mask)
    output = render_compact_marker_mask(
        np.asarray(vendor_base),
        marker,
        BROWN_STYLE_MARKER,
    )
    return _with_boundary(preprocessed, output) if include_boundary else output


__all__ = ["render_vendor_brown_result", "render_vendor_red_result"]
