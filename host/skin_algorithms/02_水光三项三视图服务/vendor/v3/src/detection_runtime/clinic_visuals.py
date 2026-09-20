from __future__ import annotations

"""Frozen four-light display helpers extracted from the validated runtime."""

from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from src.engines.clinic_fluorescence_porphyrin_engine import FluorescencePorphyrinAnalyzer
from src.engines.uv365_spots_engine import UV365SpotsAnalyzer
from src.engines.visia_regions import VISIA_BOUNDARY_COLOR, build_visia_regions, draw_region_boundaries
from src.preprocess.image_preprocessor import LEFT_EYE, LIPS, RIGHT_EYE, PreprocessResultV2


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _regions(pre: PreprocessResultV2) -> Any:
    return build_visia_regions(
        pre.analysis_image, pre.skin_mask, pre.landmarks, pre.quality_flags,
        include_chin=True, mode="full", feature_margin_px=10,
    )


def with_public_visia_boundary(pre: PreprocessResultV2, image: np.ndarray) -> np.ndarray:
    regions = _regions(pre)
    return draw_region_boundaries(
        image, regions.display_regions, color=VISIA_BOUNDARY_COLOR, thickness=2,
        partial_face=regions.partial_face, nose_style="hidden", landmarks=pre.landmarks,
        closed_contour=regions.display_contour, separator_contour=regions.display_separator,
    )


def _fill_landmark_polygon(
    target: np.ndarray,
    landmarks: np.ndarray,
    indices: tuple[int, ...] | list[int],
) -> None:
    points = np.rint(landmarks[np.asarray(indices, dtype=np.int32), :2]).astype(np.int32)
    cv2.fillPoly(target, [points], 255, lineType=cv2.LINE_AA)


def _porphyrin_domain(
    pre: PreprocessResultV2,
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray]:
    standard = _regions(pre)
    display_scope = np.zeros(pre.skin_mask.shape, np.uint8)
    cv2.fillConvexPoly(
        display_scope,
        cv2.convexHull(np.rint(pre.landmarks[:, :2]).astype(np.int32)),
        255,
    )
    display_scope = cv2.bitwise_or(
        display_scope,
        _binary(standard.display_regions.get("forehead", np.zeros_like(display_scope))),
    )
    debug = getattr(pre, "_debug_masks", {}) or {}
    compact_features = _binary(standard.feature_exclusion_mask)
    for indices in (LEFT_EYE, RIGHT_EYE, LIPS):
        local = np.zeros_like(display_scope)
        _fill_landmark_polygon(local, pre.landmarks, indices)
        local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        compact_features = cv2.bitwise_or(compact_features, local)
    compact_features = cv2.bitwise_or(
        compact_features,
        _binary(debug.get("nostril_mask", np.zeros_like(display_scope))),
    )
    domain = cv2.bitwise_and(
        _binary(standard.analysis_mask),
        cv2.bitwise_not(compact_features),
    )
    occupied = np.zeros_like(domain)
    fluorescence_regions: dict[str, np.ndarray] = {}
    for name in ("nose", "forehead", "left_cheek", "right_cheek", "chin"):
        local = cv2.bitwise_and(_binary(standard.regions[name]), domain)
        local = cv2.bitwise_and(local, cv2.bitwise_not(occupied))
        fluorescence_regions[name] = local
        occupied = cv2.bitwise_or(occupied, local)
    return domain, compact_features, replace(
        standard,
        analysis_mask=occupied,
        feature_exclusion_mask=compact_features,
        regions=fluorescence_regions,
    ), display_scope


def uv_results(pre: PreprocessResultV2) -> tuple[Any, Any, Any]:
    regions = _regions(pre)
    debug = getattr(pre, "_debug_masks", {}) or {}
    exclusion = cv2.bitwise_or(
        _binary(debug.get("feature_exclusions", np.zeros((1024, 1024), np.uint8))),
        _binary(debug.get("hair_mask", np.zeros((1024, 1024), np.uint8))),
    )
    (
        porphyrin_domain,
        compact_features,
        porphyrin_regions,
        porphyrin_display_scope,
    ) = _porphyrin_domain(pre)
    fluorescence = FluorescencePorphyrinAnalyzer().detect(
        pre.analysis_image, porphyrin_regions, pre.landmarks,
        instance_exclusion_mask=compact_features,
        display_scope_mask=porphyrin_display_scope,
        analysis_domain_mask=porphyrin_domain,
    )
    uv = UV365SpotsAnalyzer().detect(
        pre.analysis_image, regions, pre.landmarks,
        porphyrin_mask=cv2.bitwise_or(
            fluorescence.formal_instance_mask,
            fluorescence.accepted_visible_support_mask,
        ),
        display_base=pre.display_image,
        instance_exclusion_mask=exclusion,
    )
    return regions, uv, fluorescence


__all__ = ["uv_results", "with_public_visia_boundary"]
