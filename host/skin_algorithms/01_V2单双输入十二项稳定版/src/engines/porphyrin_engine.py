# -*- coding: utf-8 -*-
"""Visible Porphyrin Proxy V1 for ordinary RGB face photographs.

This module detects *visible proxy candidates* with a pore-like dark core,
centre-to-ring luminance contrast and locally unusual red/brown/blue chroma.
It does not use UV illumination, fluorescence or microbiology information and
must never be presented as VISIA Porphyrins or a medical measurement.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from skimage.feature import peak_local_max

from src.engines.visia_regions import (
    REGION_LABELS,
    REGION_ORDER,
    VISIA_BOUNDARY_COLOR,
    assign_region,
    build_nasolabial_shadow_mask,
    build_visia_regions,
    compact_region_counts,
    draw_region_boundaries,
    masked_gaussian,
    regional_positive_z,
    robust_unit_map,
)
from src.preprocess.image_preprocessor import ImagePreprocessor, PreprocessResultV2
from src.utils.io_utils import cv_imread, cv_imwrite


class Config:
    """Visible Porphyrin Proxy V1 defaults for a 1024px Preprocessor V2 image.

    The values below describe deterministic engineering evidence, not a
    biological concentration model.  Keep the four fusion weights close to a
    total of one when tuning them.
    """

    # A. Pore-like dark-core evidence.  The kernels match the visible pores
    # detector, but proxy candidates must also pass colour and ring checks.
    BLACKHAT_KERNEL_SIZES = (5, 9, 13)
    PORE_SAFE_ERODE_KERNEL = 13
    PORE_ROBUST_LOW_PERCENTILE = 1.0
    PORE_ROBUST_HIGH_PERCENTILE = 99.5
    MIN_PORE_RESPONSE = 0.65

    # B. Local chroma anomaly.  HSV is deliberately *not* a purple detector;
    # it is only used below for highlight/shadow artefact exclusion.
    COLOR_BACKGROUND_SIGMA = 8.0
    LAB_A_LOCAL_WEIGHT = 0.46
    LAB_B_LOCAL_WEIGHT = 0.20
    RGB_RED_GREEN_WEIGHT = 0.18
    RGB_RED_BLUE_WEIGHT = 0.16
    COLOR_ROBUST_LOW_PERCENTILE = 2.0
    COLOR_ROBUST_HIGH_PERCENTILE = 99.0
    MIN_COLOR_RESPONSE = 0.55

    # C. Vectorised dark-centre / brighter-ring evidence.  All kernels are
    # normalized and evaluated through masked convolution, never Python pixel
    # loops.  The contrast unit is normalised LAB L in [0, 1].
    CENTER_RADIUS = 2.0
    RING_INNER_RADIUS = 3.0
    RING_OUTER_RADIUS = 6.0
    MIN_CENTER_RING_CONTRAST = 0.025
    RING_ROBUST_LOW_PERCENTILE = 1.0
    RING_ROBUST_HIGH_PERCENTILE = 99.3

    # D. Multi-evidence fusion.  T-zone is a weak engineering prior only;
    # location alone can never create a candidate.
    PORE_WEIGHT = 0.40
    CENTER_RING_WEIGHT = 0.25
    LOCAL_CHROMA_WEIGHT = 0.25
    T_ZONE_PRIOR_WEIGHT = 0.10
    T_ZONE_PRIOR_VALUE = 1.0
    NON_T_ZONE_PRIOR_VALUE = 0.0
    SCORE_ROBUST_LOW_PERCENTILE = 3.0
    SCORE_ROBUST_HIGH_PERCENTILE = 99.4

    # E. Candidate extraction and instance display.
    PEAK_MIN_DISTANCE = 10
    PEAK_THRESHOLD = 0.85
    MAX_FEATURES = 2500
    MIN_INSTANCE_CONFIDENCE = 0.78
    MIN_RADIUS_PX = 2
    MAX_RADIUS_PX = 5

    # F. Candidate-only rejection.  These values never alter the continuous
    # score map or render base image; they only remove invalid point instances.
    FEATURE_EXCLUSION_MARGIN_PX = 30
    NASOLABIAL_CORRIDOR_RADIUS_PX = 25
    NASOLABIAL_SHADOW_MARGIN_PX = 5
    HIGHLIGHT_VALUE = 0.96
    HIGHLIGHT_MAX_SATURATION = 0.22
    DEEP_SHADOW_VALUE = 0.14
    LOW_SATURATION_VALUE = 0.985
    LOW_SATURATION_MAX = 0.035

    # OpenCV BGR colours.  They affect only overlay readability.
    INSTANCE_COLOR = (205, 95, 170)  # cold purple / blue-purple
    REGION_COLOR = VISIA_BOUNDARY_COLOR
    OVERLAY_RADIUS = 2
    REGION_THICKNESS = 3


@dataclass
class PorphyrinResult:
    """Serializable result container for Visible Porphyrin Proxy V1."""

    porphyrin_count: int
    porphyrin_area_ratio: float
    porphyrin_locations: list[dict[str, Any]]
    region_distribution: dict[str, dict[str, Any]]
    mean_porphyrin_score: float
    p90_porphyrin_score: float
    porphyrin_burden: float
    porphyrin_score_map: np.ndarray
    porphyrin_mask: np.ndarray
    overlay: np.ndarray
    region_debug: np.ndarray
    analysis_mask: np.ndarray
    feature_exclusion_mask: np.ndarray
    partial_face: bool
    quality_score: float
    quality_status: str
    quality_flags: list[str]
    cross_validation: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        """Return JSON-safe metrics only; images and NumPy arrays stay out."""
        return {
            "algorithm": "Visible Porphyrin Proxy V1",
            "definition": (
                "普通RGB白光照片中的可见紫质代理特征：毛孔样暗核、"
                "中心—环结构与局部相对色度异常的工程候选。"
            ),
            "porphyrin_count": int(self.porphyrin_count),
            "porphyrin_area_ratio": float(self.porphyrin_area_ratio),
            "porphyrin_locations": self.porphyrin_locations,
            "region_distribution": self.region_distribution,
            "mean_porphyrin_score": float(self.mean_porphyrin_score),
            "p90_porphyrin_score": float(self.p90_porphyrin_score),
            "porphyrin_burden": float(self.porphyrin_burden),
            "partial_face": bool(self.partial_face),
            "quality_score": float(self.quality_score),
            "quality_status": str(self.quality_status),
            "quality_flags": list(self.quality_flags),
            "cross_validation": self.cross_validation,
            "confidence_definition": (
                "confidence 是确定性工程候选评分，不是医学概率、"
                "真实卟啉浓度或疾病概率。"
            ),
            "disclaimer": (
                "该结果不是 VISIA 官方 Porphyrins、UV 荧光紫质检测、"
                "痤疮丙酸杆菌检测、医学诊断或真实卟啉含量测量。"
            ),
        }


class PorphyrinAnalyzer:
    """Detector using only the algorithm-safe fields of PreprocessResultV2."""

    _HARD_FAILURE_FLAGS = {
        "NO_FACE",
        "MULTIPLE_FACES",
        "INVALID_IMAGE",
        "INVALID_FACE_GEOMETRY",
    }

    def __init__(self) -> None:
        self.last_result: PorphyrinResult | None = None
        self.last_regions = None
        self._compat_preprocessor: ImagePreprocessor | None = None

    @staticmethod
    def _sanitize_analysis_image(image: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
        """Remove mask-exterior influence without reading ``display_image``."""
        safe = np.asarray(image).copy()
        valid = skin_mask > 0
        if not np.any(valid):
            safe[:] = 0
            return safe
        fill = np.median(safe[valid], axis=0).astype(np.uint8)
        safe[~valid] = fill
        return safe

    @staticmethod
    def _safe_analysis_mask(mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (Config.PORE_SAFE_ERODE_KERNEL, Config.PORE_SAFE_ERODE_KERNEL),
        )
        return cv2.erode((mask > 0).astype(np.uint8) * 255, kernel)

    @staticmethod
    def _masked_filter(channel: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Masked normalized convolution for a disk/ring kernel."""
        valid = (mask > 0).astype(np.float32)
        values = channel.astype(np.float32)
        numerator = cv2.filter2D(values * valid, cv2.CV_32F, kernel)
        denominator = cv2.filter2D(valid, cv2.CV_32F, kernel)
        output = numerator / np.maximum(denominator, 1e-6)
        output[mask == 0] = 0.0
        return output.astype(np.float32)

    @staticmethod
    def _radial_kernel(
        radius: float,
        *,
        inner_radius: float = -1.0,
    ) -> np.ndarray:
        size = int(np.ceil(radius) * 2 + 1)
        yy, xx = np.indices((size, size), dtype=np.float32)
        centre = float(size // 2)
        distance = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
        if inner_radius >= 0.0:
            kernel = ((distance >= inner_radius) & (distance <= radius)).astype(np.float32)
        else:
            kernel = (distance <= radius).astype(np.float32)
        total = float(kernel.sum())
        if total <= 0.0:
            raise ValueError("invalid radial convolution kernel")
        return kernel / total

    @staticmethod
    def _pore_like_score(
        image: np.ndarray,
        regions,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return robust pore score, raw response and per-pixel dominant scale."""
        mask = regions.analysis_mask
        safe_mask = PorphyrinAnalyzer._safe_analysis_mask(mask)
        l_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        filled = l_channel.copy()
        values = l_channel[mask > 0]
        fill_value = int(np.median(values)) if values.size else 128
        filled[mask == 0] = fill_value

        responses: list[np.ndarray] = []
        for size in Config.BLACKHAT_KERNEL_SIZES:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            response = cv2.morphologyEx(filled, cv2.MORPH_BLACKHAT, kernel)
            responses.append(response.astype(np.float32) / 255.0)
        stack = np.stack(responses, axis=0)
        raw = 0.70 * stack.max(axis=0) + 0.30 * stack.mean(axis=0)
        raw[safe_mask == 0] = 0.0
        robust = regional_positive_z(raw, regions, sigma_floor=0.005)
        score = robust_unit_map(
            robust,
            safe_mask,
            Config.PORE_ROBUST_LOW_PERCENTILE,
            Config.PORE_ROBUST_HIGH_PERCENTILE,
        )
        score[safe_mask == 0] = 0.0
        dominant_scale = np.argmax(stack, axis=0).astype(np.uint8)
        return score.astype(np.float32), raw.astype(np.float32), dominant_scale

    @staticmethod
    def _local_colour_score(
        image: np.ndarray,
        regions,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Local relative chroma anomaly without absolute HSV purple thresholds."""
        mask = regions.analysis_mask
        safe_mask = PorphyrinAnalyzer._safe_analysis_mask(mask)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = (lab[:, :, 1] - 128.0) / 127.0
        b = (lab[:, :, 2] - 128.0) / 127.0
        bgr = image.astype(np.float32) / 255.0
        blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        total = np.maximum(red + green + blue, 1e-6)
        nr, ng, nb = red / total, green / total, blue / total
        red_green = nr - ng
        red_blue = nr - nb

        a_bg = masked_gaussian(a, mask, Config.COLOR_BACKGROUND_SIGMA)
        b_bg = masked_gaussian(b, mask, Config.COLOR_BACKGROUND_SIGMA)
        rg_bg = masked_gaussian(red_green, mask, Config.COLOR_BACKGROUND_SIGMA)
        rb_bg = masked_gaussian(red_blue, mask, Config.COLOR_BACKGROUND_SIGMA)
        a_residual = np.maximum(a - a_bg, 0.0)
        b_residual = np.abs(b - b_bg)
        rg_residual = np.maximum(red_green - rg_bg, 0.0)
        rb_residual = np.maximum(red_blue - rb_bg, 0.0)
        raw = (
            Config.LAB_A_LOCAL_WEIGHT * a_residual
            + Config.LAB_B_LOCAL_WEIGHT * b_residual
            + Config.RGB_RED_GREEN_WEIGHT * rg_residual
            + Config.RGB_RED_BLUE_WEIGHT * rb_residual
        )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        highlight = (value >= Config.HIGHLIGHT_VALUE) & (saturation <= Config.HIGHLIGHT_MAX_SATURATION)
        deep_shadow = value <= Config.DEEP_SHADOW_VALUE
        low_saturation_artifact = (
            (value >= Config.LOW_SATURATION_VALUE)
            & (saturation <= Config.LOW_SATURATION_MAX)
        )
        raw[highlight | deep_shadow | low_saturation_artifact] = 0.0
        raw[safe_mask == 0] = 0.0
        robust = regional_positive_z(raw, regions, sigma_floor=0.003)
        score = robust_unit_map(
            robust,
            safe_mask,
            Config.COLOR_ROBUST_LOW_PERCENTILE,
            Config.COLOR_ROBUST_HIGH_PERCENTILE,
        )
        score[safe_mask == 0] = 0.0
        return score.astype(np.float32), raw.astype(np.float32), saturation, value

    @staticmethod
    def _center_ring_score(
        image: np.ndarray,
        regions,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised ring-minus-centre luminance contrast for every pixel."""
        mask = regions.analysis_mask
        safe_mask = PorphyrinAnalyzer._safe_analysis_mask(mask)
        l_channel = (
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            / 255.0
        )
        centre_kernel = PorphyrinAnalyzer._radial_kernel(Config.CENTER_RADIUS)
        ring_kernel = PorphyrinAnalyzer._radial_kernel(
            Config.RING_OUTER_RADIUS,
            inner_radius=Config.RING_INNER_RADIUS,
        )
        centre = PorphyrinAnalyzer._masked_filter(l_channel, mask, centre_kernel)
        ring = PorphyrinAnalyzer._masked_filter(l_channel, mask, ring_kernel)
        raw = np.maximum(ring - centre, 0.0)
        raw[safe_mask == 0] = 0.0
        robust = regional_positive_z(raw, regions, sigma_floor=0.002)
        score = robust_unit_map(
            robust,
            safe_mask,
            Config.RING_ROBUST_LOW_PERCENTILE,
            Config.RING_ROBUST_HIGH_PERCENTILE,
        )
        score[safe_mask == 0] = 0.0
        return score.astype(np.float32), raw.astype(np.float32)

    @staticmethod
    def _t_zone_prior(regions) -> np.ndarray:
        prior = np.zeros_like(regions.analysis_mask, dtype=np.float32)
        for name in ("forehead", "nose", "chin"):
            mask = regions.regions.get(name)
            if mask is not None:
                prior[mask > 0] = Config.T_ZONE_PRIOR_VALUE
        prior[regions.analysis_mask == 0] = 0.0
        return prior

    @staticmethod
    def _fusion_score(
        pore_score: np.ndarray,
        colour_score: np.ndarray,
        ring_score: np.ndarray,
        regions,
    ) -> np.ndarray:
        prior = PorphyrinAnalyzer._t_zone_prior(regions)
        raw = (
            Config.PORE_WEIGHT * pore_score
            + Config.CENTER_RING_WEIGHT * ring_score
            + Config.LOCAL_CHROMA_WEIGHT * colour_score
            + Config.T_ZONE_PRIOR_WEIGHT * prior
        )
        # Percentiles are evaluated inside the valid analysis mask only.
        score = robust_unit_map(
            raw,
            regions.analysis_mask,
            Config.SCORE_ROBUST_LOW_PERCENTILE,
            Config.SCORE_ROBUST_HIGH_PERCENTILE,
        )
        score[regions.analysis_mask == 0] = 0.0
        return score.astype(np.float32)

    @staticmethod
    def _candidate_instances(
        score: np.ndarray,
        pore_score: np.ndarray,
        colour_score: np.ndarray,
        ring_score: np.ndarray,
        ring_raw: np.ndarray,
        dominant_scale: np.ndarray,
        saturation: np.ndarray,
        value: np.ndarray,
        regions,
        nasolabial_shadow: np.ndarray,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        coordinates = peak_local_max(
            score,
            min_distance=Config.PEAK_MIN_DISTANCE,
            threshold_abs=Config.PEAK_THRESHOLD,
            labels=(regions.analysis_mask > 0).astype(np.uint8),
            exclude_border=False,
            num_peaks=Config.MAX_FEATURES,
        )
        locations: list[dict[str, Any]] = []
        instance_mask = np.zeros(score.shape, dtype=np.uint8)
        kernel_radii = [(size - 1) // 2 for size in Config.BLACKHAT_KERNEL_SIZES]

        for y, x in coordinates:
            if regions.feature_exclusion_mask[y, x] > 0 or nasolabial_shadow[y, x] > 0:
                continue
            if value[y, x] <= Config.DEEP_SHADOW_VALUE:
                continue
            if value[y, x] >= Config.HIGHLIGHT_VALUE and saturation[y, x] <= Config.HIGHLIGHT_MAX_SATURATION:
                continue
            if pore_score[y, x] < Config.MIN_PORE_RESPONSE:
                continue
            if colour_score[y, x] < Config.MIN_COLOR_RESPONSE:
                continue
            if ring_raw[y, x] < Config.MIN_CENTER_RING_CONTRAST:
                continue
            region = assign_region(float(x), float(y), regions.regions)
            if region == "other":
                continue
            proxy_score = float(score[y, x])
            confidence = float(
                np.clip(
                    0.08
                    + 0.40 * proxy_score
                    + 0.27 * float(pore_score[y, x])
                    + 0.17 * float(ring_score[y, x])
                    + 0.08 * float(colour_score[y, x]),
                    0.0,
                    1.0,
                )
            )
            if confidence < Config.MIN_INSTANCE_CONFIDENCE:
                continue
            scale_index = int(dominant_scale[y, x])
            radius = int(
                np.clip(
                    max(1, kernel_radii[scale_index] // 2),
                    Config.MIN_RADIUS_PX,
                    Config.MAX_RADIUS_PX,
                )
            )
            x_int, y_int = int(x), int(y)
            cv2.circle(instance_mask, (x_int, y_int), radius, 255, -1, lineType=cv2.LINE_AA)
            locations.append(
                {
                    "id": len(locations) + 1,
                    "centroid": [x_int, y_int],
                    "bbox": [x_int - radius, y_int - radius, 2 * radius + 1, 2 * radius + 1],
                    "radius_px": radius,
                    "dominant_scale_kernel": int(Config.BLACKHAT_KERNEL_SIZES[scale_index]),
                    "region": region,
                    "pore_response": round(float(pore_score[y, x]), 6),
                    "color_response": round(float(colour_score[y, x]), 6),
                    "center_ring_contrast": round(float(ring_raw[y, x]), 6),
                    "porphyrin_score": round(proxy_score, 6),
                    "confidence": round(confidence, 4),
                    "confidence_definition": "确定性工程候选评分，非医学概率",
                }
            )
        return locations, (instance_mask > 0).astype(np.uint8) * 255

    @staticmethod
    def _region_distribution(
        locations: list[dict[str, Any]],
        instance_mask: np.ndarray,
        score: np.ndarray,
        regions,
    ) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        t_region_names = {"forehead", "nose", "chin"}
        aggregate_masks: dict[str, np.ndarray] = {}
        t_mask = np.zeros_like(regions.analysis_mask)
        non_t_mask = np.zeros_like(regions.analysis_mask)
        for name in REGION_ORDER:
            region_mask = regions.regions[name]
            if name in t_region_names:
                t_mask = cv2.bitwise_or(t_mask, region_mask)
            else:
                non_t_mask = cv2.bitwise_or(non_t_mask, region_mask)
        aggregate_masks["t_zone"] = t_mask
        aggregate_masks["non_t_zone"] = non_t_mask

        def summarize(name: str, label: str, region_mask: np.ndarray) -> dict[str, Any]:
            valid = region_mask > 0
            local = [item for item in locations if item["region"] == name] if name in REGION_ORDER else [
                item
                for item in locations
                if (item["region"] in t_region_names) == (name == "t_zone")
            ]
            area = int(np.count_nonzero((instance_mask > 0) & valid))
            region_area = max(int(np.count_nonzero(valid)), 1)
            values = score[valid]
            return {
                "label": label,
                "count": len(local),
                "area": area,
                "area_ratio": round(float(area / region_area), 8),
                "mean_score": round(float(np.mean(values)) if values.size else 0.0, 6),
            }

        for name in REGION_ORDER:
            output[name] = summarize(name, REGION_LABELS[name], regions.regions[name])
        output["t_zone"] = summarize("t_zone", "T区（工程代理）", aggregate_masks["t_zone"])
        output["non_t_zone"] = summarize("non_t_zone", "非T区（工程代理）", aggregate_masks["non_t_zone"])
        return output

    @staticmethod
    def _region_debug(image: np.ndarray, regions) -> np.ndarray:
        debug = image.copy()
        tint = np.zeros_like(debug)
        tint[regions.analysis_mask > 0] = (100, 35, 80)
        debug = cv2.addWeighted(debug, 0.80, tint, 0.20, 0.0)
        return draw_region_boundaries(
            debug,
            regions.display_regions,
            Config.REGION_COLOR,
            thickness=Config.REGION_THICKNESS,
            partial_face=regions.partial_face,
            closed_contour=getattr(regions, "display_contour", None),
            separator_contour=getattr(regions, "display_separator", None),
        )

    @staticmethod
    def _empty_result(preprocess_result: PreprocessResultV2) -> PorphyrinResult:
        shape = preprocess_result.skin_mask.shape
        blank_mask = np.zeros(shape, dtype=np.uint8)
        blank_score = np.zeros(shape, dtype=np.float32)
        image = np.asarray(preprocess_result.analysis_image).copy()
        return PorphyrinResult(
            porphyrin_count=0,
            porphyrin_area_ratio=0.0,
            porphyrin_locations=[],
            region_distribution={},
            mean_porphyrin_score=0.0,
            p90_porphyrin_score=0.0,
            porphyrin_burden=0.0,
            porphyrin_score_map=blank_score,
            porphyrin_mask=blank_mask,
            overlay=image,
            region_debug=image.copy(),
            analysis_mask=blank_mask,
            feature_exclusion_mask=blank_mask,
            partial_face="PARTIAL_FACE" in set(preprocess_result.quality_flags),
            quality_score=float(preprocess_result.quality_score),
            quality_status=str(preprocess_result.quality_status),
            quality_flags=list(preprocess_result.quality_flags),
        )

    def detect_porphyrins(self, preprocess_result: PreprocessResultV2) -> PorphyrinResult:
        """Detect proxy candidates from analysis image, skin mask and landmarks only."""
        image = np.asarray(preprocess_result.analysis_image)
        skin_mask = np.asarray(preprocess_result.skin_mask)
        landmarks = np.asarray(preprocess_result.landmarks)
        if (
            self._HARD_FAILURE_FLAGS.intersection(preprocess_result.quality_flags)
            or image.ndim != 3
            or image.shape[:2] != skin_mask.shape
            or landmarks.ndim != 2
            or landmarks.shape[1] < 2
            or np.count_nonzero(skin_mask) == 0
        ):
            result = self._empty_result(preprocess_result)
            self.last_result = result
            return result

        # Detection sees a background-invariant working image.  The original
        # analysis image remains the aligned visual base for the final overlay.
        working_image = self._sanitize_analysis_image(image, skin_mask)
        regions = build_visia_regions(
            working_image,
            skin_mask,
            landmarks,
            preprocess_result.quality_flags,
            include_chin=True,
            mode="full",
            feature_margin_px=Config.FEATURE_EXCLUSION_MARGIN_PX,
        )
        if np.count_nonzero(regions.analysis_mask) < 500:
            result = self._empty_result(preprocess_result)
            self.last_result = result
            return result

        pore_score, _pore_raw, dominant_scale = self._pore_like_score(working_image, regions)
        colour_score, _colour_raw, saturation, value = self._local_colour_score(working_image, regions)
        ring_score, ring_raw = self._center_ring_score(working_image, regions)
        score = self._fusion_score(pore_score, colour_score, ring_score, regions)
        nasolabial_shadow = build_nasolabial_shadow_mask(
            working_image,
            landmarks,
            regions.analysis_mask,
            radius_px=Config.NASOLABIAL_CORRIDOR_RADIUS_PX,
            output_margin_px=Config.NASOLABIAL_SHADOW_MARGIN_PX,
        )
        locations, instance_mask = self._candidate_instances(
            score,
            pore_score,
            colour_score,
            ring_score,
            ring_raw,
            dominant_scale,
            saturation,
            value,
            regions,
            nasolabial_shadow,
        )

        overlay = image.copy()
        for item in locations:
            x, y = item["centroid"]
            cv2.circle(
                overlay,
                (int(x), int(y)),
                Config.OVERLAY_RADIUS,
                Config.INSTANCE_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )
        overlay = draw_region_boundaries(
            overlay,
            regions.display_regions,
            Config.REGION_COLOR,
            thickness=Config.REGION_THICKNESS,
            partial_face=regions.partial_face,
            closed_contour=getattr(regions, "display_contour", None),
            separator_contour=getattr(regions, "display_separator", None),
        )

        valid = regions.analysis_mask > 0
        values = score[valid]
        analysis_area = max(int(np.count_nonzero(valid)), 1)
        area_ratio = float(np.count_nonzero(instance_mask) / analysis_area)
        result = PorphyrinResult(
            porphyrin_count=len(locations),
            porphyrin_area_ratio=round(area_ratio, 8),
            porphyrin_locations=locations,
            region_distribution=self._region_distribution(locations, instance_mask, score, regions),
            mean_porphyrin_score=round(float(np.mean(values)) if values.size else 0.0, 6),
            p90_porphyrin_score=round(float(np.percentile(values, 90)) if values.size else 0.0, 6),
            porphyrin_burden=round(float(np.sum(values) / analysis_area) if values.size else 0.0, 6),
            porphyrin_score_map=score,
            porphyrin_mask=instance_mask,
            overlay=overlay,
            region_debug=self._region_debug(image, regions),
            analysis_mask=regions.analysis_mask,
            feature_exclusion_mask=regions.feature_exclusion_mask,
            partial_face=regions.partial_face,
            quality_score=float(preprocess_result.quality_score),
            quality_status=str(preprocess_result.quality_status),
            quality_flags=list(preprocess_result.quality_flags),
        )
        self.last_regions = regions
        self.last_result = result
        return result

    @staticmethod
    def _score_visual(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gray = np.rint(np.clip(score, 0.0, 1.0) * 255.0).astype(np.uint8)
        visual = cv2.applyColorMap(gray, cv2.COLORMAP_TWILIGHT_SHIFTED)
        visual[mask == 0] = 0
        return visual

    @staticmethod
    def save_result(
        result: PorphyrinResult,
        output_dir: str,
        source_name: str,
        *,
        test_input: np.ndarray | None = None,
        flat_output: bool = False,
    ) -> dict[str, str]:
        """Persist deterministic local artifacts; all metrics are JSON-safe."""
        base_name = os.path.splitext(os.path.basename(source_name))[0]
        sample_dir = output_dir if flat_output else os.path.join(output_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)
        paths = {
            "test_input": os.path.join(sample_dir, "00_test_input.jpg"),
            "overlay": os.path.join(sample_dir, "01_紫质代理检测结果图.jpg"),
            "score": os.path.join(sample_dir, "02_紫质代理强度图.png"),
            "mask": os.path.join(sample_dir, "03_紫质代理实例Mask.png"),
            "region_debug": os.path.join(sample_dir, "04_紫质代理区域调试图.jpg"),
            "csv": os.path.join(sample_dir, "紫质代理量化指标.csv"),
            "json": os.path.join(sample_dir, "紫质代理量化指标.json"),
        }
        if test_input is not None:
            cv_imwrite(paths["test_input"], test_input)
        cv_imwrite(paths["overlay"], result.overlay)
        cv_imwrite(paths["score"], PorphyrinAnalyzer._score_visual(result.porphyrin_score_map, result.analysis_mask))
        cv_imwrite(paths["mask"], (result.porphyrin_mask > 0).astype(np.uint8) * 255)
        cv_imwrite(paths["region_debug"], result.region_debug)
        headers, values = compact_region_counts(result.porphyrin_locations, result.partial_face)
        with open(paths["csv"], "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerow(values)
        with open(paths["json"], "w", encoding="utf-8") as handle:
            json.dump(result.metrics(), handle, ensure_ascii=False, indent=2)
        return paths

    def _get_compat_preprocessor(self) -> ImagePreprocessor:
        if self._compat_preprocessor is None:
            self._compat_preprocessor = ImagePreprocessor(None, None, None)
        return self._compat_preprocessor

    def process_preprocess_result(
        self,
        preprocess_result: PreprocessResultV2,
        output_dir: str,
        source_name: str,
    ) -> str | None:
        if preprocess_result.quality_status == "REJECT":
            return None
        result = self.detect_porphyrins(preprocess_result)
        paths = self.save_result(result, output_dir, source_name)
        return paths["overlay"]

    def process_image(self, image_path: str, output_dir: str) -> str | None:
        """Compatibility adapter for callers that still pass an image path."""
        image = cv_imread(image_path)
        if image is None:
            return None
        preprocess_result = self._get_compat_preprocessor().preprocess_image(image)
        return self.process_preprocess_result(preprocess_result, output_dir, image_path)

    def close(self) -> None:
        if self._compat_preprocessor is not None:
            self._compat_preprocessor.close()
            self._compat_preprocessor = None


def detect_porphyrins(preprocess_result: PreprocessResultV2) -> PorphyrinResult:
    """Module-level convenience API used by local validation code."""
    return PorphyrinAnalyzer().detect_porphyrins(preprocess_result)


# Compatibility typo retained only for historical local scripts.
detect_porhyrins = detect_porphyrins
