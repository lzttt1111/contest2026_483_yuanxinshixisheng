# -*- coding: utf-8 -*-
"""UV-like 底图上的独立紫外线色斑工程检测。

本引擎把输入底图视为已完成 UV 风格化的标准化人脸图，只检测相对周围
更暗、且获得多个物理尺度支持的局部色素结构。它不复用 Brown/Spots 的
最终实例，也不硬编码任何 VISIA 样本数量。

由于输入仍来源于普通白光 RGB，本结果只能称为 ``UV-like Spots Proxy``，
不能解释为真实 UV 隐藏色素、黑色素深度或 VISIA 官方 UV Spots。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.feature import peak_local_max

from src.engines.visia_regions import (
    REGION_LABELS,
    REGION_ORDER,
    VISIA_BOUNDARY_COLOR,
    assign_region,
    draw_region_boundaries,
    masked_gaussian,
    regional_positive_z,
    robust_unit_map,
)


@dataclass(frozen=True)
class UVSpotsConfig:
    """UV-like 色斑检测集中调参区，默认按 1024×1024 输入设置。"""

    # 局部背景尺度。调大更关注大范围色素，调小更容易检出毛孔和噪声。
    background_sigmas: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)
    scale_z_threshold: float = 1.20
    minimum_scale_votes: int = 2
    safe_mask_erode_kernel: int = 11

    # 暗结构形态尺度；作为局部暗度证据的补充而不是唯一判据。
    blackhat_kernel_sizes: tuple[int, ...] = (7, 13, 23, 39)
    multiscale_weight: float = 0.74
    blackhat_weight: float = 0.26

    # 局部峰与实例提取。降低阈值会增加召回，也会增加毛孔/噪声候选。
    peak_threshold: float = 0.86
    peak_min_distance: int = 8
    maximum_features: int = 800
    support_relative_threshold: float = 0.90
    support_absolute_threshold: float = 0.72
    support_radius_px: int = 6

    # 形状过滤只作用于实例，不改变连续分数图。
    minimum_area_px: int = 4
    maximum_area_px: int = 100
    minimum_mean_score: float = 0.70
    minimum_solidity: float = 0.12
    maximum_aspect_ratio: float = 4.5
    display_min_radius_px: int = 2
    display_max_radius_px: int = 8

    # VISIA-like 展示颜色，OpenCV BGR。
    instance_fill_bgr: tuple[int, int, int] = (0, 192, 255)
    instance_outline_bgr: tuple[int, int, int] = (0, 232, 255)
    fill_alpha: float = 0.88
    region_color_bgr: tuple[int, int, int] = VISIA_BOUNDARY_COLOR
    region_thickness: int = 2
    nose_style: str = "hidden"


@dataclass
class UVSpotsResult:
    overlay: np.ndarray
    instance_mask: np.ndarray
    score_map: np.ndarray
    instances: list[dict[str, Any]]
    region_distribution: dict[str, dict[str, Any]]
    # Internal measurement evidence; not the circular public display marker.
    measurement_support_mask: np.ndarray | None = None


MEASUREMENT_SUPPORT_DEFINITION = "accepted_detector_support_union_v1"


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _region_distribution(
    instances: list[dict[str, Any]],
    instance_mask: np.ndarray,
    score_map: np.ndarray,
    regions,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in REGION_ORDER:
        region_mask = regions.regions[name] > 0
        area = int(np.count_nonzero((instance_mask > 0) & region_mask))
        region_area = int(np.count_nonzero(region_mask))
        local = [item for item in instances if item["region"] == name]
        values = score_map[region_mask]
        output[name] = {
            "label": REGION_LABELS[name],
            "count": len(local),
            "area": area,
            "area_ratio": round(area / max(region_area, 1), 8),
            "mean_score": round(float(np.mean(values)) if values.size else 0.0, 6),
        }
    return output


class UVSpotsAnalyzer:
    """在 UV-like 灰黑底图上提取多尺度局部暗斑实例。"""

    def __init__(self, config: UVSpotsConfig | None = None) -> None:
        self.config = config or UVSpotsConfig()

    def _score_map(self, uv_like_base: np.ndarray, regions) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        gray = cv2.cvtColor(uv_like_base, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        analysis_mask = _binary_mask(regions.analysis_mask)
        analysis_mask = cv2.erode(
            analysis_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (config.safe_mask_erode_kernel, config.safe_mask_erode_kernel),
            ),
        )
        values = gray[analysis_mask > 0]
        filled = gray.copy()
        filled[analysis_mask == 0] = float(np.median(values)) if values.size else 0.5

        scale_scores: list[np.ndarray] = []
        scale_votes = np.zeros(gray.shape, dtype=np.uint8)
        for sigma in config.background_sigmas:
            background = masked_gaussian(filled, analysis_mask, sigma)
            dark = np.maximum(background - filled, 0.0)
            z = regional_positive_z(dark, regions, sigma_floor=0.0025)
            local_score = robust_unit_map(z, analysis_mask, 1.0, 99.2)
            local_score[analysis_mask == 0] = 0.0
            scale_scores.append(local_score.astype(np.float32))
            scale_votes += (z >= config.scale_z_threshold).astype(np.uint8)

        scale_stack = np.stack(scale_scores, axis=0)
        multiscale = 0.62 * np.max(scale_stack, axis=0) + 0.38 * np.mean(
            scale_stack,
            axis=0,
        )
        vote_support = np.clip(
            scale_votes.astype(np.float32) / max(config.minimum_scale_votes, 1),
            0.0,
            1.0,
        )
        multiscale *= 0.55 + 0.45 * vote_support

        gray_u8 = np.rint(filled * 255.0).astype(np.uint8)
        blackhat_scales: list[np.ndarray] = []
        for size in config.blackhat_kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            response = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel)
            blackhat_scales.append(response.astype(np.float32) / 255.0)
        blackhat_raw = np.max(np.stack(blackhat_scales, axis=0), axis=0)
        blackhat = robust_unit_map(blackhat_raw, analysis_mask, 2.0, 99.4)

        score = np.clip(
            config.multiscale_weight * multiscale
            + config.blackhat_weight * blackhat,
            0.0,
            1.0,
        ).astype(np.float32)
        score[(analysis_mask == 0) | (scale_votes < config.minimum_scale_votes)] = 0.0
        return score, scale_votes

    @staticmethod
    def _component_at_peak(
        score: np.ndarray,
        analysis_mask: np.ndarray,
        x: int,
        y: int,
        radius: int,
        threshold: float,
    ) -> np.ndarray:
        height, width = score.shape
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        local = (
            (score[y0:y1, x0:x1] >= threshold)
            & (analysis_mask[y0:y1, x0:x1] > 0)
        ).astype(np.uint8)
        if local[y - y0, x - x0] == 0:
            local[y - y0, x - x0] = 1
        _, labels = cv2.connectedComponents(local, connectivity=8)
        label = int(labels[y - y0, x - x0])
        result = np.zeros_like(analysis_mask, dtype=np.uint8)
        if label > 0:
            result[y0:y1, x0:x1][labels == label] = 255
        return result

    def detect(
        self,
        uv_like_base: np.ndarray,
        regions,
        landmarks: np.ndarray,
        *,
        display_base: np.ndarray | None = None,
        instance_exclusion_mask: np.ndarray | None = None,
    ) -> UVSpotsResult:
        config = self.config
        analysis_mask = _binary_mask(regions.analysis_mask)
        analysis_mask = cv2.erode(
            analysis_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (config.safe_mask_erode_kernel, config.safe_mask_erode_kernel),
            ),
        )
        score, scale_votes = self._score_map(uv_like_base, regions)
        coordinates = peak_local_max(
            score,
            min_distance=config.peak_min_distance,
            threshold_abs=config.peak_threshold,
            exclude_border=False,
            labels=(analysis_mask > 0).astype(np.uint8),
            num_peaks=config.maximum_features,
        )

        output_mask = np.zeros_like(analysis_mask)
        measurement_support = np.zeros_like(analysis_mask)
        claimed = np.zeros_like(analysis_mask)
        exclusion = (
            _binary_mask(instance_exclusion_mask)
            if instance_exclusion_mask is not None
            else np.zeros_like(analysis_mask)
        )
        instances: list[dict[str, Any]] = []
        for y_raw, x_raw in coordinates:
            x, y = int(x_raw), int(y_raw)
            if (
                analysis_mask[y, x] == 0
                or exclusion[y, x] > 0
                or claimed[y, x] > 0
            ):
                continue
            peak_score = float(score[y, x])
            threshold = max(
                config.support_absolute_threshold,
                config.support_relative_threshold * peak_score,
            )
            component_mask = self._component_at_peak(
                score,
                analysis_mask,
                x,
                y,
                config.support_radius_px,
                threshold,
            )
            component = component_mask > 0
            area = int(np.count_nonzero(component))
            if area < config.minimum_area_px or area > config.maximum_area_px:
                continue
            contours, _ = cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour_area = max(float(cv2.contourArea(contour)), 1.0)
            hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            solidity = contour_area / hull_area
            mean_score = float(np.mean(score[component]))
            if (
                mean_score < config.minimum_mean_score
                or solidity < config.minimum_solidity
                or aspect > config.maximum_aspect_ratio
            ):
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-6:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx, cy = float(x), float(y)
            region = assign_region(cx, cy, regions.regions)
            if region == "other":
                continue
            # Keep accepted detector components before any display-radius
            # conversion. Existing public masks/counts remain untouched.
            measurement_support[component & (exclusion == 0)] = 255
            support_area = area
            display_radius = int(
                np.clip(
                    round(math.sqrt(support_area / math.pi)),
                    config.display_min_radius_px,
                    config.display_max_radius_px,
                )
            )
            display_component = np.zeros_like(output_mask)
            cv2.circle(display_component, (x, y), display_radius, 255, -1)
            display_component = cv2.bitwise_and(display_component, analysis_mask)
            display_pixels = display_component > 0
            display_area = int(np.count_nonzero(display_pixels))
            output_mask[display_pixels] = 255
            claimed[display_pixels] = 255
            instances.append(
                {
                    "id": len(instances) + 1,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "area": display_area,
                    "support_area": support_area,
                    "equivalent_diameter": round(
                        2.0 * math.sqrt(display_area / math.pi),
                        3,
                    ),
                    "region": region,
                    "mean_score": round(mean_score, 6),
                    "peak_score": round(peak_score, 6),
                    "scale_votes": int(scale_votes[y, x]),
                    "solidity": round(solidity, 4),
                    "aspect_ratio": round(aspect, 4),
                    "confidence": round(float(np.clip(0.35 + 0.65 * mean_score, 0.0, 1.0)), 4),
                    "confidence_definition": "确定性工程候选评分，非医学概率",
                }
            )

        overlay = (
            np.asarray(display_base).copy()
            if display_base is not None
            else uv_like_base.copy()
        )
        pixels = output_mask > 0
        if np.any(pixels):
            fill = np.asarray(config.instance_fill_bgr, dtype=np.float32)
            overlay[pixels] = np.clip(
                (1.0 - config.fill_alpha) * overlay[pixels].astype(np.float32)
                + config.fill_alpha * fill,
                0.0,
                255.0,
            ).astype(np.uint8)
        contours, _ = cv2.findContours(output_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            overlay,
            contours,
            -1,
            config.instance_outline_bgr,
            1,
            lineType=cv2.LINE_AA,
        )
        overlay = draw_region_boundaries(
            overlay,
            regions.display_regions,
            color=config.region_color_bgr,
            thickness=config.region_thickness,
            partial_face=regions.partial_face,
            nose_style=config.nose_style,
            landmarks=np.asarray(landmarks),
            closed_contour=getattr(regions, "display_contour", None),
            separator_contour=getattr(regions, "display_separator", None),
        )
        return UVSpotsResult(
            overlay=overlay,
            instance_mask=_binary_mask(output_mask),
            score_map=np.clip(score, 0.0, 1.0).astype(np.float32),
            instances=instances,
            measurement_support_mask=measurement_support,
            region_distribution=_region_distribution(
                instances,
                output_mask,
                score,
                regions,
            ),
        )
