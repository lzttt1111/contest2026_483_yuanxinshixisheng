# -*- coding: utf-8 -*-
"""标准化荧光 UV 底图上的紫质特征检测。

检测器只接受已经完成几何对齐和荧光风格调光的 BGR 图像。当前可输入
工程生成的荧光 UV 底图；后续替换为真实 UV 荧光图时无需改变实例提取
和量化接口。五官、胡须等禁区只在候选实例阶段删除，不参与底图生成，
也不修改连续分数图。
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
class FluorescencePorphyrinConfig:
    """按 1024×1024 标准化荧光 UV 图设置的工程参数。"""

    # 小型荧光结构相对局部背景的物理尺度。
    background_sigmas: tuple[float, ...] = (2.5, 5.0, 9.0, 15.0)
    scale_z_threshold: float = 1.25
    minimum_scale_votes: int = 2
    safe_mask_erode_kernel: int = 9

    # 分数融合。真实荧光图的暖色证据权重更高；工程底图仍可由亮度证据
    # 形成候选，因此更换真实 UV 图时不必重新设计检测框架。
    local_brightness_weight: float = 0.56
    warm_fluorescence_weight: float = 0.30
    local_saturation_weight: float = 0.14

    # 局部峰和实例形态。
    peak_threshold: float = 0.78
    peak_min_distance: int = 6
    maximum_features: int = 1200
    support_relative_threshold: float = 0.88
    support_absolute_threshold: float = 0.64
    support_radius_px: int = 6
    minimum_area_px: int = 3
    maximum_area_px: int = 110
    minimum_mean_score: float = 0.62
    minimum_solidity: float = 0.10
    maximum_aspect_ratio: float = 4.5
    display_min_radius_px: int = 2
    display_max_radius_px: int = 4

    # VISIA-like 黄色实例和青色区域边界，OpenCV BGR。
    instance_fill_bgr: tuple[int, int, int] = (0, 230, 255)
    instance_outline_bgr: tuple[int, int, int] = (0, 250, 255)
    fill_alpha: float = 0.92
    region_color_bgr: tuple[int, int, int] = VISIA_BOUNDARY_COLOR
    region_thickness: int = 2
    nose_style: str = "hidden"


@dataclass
class FluorescencePorphyrinResult:
    overlay: np.ndarray
    instance_mask: np.ndarray
    score_map: np.ndarray
    locations: list[dict[str, Any]]
    region_distribution: dict[str, dict[str, Any]]


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _distribution(
    locations: list[dict[str, Any]],
    instance_mask: np.ndarray,
    score_map: np.ndarray,
    regions,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in REGION_ORDER:
        region_mask = regions.regions[name] > 0
        local = [item for item in locations if item["region"] == name]
        region_area = int(np.count_nonzero(region_mask))
        instance_area = int(np.count_nonzero((instance_mask > 0) & region_mask))
        values = score_map[region_mask]
        output[name] = {
            "label": REGION_LABELS[name],
            "count": len(local),
            "area": instance_area,
            "area_ratio": round(instance_area / max(region_area, 1), 8),
            "mean_score": round(float(np.mean(values)) if values.size else 0.0, 6),
        }
    return output


class FluorescencePorphyrinAnalyzer:
    """从荧光 UV 底图提取局部荧光紫质候选。"""

    def __init__(self, config: FluorescencePorphyrinConfig | None = None) -> None:
        self.config = config or FluorescencePorphyrinConfig()

    def _score_map(self, image: np.ndarray, regions) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        analysis_mask = _binary_mask(regions.analysis_mask)
        safe_mask = cv2.erode(
            analysis_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (config.safe_mask_erode_kernel, config.safe_mask_erode_kernel),
            ),
        )
        bgr = image.astype(np.float32) / 255.0
        blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        total = np.maximum(red + green + blue, 1e-6)
        warm = (red + 0.42 * green - 0.58 * blue) / total
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0

        valid_values = luminance[safe_mask > 0]
        filled_l = luminance.copy()
        filled_l[safe_mask == 0] = float(np.median(valid_values)) if valid_values.size else 0.25

        brightness_scores: list[np.ndarray] = []
        warm_scores: list[np.ndarray] = []
        saturation_scores: list[np.ndarray] = []
        scale_votes = np.zeros(luminance.shape, dtype=np.uint8)
        for sigma in config.background_sigmas:
            bright_raw = np.maximum(filled_l - masked_gaussian(filled_l, safe_mask, sigma), 0.0)
            warm_raw = np.maximum(warm - masked_gaussian(warm, safe_mask, sigma), 0.0)
            saturation_raw = np.maximum(
                saturation - masked_gaussian(saturation, safe_mask, sigma),
                0.0,
            )
            bright_z = regional_positive_z(bright_raw, regions, sigma_floor=0.002)
            warm_z = regional_positive_z(warm_raw, regions, sigma_floor=0.002)
            saturation_z = regional_positive_z(
                saturation_raw,
                regions,
                sigma_floor=0.002,
            )
            brightness_scores.append(robust_unit_map(bright_z, safe_mask, 1.0, 99.4))
            warm_scores.append(robust_unit_map(warm_z, safe_mask, 1.0, 99.4))
            saturation_scores.append(
                robust_unit_map(saturation_z, safe_mask, 1.0, 99.4)
            )
            scale_votes += (
                np.maximum(bright_z, 0.72 * warm_z) >= config.scale_z_threshold
            ).astype(np.uint8)

        brightness = np.max(np.stack(brightness_scores, axis=0), axis=0)
        warm_score = np.max(np.stack(warm_scores, axis=0), axis=0)
        saturation_score = np.max(np.stack(saturation_scores, axis=0), axis=0)
        vote_support = np.clip(
            scale_votes.astype(np.float32) / max(config.minimum_scale_votes, 1),
            0.0,
            1.0,
        )
        score = np.clip(
            config.local_brightness_weight * brightness
            + config.warm_fluorescence_weight * warm_score
            + config.local_saturation_weight * saturation_score,
            0.0,
            1.0,
        )
        score *= 0.55 + 0.45 * vote_support
        score[(safe_mask == 0) | (scale_votes < config.minimum_scale_votes)] = 0.0
        return score.astype(np.float32), scale_votes

    @staticmethod
    def _support_component(
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
        local[y - y0, x - x0] = 1
        _, labels = cv2.connectedComponents(local, connectivity=8)
        selected = int(labels[y - y0, x - x0])
        output = np.zeros_like(analysis_mask)
        if selected > 0:
            output[y0:y1, x0:x1][labels == selected] = 255
        return output

    def detect(
        self,
        fluorescence_base: np.ndarray,
        regions,
        landmarks: np.ndarray,
        *,
        instance_exclusion_mask: np.ndarray | None = None,
    ) -> FluorescencePorphyrinResult:
        config = self.config
        analysis_mask = _binary_mask(regions.analysis_mask)
        score, scale_votes = self._score_map(fluorescence_base, regions)
        exclusion = (
            _binary_mask(instance_exclusion_mask)
            if instance_exclusion_mask is not None
            else np.zeros_like(analysis_mask)
        )
        coordinates = peak_local_max(
            score,
            min_distance=config.peak_min_distance,
            threshold_abs=config.peak_threshold,
            exclude_border=False,
            labels=(analysis_mask > 0).astype(np.uint8),
            num_peaks=config.maximum_features,
        )
        mask = np.zeros_like(analysis_mask)
        claimed = np.zeros_like(analysis_mask)
        locations: list[dict[str, Any]] = []
        for y_raw, x_raw in coordinates:
            x, y = int(x_raw), int(y_raw)
            # 五官与胡须禁区只在这里删除实例，不修改底图和连续分数图。
            if analysis_mask[y, x] == 0 or exclusion[y, x] > 0 or claimed[y, x] > 0:
                continue
            peak_score = float(score[y, x])
            threshold = max(
                config.support_absolute_threshold,
                config.support_relative_threshold * peak_score,
            )
            component = self._support_component(
                score,
                analysis_mask,
                x,
                y,
                config.support_radius_px,
                threshold,
            )
            pixels = component > 0
            support_area = int(np.count_nonzero(pixels))
            if support_area < config.minimum_area_px or support_area > config.maximum_area_px:
                continue
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
            contour_area = max(float(cv2.contourArea(contour)), 1.0)
            bx, by, bw, bh = cv2.boundingRect(contour)
            solidity = contour_area / hull_area
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            mean_score = float(np.mean(score[pixels]))
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
            radius = int(
                np.clip(
                    round(math.sqrt(support_area / math.pi)),
                    config.display_min_radius_px,
                    config.display_max_radius_px,
                )
            )
            marker = np.zeros_like(mask)
            cv2.circle(marker, (x, y), radius, 255, -1, lineType=cv2.LINE_AA)
            marker = cv2.bitwise_and(marker, analysis_mask)
            marker_pixels = marker > 0
            mask[marker_pixels] = 255
            claimed[marker_pixels] = 255
            locations.append(
                {
                    "id": len(locations) + 1,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "area": int(np.count_nonzero(marker_pixels)),
                    "support_area": support_area,
                    "region": region,
                    "mean_score": round(mean_score, 6),
                    "peak_score": round(peak_score, 6),
                    "scale_votes": int(scale_votes[y, x]),
                    "confidence": round(float(np.clip(0.30 + 0.70 * mean_score, 0.0, 1.0)), 4),
                    "confidence_definition": "确定性工程候选评分，非医学概率",
                }
            )

        overlay = fluorescence_base.copy()
        marker_pixels = mask > 0
        if np.any(marker_pixels):
            fill = np.asarray(config.instance_fill_bgr, dtype=np.float32)
            overlay[marker_pixels] = np.clip(
                (1.0 - config.fill_alpha) * overlay[marker_pixels].astype(np.float32)
                + config.fill_alpha * fill,
                0.0,
                255.0,
            ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            closed_contour=regions.display_contour,
            separator_contour=regions.display_separator,
        )
        return FluorescencePorphyrinResult(
            overlay=overlay,
            instance_mask=_binary_mask(mask),
            score_map=np.clip(score, 0.0, 1.0).astype(np.float32),
            locations=locations,
            region_distribution=_distribution(locations, mask, score, regions),
        )


__all__ = [
    "FluorescencePorphyrinAnalyzer",
    "FluorescencePorphyrinConfig",
    "FluorescencePorphyrinResult",
]
