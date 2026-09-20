# -*- coding: utf-8 -*-
"""DermaVision Surface Gloss V1.

用于标准化白光人脸照片中的“表面油光/镜面反射样”检测。

边界：
- 不测量绝对皮脂量；
- 不测量单位时间皮脂分泌速率；
- 汗液、护肤品、防晒、底妆、水膜、光源、曝光、拍摄角度都会影响结果；
- 本引擎只实现医生油脂方案中“表面油光”60%的底层证据；
- 毛囊卟啉40%必须继续由独立紫质/荧光模块提供，后续再由 oil_tendency_engine 融合。

医生核心指标：
1. 油光面积占比；
2. 高强度油光面积占比；
3. P50 油光强度；
4. P90 油光强度；
5. 最大连续油光区域面积占比。

兼容：
- PreprocessResultV2
- build_visia_regions()
- masked_gaussian()
- regional_positive_z()
- robust_unit_map()
- cv_imwrite()
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.engines.visia_regions import (
    VISIA_BOUNDARY_COLOR,
    VisiaRegionSet,
    build_visia_regions,
    masked_gaussian,
)
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.utils.io_utils import cv_imwrite


class Config:
    """1024×1024 Preprocessor V2 默认参数。"""

    # ------------------------------------------------------------------
    # Oil-specific analysis domain
    # ------------------------------------------------------------------
    # 皮肤Mask本身已排除眼/眉/唇/鼻孔/毛发，但油光对边缘反射特别敏感，
    # 因此再向内缩一圈并扩大现有排除Mask。
    OUTER_BOUNDARY_MARGIN_PX = 10
    FEATURE_SAFETY_DILATE_PX = 5
    HAIR_SAFETY_DILATE_PX = 7
    NOSTRIL_SAFETY_DILATE_PX = 5

    # 近白剪裁像素不参与正式油光统计；它们只记录“曝光/强反光不可靠”。
    CLIP_CHANNEL_THRESHOLD = 250
    CLIP_DILATE_PX = 2
    CLIP_WARNING_RATIO = 0.012

    # ------------------------------------------------------------------
    # Photometric field correction
    # ------------------------------------------------------------------
    # 先估计超大尺度照明场，压掉“整块脸因为曲率/灯光更亮”的低频梯度。
    ILLUMINATION_FIELD_SIGMA = 135.0
    ILLUMINATION_RATIO_MIN = 0.55
    ILLUMINATION_RATIO_MAX = 1.65

    # ------------------------------------------------------------------
    # Broad sheen: 宽域油膜光泽
    # ------------------------------------------------------------------
    # V3 使用多尺度“中位”响应，要求跨尺度持续存在，而不是任一尺度亮就算。
    BROAD_BACKGROUND_SIGMAS = (24.0, 42.0, 68.0)
    BROAD_RATIO_LOW = 0.018
    BROAD_RATIO_HIGH = 0.105

    # Broad sheen 也需要局部去色/镜面邻域支持，避免普通亮皮肤大片进入Mask。
    BROAD_SATURATION_DROP_LOW = 0.004
    BROAD_SATURATION_DROP_HIGH = 0.055
    BROAD_CHROMA_DROP_LOW = 0.004
    BROAD_CHROMA_DROP_HIGH = 0.050
    BROAD_COLOR_BACKGROUND_SIGMA = 28.0

    # ------------------------------------------------------------------
    # Specular highlight: 局部镜面反射
    # ------------------------------------------------------------------
    SPECULAR_BACKGROUND_SIGMAS = (8.0, 14.0, 22.0)
    SPECULAR_RATIO_LOW = 0.020
    SPECULAR_RATIO_HIGH = 0.155

    # 镜面反射相对于周围正常皮肤通常“更接近光源颜色”，表现为饱和度/色度下降。
    SATURATION_DROP_LOW = 0.010
    SATURATION_DROP_HIGH = 0.100
    CHROMA_DROP_LOW = 0.008
    CHROMA_DROP_HIGH = 0.080
    SPECULAR_COLOR_BACKGROUND_SIGMA = 16.0

    # 将局部镜面反射响应扩散成“附近存在镜面核心”的软支持，
    # Broad sheen 若完全没有这类支持则明显降权。
    SPECULAR_NEIGHBOR_SIGMA = 8.0
    SPECULAR_NEIGHBOR_GAIN = 2.4

    # 抑制强边缘/线状结构，避免眉毛、发丝、鼻孔边缘等尖锐结构进入Broad sheen。
    GRADIENT_LOW = 0.016
    GRADIENT_HIGH = 0.075

    # 两个通道的最终融合。V3适度降低Broad占比，减少大面积泛黄。
    BROAD_SHEEN_WEIGHT = 0.54
    SPECULAR_WEIGHT = 0.46

    # 固定跨图强度标尺：
    # relative_luminance_excess = (Lcorr - local_background) / local_background
    RAW_INTENSITY_SCALE = 0.16

    # ------------------------------------------------------------------
    # Hysteresis / component filtering
    # ------------------------------------------------------------------
    SUPPORT_THRESHOLD = 0.31
    SEED_THRESHOLD = 0.49
    HIGH_GLOSS_INTENSITY_THRESHOLD = 0.68

    SUPPORT_CLOSE_KERNEL = 5
    SUPPORT_OPEN_KERNEL = 3
    MIN_COMPONENT_AREA_PX = 18
    MIN_SEED_PIXELS = 2
    MAX_COMPONENT_AREA_RATIO = 0.20

    # 小型细长组件大概率是发丝高光/边缘反光；大片鼻梁/颧部油膜不按该规则删除。
    LINE_REJECT_MAX_AREA_PX = 420
    LINE_REJECT_ASPECT_RATIO = 4.5
    LINE_REJECT_MIN_MINOR_AXIS_PX = 3
    COMPONENT_MIN_MEAN_SCORE = 0.34
    COMPONENT_MIN_SPECULAR_PEAK = 0.16
    COMPONENT_MIN_SPECULAR_FRACTION = 0.012

    # 九区拆分：从现有左右cheek到鼻部的距离拆成鼻旁、内侧、外侧。
    NASAL_SIDE_DISTANCE_PX = 48.0
    INNER_CHEEK_DISTANCE_PX = 120.0

    # T区额部只取中央部分。
    T_FOREHEAD_HALF_WIDTH_RATIO = 0.24

    # 展示
    GLOSS_FILL_BGR = (0, 205, 255)
    HIGH_GLOSS_BGR = (0, 255, 255)
    GLOSS_CONTOUR_BGR = (0, 150, 255)
    GLOSS_ALPHA = 0.58
    REGION_COLOR_BGR = VISIA_BOUNDARY_COLOR
    REGION_THICKNESS = 2

    ALGORITHM_ID = "surface_gloss"
    ALGORITHM_VERSION = "SurfaceGloss-V3.0"
    REGION_SCHEMA_VERSION = "OilRegions-9Z-V1"


MEDICAL_REGION_ORDER = (
    "forehead",
    "nose",
    "left_nasal_side",
    "right_nasal_side",
    "left_inner_cheek",
    "right_inner_cheek",
    "left_outer_cheek",
    "right_outer_cheek",
    "chin",
)

MEDICAL_REGION_LABELS = {
    "forehead": "额部",
    "nose": "鼻部",
    "left_nasal_side": "画面左鼻旁",
    "right_nasal_side": "画面右鼻旁",
    "left_inner_cheek": "画面左内侧面颊",
    "right_inner_cheek": "画面右内侧面颊",
    "left_outer_cheek": "画面左外侧面颊",
    "right_outer_cheek": "画面右外侧面颊",
    "chin": "下巴",
}


@dataclass(frozen=True)
class OilRegionSet:
    analysis_mask: np.ndarray
    regions: dict[str, np.ndarray]
    t_zone_mask: np.ndarray
    cheek_zone_mask: np.ndarray
    partial_face: bool
    schema_version: str = Config.REGION_SCHEMA_VERSION


@dataclass
class SurfaceGlossResult:
    # 固定0~1跨图可比响应，不做单图百分位拉伸。
    gloss_score_map: np.ndarray
    gloss_intensity_map: np.ndarray
    broad_sheen_map: np.ndarray
    specular_map: np.ndarray
    raw_relative_luminance_map: np.ndarray
    illumination_corrected_luminance_map: np.ndarray
    broad_color_support_map: np.ndarray
    specular_neighborhood_map: np.ndarray

    gloss_analysis_mask: np.ndarray
    unreliable_highlight_mask: np.ndarray
    gloss_mask: np.ndarray
    high_gloss_mask: np.ndarray

    overlay: np.ndarray
    region_debug: np.ndarray

    valid_skin_area_px: int
    gloss_area_px: int
    gloss_area_ratio: float
    high_gloss_area_px: int
    high_gloss_area_ratio: float

    p50_gloss_intensity: float
    p90_gloss_intensity: float
    mean_gloss_intensity: float
    max_gloss_intensity: float

    p50_relative_luminance_excess: float
    p90_relative_luminance_excess: float

    patch_count: int
    p50_patch_area_px: float
    p90_patch_area_px: float
    largest_gloss_component_area_px: int
    largest_gloss_component_area_ratio: float

    clipped_highlight_area_px: int
    clipped_highlight_ratio: float
    lighting_risk: bool

    region_metrics: dict[str, dict[str, Any]]
    t_zone_summary: dict[str, Any]
    cheek_zone_summary: dict[str, Any]
    left_right_summary: dict[str, Any]

    quality_score: float
    quality_status: str
    quality_flags: list[str]
    partial_face: bool
    runtime_ms: float
    warnings: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, Any]:
        return {
            "algorithm": Config.ALGORITHM_ID,
            "algorithm_version": Config.ALGORITHM_VERSION,
            "region_schema_version": Config.REGION_SCHEMA_VERSION,
            "measurement_definition": (
                "标准化白光照片中的表面油光/镜面反射样图像表型，不等同于绝对皮脂量"
            ),
            "intensity_definition": (
                "固定尺度的局部相对反射响应；不使用单张图百分位拉满，"
                "可用于后续历史人群ECDF评分"
            ),
            "full_face": {
                "valid_skin_area_px": int(self.valid_skin_area_px),
                "gloss_area_px": int(self.gloss_area_px),
                "gloss_area_ratio": round(float(self.gloss_area_ratio), 8),
                "high_gloss_area_px": int(self.high_gloss_area_px),
                "high_gloss_area_ratio": round(float(self.high_gloss_area_ratio), 8),
                "p50_gloss_intensity": round(float(self.p50_gloss_intensity), 6),
                "p90_gloss_intensity": round(float(self.p90_gloss_intensity), 6),
                "mean_gloss_intensity": round(float(self.mean_gloss_intensity), 6),
                "max_gloss_intensity": round(float(self.max_gloss_intensity), 6),
                "p50_relative_luminance_excess": round(
                    float(self.p50_relative_luminance_excess), 6
                ),
                "p90_relative_luminance_excess": round(
                    float(self.p90_relative_luminance_excess), 6
                ),
                "patch_count": int(self.patch_count),
                "p50_patch_area_px": round(float(self.p50_patch_area_px), 3),
                "p90_patch_area_px": round(float(self.p90_patch_area_px), 3),
                "largest_gloss_component_area_px": int(
                    self.largest_gloss_component_area_px
                ),
                "largest_gloss_component_area_ratio": round(
                    float(self.largest_gloss_component_area_ratio), 8
                ),
                "unreliable_highlight_area_px": int(self.clipped_highlight_area_px),
                "unreliable_highlight_ratio": round(
                    float(self.clipped_highlight_ratio), 8
                ),
            },
            # 正好覆盖医生方案“表面油光60%”的五个核心评分输入。
            "doctor_core_inputs": {
                "surface_gloss_coverage": {
                    "gloss_area_ratio": round(float(self.gloss_area_ratio), 8),
                    "high_gloss_area_ratio": round(float(self.high_gloss_area_ratio), 8),
                },
                "surface_gloss_intensity": {
                    "p50_gloss_intensity": round(float(self.p50_gloss_intensity), 6),
                    "p90_gloss_intensity": round(float(self.p90_gloss_intensity), 6),
                },
                "surface_gloss_continuity": {
                    "largest_gloss_component_area_ratio": round(
                        float(self.largest_gloss_component_area_ratio), 8
                    )
                },
            },
            "auxiliary_cross_image_metrics": {
                "p50_relative_luminance_excess": round(
                    float(self.p50_relative_luminance_excess), 6
                ),
                "p90_relative_luminance_excess": round(
                    float(self.p90_relative_luminance_excess), 6
                ),
            },
            "region_metrics": self.region_metrics,
            "t_zone_summary": self.t_zone_summary,
            "cheek_zone_summary": self.cheek_zone_summary,
            "left_right_summary": self.left_right_summary,
            "quality": {
                "score": float(self.quality_score),
                "status": self.quality_status,
                "flags": list(self.quality_flags),
                "partial_face": bool(self.partial_face),
                "lighting_risk": bool(self.lighting_risk),
                "warnings": list(self.warnings),
            },
            "runtime_ms": round(float(self.runtime_ms), 3),
            "limitations": [
                "结果描述当前照片中的表面油亮外观，不测量绝对皮脂量或皮脂分泌速率。",
                "汗液、护肤品、防晒、底妆、水膜、曝光和拍摄角度均可能改变油光结果。",
                "近白剪裁/过曝区域从正式油光统计中剔除，单独记录为不可靠高光。",
                "本引擎不包含毛囊卟啉；卟啉应在后续油脂倾向融合层中作为独立40%证据使用。",
            ],
        }


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _odd_kernel(size: int) -> int:
    value = max(1, int(size))
    return value if value % 2 == 1 else value + 1


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(np.float32), float(q)))


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Fixed cross-image mapping. No per-image percentile normalization."""
    denom = max(float(high) - float(low), 1e-6)
    x = np.clip((values.astype(np.float32) - float(low)) / denom, 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    radius = max(0, int(radius_px))
    if radius <= 0:
        return _binary_mask(mask)
    return cv2.dilate(
        _binary_mask(mask),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        ),
    )


def _erode_by_distance(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Erode irregular face mask by a true pixel-distance margin."""
    binary = _binary_mask(mask)
    if int(margin_px) <= 0 or np.count_nonzero(binary) == 0:
        return binary
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return ((distance > float(margin_px)).astype(np.uint8) * 255)


def _component_areas(mask: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    return [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, count)
    ]


def _draw_mask_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    contours, _ = cv2.findContours(
        _binary_mask(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(
            image,
            contours,
            -1,
            color,
            int(thickness),
            lineType=cv2.LINE_AA,
        )


def _mask_distance_to(source_mask: np.ndarray) -> np.ndarray:
    source = _binary_mask(source_mask)
    inverted = cv2.bitwise_not(source)
    return cv2.distanceTransform(inverted, cv2.DIST_L2, 5).astype(np.float32)


def build_oil_regions(
    base_regions: VisiaRegionSet,
    landmarks: np.ndarray,
) -> OilRegionSet:
    """在现有VISIA五区之上构建医生油脂报告九区。

    左右语义沿用现有 visia_regions.py：left/right 是画面左右。
    """
    analysis_mask = _binary_mask(base_regions.analysis_mask)
    shape = analysis_mask.shape
    zeros = np.zeros(shape, dtype=np.uint8)

    forehead = _binary_mask(base_regions.regions.get("forehead", zeros))
    nose = _binary_mask(base_regions.regions.get("nose", zeros))
    chin = _binary_mask(base_regions.regions.get("chin", zeros))
    left_cheek = _binary_mask(base_regions.regions.get("left_cheek", zeros))
    right_cheek = _binary_mask(base_regions.regions.get("right_cheek", zeros))

    distance = _mask_distance_to(nose)

    def split_cheek(
        cheek: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = cheek > 0
        nasal = (
            valid & (distance <= float(Config.NASAL_SIDE_DISTANCE_PX))
        ).astype(np.uint8) * 255
        inner = (
            valid
            & (distance > float(Config.NASAL_SIDE_DISTANCE_PX))
            & (distance <= float(Config.INNER_CHEEK_DISTANCE_PX))
        ).astype(np.uint8) * 255
        outer = (
            valid & (distance > float(Config.INNER_CHEEK_DISTANCE_PX))
        ).astype(np.uint8) * 255
        return nasal, inner, outer

    left_nasal, left_inner, left_outer = split_cheek(left_cheek)
    right_nasal, right_inner, right_outer = split_cheek(right_cheek)

    regions = {
        "forehead": forehead,
        "nose": nose,
        "left_nasal_side": left_nasal,
        "right_nasal_side": right_nasal,
        "left_inner_cheek": left_inner,
        "right_inner_cheek": right_inner,
        "left_outer_cheek": left_outer,
        "right_outer_cheek": right_outer,
        "chin": chin,
    }

    # T区：额部中央 + 鼻部 + 下巴
    yy, xx = np.indices(shape)
    points = np.column_stack(np.where(forehead > 0))
    if points.size:
        xs = points[:, 1]
        face_width = max(float(xs.max() - xs.min() + 1), 1.0)
        landmarks_arr = np.asarray(landmarks)
        center_x = (
            float(landmarks_arr[1, 0])
            if landmarks_arr.ndim == 2 and len(landmarks_arr) > 1
            else float(np.median(xs))
        )
        half_width = Config.T_FOREHEAD_HALF_WIDTH_RATIO * face_width
        central_forehead = (
            (forehead > 0)
            & (xx >= center_x - half_width)
            & (xx <= center_x + half_width)
        ).astype(np.uint8) * 255
    else:
        central_forehead = zeros.copy()

    t_zone = cv2.bitwise_or(central_forehead, nose)
    t_zone = cv2.bitwise_or(t_zone, chin)

    cheek_zone = zeros.copy()
    for name in (
        "left_nasal_side",
        "right_nasal_side",
        "left_inner_cheek",
        "right_inner_cheek",
        "left_outer_cheek",
        "right_outer_cheek",
    ):
        cheek_zone = cv2.bitwise_or(cheek_zone, regions[name])

    return OilRegionSet(
        analysis_mask=analysis_mask,
        regions=regions,
        t_zone_mask=_binary_mask(t_zone),
        cheek_zone_mask=_binary_mask(cheek_zone),
        partial_face=bool(base_regions.partial_face),
    )


class SurfaceGlossAnalyzer:
    """白光图像的表面油光/镜面反射样检测器。"""

    @staticmethod
    def _validate_input(preprocess_result: PreprocessResultV2) -> None:
        image = np.asarray(preprocess_result.analysis_image)
        mask = np.asarray(preprocess_result.skin_mask)
        landmarks = np.asarray(preprocess_result.landmarks)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("analysis_image must be BGR HxWx3")
        if image.shape[:2] != mask.shape:
            raise ValueError("analysis_image and skin_mask shapes do not match")
        if landmarks.ndim != 2 or landmarks.shape[1] < 2 or len(landmarks) < 335:
            raise ValueError("aligned MediaPipe landmarks are required")

    @staticmethod
    def _photometric_channels(image: np.ndarray) -> dict[str, np.ndarray]:
        image_u8 = np.asarray(image, dtype=np.uint8)
        lab = cv2.cvtColor(image_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(image_u8, cv2.COLOR_BGR2HSV).astype(np.float32)

        l = lab[:, :, 0] / 255.0
        a = (lab[:, :, 1] - 128.0) / 127.0
        b = (lab[:, :, 2] - 128.0) / 127.0
        chroma = np.clip(np.sqrt(a * a + b * b) / np.sqrt(2.0), 0.0, 1.0)
        saturation = hsv[:, :, 1] / 255.0

        return {
            "l": l.astype(np.float32),
            "saturation": saturation.astype(np.float32),
            "chroma": chroma.astype(np.float32),
        }

    @staticmethod
    def _clipped_highlight_mask(
        image: np.ndarray,
        base_mask: np.ndarray,
    ) -> np.ndarray:
        image_u8 = np.asarray(image, dtype=np.uint8)
        clipped = (
            np.all(
                image_u8 >= int(Config.CLIP_CHANNEL_THRESHOLD),
                axis=2,
            )
            & (base_mask > 0)
        )
        mask = clipped.astype(np.uint8) * 255
        return _dilate_mask(mask, Config.CLIP_DILATE_PX)

    @staticmethod
    def _build_gloss_analysis_mask(
        preprocess_result: PreprocessResultV2,
        analysis_mask: np.ndarray,
        unreliable_highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build oil-specific safe skin domain.

        Uses the exact private debug masks already attached by Preprocessor V2
        when available; falls back to analysis_mask only for compatibility.
        """
        safe = _erode_by_distance(
            analysis_mask,
            Config.OUTER_BOUNDARY_MARGIN_PX,
        )

        debug_masks = getattr(preprocess_result, "_debug_masks", {}) or {}
        shape = safe.shape

        exclusions = np.zeros(shape, dtype=np.uint8)
        feature = debug_masks.get("feature_exclusions")
        hair = debug_masks.get("hair_mask")
        nostril = debug_masks.get("nostril_mask")

        if isinstance(feature, np.ndarray) and feature.shape == shape:
            exclusions = cv2.bitwise_or(
                exclusions,
                _dilate_mask(feature, Config.FEATURE_SAFETY_DILATE_PX),
            )
        if isinstance(hair, np.ndarray) and hair.shape == shape:
            exclusions = cv2.bitwise_or(
                exclusions,
                _dilate_mask(hair, Config.HAIR_SAFETY_DILATE_PX),
            )
        if isinstance(nostril, np.ndarray) and nostril.shape == shape:
            exclusions = cv2.bitwise_or(
                exclusions,
                _dilate_mask(nostril, Config.NOSTRIL_SAFETY_DILATE_PX),
            )

        safe = cv2.bitwise_and(safe, cv2.bitwise_not(exclusions))

        # 人脸分割在VISIA下巴托、白色夹具和下颌边缘附近偶尔会将
        # 非皮肤高光并入前景。正式油光域以MediaPipe 152下颌底点为硬边界，
        # 仅保留少量抗栅格余量，避免把采集夹具当成高亮油光。
        landmarks = np.asarray(getattr(preprocess_result, "landmarks", np.empty((0, 2))))
        if landmarks.ndim == 2 and landmarks.shape[1] >= 2 and len(landmarks) > 152:
            # 下颌点152通常正好落在下巴托上缘；向上保守回收36px，
            # 尽量保留下巴皮肤，同时将VISIA夹具从正式测量域移除。
            jaw_bottom = int(np.clip(round(float(landmarks[152, 1])) - 36, 0, shape[0]))
            safe[jaw_bottom:, :] = 0

        if (
            isinstance(unreliable_highlight_mask, np.ndarray)
            and unreliable_highlight_mask.shape == shape
        ):
            safe = cv2.bitwise_and(
                safe,
                cv2.bitwise_not(_binary_mask(unreliable_highlight_mask)),
            )

        return _binary_mask(safe)

    @staticmethod
    def _fill_mask_outside(
        channel: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        valid = mask > 0
        output = channel.astype(np.float32).copy()
        if np.count_nonzero(valid) == 0:
            output.fill(0.0)
            return output
        median = float(np.median(output[valid]))
        output[~valid] = median
        return output

    @staticmethod
    def _relative_brightness_responses(
        l_channel: np.ndarray,
        mask: np.ndarray,
        sigmas: tuple[float, ...],
    ) -> list[np.ndarray]:
        filled = SurfaceGlossAnalyzer._fill_mask_outside(l_channel, mask)
        responses: list[np.ndarray] = []
        for sigma in sigmas:
            background = masked_gaussian(
                filled,
                mask,
                float(sigma),
            )
            ratio = np.maximum(
                (filled - background) / np.maximum(background, 0.06),
                0.0,
            )
            ratio[mask == 0] = 0.0
            responses.append(ratio.astype(np.float32))
        return responses

    @staticmethod
    def _multiscale_consensus(
        responses: list[np.ndarray],
        mode: str = "median",
    ) -> np.ndarray:
        if not responses:
            raise ValueError("responses must not be empty")
        stack = np.stack(responses, axis=0)
        if mode == "max":
            return np.max(stack, axis=0).astype(np.float32)
        if mode == "median":
            return np.median(stack, axis=0).astype(np.float32)
        raise ValueError(f"Unsupported consensus mode: {mode}")

    @staticmethod
    def _illumination_corrected_luminance(
        l_channel: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Remove very-low-frequency illumination/face-curvature field."""
        filled = SurfaceGlossAnalyzer._fill_mask_outside(l_channel, mask)
        field = masked_gaussian(
            filled,
            mask,
            float(Config.ILLUMINATION_FIELD_SIGMA),
        )
        corrected = filled / np.maximum(field, 0.08)
        corrected = np.clip(
            corrected,
            float(Config.ILLUMINATION_RATIO_MIN),
            float(Config.ILLUMINATION_RATIO_MAX),
        )
        corrected[mask == 0] = 0.0
        return corrected.astype(np.float32)

    @staticmethod
    def _gradient_smooth_support(
        l_channel: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        l = SurfaceGlossAnalyzer._fill_mask_outside(l_channel, mask)
        gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        edge = _smoothstep(
            grad,
            Config.GRADIENT_LOW,
            Config.GRADIENT_HIGH,
        )
        support = 1.0 - edge
        support[mask == 0] = 0.0
        return np.clip(support, 0.0, 1.0).astype(np.float32)

    def _build_score_maps(
        self,
        image: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return fixed-scale Broad/Specular/Intensity maps.

        V3 changes:
        1) very-low-frequency illumination field is removed first;
        2) Broad Sheen uses multi-scale median response, not max response;
        3) Broad Sheen must be supported by local chroma/saturation wash-out
           or by nearby Specular cores;
        4) no per-image percentile normalization is used anywhere.
        """
        channels = self._photometric_channels(image)
        l = channels["l"]
        saturation = channels["saturation"]
        chroma = channels["chroma"]

        if np.count_nonzero(analysis_mask) < 500:
            zeros = np.zeros(l.shape, dtype=np.float32)
            return {
                "score": zeros.copy(),
                "intensity": zeros.copy(),
                "broad": zeros.copy(),
                "specular": zeros.copy(),
                "raw_relative_luminance": zeros.copy(),
                "illumination_corrected_luminance": zeros.copy(),
                "broad_color_support": zeros.copy(),
                "specular_neighborhood": zeros.copy(),
            }

        l_corr = self._illumination_corrected_luminance(
            l,
            analysis_mask,
        )

        broad_responses = self._relative_brightness_responses(
            l_corr,
            analysis_mask,
            tuple(float(x) for x in Config.BROAD_BACKGROUND_SIGMAS),
        )
        spec_responses = self._relative_brightness_responses(
            l_corr,
            analysis_mask,
            tuple(float(x) for x in Config.SPECULAR_BACKGROUND_SIGMAS),
        )

        # Broad response requires persistence across scales.
        broad_ratio = self._multiscale_consensus(
            broad_responses,
            mode="median",
        )
        # Specular core can be sharp at one specific small scale.
        spec_ratio = self._multiscale_consensus(
            spec_responses,
            mode="max",
        )

        broad_brightness = _smoothstep(
            broad_ratio,
            Config.BROAD_RATIO_LOW,
            Config.BROAD_RATIO_HIGH,
        )
        spec_brightness = _smoothstep(
            spec_ratio,
            Config.SPECULAR_RATIO_LOW,
            Config.SPECULAR_RATIO_HIGH,
        )

        sat_filled = self._fill_mask_outside(saturation, analysis_mask)
        chroma_filled = self._fill_mask_outside(chroma, analysis_mask)

        # Specular local colour wash-out.
        sat_bg_spec = masked_gaussian(
            sat_filled,
            analysis_mask,
            float(Config.SPECULAR_COLOR_BACKGROUND_SIGMA),
        )
        chroma_bg_spec = masked_gaussian(
            chroma_filled,
            analysis_mask,
            float(Config.SPECULAR_COLOR_BACKGROUND_SIGMA),
        )
        sat_drop_spec = np.maximum(sat_bg_spec - sat_filled, 0.0)
        chroma_drop_spec = np.maximum(chroma_bg_spec - chroma_filled, 0.0)

        sat_gate_spec = _smoothstep(
            sat_drop_spec,
            Config.SATURATION_DROP_LOW,
            Config.SATURATION_DROP_HIGH,
        )
        chroma_gate_spec = _smoothstep(
            chroma_drop_spec,
            Config.CHROMA_DROP_LOW,
            Config.CHROMA_DROP_HIGH,
        )
        spec_color_gate = np.sqrt(
            np.clip(
                0.5 * sat_gate_spec + 0.5 * chroma_gate_spec,
                0.0,
                1.0,
            )
        ).astype(np.float32)

        specular = np.sqrt(
            np.clip(spec_brightness, 0.0, 1.0)
            * np.clip(spec_color_gate, 0.0, 1.0)
        ).astype(np.float32)

        # Broader colour wash-out support, intentionally more sensitive but
        # still local-relative rather than absolute "low saturation".
        sat_bg_broad = masked_gaussian(
            sat_filled,
            analysis_mask,
            float(Config.BROAD_COLOR_BACKGROUND_SIGMA),
        )
        chroma_bg_broad = masked_gaussian(
            chroma_filled,
            analysis_mask,
            float(Config.BROAD_COLOR_BACKGROUND_SIGMA),
        )
        sat_drop_broad = np.maximum(sat_bg_broad - sat_filled, 0.0)
        chroma_drop_broad = np.maximum(chroma_bg_broad - chroma_filled, 0.0)

        sat_gate_broad = _smoothstep(
            sat_drop_broad,
            Config.BROAD_SATURATION_DROP_LOW,
            Config.BROAD_SATURATION_DROP_HIGH,
        )
        chroma_gate_broad = _smoothstep(
            chroma_drop_broad,
            Config.BROAD_CHROMA_DROP_LOW,
            Config.BROAD_CHROMA_DROP_HIGH,
        )
        broad_color_support = np.sqrt(
            np.clip(
                0.5 * sat_gate_broad + 0.5 * chroma_gate_broad,
                0.0,
                1.0,
            )
        ).astype(np.float32)

        # Nearby specular cores are strong evidence that a broader bright
        # plateau is actual oil-film sheen rather than simple facial curvature.
        specular_neighborhood = cv2.GaussianBlur(
            specular,
            (0, 0),
            sigmaX=float(Config.SPECULAR_NEIGHBOR_SIGMA),
            sigmaY=float(Config.SPECULAR_NEIGHBOR_SIGMA),
            borderType=cv2.BORDER_REFLECT,
        ).astype(np.float32)
        specular_neighborhood = np.clip(
            specular_neighborhood * float(Config.SPECULAR_NEIGHBOR_GAIN),
            0.0,
            1.0,
        )
        specular_neighborhood[analysis_mask == 0] = 0.0

        smooth_support = self._gradient_smooth_support(
            l_corr,
            analysis_mask,
        )

        # Key V3 gate:
        # - broad brightness alone is not enough;
        # - it needs either local colour wash-out or nearby specular cores.
        broad_support_gate = np.maximum(
            broad_color_support,
            specular_neighborhood,
        )
        broad = (
            broad_brightness
            * (0.20 + 0.80 * broad_support_gate)
            * (0.55 + 0.45 * smooth_support)
        ).astype(np.float32)

        # Suppress weak unsupported broad fields aggressively.
        broad *= np.clip(
            0.35 + 0.65 * np.sqrt(np.maximum(broad_support_gate, 0.0)),
            0.0,
            1.0,
        ).astype(np.float32)

        score = (
            Config.BROAD_SHEEN_WEIGHT * broad
            + Config.SPECULAR_WEIGHT * specular
        )
        score = np.clip(score, 0.0, 1.0).astype(np.float32)

        # Cross-image raw intensity keeps physical-ish relative excess.
        raw_relative = np.maximum(
            broad_ratio,
            spec_ratio,
        ).astype(np.float32)
        raw_relative[analysis_mask == 0] = 0.0

        intensity = (
            1.0
            - np.exp(
                -np.maximum(raw_relative, 0.0)
                / max(float(Config.RAW_INTENSITY_SCALE), 1e-6)
            )
        ).astype(np.float32)
        intensity = np.clip(intensity, 0.0, 1.0)

        for output in (
            score,
            intensity,
            broad,
            specular,
            raw_relative,
            broad_color_support,
            specular_neighborhood,
        ):
            output[analysis_mask == 0] = 0.0

        return {
            "score": score,
            "intensity": intensity,
            "broad": np.clip(broad, 0.0, 1.0).astype(np.float32),
            "specular": np.clip(specular, 0.0, 1.0).astype(np.float32),
            "raw_relative_luminance": raw_relative,
            "illumination_corrected_luminance": l_corr,
            "broad_color_support": np.clip(
                broad_color_support, 0.0, 1.0
            ).astype(np.float32),
            "specular_neighborhood": specular_neighborhood,
        }

    @staticmethod
    def _hysteresis_mask(
        score: np.ndarray,
        specular: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        support = (
            (score >= float(Config.SUPPORT_THRESHOLD))
            & (analysis_mask > 0)
        ).astype(np.uint8) * 255

        seed = (
            (score >= float(Config.SEED_THRESHOLD))
            & (analysis_mask > 0)
        ).astype(np.uint8) * 255

        support = cv2.morphologyEx(
            support,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_kernel(Config.SUPPORT_CLOSE_KERNEL),) * 2,
            ),
        )
        support = cv2.morphologyEx(
            support,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_kernel(Config.SUPPORT_OPEN_KERNEL),) * 2,
            ),
        )
        support = cv2.bitwise_and(support, analysis_mask)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (support > 0).astype(np.uint8),
            connectivity=8,
        )

        valid_area = max(int(np.count_nonzero(analysis_mask)), 1)
        max_area = max(
            1,
            int(Config.MAX_COMPONENT_AREA_RATIO * valid_area),
        )

        output = np.zeros_like(support)

        for label in range(1, count):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])

            if area < int(Config.MIN_COMPONENT_AREA_PX) or area > max_area:
                continue

            seed_pixels = int(np.count_nonzero(component & (seed > 0)))
            if seed_pixels < int(Config.MIN_SEED_PIXELS):
                continue

            major = max(width, height)
            minor = max(1, min(width, height))
            aspect_ratio = major / minor

            if (
                area <= int(Config.LINE_REJECT_MAX_AREA_PX)
                and (
                    aspect_ratio >= float(Config.LINE_REJECT_ASPECT_RATIO)
                    or minor <= int(Config.LINE_REJECT_MIN_MINOR_AXIS_PX)
                )
            ):
                continue

            component_scores = score[component]
            mean_score = (
                float(np.mean(component_scores))
                if component_scores.size else 0.0
            )

            component_specular = specular[component]
            specular_peak = (
                float(np.max(component_specular))
                if component_specular.size else 0.0
            )
            specular_fraction = (
                float(np.mean(
                    component_specular >= float(Config.COMPONENT_MIN_SPECULAR_PEAK)
                ))
                if component_specular.size else 0.0
            )

            # A component with only mediocre Broad response and absolutely no
            # specular-like core is likely facial curvature/illumination.
            if (
                mean_score < float(Config.COMPONENT_MIN_MEAN_SCORE)
                and (
                    specular_peak < float(Config.COMPONENT_MIN_SPECULAR_PEAK)
                    or specular_fraction
                    < float(Config.COMPONENT_MIN_SPECULAR_FRACTION)
                )
            ):
                continue

            output[component] = 255

        return _binary_mask(output)

    @staticmethod
    def _region_metrics(
        region_mask: np.ndarray,
        valid_mask: np.ndarray,
        gloss_mask: np.ndarray,
        high_mask: np.ndarray,
        intensity_map: np.ndarray,
    ) -> dict[str, Any]:
        region = (region_mask > 0) & (valid_mask > 0)
        valid_area = int(np.count_nonzero(region))

        if valid_area == 0:
            return {
                "status": "NOT_VISIBLE",
                "valid_area_px": 0,
                "gloss_area_px": 0,
                "gloss_area_ratio": None,
                "high_gloss_area_px": 0,
                "high_gloss_area_ratio": None,
                "p50_gloss_intensity": None,
                "p90_gloss_intensity": None,
                "largest_gloss_component_area_px": 0,
                "largest_gloss_component_area_ratio": None,
                "patch_count": 0,
            }

        local_gloss = (gloss_mask > 0) & region
        local_high = (high_mask > 0) & region

        gloss_area = int(np.count_nonzero(local_gloss))
        high_area = int(np.count_nonzero(local_high))
        values = intensity_map[local_gloss]

        local_mask = local_gloss.astype(np.uint8) * 255
        areas = _component_areas(local_mask)
        largest = max(areas) if areas else 0

        return {
            "status": "VALID",
            "valid_area_px": valid_area,
            "gloss_area_px": gloss_area,
            "gloss_area_ratio": round(gloss_area / valid_area, 8),
            "high_gloss_area_px": high_area,
            "high_gloss_area_ratio": round(high_area / valid_area, 8),
            "p50_gloss_intensity": round(_safe_percentile(values, 50), 6),
            "p90_gloss_intensity": round(_safe_percentile(values, 90), 6),
            "largest_gloss_component_area_px": int(largest),
            "largest_gloss_component_area_ratio": round(
                largest / valid_area,
                8,
            ),
            "patch_count": len(areas),
        }


    @staticmethod
    def _left_right_summary(
        region_metrics: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        pairs = {
            "nasal_side": (
                "left_nasal_side",
                "right_nasal_side",
            ),
            "inner_cheek": (
                "left_inner_cheek",
                "right_inner_cheek",
            ),
            "outer_cheek": (
                "left_outer_cheek",
                "right_outer_cheek",
            ),
        }

        output: dict[str, Any] = {}
        for name, (left_name, right_name) in pairs.items():
            left = region_metrics[left_name]
            right = region_metrics[right_name]

            left_ratio = left.get("gloss_area_ratio")
            right_ratio = right.get("gloss_area_ratio")
            left_p90 = left.get("p90_gloss_intensity")
            right_p90 = right.get("p90_gloss_intensity")

            output[name] = {
                "left_region": left_name,
                "right_region": right_name,
                "gloss_area_ratio_difference": (
                    None
                    if left_ratio is None or right_ratio is None
                    else round(
                        float(right_ratio) - float(left_ratio),
                        8,
                    )
                ),
                "p90_intensity_difference": (
                    None
                    if left_p90 is None or right_p90 is None
                    else round(
                        float(right_p90) - float(left_p90),
                        6,
                    )
                ),
            }
        return output

    @staticmethod
    def _render_overlay(
        image: np.ndarray,
        gloss_mask: np.ndarray,
        high_mask: np.ndarray,
        oil_regions: OilRegionSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        overlay = np.asarray(image).copy()

        pixels = gloss_mask > 0
        if np.any(pixels):
            fill = np.asarray(
                Config.GLOSS_FILL_BGR,
                dtype=np.float32,
            )
            overlay[pixels] = np.clip(
                (1.0 - Config.GLOSS_ALPHA)
                * overlay[pixels].astype(np.float32)
                + Config.GLOSS_ALPHA * fill,
                0,
                255,
            ).astype(np.uint8)

        high_pixels = high_mask > 0
        if np.any(high_pixels):
            high = np.asarray(
                Config.HIGH_GLOSS_BGR,
                dtype=np.float32,
            )
            overlay[high_pixels] = np.clip(
                0.35 * overlay[high_pixels].astype(np.float32)
                + 0.65 * high,
                0,
                255,
            ).astype(np.uint8)

        _draw_mask_contours(
            overlay,
            gloss_mask,
            Config.GLOSS_CONTOUR_BGR,
            1,
        )

        debug = np.asarray(image).copy()
        for name in MEDICAL_REGION_ORDER:
            _draw_mask_contours(
                debug,
                oil_regions.regions[name],
                Config.REGION_COLOR_BGR,
                Config.REGION_THICKNESS,
            )

        return overlay, debug

    def detect_surface_gloss(
        self,
        preprocess_result: PreprocessResultV2,
    ) -> SurfaceGlossResult:
        self._validate_input(preprocess_result)
        started = time.perf_counter()

        image = np.asarray(
            preprocess_result.analysis_image,
            dtype=np.uint8,
        )

        base_regions = build_visia_regions(
            image,
            preprocess_result.skin_mask,
            preprocess_result.landmarks,
            preprocess_result.quality_flags,
            include_chin=True,
            mode="full",
            feature_margin_px=10,
        )

        # Stage 1: strict skin/feature/hair/boundary mask.
        pre_clip_mask = self._build_gloss_analysis_mask(
            preprocess_result,
            base_regions.analysis_mask,
            unreliable_highlight_mask=None,
        )

        # Stage 2: clipped/near-white pixels are unreliable measurement, not oil.
        unreliable_highlight_mask = self._clipped_highlight_mask(
            image,
            pre_clip_mask,
        )

        analysis_mask = self._build_gloss_analysis_mask(
            preprocess_result,
            base_regions.analysis_mask,
            unreliable_highlight_mask=unreliable_highlight_mask,
        )

        maps = self._build_score_maps(
            image,
            analysis_mask,
        )
        score = maps["score"]
        intensity = maps["intensity"]
        broad = maps["broad"]
        specular = maps["specular"]
        raw_relative = maps["raw_relative_luminance"]

        gloss_mask = self._hysteresis_mask(
            score,
            specular,
            analysis_mask,
        )

        high_mask = (
            (intensity >= float(Config.HIGH_GLOSS_INTENSITY_THRESHOLD))
            & (gloss_mask > 0)
        ).astype(np.uint8) * 255

        # Safety: final masks can never leak outside the dedicated oil domain.
        gloss_mask = cv2.bitwise_and(gloss_mask, analysis_mask)
        high_mask = cv2.bitwise_and(high_mask, analysis_mask)

        valid_area = int(np.count_nonzero(analysis_mask))
        gloss_area = int(np.count_nonzero(gloss_mask))
        high_area = int(np.count_nonzero(high_mask))

        pre_clip_area = max(int(np.count_nonzero(pre_clip_mask)), 1)
        clipped_area = int(np.count_nonzero(unreliable_highlight_mask))

        intensity_values = intensity[gloss_mask > 0]
        raw_relative_values = raw_relative[gloss_mask > 0]

        areas = _component_areas(gloss_mask)
        largest_area = max(areas) if areas else 0

        oil_regions = build_oil_regions(
            base_regions,
            preprocess_result.landmarks,
        )

        region_metrics: dict[str, dict[str, Any]] = {}
        for name in MEDICAL_REGION_ORDER:
            region_metrics[name] = {
                "label": MEDICAL_REGION_LABELS[name],
                **self._region_metrics(
                    oil_regions.regions[name],
                    analysis_mask,
                    gloss_mask,
                    high_mask,
                    intensity,
                ),
            }

        t_zone_summary = {
            "label": "T区工程汇总（额部中央+鼻部+下巴）",
            **self._region_metrics(
                oil_regions.t_zone_mask,
                analysis_mask,
                gloss_mask,
                high_mask,
                intensity,
            ),
        }

        cheek_zone_summary = {
            "label": "面颊区工程汇总（鼻旁+内侧面颊+外侧面颊）",
            **self._region_metrics(
                oil_regions.cheek_zone_mask,
                analysis_mask,
                gloss_mask,
                high_mask,
                intensity,
            ),
        }

        left_right = self._left_right_summary(
            region_metrics
        )

        clipped_ratio = clipped_area / pre_clip_area
        lighting_risk = clipped_ratio >= float(Config.CLIP_WARNING_RATIO)

        warnings: list[str] = []
        if lighting_risk:
            warnings.append(
                "UNRELIABLE_HIGHLIGHT_RISK: 近白剪裁区域已从正式油光统计中剔除"
            )
        if preprocess_result.quality_status == "WARNING":
            warnings.append(
                "PREPROCESS_WARNING: 预处理质量状态为WARNING"
            )
        if base_regions.partial_face:
            warnings.append(
                "PARTIAL_FACE: 部分左右/分区比较仅供参考"
            )
        if valid_area < 500:
            warnings.append(
                "INSUFFICIENT_GLOSS_DOMAIN: 油光专属有效皮肤区域过小"
            )

        overlay, region_debug = self._render_overlay(
            image,
            gloss_mask,
            high_mask,
            oil_regions,
        )

        # Draw unreliable highlights in red only on debug image, never as oil.
        _draw_mask_contours(
            region_debug,
            unreliable_highlight_mask,
            (0, 0, 255),
            1,
        )

        return SurfaceGlossResult(
            gloss_score_map=score,
            gloss_intensity_map=intensity,
            broad_sheen_map=broad,
            specular_map=specular,
            raw_relative_luminance_map=raw_relative,
            illumination_corrected_luminance_map=maps["illumination_corrected_luminance"],
            broad_color_support_map=maps["broad_color_support"],
            specular_neighborhood_map=maps["specular_neighborhood"],
            gloss_analysis_mask=_binary_mask(analysis_mask),
            unreliable_highlight_mask=_binary_mask(unreliable_highlight_mask),
            gloss_mask=_binary_mask(gloss_mask),
            high_gloss_mask=_binary_mask(high_mask),
            overlay=overlay,
            region_debug=region_debug,
            valid_skin_area_px=valid_area,
            gloss_area_px=gloss_area,
            gloss_area_ratio=(
                gloss_area / valid_area
                if valid_area else 0.0
            ),
            high_gloss_area_px=high_area,
            high_gloss_area_ratio=(
                high_area / valid_area
                if valid_area else 0.0
            ),
            p50_gloss_intensity=_safe_percentile(
                intensity_values,
                50,
            ),
            p90_gloss_intensity=_safe_percentile(
                intensity_values,
                90,
            ),
            mean_gloss_intensity=(
                float(np.mean(intensity_values))
                if intensity_values.size else 0.0
            ),
            max_gloss_intensity=(
                float(np.max(intensity_values))
                if intensity_values.size else 0.0
            ),
            p50_relative_luminance_excess=_safe_percentile(
                raw_relative_values,
                50,
            ),
            p90_relative_luminance_excess=_safe_percentile(
                raw_relative_values,
                90,
            ),
            patch_count=len(areas),
            p50_patch_area_px=_safe_percentile(
                np.asarray(areas, dtype=np.float32),
                50,
            ),
            p90_patch_area_px=_safe_percentile(
                np.asarray(areas, dtype=np.float32),
                90,
            ),
            largest_gloss_component_area_px=int(
                largest_area
            ),
            largest_gloss_component_area_ratio=(
                largest_area / valid_area
                if valid_area else 0.0
            ),
            clipped_highlight_area_px=clipped_area,
            clipped_highlight_ratio=clipped_ratio,
            lighting_risk=bool(lighting_risk),
            region_metrics=region_metrics,
            t_zone_summary=t_zone_summary,
            cheek_zone_summary=cheek_zone_summary,
            left_right_summary=left_right,
            quality_score=float(
                preprocess_result.quality_score
            ),
            quality_status=str(
                preprocess_result.quality_status
            ),
            quality_flags=list(
                preprocess_result.quality_flags
            ),
            partial_face=bool(
                base_regions.partial_face
            ),
            runtime_ms=(
                time.perf_counter() - started
            ) * 1000.0,
            warnings=warnings,
        )


    @staticmethod
    def _write_csv(
        result: SurfaceGlossResult,
        path: str,
    ) -> None:
        rows: list[list[Any]] = [
            ["范围", "指标", "结果", "单位"],
            ["全面部", "有效皮肤面积", result.valid_skin_area_px, "像素"],
            ["全面部", "油光面积", result.gloss_area_px, "像素"],
            ["全面部", "油光面积占比", result.gloss_area_ratio, "比例"],
            ["全面部", "高强度油光面积", result.high_gloss_area_px, "像素"],
            ["全面部", "高强度油光面积占比", result.high_gloss_area_ratio, "比例"],
            ["全面部", "P50油光强度", result.p50_gloss_intensity, "0~1固定尺度工程值"],
            ["全面部", "P90油光强度", result.p90_gloss_intensity, "0~1固定尺度工程值"],
            ["全面部", "P50局部相对亮度增量", result.p50_relative_luminance_excess, "比例"],
            ["全面部", "P90局部相对亮度增量", result.p90_relative_luminance_excess, "比例"],
            [
                "全面部",
                "最大连续油光区域面积",
                result.largest_gloss_component_area_px,
                "像素",
            ],
            [
                "全面部",
                "最大连续油光区域面积占比",
                result.largest_gloss_component_area_ratio,
                "比例",
            ],
            ["全面部", "油光斑块数量", result.patch_count, "个"],
        ]

        for name in MEDICAL_REGION_ORDER:
            item = result.region_metrics[name]
            label = item["label"]
            rows.extend(
                [
                    [label, "有效面积", item["valid_area_px"], "像素"],
                    [label, "油光面积占比", item["gloss_area_ratio"], "比例"],
                    [label, "高强度油光面积占比", item["high_gloss_area_ratio"], "比例"],
                    [label, "P50油光强度", item["p50_gloss_intensity"], "0~1工程值"],
                    [label, "P90油光强度", item["p90_gloss_intensity"], "0~1工程值"],
                    [
                        label,
                        "最大连续油光区域面积占比",
                        item["largest_gloss_component_area_ratio"],
                        "比例",
                    ],
                ]
            )

        with open(
            path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            csv.writer(handle).writerows(rows)

    @staticmethod
    def save_result(
        result: SurfaceGlossResult,
        output_dir: str,
        base_name: str,
    ) -> dict[str, str]:
        sample_dir = os.path.join(
            output_dir,
            base_name,
        )
        os.makedirs(
            sample_dir,
            exist_ok=True,
        )

        paths = {
            "overlay": os.path.join(
                sample_dir,
                "01_表面油光检测结果图.jpg",
            ),
            "score": os.path.join(
                sample_dir,
                "02_表面油光强度图.png",
            ),
            "mask": os.path.join(
                sample_dir,
                "03_表面油光Mask.png",
            ),
            "high_mask": os.path.join(
                sample_dir,
                "04_高强度油光Mask.png",
            ),
            "region_debug": os.path.join(
                sample_dir,
                "05_油脂九分区调试图.jpg",
            ),
            "analysis_mask": os.path.join(
                sample_dir,
                "06_油光有效分析Mask.png",
            ),
            "unreliable_highlight": os.path.join(
                sample_dir,
                "07_不可靠高光Mask.png",
            ),
            "broad_sheen": os.path.join(
                sample_dir,
                "08_BroadSheen响应图.png",
            ),
            "specular": os.path.join(
                sample_dir,
                "09_Specular响应图.png",
            ),
            "raw_relative": os.path.join(
                sample_dir,
                "10_局部相对亮度增量图.png",
            ),
            "illumination_corrected": os.path.join(
                sample_dir,
                "11_光照场校正亮度图.png",
            ),
            "broad_color_support": os.path.join(
                sample_dir,
                "12_BroadSheen去色支持图.png",
            ),
            "specular_neighborhood": os.path.join(
                sample_dir,
                "13_Specular邻域支持图.png",
            ),
            "json": os.path.join(
                sample_dir,
                "表面油光量化指标.json",
            ),
            "csv": os.path.join(
                sample_dir,
                "表面油光量化指标.csv",
            ),
        }

        cv_imwrite(
            paths["overlay"],
            result.overlay,
        )
        cv_imwrite(
            paths["score"],
            np.rint(
                np.clip(
                    result.gloss_score_map,
                    0.0,
                    1.0,
                ) * 255.0
            ).astype(np.uint8),
        )
        cv_imwrite(
            paths["mask"],
            result.gloss_mask,
        )
        cv_imwrite(
            paths["high_mask"],
            result.high_gloss_mask,
        )
        cv_imwrite(
            paths["region_debug"],
            result.region_debug,
        )
        cv_imwrite(
            paths["analysis_mask"],
            result.gloss_analysis_mask,
        )
        cv_imwrite(
            paths["unreliable_highlight"],
            result.unreliable_highlight_mask,
        )
        cv_imwrite(
            paths["broad_sheen"],
            np.rint(np.clip(result.broad_sheen_map, 0.0, 1.0) * 255.0).astype(np.uint8),
        )
        cv_imwrite(
            paths["specular"],
            np.rint(np.clip(result.specular_map, 0.0, 1.0) * 255.0).astype(np.uint8),
        )
        # 仅用于调试显示；JSON中的P50/P90仍保存未拉伸的原始比例值。
        raw_display = np.rint(
            np.clip(
                result.raw_relative_luminance_map
                / max(float(Config.RAW_INTENSITY_SCALE) * 2.0, 1e-6),
                0.0,
                1.0,
            ) * 255.0
        ).astype(np.uint8)
        cv_imwrite(
            paths["raw_relative"],
            raw_display,
        )
        illum = result.illumination_corrected_luminance_map
        illum_display = np.rint(
            np.clip((illum - 0.70) / 0.60, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        cv_imwrite(
            paths["illumination_corrected"],
            illum_display,
        )
        cv_imwrite(
            paths["broad_color_support"],
            np.rint(
                np.clip(result.broad_color_support_map, 0.0, 1.0) * 255.0
            ).astype(np.uint8),
        )
        cv_imwrite(
            paths["specular_neighborhood"],
            np.rint(
                np.clip(result.specular_neighborhood_map, 0.0, 1.0) * 255.0
            ).astype(np.uint8),
        )

        with open(
            paths["json"],
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result.metrics(),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        SurfaceGlossAnalyzer._write_csv(
            result,
            paths["csv"],
        )
        return paths

    def process_preprocess_result(
        self,
        preprocess_result: PreprocessResultV2,
        output_dir: str,
        source_name: str,
    ) -> str | None:
        """和现有 Pores/Spots V2 Engine 保持同一入口风格。"""
        if preprocess_result.quality_status == "REJECT":
            print(
                f"⚠️ 跳过 {source_name}: V2质量门禁拒绝 "
                f"[{','.join(preprocess_result.quality_flags)}]"
            )
            return None

        base_name = os.path.splitext(
            os.path.basename(source_name)
        )[0]

        result = self.detect_surface_gloss(
            preprocess_result
        )

        paths = self.save_result(
            result,
            output_dir,
            base_name,
        )

        print(
            f"✨ Surface Gloss V3: "
            f"油光面积率 {result.gloss_area_ratio:.2%} | "
            f"高强度 {result.high_gloss_area_ratio:.2%} | "
            f"P90强度 {result.p90_gloss_intensity:.3f} | "
            f"最大连续区 {result.largest_gloss_component_area_ratio:.2%}"
        )
        print(
            f"✅ 生成完毕 -> "
            f"{base_name}/{os.path.basename(paths['overlay'])}"
        )
        return paths["overlay"]


def detect_surface_gloss(
    preprocess_result: PreprocessResultV2,
) -> SurfaceGlossResult:
    return SurfaceGlossAnalyzer().detect_surface_gloss(
        preprocess_result
    )


__all__ = [
    "Config",
    "OilRegionSet",
    "SurfaceGlossResult",
    "SurfaceGlossAnalyzer",
    "build_oil_regions",
    "detect_surface_gloss",
]
