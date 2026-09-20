# -*- coding: utf-8 -*-
"""DermaVision 标准化 UV / 荧光 UV 双通道 Purple Analysis V1.

本模块把三项职责严格分离：

1. :class:`PurpleBaseStyleRenderer` 生成 UV 与荧光 UV 两张标准化底图；
2. :class:`PorphyrinProxyDetector` 复用现有 ``PorphyrinAnalyzer``，但把
   荧光 UV 底图作为检测图；
3. :class:`UVSpotsAnalyzer` 把 UV-like 底图视为标准 UV 输入，独立提取
   多尺度局部暗斑代理，不复用 Spots/Brown 的最终实例。

当前由白光 ``analysis_image`` 确定性生成两张标准化工程底图，检测器只
读取对应底图、skin_mask、landmarks 和质量字段，不读取 display_image。
后续可直接用真实 UV 与荧光 UV 图替换底图生成步骤。当前输出不是 VISIA
官方指标、细菌检测、皮下黑色素测量或医学诊断。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from typing_extensions import assert_never

from src.capture_profile import CaptureProfile
from src.consumer_pigment.formal_markers import (
    ConsumerMarkerInput,
    ConsumerPigmentMarkerKind,
    project_consumer_markers,
)
from src.consumer_pigment.markers import (
    PORPHYRIN_STYLE_MARKER,
    UV_STYLE_MARKER,
    render_compact_marker_mask,
)
from src.consumer_pigment.specular import build_specular_evidence
from src.engines.rgb_proxy_fluorescence_porphyrin_engine import (
    RGB_PROXY_PORPHYRIN_CONFIG,
    RGBProxyFluorescencePorphyrinAnalyzer,
)
from src.engines.porphyrin_engine import PorphyrinAnalyzer, PorphyrinResult
from src.engines.purple_base_renderer import PurpleBaseStyleRenderer
from src.engines.uv_spots_engine import UVSpotsAnalyzer
from src.engines.visia_regions import (
    REGION_LABELS,
    REGION_ORDER,
    VISIA_BOUNDARY_COLOR,
    VisiaRegionSet,
    assign_region,
    build_moustache_feature_exclusion_mask,
    build_nasolabial_shadow_mask,
    build_visia_regions,
    draw_region_boundaries,
    masked_gaussian,
    robust_unit_map,
)
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.medical_v2_delivery import combine_documents
from src.utils.detailed_metrics import (
    METRIC_VERSION,
    build_medical_payload,
    json_safe,
    try_write_medical_metrics_v2,
)
from src.utils.io_utils import cv_imwrite


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PurpleAnalysisConfig:
    """Purple Analysis V1 的集中调参区。"""

    # 共享分析区域。30px 与项目中的实例五官安全区保持同一量级。
    shared_feature_margin_px: int = 30
    minimum_analysis_pixels: int = 500

    # UV-like 底图：只改变展示，不改变检测分数。
    base_low_percentile: float = 3.0
    base_high_percentile: float = 98.5
    base_structure_low_percentile: float = 35.0
    base_structure_high_percentile: float = 99.3
    base_blackhat_kernels: tuple[int, ...] = (7, 13, 23)
    base_clahe_clip_limit: float = 2.2
    base_clahe_grid_size: tuple[int, int] = (8, 8)
    base_render_feather_sigma: float = 1.4
    base_signed_detail_gain: float = 42.0
    base_feature_structure_scale: float = 0.18

    # OpenCV BGR 的分段紫蓝 LUT 锚点。
    base_lut_positions: tuple[float, ...] = (0.0, 0.22, 0.50, 0.78, 1.0)
    base_lut_colors_bgr: tuple[tuple[int, int, int], ...] = (
        (10, 3, 7),
        (42, 7, 20),
        (92, 17, 62),
        (155, 50, 128),
        (218, 132, 205),
    )

    # 紫质 Overlay。
    porphyrin_marker_color: tuple[int, int, int] = (0, 236, 255)
    porphyrin_outline_color: tuple[int, int, int] = (0, 178, 255)
    porphyrin_marker_dilate_kernel: int = 3

    # UV Spots 连续融合。
    uv_spots_brown_weight: float = 0.55
    uv_spots_spots_weight: float = 0.25
    uv_spots_local_weight: float = 0.20
    uv_spots_candidate_threshold: float = 0.42
    uv_spots_strong_local_threshold: float = 0.70
    uv_spots_min_mean_score: float = 0.34

    # Spots 实例筛选。
    spots_min_confidence: float = 0.24
    spots_min_brownness: float = 0.72
    spots_min_regional_delta_e: float = 1.20
    spots_min_dark_evidence: float = 0.75
    spots_min_yellow_evidence: float = 0.55
    spots_red_dominance_ratio: float = 1.20

    # UV Spots 形状过滤。
    uv_spots_min_area_px: int = 10
    uv_spots_max_area_ratio: float = 0.010
    uv_spots_min_solidity: float = 0.14
    uv_spots_min_extent: float = 0.07
    uv_spots_max_aspect_ratio: float = 4.2
    uv_spots_small_pore_area_px: int = 90
    uv_spots_pore_min_circularity: float = 0.42
    uv_spots_pore_overlap_ratio: float = 0.20

    # 高光、阴影、法令纹和紫质交叉抑制。
    highlight_value: float = 0.94
    highlight_max_saturation: float = 0.24
    deep_shadow_value: float = 0.15
    nasolabial_corridor_radius_px: int = 25
    nasolabial_shadow_margin_px: int = 5
    porphyrin_suppression_radius_px: int = 3
    porphyrin_score_scale: float = 0.30

    # UV Spots Overlay。
    uv_spots_fill_color: tuple[int, int, int] = (0, 175, 255)
    uv_spots_outline_color: tuple[int, int, int] = (0, 220, 255)
    uv_spots_fill_alpha: float = 0.82
    region_color: tuple[int, int, int] = VISIA_BOUNDARY_COLOR
    region_thickness: int = 2
    nose_style: str = "smooth"


@dataclass
class PurpleAnalysisResult:
    uv_like_base: np.ndarray
    porphyrin_base: np.ndarray

    porphyrin_overlay: np.ndarray
    porphyrin_mask: np.ndarray
    porphyrin_score_map: np.ndarray
    porphyrin_locations: list[dict[str, Any]]
    porphyrin_region_distribution: dict[str, dict[str, Any]]
    porphyrin_count: int
    porphyrin_area_ratio: float

    uv_spots_overlay: np.ndarray
    uv_spots_mask: np.ndarray
    uv_spots_score_map: np.ndarray
    uv_spots_instances: list[dict[str, Any]]
    uv_spots_region_distribution: dict[str, dict[str, Any]]
    uv_spots_count: int
    uv_spots_area_ratio: float

    analysis_mask: np.ndarray
    feature_exclusion_mask: np.ndarray
    region_debug: np.ndarray
    landmarks: np.ndarray

    quality_score: float
    quality_status: str
    quality_flags: list[str]
    partial_face: bool
    runtime_ms: float

    def metrics(self) -> dict[str, Any]:
        """返回不含 NumPy 数组、可直接 JSON 序列化的指标。"""
        return {
            "algorithm": "Standardized UV Purple Analysis V1",
            "definition": (
                "标准化UV底图上的紫外线色斑工程检测，以及标准化荧光UV"
                "底图上的紫质荧光工程检测。当前底图由普通RGB照片调光生成，"
                "后续可替换为真实UV采集图。"
            ),
            "porphyrin_count": int(self.porphyrin_count),
            "porphyrin_area_ratio": float(self.porphyrin_area_ratio),
            "porphyrin_locations": self.porphyrin_locations,
            "porphyrin_region_distribution": self.porphyrin_region_distribution,
            "uv_spots_count": int(self.uv_spots_count),
            "uv_spots_area_ratio": float(self.uv_spots_area_ratio),
            "uv_spots_instances": self.uv_spots_instances,
            "uv_spots_region_distribution": self.uv_spots_region_distribution,
            "t_zone_summary": {
                "porphyrin": self.porphyrin_region_distribution.get("t_zone", {}),
                "uv_spots": self.uv_spots_region_distribution.get("t_zone", {}),
            },
            "non_t_zone_summary": {
                "porphyrin": self.porphyrin_region_distribution.get("non_t_zone", {}),
                "uv_spots": self.uv_spots_region_distribution.get("non_t_zone", {}),
            },
            "quality_score": float(self.quality_score),
            "quality_status": str(self.quality_status),
            "quality_flags": list(self.quality_flags),
            "partial_face": bool(self.partial_face),
            "runtime_ms": round(float(self.runtime_ms), 3),
            "confidence_definition": (
                "所有 confidence 均为确定性工程候选评分，不是医学概率。"
            ),
            "disclaimer": (
                "UV-like 底图不是真实UV成像；紫质结果不是VISIA官方"
                "Porphyrins、荧光或细菌检测；UV Spots结果不是VISIA官方"
                "UV Spots、皮下黑色素测量或医学诊断。"
            ),
        }

    @staticmethod
    def _medical_instances(
        instances: list[dict[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        """把两类检测实例转换为五项医学量化工具的统一输入。"""
        output: list[dict[str, Any]] = []
        for index, raw in enumerate(instances, start=1):
            item = dict(raw)
            item["id"] = int(item.get("id", index))
            item["area"] = float(item.get("area", 0.0))
            item["intensity"] = float(
                item.get("mean_score", item.get("confidence", 0.0))
            )
            item["confidence"] = float(item.get("confidence", 0.0))
            item["morphology"] = kind
            output.append(item)
        return output

    def medical_payloads(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """返回 UV 色斑与紫质的独立医学统计源，不改变公开 metrics。"""
        common_quality = {
            "图像质量分数": float(self.quality_score),
            "图像质量状态": str(self.quality_status),
            "图像质量提示": list(self.quality_flags),
            "局部脸": bool(self.partial_face),
        }
        limitations = [
            "当前底图由普通白光RGB照片确定性调光生成，不是真实UV或荧光UV采集。",
            "紫外线色斑为工程代理，不能解释为皮下黑色素含量或VISIA官方UV Spots。",
            "紫质为工程代理，不能解释为细菌数量、卟啉浓度或医学诊断。",
        ]
        uv_payload = build_medical_payload(
            project="purple_uv_spots",
            project_label="标准化UV紫外线色斑工程代理",
            analysis_mask=self.analysis_mask,
            landmarks=self.landmarks,
            instances=self._medical_instances(
                self.uv_spots_instances,
                "紫外线色斑实例",
            ),
            score_map=self.uv_spots_score_map,
            instance_mask=self.uv_spots_mask,
            quality_control=common_quality,
            limitations=limitations,
        )
        porphyrin_payload = build_medical_payload(
            project="purple_porphyrin",
            project_label="标准化荧光UV紫质工程代理",
            analysis_mask=self.analysis_mask,
            landmarks=self.landmarks,
            instances=self._medical_instances(
                self.porphyrin_locations,
                "紫质实例",
            ),
            score_map=self.porphyrin_score_map,
            instance_mask=self.porphyrin_mask,
            quality_control=common_quality,
            limitations=limitations,
        )
        return uv_payload, porphyrin_payload

    def medical_metrics(self) -> dict[str, Any]:
        """生成医生使用的 ``medical_metrics_v1`` 紫区宽表。"""
        uv_payload, porphyrin_payload = self.medical_payloads()
        uv_rows = [uv_payload["总体指标"], *uv_payload["分区指标"]]
        porphyrin_rows = [
            porphyrin_payload["总体指标"],
            *porphyrin_payload["分区指标"],
        ]
        if len(uv_rows) != len(porphyrin_rows):
            raise ValueError("紫外线色斑和紫质医学分区数量不一致")

        shared_columns = (
            "检测范围",
            "评估状态",
            "有效皮肤面积（像素）",
        )
        metric_columns = (
            "特征数量（个）",
            "单位面积密度（个/10万有效皮肤像素）",
            "特征总面积（像素）",
            "特征面积占比",
            "P50单体面积（像素）",
            "P90单体面积（像素）",
            "最大单体面积（像素）",
            "平均强度（0～1）",
            "P50强度（0～1）",
            "P90强度（0～1）",
            "P95强度（0～1）",
            "实例P50强度（0～1）",
            "实例P90强度（0～1）",
            "主要集中区域",
            "画面左右面颊数量密度差异",
            "画面左右面颊面积占比差异",
        )
        combined_rows: list[dict[str, Any]] = []
        for uv_row, porphyrin_row in zip(uv_rows, porphyrin_rows):
            if uv_row.get("检测范围") != porphyrin_row.get("检测范围"):
                raise ValueError("紫外线色斑和紫质医学分区名称不一致")
            row = {
                name: uv_row.get(name, "")
                for name in shared_columns
            }
            for prefix, source in (
                ("紫外线色斑", uv_row),
                ("紫质", porphyrin_row),
            ):
                for name in metric_columns:
                    row[f"{prefix}{name}"] = source.get(name, "")
            combined_rows.append(row)

        return json_safe({
            "检测项目": "标准化UV紫外线色斑与紫质工程分析",
            "指标版本": METRIC_VERSION,
            "总体指标": combined_rows[0],
            "分区指标": combined_rows[1:],
        })

    @staticmethod
    def _public_count_summary(
        total: int,
        distribution: dict[str, dict[str, Any]],
        partial_face: bool,
    ) -> dict[str, int]:
        """构建前端展示用的英文计数摘要。"""

        summary = {"total": int(total)}
        if not partial_face:
            for name in ("forehead", "left_cheek", "right_cheek", "nose", "chin"):
                summary[name] = int(distribution.get(name, {}).get("count", 0))
        return summary

    def public_metrics(self) -> dict[str, Any]:
        """返回云端正式透传的精简英文 JSON。

        这里只保留用户简表能够展示的两项总数和五分区数量。面积、强度、
        分位数及医生统计继续写入独立医学 CSV，不进入前端 ``metrics``。
        """

        public: dict[str, int] = {}
        for prefix, summary in (
            (
                "uv_spots",
                self._public_count_summary(
                    self.uv_spots_count,
                    self.uv_spots_region_distribution,
                    self.partial_face,
                ),
            ),
            (
                "porphyrin",
                self._public_count_summary(
                    self.porphyrin_count,
                    self.porphyrin_region_distribution,
                    self.partial_face,
                ),
            ),
        ):
            for region, value in summary.items():
                public[f"{prefix}_{region}"] = int(value)
        return public

    # 兼容旧的本地调用名；正式云端与保存逻辑使用 public_metrics()。
    def compact_metrics(self) -> dict[str, Any]:
        return self.public_metrics()

    def user_summary_rows(self) -> tuple[list[str], list[list[Any]]]:
        """返回面向用户的紫区简表，不改变公开 JSON 的既有结构。

        紫区包含紫外线色斑和紫质两个检测项目，因此以两个数据行展示；
        完整脸显示五个常用分区，局部脸只展示总计，避免把不可见区域写成 0。
        """
        public = self.public_metrics()
        columns = (
            ("总计", "total"),
            ("额头", "forehead"),
            ("左脸颊", "left_cheek"),
            ("右脸颊", "right_cheek"),
            ("鼻部", "nose"),
            ("下巴", "chin"),
        )
        active_columns = columns[:1] if self.partial_face else columns
        headers = ["检测项目", *(label for label, _ in active_columns)]
        return headers, [
            [
                "紫外线色斑",
                *(public[f"uv_spots_{key}"] for _, key in active_columns),
            ],
            [
                "紫质",
                *(public[f"porphyrin_{key}"] for _, key in active_columns),
            ],
        ]


@dataclass
class _DetectionPreprocessView:
    """只暴露算法允许读取的字段，并替换本次检测的标准化底图。

    紫质引擎继续复用现有 ``PorphyrinAnalyzer``，但其 ``analysis_image``
    明确指向荧光 UV 底图。该适配层没有 ``display_image``，因此不会把
    展示图或原始白光颜色意外带回特征打分。
    """

    analysis_image: np.ndarray
    skin_mask: np.ndarray
    landmarks: np.ndarray
    quality_score: float
    quality_status: str
    quality_flags: list[str]


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _masked_percentile_parameters(
    values: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float32)[mask > 0]
    if samples.size < 32:
        return 0.0, 1.0
    lo, hi = np.percentile(samples, [low, high])
    return float(lo), max(float(hi), float(lo) + 1e-6)


def _normalize_with_parameters(
    values: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    return np.clip(
        (np.asarray(values, dtype=np.float32) - low) / max(high - low, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)


def _full_render_mask(regions: VisiaRegionSet) -> np.ndarray:
    output = np.zeros_like(regions.analysis_mask, dtype=np.uint8)
    for mask in regions.display_regions.values():
        output = cv2.bitwise_or(output, _binary_mask(mask))
    return output


def _piecewise_bgr_lut(
    values: np.ndarray,
    positions: tuple[float, ...],
    colors_bgr: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    values_f = np.clip(values.astype(np.float32), 0.0, 1.0)
    xp = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(colors_bgr, dtype=np.float32)
    channels = [
        np.interp(values_f, xp, colors[:, index]).astype(np.float32)
        for index in range(3)
    ]
    return np.stack(channels, axis=2)


def _safe_region_image(image: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """仅用于共享区域构建，防止 Mask 外像素影响鼻孔局部判断。"""
    safe = np.asarray(image).copy()
    valid = skin_mask > 0
    if np.any(valid):
        safe[~valid] = np.median(safe[valid], axis=0).astype(np.uint8)
    else:
        safe[:] = 0
    return safe


def _region_distribution(
    instances: list[dict[str, Any]],
    instance_mask: np.ndarray,
    score_map: np.ndarray,
    regions: VisiaRegionSet,
) -> dict[str, dict[str, Any]]:
    t_names = {"forehead", "nose", "chin"}
    output: dict[str, dict[str, Any]] = {}

    def summarize(
        name: str,
        label: str,
        region_mask: np.ndarray,
        accepted_names: set[str],
    ) -> dict[str, Any]:
        valid = region_mask > 0
        local = [item for item in instances if item.get("region") in accepted_names]
        area = int(np.count_nonzero((instance_mask > 0) & valid))
        region_area = max(int(np.count_nonzero(valid)), 1)
        values = score_map[valid]
        return {
            "label": label,
            "count": len(local),
            "area": area,
            "area_ratio": round(area / region_area, 8),
            "mean_score": round(float(np.mean(values)) if values.size else 0.0, 6),
        }

    for name in REGION_ORDER:
        output[name] = summarize(
            name,
            REGION_LABELS[name],
            regions.regions[name],
            {name},
        )

    t_mask = np.zeros_like(regions.analysis_mask)
    non_t_mask = np.zeros_like(regions.analysis_mask)
    for name in REGION_ORDER:
        if name in t_names:
            t_mask = cv2.bitwise_or(t_mask, regions.regions[name])
        else:
            non_t_mask = cv2.bitwise_or(non_t_mask, regions.regions[name])
    output["t_zone"] = summarize("t_zone", "T区（工程代理）", t_mask, t_names)
    output["non_t_zone"] = summarize(
        "non_t_zone",
        "非T区（工程代理）",
        non_t_mask,
        set(REGION_ORDER) - t_names,
    )
    return output


class UVLikeBaseRenderer:
    """从白光 analysis_image 生成仅用于展示的 UV-like 紫蓝底图。"""

    def __init__(self, config: PurpleAnalysisConfig | None = None) -> None:
        self.config = config or PurpleAnalysisConfig()

    def render(
        self,
        analysis_image: np.ndarray,
        skin_mask: np.ndarray,
        statistics_mask: np.ndarray,
        full_face_render_mask: np.ndarray,
        feature_exclusion_mask: np.ndarray,
    ) -> np.ndarray:
        image = np.asarray(analysis_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("analysis_image 必须是 H×W×3 BGR 图像")

        stats = cv2.bitwise_and(_binary_mask(skin_mask), _binary_mask(statistics_mask))
        render = _binary_mask(full_face_render_mask)
        if np.count_nonzero(stats) < self.config.minimum_analysis_pixels:
            return np.zeros_like(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_u8 = lab[:, :, 0]
        l_float = l_u8.astype(np.float32) / 255.0
        valid_l = l_u8[stats > 0]
        fill_value = int(np.median(valid_l)) if valid_l.size else 128
        filled_l = l_u8.copy()
        filled_l[stats == 0] = fill_value

        clahe = cv2.createCLAHE(
            clipLimit=self.config.base_clahe_clip_limit,
            tileGridSize=self.config.base_clahe_grid_size,
        ).apply(filled_l)
        clahe_f = clahe.astype(np.float32) / 255.0

        # 多尺度局部结构。所有用于归一化的统计只来自 stats。
        blackhats: list[np.ndarray] = []
        for size in self.config.base_blackhat_kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            blackhats.append(
                cv2.morphologyEx(filled_l, cv2.MORPH_BLACKHAT, kernel)
                .astype(np.float32)
                / 255.0
            )
        blackhat = np.max(np.stack(blackhats, axis=0), axis=0)

        fine = cv2.GaussianBlur(clahe_f, (0, 0), sigmaX=0.9, sigmaY=0.9)
        medium = cv2.GaussianBlur(clahe_f, (0, 0), sigmaX=3.2, sigmaY=3.2)
        broad = cv2.GaussianBlur(clahe_f, (0, 0), sigmaX=10.0, sigmaY=10.0)
        dog = np.abs(fine - medium) + 0.45 * np.abs(medium - broad)
        laplacian = np.abs(cv2.Laplacian(fine, cv2.CV_32F, ksize=3))

        local_mean = masked_gaussian(l_float, stats, 3.5)
        local_second = masked_gaussian(l_float * l_float, stats, 3.5)
        local_std = np.sqrt(np.maximum(local_second - local_mean * local_mean, 0.0))

        blackhat_n = robust_unit_map(blackhat, stats, 25.0, 99.4)
        dog_n = robust_unit_map(dog, stats, 30.0, 99.4)
        lap_n = robust_unit_map(laplacian, stats, 40.0, 99.5)
        std_n = robust_unit_map(local_std, stats, 25.0, 99.0)
        structure = np.clip(
            0.36 * blackhat_n + 0.30 * dog_n + 0.18 * lap_n + 0.16 * std_n,
            0.0,
            1.0,
        )

        # 五官位置仍显示，但抑制纹理增强，避免眼睫、唇纹和鼻孔被过度强化。
        feature_pixels = feature_exclusion_mask > 0
        structure[feature_pixels] *= self.config.base_feature_structure_scale

        tone_low, tone_high = _masked_percentile_parameters(
            clahe_f,
            stats,
            self.config.base_low_percentile,
            self.config.base_high_percentile,
        )
        tone = _normalize_with_parameters(clahe_f, tone_low, tone_high)
        display_level = np.clip(0.78 * tone + 0.22 * structure, 0.0, 1.0)
        canvas = _piecewise_bgr_lut(
            display_level,
            self.config.base_lut_positions,
            self.config.base_lut_colors_bgr,
        )

        # 真实亮度的有符号细节只用于显示，仍使用 stats 内分位数控制幅度。
        signed = fine - medium
        signed_values = signed[stats > 0]
        signed_limit = (
            float(np.percentile(np.abs(signed_values), 99.0))
            if signed_values.size
            else 0.02
        )
        signed_normalized = np.clip(
            signed / max(signed_limit, 1e-6),
            -1.0,
            1.0,
        )
        signed_normalized[feature_pixels] *= 0.35
        canvas += (
            signed_normalized[:, :, None]
            * self.config.base_signed_detail_gain
        )

        alpha = cv2.GaussianBlur(
            (render > 0).astype(np.float32),
            (0, 0),
            sigmaX=self.config.base_render_feather_sigma,
            sigmaY=self.config.base_render_feather_sigma,
        )
        output = canvas * np.clip(alpha[:, :, None], 0.0, 1.0)
        return np.clip(output, 0.0, 255.0).astype(np.uint8)


class PorphyrinProxyDetector:
    """现有 PorphyrinAnalyzer 的显示适配器，不重复实现检测算法。"""

    def __init__(
        self,
        analyzer: PorphyrinAnalyzer,
        config: PurpleAnalysisConfig | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.config = config or PurpleAnalysisConfig()

    def detect(
        self,
        preprocess_result: PreprocessResultV2,
        fluorescence_base: np.ndarray,
        regions: VisiaRegionSet,
        existing_result: PorphyrinResult | None = None,
        instance_exclusion_mask: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        PorphyrinResult,
    ]:
        detection_input = _DetectionPreprocessView(
            analysis_image=np.asarray(fluorescence_base),
            skin_mask=_binary_mask(preprocess_result.skin_mask),
            landmarks=np.asarray(preprocess_result.landmarks, dtype=np.float32),
            quality_score=float(preprocess_result.quality_score),
            quality_status=str(preprocess_result.quality_status),
            quality_flags=list(preprocess_result.quality_flags),
        )
        result = existing_result or self.analyzer.detect_porphyrins(detection_input)
        score = np.clip(
            np.asarray(result.porphyrin_score_map, dtype=np.float32),
            0.0,
            1.0,
        )
        score[regions.analysis_mask == 0] = 0.0
        mask = np.zeros_like(regions.analysis_mask, dtype=np.uint8)
        exclusion = _binary_mask(regions.feature_exclusion_mask)
        if instance_exclusion_mask is not None:
            exclusion = cv2.bitwise_or(
                exclusion,
                _binary_mask(instance_exclusion_mask),
            )

        locations: list[dict[str, Any]] = []
        height, width = mask.shape
        for raw in result.porphyrin_locations:
            item = dict(raw)
            centroid = item.get("centroid", [])
            if len(centroid) != 2:
                continue
            x = int(np.clip(round(float(centroid[0])), 0, width - 1))
            y = int(np.clip(round(float(centroid[1])), 0, height - 1))
            if (
                result.analysis_mask[y, x] == 0
                or regions.analysis_mask[y, x] == 0
                or exclusion[y, x] > 0
            ):
                continue
            item["centroid"] = [x, y]
            item["confidence_definition"] = "确定性工程候选评分，非医学概率"
            locations.append(item)
            radius = int(np.clip(item.get("radius_px", 2), 1, 8))
            cv2.circle(mask, (x, y), radius, 255, -1, lineType=cv2.LINE_AA)
        mask = cv2.bitwise_and(mask, _binary_mask(regions.analysis_mask))

        overlay = fluorescence_base.copy()
        marker = mask
        kernel_size = max(1, int(self.config.porphyrin_marker_dilate_kernel))
        if kernel_size > 1:
            marker = cv2.dilate(
                marker,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size),
                ),
            )
        overlay[marker > 0] = self.config.porphyrin_marker_color
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            self.config.porphyrin_outline_color,
            1,
            lineType=cv2.LINE_AA,
        )
        overlay = draw_region_boundaries(
            overlay,
            regions.display_regions,
            color=self.config.region_color,
            thickness=self.config.region_thickness,
            partial_face=regions.partial_face,
            nose_style=self.config.nose_style,
            landmarks=np.asarray(preprocess_result.landmarks),
            closed_contour=getattr(regions, "display_contour", None),
            separator_contour=getattr(regions, "display_separator", None),
        )
        return (
            overlay,
            mask,
            score,
            locations,
            dict(result.region_distribution),
            result,
        )


class UVSpotsProxyDetector:
    """已停用的旧 Brown/Spots 融合实验，仅保留源码回溯兼容。

    正式流程使用 :class:`src.engines.uv_spots_engine.UVSpotsAnalyzer`。
    """

    def __init__(self, config: PurpleAnalysisConfig | None = None) -> None:
        self.config = config or PurpleAnalysisConfig()

    def _select_spot(self, item: dict[str, Any]) -> bool:
        confidence = float(item.get("confidence", 0.0))
        if confidence < self.config.spots_min_confidence:
            return False
        spot_type = str(item.get("spot_type", "small"))
        brownness = float(item.get("brownness_score", 0.0))
        regional_delta_e = float(item.get("regional_deltaE", 0.0))
        local_contrast = float(item.get("local_contrast", 0.0))
        red = float(item.get("salient_red_difference", 0.0))
        yellow = float(item.get("salient_yellow_difference", 0.0))
        dark = float(item.get("salient_dark_difference", 0.0))
        prominent_red = bool(item.get("prominent_red_rescue", False))

        pigment_support = (
            brownness >= self.config.spots_min_brownness
            or regional_delta_e >= self.config.spots_min_regional_delta_e
            or dark >= self.config.spots_min_dark_evidence
            or yellow >= self.config.spots_min_yellow_evidence
            or (
                spot_type in {"large", "salient"}
                and local_contrast >= self.config.spots_min_dark_evidence
            )
        )
        if not pigment_support:
            return False

        red_dominant = red > self.config.spots_red_dominance_ratio * max(yellow, 1e-6)
        if prominent_red and red_dominant and brownness < self.config.spots_min_brownness:
            return False
        return True

    @staticmethod
    def _fallback_item_mask(
        shape: tuple[int, int],
        item: dict[str, Any],
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        bbox = item.get("bbox", [])
        centroid = item.get("centroid", [])
        if len(centroid) != 2:
            return mask
        cx = int(np.clip(round(float(centroid[0])), 0, shape[1] - 1))
        cy = int(np.clip(round(float(centroid[1])), 0, shape[0] - 1))
        area = max(float(item.get("area", 1.0)), 1.0)
        radius = max(2, int(round(math.sqrt(area / math.pi))))
        if len(bbox) == 4:
            width = max(2, int(round(float(bbox[2]))))
            height = max(2, int(round(float(bbox[3]))))
            cv2.ellipse(
                mask,
                (cx, cy),
                (max(1, width // 2), max(1, height // 2)),
                0,
                0,
                360,
                255,
                -1,
            )
        else:
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask

    def _selected_spot_support(
        self,
        spots_result: SpotsResultV2,
        analysis_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        final_mask = _binary_mask(spots_result.filtered_mask)
        _, labels = cv2.connectedComponents((final_mask > 0).astype(np.uint8))
        selected_mask = np.zeros_like(final_mask)
        support = np.zeros(final_mask.shape, dtype=np.float32)
        selected_items: list[dict[str, Any]] = []
        height, width = final_mask.shape

        for raw in spots_result.spot_locations:
            item = dict(raw)
            if not self._select_spot(item):
                continue
            centroid = item.get("centroid", [])
            if len(centroid) != 2:
                continue
            x = int(np.clip(round(float(centroid[0])), 0, width - 1))
            y = int(np.clip(round(float(centroid[1])), 0, height - 1))
            if analysis_mask[y, x] == 0:
                continue
            label = int(labels[y, x])
            if label > 0:
                component = labels == label
            else:
                component = self._fallback_item_mask(final_mask.shape, item) > 0
            component &= analysis_mask > 0
            if not np.any(component):
                continue
            selected_mask[component] = 255
            confidence = float(np.clip(item.get("confidence", 0.0), 0.0, 1.0))
            support[component] = np.maximum(support[component], max(confidence, 0.35))
            selected_items.append(item)
        return selected_mask, support.astype(np.float32), selected_items

    @staticmethod
    def _line_kernel(length: int, angle: int) -> np.ndarray:
        size = length if length % 2 else length + 1
        kernel = np.zeros((size, size), dtype=np.uint8)
        centre = (size - 1) * 0.5
        radius = centre - 1.0
        theta = np.deg2rad(float(angle))
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)
        start = (int(round(centre - dx)), int(round(centre - dy)))
        end = (int(round(centre + dx)), int(round(centre + dy)))
        cv2.line(kernel, start, end, 1, 1, cv2.LINE_8)
        return kernel

    @classmethod
    def _line_response(
        cls,
        image: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        l_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        valid = l_channel[analysis_mask > 0]
        filled = l_channel.copy()
        filled[analysis_mask == 0] = int(np.median(valid)) if valid.size else 128
        response = np.zeros(l_channel.shape, dtype=np.float32)
        for length in (9, 17):
            for angle in (0, 30, 60, 90, 120, 150):
                local = cv2.morphologyEx(
                    filled,
                    cv2.MORPH_BLACKHAT,
                    cls._line_kernel(length, angle),
                ).astype(np.float32)
                response = np.maximum(response, local)
        response[analysis_mask == 0] = 0.0
        return robust_unit_map(response, analysis_mask, 45.0, 99.5).astype(np.float32)

    @staticmethod
    def _local_pigmentation_score(
        image: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0] / 255.0
        a = (lab[:, :, 1] - 128.0) / 127.0
        b = (lab[:, :, 2] - 128.0) / 127.0

        raw_scales: list[np.ndarray] = []
        for sigma in (8.0, 18.0, 32.0):
            bg_l = masked_gaussian(l, analysis_mask, sigma)
            bg_a = masked_gaussian(a, analysis_mask, sigma)
            bg_b = masked_gaussian(b, analysis_mask, sigma)
            dark = np.maximum(bg_l - l, 0.0)
            red_brown = np.maximum(a - bg_a, 0.0)
            yellow_brown = np.maximum(b - bg_b, 0.0)
            chroma = np.sqrt(red_brown * red_brown + yellow_brown * yellow_brown)
            raw_scales.append(
                0.46 * dark / 0.040
                + 0.22 * yellow_brown / 0.026
                + 0.16 * red_brown / 0.026
                + 0.16 * chroma / 0.035
            )

        l_u8 = lab[:, :, 0].astype(np.uint8)
        valid_l = l_u8[analysis_mask > 0]
        filled = l_u8.copy()
        filled[analysis_mask == 0] = int(np.median(valid_l)) if valid_l.size else 128
        blackhats: list[np.ndarray] = []
        for size in (9, 17, 31):
            blackhats.append(
                cv2.morphologyEx(
                    filled,
                    cv2.MORPH_BLACKHAT,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                ).astype(np.float32)
                / 255.0
            )
        blackhat = np.max(np.stack(blackhats, axis=0), axis=0)
        dog = np.maximum(
            masked_gaussian(l, analysis_mask, 4.0) - l,
            0.0,
        )
        raw = (
            0.65 * np.max(np.stack(raw_scales, axis=0), axis=0)
            + 0.20 * blackhat / 0.08
            + 0.15 * dog / 0.04
        )
        raw[analysis_mask == 0] = 0.0
        return robust_unit_map(raw, analysis_mask, 2.0, 99.2).astype(np.float32)

    def detect(
        self,
        preprocess_result: PreprocessResultV2,
        regions: VisiaRegionSet,
        uv_like_base: np.ndarray,
        porphyrin_result: PorphyrinResult,
        spots_result: SpotsResultV2,
        brown_result: BrownResult,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        image = np.asarray(preprocess_result.analysis_image)
        analysis_mask = _binary_mask(regions.analysis_mask)
        selected_spots_mask, spots_support, _ = self._selected_spot_support(
            spots_result,
            analysis_mask,
        )
        brown_score = np.clip(
            np.asarray(brown_result.brown_score_map, dtype=np.float32),
            0.0,
            1.0,
        )
        brown_score[analysis_mask == 0] = 0.0
        local_score = self._local_pigmentation_score(image, analysis_mask)
        line_response = self._line_response(image, analysis_mask)

        fused = np.clip(
            self.config.uv_spots_brown_weight * brown_score
            + self.config.uv_spots_spots_weight * spots_support
            + self.config.uv_spots_local_weight * local_score,
            0.0,
            1.0,
        ).astype(np.float32)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        highlight = (
            (value >= self.config.highlight_value)
            & (saturation <= self.config.highlight_max_saturation)
        )
        deep_shadow = value <= self.config.deep_shadow_value
        nasolabial_shadow = build_nasolabial_shadow_mask(
            image,
            np.asarray(preprocess_result.landmarks),
            analysis_mask,
            radius_px=self.config.nasolabial_corridor_radius_px,
            output_margin_px=self.config.nasolabial_shadow_margin_px,
            restrict_frangi_to_corridor=True,
        )

        porphyrin_core = _binary_mask(porphyrin_result.porphyrin_mask)
        radius = max(0, int(self.config.porphyrin_suppression_radius_px))
        porphyrin_near = porphyrin_core
        if radius > 0:
            porphyrin_near = cv2.dilate(
                porphyrin_core,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * radius + 1, 2 * radius + 1),
                ),
            )
        fused[porphyrin_near > 0] *= self.config.porphyrin_score_scale
        fused[line_response >= 0.85] *= 0.25
        fused[
            (analysis_mask == 0)
            | (regions.feature_exclusion_mask > 0)
            | highlight
            | deep_shadow
            | (nasolabial_shadow > 0)
        ] = 0.0

        seed_mask = cv2.bitwise_or(
            _binary_mask(brown_result.instance_mask),
            selected_spots_mask,
        )
        candidate = (
            (
                ((seed_mask > 0) & (fused >= self.config.uv_spots_candidate_threshold))
                | (fused >= self.config.uv_spots_strong_local_threshold)
            )
            & (analysis_mask > 0)
        ).astype(np.uint8) * 255
        candidate[porphyrin_core > 0] = 0
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        candidate = cv2.bitwise_and(candidate, analysis_mask)

        contours, _ = cv2.findContours(
            candidate,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        analysis_area = max(int(np.count_nonzero(analysis_mask)), 1)
        max_area = max(
            self.config.uv_spots_min_area_px + 1,
            int(self.config.uv_spots_max_area_ratio * analysis_area),
        )
        output_mask = np.zeros_like(candidate)
        instances: list[dict[str, Any]] = []

        for contour in contours:
            component_mask = np.zeros_like(candidate)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            component = (component_mask > 0) & (candidate > 0)
            area = int(np.count_nonzero(component))
            if area < self.config.uv_spots_min_area_px or area > max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            aspect = max(width, height) / max(min(width, height), 1)
            if aspect > self.config.uv_spots_max_aspect_ratio:
                continue
            line_values = line_response[component]
            line_p90 = (
                float(np.percentile(line_values, 90.0))
                if line_values.size
                else 0.0
            )
            if aspect >= 1.45 and line_p90 >= 0.55:
                continue
            contour_area = max(float(cv2.contourArea(contour)), 1.0)
            hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
            solidity = contour_area / hull_area
            extent = contour_area / max(width * height, 1)
            circularity = 4.0 * math.pi * contour_area / max(perimeter * perimeter, 1e-6)
            if (
                solidity < self.config.uv_spots_min_solidity
                or extent < self.config.uv_spots_min_extent
            ):
                continue
            mean_score = float(np.mean(fused[component]))
            if mean_score < self.config.uv_spots_min_mean_score:
                continue

            porphyrin_overlap = float(
                np.count_nonzero(component & (porphyrin_near > 0)) / max(area, 1)
            )
            if (
                area <= self.config.uv_spots_small_pore_area_px
                and circularity >= self.config.uv_spots_pore_min_circularity
                and porphyrin_overlap >= self.config.uv_spots_pore_overlap_ratio
            ):
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) <= 1e-6:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            px = int(np.clip(round(cx), 0, candidate.shape[1] - 1))
            py = int(np.clip(round(cy), 0, candidate.shape[0] - 1))
            if analysis_mask[py, px] == 0:
                continue
            region = assign_region(cx, cy, regions.regions)
            if region == "other":
                continue

            brown_overlap = float(
                np.count_nonzero(component & (brown_result.instance_mask > 0))
                / max(area, 1)
            )
            spots_overlap = float(
                np.count_nonzero(component & (selected_spots_mask > 0))
                / max(area, 1)
            )
            confidence = float(
                np.clip(
                    0.20
                    + 0.52 * mean_score
                    + 0.10 * min(solidity, 1.0)
                    + 0.10 * min(extent, 1.0)
                    + 0.04 * min(brown_overlap, 1.0)
                    + 0.04 * min(spots_overlap, 1.0),
                    0.0,
                    1.0,
                )
            )
            output_mask[component] = 255
            instances.append(
                {
                    "id": len(instances) + 1,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "area": area,
                    "region": region,
                    "mean_score": round(mean_score, 6),
                    "circularity": round(float(circularity), 4),
                    "solidity": round(float(solidity), 4),
                    "extent": round(float(extent), 4),
                    "aspect_ratio": round(float(aspect), 4),
                    "line_response_p90": round(line_p90, 4),
                    "brown_support_ratio": round(brown_overlap, 4),
                    "spots_support_ratio": round(spots_overlap, 4),
                    "porphyrin_overlap_ratio": round(porphyrin_overlap, 4),
                    "confidence": round(confidence, 4),
                    "confidence_definition": "确定性工程候选评分，非医学概率",
                }
            )

        output_mask = _binary_mask(output_mask)
        overlay = uv_like_base.copy()
        marker_pixels = output_mask > 0
        if np.any(marker_pixels):
            fill_color = np.asarray(self.config.uv_spots_fill_color, dtype=np.float32)
            overlay[marker_pixels] = np.clip(
                (1.0 - self.config.uv_spots_fill_alpha)
                * overlay[marker_pixels].astype(np.float32)
                + self.config.uv_spots_fill_alpha * fill_color,
                0.0,
                255.0,
            ).astype(np.uint8)
        final_contours, _ = cv2.findContours(
            output_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            final_contours,
            -1,
            self.config.uv_spots_outline_color,
            1,
            lineType=cv2.LINE_AA,
        )
        overlay = draw_region_boundaries(
            overlay,
            regions.display_regions,
            color=self.config.region_color,
            thickness=self.config.region_thickness,
            partial_face=regions.partial_face,
            nose_style=self.config.nose_style,
            landmarks=np.asarray(preprocess_result.landmarks),
            closed_contour=getattr(regions, "display_contour", None),
            separator_contour=getattr(regions, "display_separator", None),
        )
        distribution = _region_distribution(
            instances,
            output_mask,
            fused,
            regions,
        )
        return overlay, output_mask, fused.astype(np.float32), instances, distribution


class PurpleAnalysisEngine:
    """统一调度已有引擎、紫光展示、交叉抑制、量化与保存。"""

    _HARD_FAILURE_FLAGS = {
        "NO_FACE",
        "MULTIPLE_FACES",
        "INVALID_IMAGE",
        "INVALID_FACE_GEOMETRY",
    }

    def __init__(
        self,
        config: PurpleAnalysisConfig | None = None,
        capture_profile: CaptureProfile = CaptureProfile.INSTITUTION,
        *,
        porphyrin_analyzer: PorphyrinAnalyzer | None = None,
        spots_analyzer: Any | None = None,
        brown_analyzer: Any | None = None,
    ) -> None:
        self.config = config or PurpleAnalysisConfig()
        self.capture_profile = capture_profile
        # 显式注入旧 PorphyrinAnalyzer 时仅用于兼容/对照测试。正式流程
        # 使用标准化荧光 UV 底图检测器，不再从白光 analysis_image 打分。
        self.porphyrin_analyzer = porphyrin_analyzer
        # 保留旧构造参数，避免调用方中断；正式 UV Spots 已改为独立底图
        # 算法，不再调用 Spots/Brown 最终实例做简单融合。
        self.spots_analyzer = spots_analyzer
        self.brown_analyzer = brown_analyzer
        self._owns_porphyrin = False
        self._owns_spots = False
        self._owns_brown = False

        self.renderer = UVLikeBaseRenderer(self.config)
        # 新底图渲染器只改变展示风格：UV Spots 使用灰黑底图，紫质使用
        # 同结构的蓝紫荧光底图。实例分数、Mask 和数量不由底图反推。
        self.base_style_renderer = PurpleBaseStyleRenderer()
        self.porphyrin_detector = (
            PorphyrinProxyDetector(self.porphyrin_analyzer, self.config)
            if self.porphyrin_analyzer is not None
            else None
        )
        # 单RGB只允许使用冻结的代理检测器，不读取365通道。
        # consumer 仅缩小显示圆点，候选阈值与计数保持原版。
        match self.capture_profile:
            case CaptureProfile.INSTITUTION:
                porphyrin_config = RGB_PROXY_PORPHYRIN_CONFIG
            case CaptureProfile.CONSUMER:
                porphyrin_config = replace(
                    RGB_PROXY_PORPHYRIN_CONFIG,
                    display_min_radius_px=1,
                    display_max_radius_px=1,
                )
            case unreachable:
                assert_never(unreachable)
        self.fluorescence_porphyrin_detector = (
            RGBProxyFluorescencePorphyrinAnalyzer(porphyrin_config)
        )
        self.uv_spots_detector = UVSpotsAnalyzer()
        self.last_result: PurpleAnalysisResult | None = None
        self.last_paths: dict[str, str] = {}

    def _validate_preprocess_result(self, preprocess_result: PreprocessResultV2) -> None:
        image = np.asarray(preprocess_result.analysis_image)
        skin_mask = np.asarray(preprocess_result.skin_mask)
        landmarks = np.asarray(preprocess_result.landmarks)
        if preprocess_result.quality_status == "REJECT":
            raise ValueError("质量状态为 REJECT，不执行 Purple Analysis")
        if self._HARD_FAILURE_FLAGS.intersection(preprocess_result.quality_flags):
            raise ValueError("存在不可分析的质量标志")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("analysis_image 必须是 H×W×3")
        if image.shape[:2] != skin_mask.shape:
            raise ValueError("analysis_image 与 skin_mask 尺寸不一致")
        if landmarks.ndim != 2 or landmarks.shape[1] < 2:
            raise ValueError("landmarks 必须是 N×2 或 N×更多列")
        if np.count_nonzero(skin_mask) < self.config.minimum_analysis_pixels:
            raise ValueError("有效 skin_mask 像素过少")

    @staticmethod
    def _foreground_mask(preprocess_result: PreprocessResultV2) -> np.ndarray:
        """取得预处理的人像前景，保留头发和五官，仅移除背景。

        新版 Preprocessor 会把人像分割结果放入只读调试 Mask。旧版没有
        该字段时，以 analysis_image 的非黑画布作为兼容前景。这里绝不
        使用 skin_mask 裁展示图，避免眼睛、眉毛、嘴唇和头发出现孔洞。
        """
        debug_masks = getattr(preprocess_result, "_debug_masks", {})
        semantic = debug_masks.get("semantic_skin_mask")
        skin_shape = np.asarray(preprocess_result.skin_mask).shape
        if semantic is not None and np.asarray(semantic).shape == skin_shape:
            return _binary_mask(np.asarray(semantic))
        image = np.asarray(preprocess_result.analysis_image)
        return (np.max(image, axis=2) > 3).astype(np.uint8) * 255

    @staticmethod
    def _region_debug(
        image: np.ndarray,
        regions: VisiaRegionSet,
        landmarks: np.ndarray,
        config: PurpleAnalysisConfig,
    ) -> np.ndarray:
        debug = image.copy()
        purple_tint = np.zeros_like(debug)
        purple_tint[regions.analysis_mask > 0] = (105, 35, 95)
        debug = cv2.addWeighted(debug, 0.78, purple_tint, 0.22, 0.0)
        exclusion_contours, _ = cv2.findContours(
            _binary_mask(regions.feature_exclusion_mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            debug,
            exclusion_contours,
            -1,
            (40, 40, 220),
            1,
            lineType=cv2.LINE_AA,
        )
        return draw_region_boundaries(
            debug,
            regions.display_regions,
            color=config.region_color,
            thickness=config.region_thickness,
            partial_face=regions.partial_face,
            nose_style=config.nose_style,
            landmarks=landmarks,
            closed_contour=getattr(regions, "display_contour", None),
            separator_contour=getattr(regions, "display_separator", None),
        )

    def analyze(
        self,
        preprocess_result: PreprocessResultV2,
        *,
        porphyrin_result: PorphyrinResult | None = None,
        spots_result: Any | None = None,
        brown_result: Any | None = None,
    ) -> PurpleAnalysisResult:
        self._validate_preprocess_result(preprocess_result)
        started_at = time.perf_counter()
        image = np.asarray(preprocess_result.analysis_image)
        skin_mask = _binary_mask(preprocess_result.skin_mask)
        landmarks = np.asarray(preprocess_result.landmarks, dtype=np.float32)

        region_image = _safe_region_image(image, skin_mask)
        regions = build_visia_regions(
            region_image,
            skin_mask,
            landmarks,
            preprocess_result.quality_flags,
            include_chin=True,
            mode="full",
            feature_margin_px=self.config.shared_feature_margin_px,
        )
        if np.count_nonzero(regions.analysis_mask) < self.config.minimum_analysis_pixels:
            raise ValueError("共享 VISIA-like analysis_mask 像素过少")

        base_styles = self.base_style_renderer.render(
            image,
            self._foreground_mask(preprocess_result),
            skin_mask,
        )
        uv_like_base = base_styles.uv_spots_base
        moustache_exclusion = build_moustache_feature_exclusion_mask(
            image.shape[:2],
            landmarks,
            regions.analysis_mask,
            margin_px=12,
        )
        instance_exclusion = cv2.bitwise_or(
            _binary_mask(regions.feature_exclusion_mask),
            _binary_mask(moustache_exclusion),
        )

        if self.porphyrin_detector is not None or porphyrin_result is not None:
            # 兼容旧调用方显式传入的检测结果；生产默认不走该分支。
            legacy_detector = self.porphyrin_detector
            if legacy_detector is None:
                legacy_detector = PorphyrinProxyDetector(PorphyrinAnalyzer(), self.config)
            porphyrin_tuple = legacy_detector.detect(
                preprocess_result,
                base_styles.porphyrin_base,
                regions,
                existing_result=porphyrin_result,
                instance_exclusion_mask=instance_exclusion,
            )
            (
                porphyrin_overlay,
                porphyrin_mask,
                porphyrin_score,
                porphyrin_locations,
                porphyrin_distribution,
                _resolved_porphyrin_result,
            ) = porphyrin_tuple
        else:
            fluorescence_result = self.fluorescence_porphyrin_detector.detect(
                base_styles.porphyrin_detection_base,
                regions,
                landmarks,
                instance_exclusion_mask=instance_exclusion,
            )
            porphyrin_overlay = fluorescence_result.overlay
            porphyrin_mask = fluorescence_result.instance_mask
            porphyrin_score = fluorescence_result.score_map
            porphyrin_locations = fluorescence_result.locations
            porphyrin_distribution = fluorescence_result.region_distribution

        uv_spots_result = self.uv_spots_detector.detect(
            base_styles.uv_spots_detection_base,
            regions,
            landmarks,
            display_base=uv_like_base,
            instance_exclusion_mask=instance_exclusion,
        )
        uv_spots_overlay = uv_spots_result.overlay
        uv_spots_mask = uv_spots_result.instance_mask
        uv_spots_score = uv_spots_result.score_map
        uv_spots_instances = uv_spots_result.instances
        uv_spots_distribution = uv_spots_result.region_distribution

        match self.capture_profile:
            case CaptureProfile.INSTITUTION:
                pass
            case CaptureProfile.CONSUMER:
                specular = build_specular_evidence(
                    image,
                    regions.analysis_mask,
                )
                formal_exclusion = cv2.bitwise_or(
                    instance_exclusion,
                    specular.mask,
                )
                uv_projection = project_consumer_markers(
                    ConsumerMarkerInput(
                        uv_spots_mask,
                        uv_spots_score,
                        regions.analysis_mask,
                        formal_exclusion,
                        regions,
                    ),
                    ConsumerPigmentMarkerKind.UV_SPOTS,
                )
                uv_spots_mask = uv_projection.marker_mask
                uv_spots_instances = [
                    {
                        "id": index,
                        "centroid": list(item.centroid),
                        "bbox": list(item.bbox),
                        "area": item.marker_area_px,
                        "support_area": item.source_area_px,
                        "equivalent_diameter": float(2 * item.marker_radius_px),
                        "region": item.region,
                        "mean_score": item.peak_score,
                        "peak_score": item.peak_score,
                        "solidity": 1.0,
                        "aspect_ratio": 1.0,
                        "confidence": item.confidence,
                        "confidence_definition": "确定性工程候选评分，非医学概率",
                    }
                    for index, item in enumerate(uv_projection.findings, 1)
                ]
                uv_spots_distribution = _region_distribution(
                    uv_spots_instances,
                    uv_spots_mask,
                    uv_spots_score,
                    regions,
                )
                uv_spots_overlay = render_compact_marker_mask(
                    uv_like_base,
                    uv_spots_mask,
                    UV_STYLE_MARKER,
                )
                porphyrin_overlay = render_compact_marker_mask(
                    base_styles.porphyrin_base,
                    porphyrin_mask,
                    PORPHYRIN_STYLE_MARKER,
                )
                uv_spots_overlay = draw_region_boundaries(
                    uv_spots_overlay,
                    regions.display_regions,
                    color=self.config.region_color,
                    thickness=self.config.region_thickness,
                    partial_face=regions.partial_face,
                    nose_style=self.config.nose_style,
                    landmarks=landmarks,
                    closed_contour=regions.display_contour,
                    separator_contour=regions.display_separator,
                )
                porphyrin_overlay = draw_region_boundaries(
                    porphyrin_overlay,
                    regions.display_regions,
                    color=self.config.region_color,
                    thickness=self.config.region_thickness,
                    partial_face=regions.partial_face,
                    nose_style=self.config.nose_style,
                    landmarks=landmarks,
                    closed_contour=regions.display_contour,
                    separator_contour=regions.display_separator,
                )
            case unreachable:
                assert_never(unreachable)

        analysis_area = max(int(np.count_nonzero(regions.analysis_mask)), 1)
        result = PurpleAnalysisResult(
            uv_like_base=uv_like_base,
            porphyrin_base=base_styles.porphyrin_base,
            porphyrin_overlay=porphyrin_overlay,
            porphyrin_mask=_binary_mask(porphyrin_mask),
            porphyrin_score_map=np.clip(porphyrin_score, 0.0, 1.0).astype(np.float32),
            porphyrin_locations=porphyrin_locations,
            porphyrin_region_distribution=porphyrin_distribution,
            porphyrin_count=len(porphyrin_locations),
            porphyrin_area_ratio=round(
                np.count_nonzero(porphyrin_mask) / analysis_area,
                8,
            ),
            uv_spots_overlay=uv_spots_overlay,
            uv_spots_mask=_binary_mask(uv_spots_mask),
            uv_spots_score_map=np.clip(uv_spots_score, 0.0, 1.0).astype(np.float32),
            uv_spots_instances=uv_spots_instances,
            uv_spots_region_distribution=uv_spots_distribution,
            uv_spots_count=len(uv_spots_instances),
            uv_spots_area_ratio=round(
                np.count_nonzero(uv_spots_mask) / analysis_area,
                8,
            ),
            analysis_mask=_binary_mask(regions.analysis_mask),
            feature_exclusion_mask=_binary_mask(instance_exclusion),
            region_debug=self._region_debug(image, regions, landmarks, self.config),
            landmarks=np.asarray(landmarks, dtype=np.float32).copy(),
            quality_score=float(preprocess_result.quality_score),
            quality_status=str(preprocess_result.quality_status),
            quality_flags=list(preprocess_result.quality_flags),
            partial_face=bool(regions.partial_face),
            runtime_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        self.last_result = result
        return result

    @staticmethod
    def save_result(
        result: PurpleAnalysisResult,
        output_dir: str,
        base_name: str,
    ) -> dict[str, str]:
        sample_dir = os.path.abspath(os.path.join(output_dir, base_name))
        os.makedirs(sample_dir, exist_ok=True)
        debug_dir = os.path.join(sample_dir, "_debug")
        os.makedirs(debug_dir, exist_ok=True)
        paths = {
            # 云端正式返回四张图：两张底图和各自带特征点的结果图。
            "uv_like_base": os.path.join(
                sample_dir,
                "01_紫外线色斑底图.png",
            ),
            "uv_spots_overlay": os.path.join(
                sample_dir,
                "02_紫外线色斑检测结果.jpg",
            ),
            "porphyrin_base": os.path.join(
                sample_dir,
                "03_紫质荧光底图.png",
            ),
            "porphyrin_overlay": os.path.join(
                sample_dir,
                "04_紫质检测结果.jpg",
            ),
            "metrics_json": os.path.join(sample_dir, "紫区量化指标.json"),
            "metrics_csv": os.path.join(sample_dir, "紫区量化指标.csv"),
            "uv_medical_v2_json": os.path.join(
                sample_dir, "紫外线色斑医学量化指标_V2.json"
            ),
            "uv_medical_v2_csv": os.path.join(
                sample_dir, "紫外线色斑医学量化指标_V2.csv"
            ),
            "porphyrin_medical_v2_json": os.path.join(
                sample_dir, "紫质医学量化指标_V2.json"
            ),
            "porphyrin_medical_v2_csv": os.path.join(
                sample_dir, "紫质医学量化指标_V2.csv"
            ),
            "medical_v2_csv": os.path.join(
                sample_dir, "紫区医学量化指标_V2.csv"
            ),
            "full_metrics_json": os.path.join(
                sample_dir, "紫区完整指标.json"
            ),
            "porphyrin_mask": os.path.join(debug_dir, "01_紫质Mask.png"),
            "porphyrin_score": os.path.join(debug_dir, "02_紫质分数图.png"),
            "uv_spots_mask": os.path.join(debug_dir, "03_UV色斑Mask.png"),
            "uv_spots_score": os.path.join(debug_dir, "04_UV色斑分数图.png"),
            "region_debug": os.path.join(debug_dir, "05_分析区域调试.jpg"),
        }
        cv_imwrite(paths["uv_like_base"], result.uv_like_base)
        cv_imwrite(paths["porphyrin_base"], result.porphyrin_base)
        cv_imwrite(paths["porphyrin_overlay"], result.porphyrin_overlay)
        cv_imwrite(paths["porphyrin_mask"], _binary_mask(result.porphyrin_mask))
        cv_imwrite(
            paths["porphyrin_score"],
            np.rint(np.clip(result.porphyrin_score_map, 0.0, 1.0) * 255.0)
            .astype(np.uint8),
        )
        cv_imwrite(paths["uv_spots_overlay"], result.uv_spots_overlay)
        cv_imwrite(paths["uv_spots_mask"], _binary_mask(result.uv_spots_mask))
        cv_imwrite(
            paths["uv_spots_score"],
            np.rint(np.clip(result.uv_spots_score_map, 0.0, 1.0) * 255.0)
            .astype(np.uint8),
        )
        cv_imwrite(paths["region_debug"], result.region_debug)

        metrics = result.public_metrics()
        with open(paths["metrics_json"], "w", encoding="utf-8") as handle:
            json.dump(
                metrics,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
        user_headers, user_rows = result.user_summary_rows()
        with open(paths["metrics_csv"], "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(user_headers)
            writer.writerows(user_rows)
        uv_payload, porphyrin_payload = result.medical_payloads()
        uv_medical_v2 = try_write_medical_metrics_v2(
            paths["uv_medical_v2_json"],
            paths["uv_medical_v2_csv"],
            uv_payload,
            "purple_uv_spots",
        )
        porphyrin_medical_v2 = try_write_medical_metrics_v2(
            paths["porphyrin_medical_v2_json"],
            paths["porphyrin_medical_v2_csv"],
            porphyrin_payload,
            "purple_porphyrin",
        )
        with open(paths["full_metrics_json"], "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "uv_spots": uv_medical_v2 or uv_payload,
                    "porphyrin": porphyrin_medical_v2 or porphyrin_payload,
                },
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
        if uv_medical_v2 is not None and porphyrin_medical_v2 is not None:
            try:
                combine_documents(
                    (
                        (paths["uv_medical_v2_json"], paths["uv_medical_v2_csv"]),
                        (
                            paths["porphyrin_medical_v2_json"],
                            paths["porphyrin_medical_v2_csv"],
                        ),
                    ),
                    paths["medical_v2_csv"],
                )
            except Exception:
                logger.exception("紫区两项医学 V2 合并失败，保留旧紫区量化")
        # 两个子项目 JSON/CSV 仅用于生成一张医生宽表。正式前端 JSON
        # 永远保持精简，不嵌入医学 V2，也不保留中间医学 JSON。
        for temporary_path in (
            paths["uv_medical_v2_json"],
            paths["porphyrin_medical_v2_json"],
            paths["uv_medical_v2_csv"],
            paths["porphyrin_medical_v2_csv"],
        ):
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        # 不向调用方返回已清理的临时路径；正式紫区只保留一个 JSON、
        # 一张用户简表 CSV 和一张医学 V2 CSV。
        for temporary_key in (
            "uv_medical_v2_json",
            "uv_medical_v2_csv",
            "porphyrin_medical_v2_json",
            "porphyrin_medical_v2_csv",
        ):
            paths.pop(temporary_key, None)
        return paths

    def process_preprocess_result(
        self,
        preprocess_result: PreprocessResultV2,
        output_dir: str,
        source_name: str,
    ) -> str | None:
        if preprocess_result.quality_status == "REJECT":
            return None
        base_name = os.path.splitext(os.path.basename(source_name))[0]
        result = self.analyze(preprocess_result)
        self.last_paths = self.save_result(result, output_dir, base_name)
        return self.last_paths["porphyrin_overlay"]

    def close(self) -> None:
        analyzers = (
            (self.porphyrin_analyzer, self._owns_porphyrin),
            (self.spots_analyzer, self._owns_spots),
            (self.brown_analyzer, self._owns_brown),
        )
        for analyzer, owned in analyzers:
            if not owned:
                continue
            close = getattr(analyzer, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def analyze_purple(
    preprocess_result: PreprocessResultV2,
) -> PurpleAnalysisResult:
    engine = PurpleAnalysisEngine()
    try:
        return engine.analyze(preprocess_result)
    finally:
        engine.close()


__all__ = [
    "PurpleAnalysisConfig",
    "UVLikeBaseRenderer",
    "PorphyrinProxyDetector",
    "UVSpotsProxyDetector",
    "PurpleAnalysisResult",
    "PurpleAnalysisEngine",
    "analyze_purple",
]
