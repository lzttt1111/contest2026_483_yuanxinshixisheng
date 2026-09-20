# -*- coding: utf-8 -*-
"""五项皮肤检测的医学可读量化与报告工具。

本模块只消费检测完成后的 score map、实例和有效皮肤 Mask，不参与候选
生成、过滤或结果图绘制。所有强度都是 0～1 工程归一化值；所有左右均
按图像坐标描述，避免自拍镜像造成解剖学左右误报。
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from src.medical_v2_schema import to_incremental_document


METRIC_VERSION = "medical_metrics_v1"
MEDICAL_V2_VERSION = "medical_metrics_v2_20260728"
DENSITY_SCALE = 100_000.0
UNAVAILABLE = "不可评估"
logger = logging.getLogger(__name__)

# 医生 2026-07-28 V2 口径。这里仅决定报告字段归类，不参与任何检测、
# 阈值、Mask 或结果图计算。未列入核心指标的现有可靠字段进入辅助指标。
V2_CORE_METRICS: dict[str, tuple[str, ...]] = {
    "redness": (
        "弥漫红区面积占比", "高强度区域面积占比", "平均强度（0～1）",
        "P90强度（0～1）", "最大连续区域面积（像素）", "红区连续性（0～1）",
        "红区边界渐变度（0～1）", "弥漫红区均匀度（0～1）",
        "局灶红色实例数量（个）", "局灶红色实例面积占比",
    ),
    "spots": (
        "单位面积密度（个/10万有效皮肤像素）", "特征面积占比",
        "P90综合色差ΔE", "点状斑点面积占比", "片状斑点面积占比",
        "融合样候选面积占比",
    ),
    "brown": (
        "连续棕色色素覆盖占比", "P90强度（0～1）", "高强度区域面积占比",
        "点状目标面积占比", "片状目标面积占比", "棕色色素综合负担",
    ),
    "texture": (
        "单位面积密度（个/10万有效皮肤像素）", "凸起样面积占比",
        "凹陷样面积占比", "P90强度（0～1）", "聚集区域数量（个）",
        "最大聚集区域特征数（个）",
    ),
    "pores": (
        "单位面积密度（个/10万有效皮肤像素）", "特征面积占比",
        "P50单体面积（像素）", "P90单体面积（像素）", "P50圆度（0～1）",
        "拉长样毛孔比例",
    ),
    "purple_uv_spots": (
        "单位面积密度（个/10万有效皮肤像素）", "特征面积占比",
        "P90强度（0～1）", "高强度目标比例", "点状目标面积占比",
        "片状目标面积占比",
    ),
    "purple_porphyrin": (
        "单位面积密度（个/10万有效皮肤像素）", "特征面积占比",
        "实例P50强度（0～1）", "实例P90强度（0～1）", "高强度目标比例",
        "聚集区域数量（个）",
    ),
}

MEDICAL_REGION_ORDER = (
    "forehead", "glabella", "nose", "image_left_nasal", "image_right_nasal",
    "image_left_zygomatic", "image_right_zygomatic", "image_left_cheek",
    "image_right_cheek", "perioral", "image_left_jaw", "image_right_jaw",
    "chin",
)
MEDICAL_REGION_LABELS = {
    "forehead": "额头",
    "glabella": "眉间",
    "nose": "鼻部",
    "image_left_nasal": "画面左鼻旁",
    "image_right_nasal": "画面右鼻旁",
    "image_left_zygomatic": "画面左颧部",
    "image_right_zygomatic": "画面右颧部",
    "image_left_cheek": "画面左面颊",
    "image_right_cheek": "画面右面颊",
    "perioral": "口周",
    "image_left_jaw": "画面左下颌",
    "image_right_jaw": "画面右下颌",
    "chin": "下巴",
}


@dataclass(frozen=True)
class MedicalRegionSet:
    """互斥且覆盖全部有效分析像素的医学报告分区。"""

    analysis_mask: np.ndarray
    regions: dict[str, np.ndarray]
    available: dict[str, bool]


def _finite_float(value: Any, decimals: int = 6) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, decimals)


def json_safe(value: Any) -> Any:
    """递归移除 NumPy 类型、NaN 与 Infinity。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return _finite_float(value)
    return value


def _percentile(values: Iterable[float] | np.ndarray, q: float) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    array = array.astype(np.float64, copy=False)
    array = array[np.isfinite(array)]
    return _finite_float(np.percentile(array, q)) if array.size else 0.0


def _landmark_xy(landmarks: np.ndarray, index: int, fallback: tuple[float, float]) -> tuple[float, float]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim == 2 and points.shape[1] >= 2 and 0 <= index < len(points):
        x, y = points[index, :2]
        if np.isfinite(x) and np.isfinite(y):
            return float(x), float(y)
    return fallback


def build_medical_report_regions(
    analysis_mask: np.ndarray,
    landmarks: np.ndarray,
) -> MedicalRegionSet:
    """建立 13 个仅用于报告的互斥分区，不改变算法检测区域。

    几何边界来自标准化人脸 landmarks，最终始终与真实 analysis_mask 相交。
    每个有效像素只归属一个区域，因此分区面积和实例数可严格回加到全面部。
    """
    mask = (np.asarray(analysis_mask) > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if xs.size < 100:
        raise ValueError("有效皮肤面积过小，无法建立医学报告分区")
    h, w = mask.shape
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    fw, fh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    cx_fallback, cy_fallback = float(np.median(xs)), float(np.median(ys))
    cx, _ = _landmark_xy(landmarks, 1, (cx_fallback, cy_fallback))
    _, brow_l = _landmark_xy(landmarks, 105, (cx - .15 * fw, y0 + .30 * fh))
    _, brow_r = _landmark_xy(landmarks, 334, (cx + .15 * fw, y0 + .30 * fh))
    brow_y = float(np.clip(min(brow_l, brow_r), y0 + .18 * fh, y0 + .46 * fh))
    _, nose_top = _landmark_xy(landmarks, 168, (cx, brow_y + .04 * fh))
    _, nose_bottom = _landmark_xy(landmarks, 2, (cx, y0 + .64 * fh))
    _, mouth_top = _landmark_xy(landmarks, 0, (cx, y0 + .70 * fh))
    _, mouth_bottom = _landmark_xy(landmarks, 17, (cx, y0 + .76 * fh))
    cheek_mid = float(np.clip((nose_bottom + mouth_top) * .5, brow_y, mouth_bottom))

    yy, xx = np.indices((h, w), dtype=np.float32)
    valid = mask > 0
    central = np.abs(xx - cx)
    side_left = xx < cx
    upper = yy < brow_y
    lower = yy > mouth_top
    mid = ~(upper | lower)

    labels = np.full((h, w), -1, dtype=np.int16)
    index = {name: i for i, name in enumerate(MEDICAL_REGION_ORDER)}

    def set_region(name: str, condition: np.ndarray) -> None:
        select = valid & (labels < 0) & condition
        labels[select] = index[name]

    # 优先分配解剖中心区，再分配左右大区，避免鼻旁同时计入面颊。
    set_region("glabella", upper & (central <= .115 * fw) & (yy >= brow_y - .16 * fh))
    set_region("forehead", upper)
    set_region(
        "nose",
        mid & (central <= .105 * fw) & (yy >= nose_top - .02 * fh) & (yy <= nose_bottom + .04 * fh),
    )
    set_region("image_left_nasal", mid & side_left & (central <= .23 * fw))
    set_region("image_right_nasal", mid & ~side_left & (central <= .23 * fw))
    set_region("image_left_zygomatic", mid & side_left & (yy <= cheek_mid))
    set_region("image_right_zygomatic", mid & ~side_left & (yy <= cheek_mid))
    set_region("image_left_cheek", mid & side_left)
    set_region("image_right_cheek", mid & ~side_left)
    # 下巴先于口周分配，避免口周吞掉 mouth_bottom 以下的中央皮肤。
    set_region("chin", lower & (central <= .285 * fw) & (yy > mouth_bottom))
    set_region(
        "perioral",
        lower & (yy <= mouth_bottom) & (central <= .285 * fw),
    )
    set_region("image_left_jaw", lower & side_left)
    set_region("image_right_jaw", lower & ~side_left)

    # 极少数几何缝隙按最近大区归属，确保总量严格守恒。
    remaining = valid & (labels < 0)
    labels[remaining & (yy < brow_y)] = index["forehead"]
    labels[remaining & (yy >= brow_y) & (yy <= mouth_top) & side_left] = index["image_left_cheek"]
    labels[remaining & (yy >= brow_y) & (yy <= mouth_top) & ~side_left] = index["image_right_cheek"]
    labels[remaining & (yy > mouth_top) & side_left] = index["image_left_jaw"]
    labels[remaining & (yy > mouth_top) & ~side_left] = index["image_right_jaw"]

    regions: dict[str, np.ndarray] = {}
    available: dict[str, bool] = {}
    # 标准化 1024 图像中，小于 100 像素的区域无法进行稳定分位统计。
    for name, label in index.items():
        region = (labels == label).astype(np.uint8) * 255
        area = int(np.count_nonzero(region))
        regions[name] = region
        available[name] = area >= 100
    return MedicalRegionSet(mask, regions, available)


def assign_medical_region(centroid: Iterable[float], regions: MedicalRegionSet) -> str:
    values = list(centroid)
    if len(values) < 2:
        return ""
    x = int(np.clip(round(float(values[0])), 0, regions.analysis_mask.shape[1] - 1))
    y = int(np.clip(round(float(values[1])), 0, regions.analysis_mask.shape[0] - 1))
    for name in MEDICAL_REGION_ORDER:
        if regions.regions[name][y, x] > 0:
            return name
    return ""


def _component_summary(mask: np.ndarray) -> tuple[int, int]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
    return len(areas), max(areas, default=0)


def _boundary_gradient(score: np.ndarray, binary_mask: np.ndarray) -> float:
    """边界内外局部响应差，0 表示无明显边界，1 表示陡峭边界。"""
    binary = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if np.count_nonzero(binary) < 8:
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 必须保持 bool dtype。uint8 与 bool 做位运算会产生 uint8 数组；随后
    # values[inside] 会被 NumPy 解释为整数高级索引，而不是布尔 Mask，导致
    # 1024 图像临时扩张成巨型数组。它既让结果失真，也会让单次红区报告
    # 额外耗时十余秒。
    inside = (binary > 0) & (
        cv2.erode(binary, kernel, iterations=1) == 0
    )
    outside = (cv2.dilate(binary, kernel, iterations=1) > 0) & (binary == 0)
    values = np.asarray(score, dtype=np.float32)
    if not np.any(inside) or not np.any(outside):
        return 0.0
    return _finite_float(np.clip(np.mean(values[inside]) - np.mean(values[outside]), 0.0, 1.0))


def _instance_values(instances: list[dict[str, Any]], key: str) -> list[float]:
    return [_finite_float(item.get(key, 0.0), 8) for item in instances]


def _normalize_instances(
    instances: list[dict[str, Any]],
    regions: MedicalRegionSet,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(instances or [], start=1):
        centroid = raw.get("centroid", raw.get("中心坐标", [0, 0]))
        if not isinstance(centroid, (list, tuple, np.ndarray)) or len(centroid) < 2:
            continue
        x, y = float(centroid[0]), float(centroid[1])
        xi = int(np.clip(round(x), 0, regions.analysis_mask.shape[1] - 1))
        yi = int(np.clip(round(y), 0, regions.analysis_mask.shape[0] - 1))
        if regions.analysis_mask[yi, xi] == 0:
            continue
        region = assign_medical_region((x, y), regions)
        item = dict(raw)
        item.update({
            "id": int(raw.get("id", raw.get("spot_id", idx))),
            "centroid": [round(x, 2), round(y, 2)],
            "region": region,
            "area": max(0.0, _finite_float(raw.get("measurement_area", raw.get("area", 1.0)), 4)),
            "intensity": np.clip(_finite_float(raw.get("intensity", raw.get("texture_score", raw.get("pore_score", raw.get("mean_brownness", raw.get("mean_redness", 0.0))))), 6), 0.0, 1.0),
            "confidence": np.clip(_finite_float(raw.get("confidence", 0.0), 6), 0.0, 1.0),
        })
        normalized.append(item)
    return normalized


def _base_row(label: str, available: bool, area: int) -> dict[str, Any]:
    return {
        "检测范围": label,
        "评估状态": "可评估" if available else UNAVAILABLE,
        "有效皮肤面积（像素）": int(area) if available else UNAVAILABLE,
    }


def _density(count: int, area: int) -> float:
    return _finite_float(count * DENSITY_SCALE / max(area, 1), 4)


def _ratio(numerator: float, denominator: float) -> float:
    return _finite_float(float(numerator) / max(float(denominator), 1.0), 6)


def _score_stats(score: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(score, dtype=np.float32)[np.asarray(mask) > 0]
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: 0.0 for key in (
            "平均强度（0～1）", "P50强度（0～1）",
            "P90强度（0～1）", "P95强度（0～1）",
        )}
    return {
        "平均强度（0～1）": _finite_float(np.mean(values)),
        "P50强度（0～1）": _percentile(values, 50),
        "P90强度（0～1）": _percentile(values, 90),
        "P95强度（0～1）": _percentile(values, 95),
    }


def _summarize_instances(instances: list[dict[str, Any]], area: int) -> dict[str, Any]:
    areas = _instance_values(instances, "area")
    strengths = _instance_values(instances, "intensity")
    total_area = _finite_float(sum(areas), 4)
    return {
        "特征数量（个）": len(instances),
        "单位面积密度（个/10万有效皮肤像素）": _density(len(instances), area),
        "特征总面积（像素）": total_area,
        "特征面积占比": _ratio(total_area, area),
        "P50单体面积（像素）": _percentile(areas, 50),
        "P90单体面积（像素）": _percentile(areas, 90),
        "最大单体面积（像素）": _finite_float(max(areas, default=0.0), 4),
        "实例平均强度（0～1）": _finite_float(np.mean(strengths) if strengths else 0.0),
        "实例P50强度（0～1）": _percentile(strengths, 50),
        "实例P90强度（0～1）": _percentile(strengths, 90),
    }


def _side_difference(rows: dict[str, dict[str, Any]], left: str, right: str, metric: str) -> float | str:
    if not rows[left]["available"] or not rows[right]["available"]:
        return UNAVAILABLE
    return _finite_float(abs(float(rows[left]["metrics"].get(metric, 0.0)) - float(rows[right]["metrics"].get(metric, 0.0))), 6)


def build_medical_payload(
    *,
    project: str,
    project_label: str,
    analysis_mask: np.ndarray,
    landmarks: np.ndarray,
    instances: list[dict[str, Any]],
    score_map: np.ndarray,
    instance_mask: np.ndarray,
    quality_control: dict[str, Any],
    limitations: list[str],
    continuous_mask: np.ndarray | None = None,
    high_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """按项目生成医学摘要、互斥分区和最小必要目标明细。"""
    region_set = build_medical_report_regions(analysis_mask, landmarks)
    normalized = _normalize_instances(instances, region_set)
    score = np.clip(np.asarray(score_map, dtype=np.float32), 0.0, 1.0)
    instance_binary = (np.asarray(instance_mask) > 0).astype(np.uint8) * 255
    continuous_binary = (
        (np.asarray(continuous_mask) > 0).astype(np.uint8) * 255
        if continuous_mask is not None else instance_binary
    )
    high_binary = (
        (np.asarray(high_mask) > 0).astype(np.uint8) * 255
        if high_mask is not None else np.zeros_like(instance_binary)
    )

    grouped: dict[str, dict[str, Any]] = {}
    all_area = int(np.count_nonzero(region_set.analysis_mask))
    scopes = [("全面部", region_set.analysis_mask, True, normalized)] + [
        (
            MEDICAL_REGION_LABELS[name], region_set.regions[name], region_set.available[name],
            [item for item in normalized if item["region"] == name],
        )
        for name in MEDICAL_REGION_ORDER
    ]
    output_rows: list[dict[str, Any]] = []
    for position, (label, region_mask, available, local_instances) in enumerate(scopes):
        region_area = int(np.count_nonzero(region_mask))
        row = _base_row(label, available, region_area)
        if not available:
            output_rows.append(row)
            if position:
                grouped[MEDICAL_REGION_ORDER[position - 1]] = {"available": False, "metrics": {}}
            continue
        summary = _summarize_instances(local_instances, region_area)
        score_stats = _score_stats(score, region_mask)
        local_continuous = cv2.bitwise_and(continuous_binary, region_mask)
        local_high = cv2.bitwise_and(high_binary, region_mask)
        continuous_area = int(np.count_nonzero(local_continuous))
        high_area = int(np.count_nonzero(local_high))
        component_count, largest_component = _component_summary(local_continuous)
        common = {
            **summary,
            **score_stats,
            "连续异常面积（像素）": continuous_area,
            "连续异常面积占比": _ratio(continuous_area, region_area),
            "高强度区域面积占比": _ratio(high_area, region_area),
            "最大连续区域面积（像素）": largest_component,
            "连续区域数量（个）": component_count,
        }
        row.update(common)

        if project == "redness":
            diffuse_values = score[local_continuous > 0]
            mad = float(np.median(np.abs(diffuse_values - np.median(diffuse_values)))) if diffuse_values.size else 0.0
            row.update({
                "弥漫红区面积（像素）": continuous_area,
                "弥漫红区面积占比": _ratio(continuous_area, region_area),
                "弥漫红区均匀度（0～1）": _finite_float(1.0 - min(mad / 0.25, 1.0)),
                "红区连续性（0～1）": _ratio(largest_component, continuous_area),
                "红区边界渐变度（0～1）": _boundary_gradient(score, local_continuous),
                "红度综合负担": _finite_float(_ratio(continuous_area, region_area) * (float(np.mean(diffuse_values)) if diffuse_values.size else 0.0)),
                "局灶红色实例数量（个）": len(local_instances),
                "局灶红色实例面积占比": summary["特征面积占比"],
            })
        elif project == "spots":
            delta_e = [_finite_float(item.get("mean_deltaE", item.get("color_difference", 0.0))) for item in local_instances]
            confidences = [_finite_float(item.get("confidence", 0.0)) for item in local_instances]
            point = [item for item in local_instances if float(item["area"]) < 90.0]
            patch = [item for item in local_instances if float(item["area"]) >= 90.0]
            types = [str(item.get("spot_type", "small")) for item in local_instances]
            fused = [item for item in local_instances if str(item.get("spot_type")) == "merged"]
            row.update({
                "平均综合色差ΔE": _finite_float(np.mean(delta_e) if delta_e else 0.0),
                "P50综合色差ΔE": _percentile(delta_e, 50),
                "P90综合色差ΔE": _percentile(delta_e, 90),
                "平均工程置信度（0～1）": _finite_float(np.mean(confidences) if confidences else 0.0),
                "点状斑点数量（个）": len(point),
                "片状斑点数量（个）": len(patch),
                "点状斑点面积占比": _ratio(sum(float(i["area"]) for i in point), region_area),
                "片状斑点面积占比": _ratio(sum(float(i["area"]) for i in patch), region_area),
                "小型候选数量（个）": sum(t == "small" for t in types),
                "大型候选数量（个）": sum(t == "large" for t in types),
                "显著颜色异常数量（个）": sum(t == "salient" for t in types),
                "融合样候选数量（个）": len(fused),
                "融合样候选面积占比": _ratio(sum(float(i["area"]) for i in fused), region_area),
                "边界清晰度中位数（0～1）": _percentile([min(_finite_float(i.get("local_contrast", 0.0)) / 10.0, 1.0) for i in local_instances], 50),
                "颜色均匀性中位数（0～1）": _percentile([i.get("uniformity_score", 0.0) for i in local_instances], 50),
            })
        elif project == "brown":
            point = [item for item in local_instances if float(item["area"]) < 160.0]
            patch = [item for item in local_instances if float(item["area"]) >= 160.0]
            row.update({
                "棕色斑数量（个）": len(local_instances),
                "棕色斑面积占比": summary["特征面积占比"],
                "连续棕色色素覆盖面积（像素）": continuous_area,
                "连续棕色色素覆盖占比": _ratio(continuous_area, region_area),
                "点状棕色目标数量（个）": len(point),
                "片状棕色目标数量（个）": len(patch),
                "点状目标面积占比": _ratio(sum(float(i["area"]) for i in point), region_area),
                "片状目标面积占比": _ratio(sum(float(i["area"]) for i in patch), region_area),
                "棕色色素综合负担": _finite_float(_ratio(continuous_area, region_area) * score_stats["P90强度（0～1）"]),
            })
        elif project == "texture":
            raised = [i for i in local_instances if i.get("texture_type") == "raised_like"]
            depressed = [i for i in local_instances if i.get("texture_type") == "depressed_like"]
            total = max(len(local_instances), 1)
            cluster_count = 0
            max_cluster_size = 0
            if local_instances:
                cluster_canvas = np.zeros(region_mask.shape, dtype=np.uint8)
                for feature_index, feature in enumerate(local_instances, start=1):
                    x, y = (int(round(v)) for v in feature["centroid"])
                    cv2.circle(cluster_canvas, (x, y), 8, 255, -1)
                component_count, cluster_labels = cv2.connectedComponents(
                    cluster_canvas, connectivity=8
                )
                sizes = []
                for label_id in range(1, component_count):
                    sizes.append(sum(
                        cluster_labels[
                            int(round(feature["centroid"][1])),
                            int(round(feature["centroid"][0])),
                        ] == label_id
                        for feature in local_instances
                    ))
                clustered_sizes = [size for size in sizes if size >= 2]
                cluster_count = len(clustered_sizes)
                max_cluster_size = max(clustered_sizes, default=0)
            row.update({
                "凸起样数量（个）": len(raised),
                "凹陷样数量（个）": len(depressed),
                "凸起样比例": _finite_float(len(raised) / total if local_instances else 0.0),
                "凹陷样比例": _finite_float(len(depressed) / total if local_instances else 0.0),
                "凸起样面积占比": _ratio(sum(float(i["area"]) for i in raised), region_area),
                "凹陷样面积占比": _ratio(sum(float(i["area"]) for i in depressed), region_area),
                "主要问题类型": "凸起样" if len(raised) > len(depressed) else ("凹陷样" if depressed else "未检出"),
                "聚集区域数量（个）": cluster_count,
                "最大聚集区域特征数（个）": max_cluster_size,
            })
        elif project == "pores":
            contrasts = [_finite_float(i.get("center_ring_contrast", 0.0)) for i in local_instances]
            circularities = [_finite_float(i.get("measurement_circularity", i.get("circularity", 0.0))) for i in local_instances]
            aspects = [_finite_float(i.get("measurement_aspect_ratio", i.get("aspect_ratio", 1.0))) for i in local_instances]
            diameters = [2.0 * math.sqrt(max(float(i["area"]), 0.0) / math.pi) for i in local_instances]
            row.update({
                "P50等效直径（像素）": _percentile(diameters, 50),
                "P90等效直径（像素）": _percentile(diameters, 90),
                "平均中心—环带视觉对比度（0～1）": _finite_float(np.mean(contrasts) if contrasts else 0.0),
                "P50中心—环带视觉对比度（0～1）": _percentile(contrasts, 50),
                "P90中心—环带视觉对比度（0～1）": _percentile(contrasts, 90),
                "P50圆度（0～1）": _percentile(circularities, 50),
                "P50长宽比": _percentile(aspects, 50),
                "低圆度毛孔比例": _finite_float(sum(v < .5 for v in circularities) / len(circularities) if circularities else 0.0),
                "椭圆形毛孔比例": _finite_float(sum(v >= 1.35 for v in aspects) / len(aspects) if aspects else 0.0),
                "拉长样毛孔比例": _finite_float(sum(v >= 2.0 for v in aspects) / len(aspects) if aspects else 0.0),
            })
        elif project in {"purple_uv_spots", "purple_porphyrin"}:
            point = [item for item in local_instances if float(item["area"]) < 160.0]
            patch = [item for item in local_instances if float(item["area"]) >= 160.0]
            strong = [item for item in local_instances if float(item["intensity"]) >= 0.70]
            dilated = cv2.dilate(
                cv2.bitwise_and(instance_binary, region_mask),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            )
            cluster_count, largest_cluster = _component_summary(dilated)
            row.update({
                "高强度目标比例": _finite_float(len(strong) / len(local_instances) if local_instances else 0.0),
                "点状目标数量（个）": len(point),
                "片状目标数量（个）": len(patch),
                "点状目标面积占比": _ratio(sum(float(i["area"]) for i in point), region_area),
                "片状目标面积占比": _ratio(sum(float(i["area"]) for i in patch), region_area),
                "聚集区域数量（个）": cluster_count,
                "最大聚集区域面积（像素）": largest_cluster,
            })
        output_rows.append(json_safe(row))
        if position:
            grouped[MEDICAL_REGION_ORDER[position - 1]] = {"available": True, "metrics": row}

    overall = dict(output_rows[0])
    # 面颊差异固定使用画面坐标，不写成患者解剖学左右。
    left, right = "image_left_cheek", "image_right_cheek"
    overall["画面左右面颊数量密度差异"] = _side_difference(grouped, left, right, "单位面积密度（个/10万有效皮肤像素）")
    overall["画面左右面颊面积占比差异"] = _side_difference(grouped, left, right, "特征面积占比")
    if project in {"redness", "brown"}:
        overall["画面左右面颊P90强度差异"] = _side_difference(grouped, left, right, "P90强度（0～1）")
    count_by_region = {
        MEDICAL_REGION_LABELS[name]: len([i for i in normalized if i["region"] == name])
        for name in MEDICAL_REGION_ORDER if region_set.available[name]
    }
    overall["主要集中区域"] = max(count_by_region, key=count_by_region.get) if any(count_by_region.values()) else "未检出"
    if project == "redness":
        central_names = {"glabella", "nose", "image_left_nasal", "image_right_nasal"}
        central_count = sum(i["region"] in central_names for i in normalized)
        overall["中央面部实例集中比例"] = _finite_float(central_count / len(normalized) if normalized else 0.0)

    detail: list[dict[str, Any]] = []
    for idx, item in enumerate(normalized, start=1):
        x, y = item["centroid"]
        detail_item = {
            "编号": idx,
            "区域": MEDICAL_REGION_LABELS.get(item["region"], item["region"]),
            "画面侧别": "画面左侧" if x < region_set.analysis_mask.shape[1] / 2 else "画面右侧",
            "中心位置（像素）": [x, y],
            "面积（像素）": item["area"],
            "强度（0～1）": item["intensity"],
            "形态": item.get("morphology", item.get("texture_type_label", item.get("texture_type", item.get("spot_type", "局灶实例")))),
            "工程置信度（0～1）": item["confidence"],
        }
        detail.append(json_safe(detail_item))

    payload = json_safe({
        "检测项目": project_label,
        "指标版本": METRIC_VERSION,
        "成像与单位说明": {
            "成像类型": "标准化普通白光RGB图像",
            "强度单位": "0～1工程归一化值",
            "面积单位": "标准化1024图像平方像素（px²）",
            "密度单位": "个/10万有效皮肤像素",
            "左右定义": "画面左侧/画面右侧，不代表患者解剖学左右",
        },
        "总体指标": overall,
        "分区指标": output_rows[1:],
        "目标明细": detail,
        "质量控制": quality_control,
        "医学局限性": limitations,
    })
    from src.doctor_v3.evidence import attach, enabled
    if enabled():
        return attach(payload, project=project, analysis_mask=analysis_mask,
                      landmarks=landmarks, instances=normalized, score_map=score_map,
                      instance_mask=instance_mask, continuous_mask=continuous_mask,
                      high_mask=high_mask, quality_control=quality_control)
    return payload


def _csv_columns(project: str) -> list[str]:
    common = [
        "检测范围", "评估状态", "有效皮肤面积（像素）", "特征数量（个）",
        "单位面积密度（个/10万有效皮肤像素）", "特征总面积（像素）",
        "特征面积占比", "P50单体面积（像素）", "P90单体面积（像素）",
        "最大单体面积（像素）", "平均强度（0～1）", "P50强度（0～1）",
        "P90强度（0～1）", "P95强度（0～1）",
        "实例P50强度（0～1）", "实例P90强度（0～1）", "主要集中区域",
        "画面左右面颊数量密度差异", "画面左右面颊面积占比差异",
    ]
    extras = {
        "redness": ["弥漫红区面积占比", "高强度区域面积占比", "最大连续区域面积（像素）", "红区连续性（0～1）", "弥漫红区均匀度（0～1）", "红度综合负担", "局灶红色实例数量（个）", "中央面部实例集中比例", "画面左右面颊P90强度差异"],
        "spots": ["平均综合色差ΔE", "P90综合色差ΔE", "平均工程置信度（0～1）", "点状斑点数量（个）", "片状斑点数量（个）", "融合样候选数量（个）", "小型候选数量（个）", "大型候选数量（个）", "显著颜色异常数量（个）", "边界清晰度中位数（0～1）", "颜色均匀性中位数（0～1）"],
        "brown": ["连续棕色色素覆盖占比", "高强度区域面积占比", "最大连续区域面积（像素）", "点状棕色目标数量（个）", "片状棕色目标数量（个）", "棕色色素综合负担", "画面左右面颊P90强度差异"],
        "texture": ["凸起样数量（个）", "凹陷样数量（个）", "凸起样比例", "凹陷样比例", "凸起样面积占比", "凹陷样面积占比", "聚集区域数量（个）", "最大聚集区域特征数（个）", "主要问题类型"],
        "pores": ["P50等效直径（像素）", "P90等效直径（像素）", "P90中心—环带视觉对比度（0～1）", "P50圆度（0～1）", "P50长宽比", "低圆度毛孔比例", "椭圆形毛孔比例", "拉长样毛孔比例"],
    }
    return common + extras[project]


def write_medical_metrics(
    json_path: str | Path,
    csv_path: str | Path,
    payload: dict[str, Any],
    project: str,
) -> None:
    """保存 CSV/JSON 完全同口径的医生摘要。

    逐目标坐标和算法质量控制只在内存中参与复算，不进入正式前端 JSON，
    避免单图产生数千行实例明细。JSON 只保留与 CSV 一一对应的总体行和
    分区行。
    """
    from src.doctor_v3.evidence import save_payload
    save_payload(Path(json_path).parent, payload, project)
    safe = json_safe(payload)
    columns = _csv_columns(project)
    rows = [safe["总体指标"], *safe["分区指标"]]
    compact_rows = [
        {key: row.get(key, "") for key in columns}
        for row in rows
    ]
    compact = {
        "检测项目": safe["检测项目"],
        "指标版本": safe["指标版本"],
        "总体指标": compact_rows[0],
        "分区指标": compact_rows[1:],
    }
    Path(json_path).write_text(
        json.dumps(
            compact,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with Path(csv_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compact_rows)


def _metric_dimension(project: str, name: str) -> str:
    if any(token in name for token in ("面积", "覆盖", "负担", "连续")):
        return "范围与负担"
    if any(token in name for token in ("强度", "红度", "色差", "对比度", "响应")):
        return "信号强度"
    if any(token in name for token in ("密度", "数量", "实例")):
        return "数量与密度"
    if any(token in name for token in ("圆度", "长宽", "点状", "片状", "融合", "形态")):
        return "形态"
    if any(token in name for token in ("左右", "集中", "区域", "聚集")):
        return "分布"
    return "辅助统计"


def _group_v2_metrics(project: str, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    core_names = set(V2_CORE_METRICS[project])
    core: dict[str, dict[str, Any]] = {}
    auxiliary: dict[str, dict[str, Any]] = {
        "分析范围": {
            "检测范围": row.get("检测范围", UNAVAILABLE),
            "评估状态": row.get("评估状态", UNAVAILABLE),
            "有效皮肤面积（像素）": row.get("有效皮肤面积（像素）", UNAVAILABLE),
        }
    }
    for name, value in row.items():
        if name in base or name.startswith("画面左右"):
            continue
        target = core if name in core_names else auxiliary
        target.setdefault(_metric_dimension(project, name), {})[name] = value
    return core, auxiliary


def build_medical_metrics_v2(payload: dict[str, Any], project: str) -> dict[str, Any]:
    """把现有可复算医学摘要转换为 V2；不携带逐实例和内部调试数组。"""
    safe = json_safe(payload)
    overall = dict(safe["总体指标"])
    core, auxiliary = _group_v2_metrics(project, overall)
    region_rows: list[dict[str, Any]] = []
    for row in safe["分区指标"]:
        region_core, region_auxiliary = _group_v2_metrics(project, row)
        region_rows.append({
            "检测范围": row.get("检测范围", UNAVAILABLE),
            "评估状态": row.get("评估状态", UNAVAILABLE),
            "有效皮肤面积（像素）": row.get("有效皮肤面积（像素）", UNAVAILABLE),
            "核心指标": region_core,
            "辅助指标": region_auxiliary,
        })
    left_right = {
        name: value for name, value in overall.items() if name.startswith("画面左右")
    }
    return json_safe({
        "检测项目": safe["检测项目"],
        "指标版本": MEDICAL_V2_VERSION,
        "评分状态": "uncalibrated",
        "成像与单位说明": safe.get("成像与单位说明", {}),
        "总体指标": {"核心指标": core, "辅助指标": auxiliary},
        "分区指标": region_rows,
        "左右比较": left_right,
        "质量控制": safe.get("质量控制", {}),
        "医学局限性": safe.get("医学局限性", []),
    })


def _v2_csv_metric_name(project: str, name: str) -> str:
    if project != "pores":
        return name
    return {
        "特征总面积（像素）": "特征总面积（像素²）",
        "P50单体面积（像素）": "P50单体面积（像素²）",
        "P90单体面积（像素）": "P90单体面积（像素²）",
        "最大单体面积（像素）": "最大单体面积（像素²）",
    }.get(name, name)


def _flatten_metric_groups(
    groups: dict[str, Any],
    prefix: str,
    project: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    base_names = {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    for metrics in groups.values():
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if name not in base_names:
                output[f"{prefix}-{_v2_csv_metric_name(project, name)}"] = value
    return output


def write_medical_metrics_v2(
    json_path: str | Path,
    csv_path: str | Path,
    payload: dict[str, Any],
    project: str,
) -> dict[str, Any]:
    """保存医生 V2 宽表和可逐项还原该宽表的紧凑 JSON。"""
    document = build_medical_metrics_v2(payload, project)
    source_rows = [{
        "检测范围": document["总体指标"]["辅助指标"]["分析范围"]["检测范围"],
        "评估状态": document["总体指标"]["辅助指标"]["分析范围"]["评估状态"],
        "有效皮肤面积（像素）": document["总体指标"]["辅助指标"]["分析范围"]["有效皮肤面积（像素）"],
        **document["总体指标"],
    }, *document["分区指标"]]
    flat_rows: list[dict[str, Any]] = []
    for row in source_rows:
        flat_rows.append({
            "检测范围": row["检测范围"],
            "评估状态": row["评估状态"],
            "有效皮肤面积（像素）": row["有效皮肤面积（像素）"],
            **_flatten_metric_groups(
                row.get("核心指标", {}),
                "核心",
                project,
            ),
            **_flatten_metric_groups(
                row.get("辅助指标", {}),
                "辅助",
                project,
            ),
        })
    columns = list(flat_rows[0])
    for row in flat_rows[1:]:
        for name in row:
            if name not in columns:
                columns.append(name)
    Path(json_path).write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with Path(csv_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    return document


def try_write_medical_metrics_v2(
    json_path: str | Path,
    csv_path: str | Path,
    payload: dict[str, Any],
    project: str,
) -> dict[str, Any] | None:
    """失败降级的旁路写入；V2 异常不得阻断旧算法结果。"""
    try:
        return write_medical_metrics_v2(json_path, csv_path, payload, project)
    except Exception:
        logger.exception("医学 V2 旁路生成失败 project=%s", project)
        for path in (Path(json_path), Path(csv_path)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("清理不完整医学 V2 文件失败: %s", path)
        return None


def embed_medical_metrics_v2(
    canonical_json_path: str | Path,
    document: dict[str, Any] | None,
    *,
    cleanup_json_paths: Iterable[str | Path] = (),
) -> bool:
    """把医学 V2 追加到原量化 JSON，并移除重复 JSON 旁路文件。

    原 JSON 的既有字段和值不做改名或覆盖，只新增
    ``medical_metrics_v2``。Worker 会在返回旧 ``metrics`` 前剥离该
    新节点，并把它作为 ``raw_result`` 的并列可选字段返回。
    """
    canonical_path = Path(canonical_json_path)
    if document is None:
        for path in cleanup_json_paths:
            candidate = Path(path)
            if candidate != canonical_path:
                candidate.unlink(missing_ok=True)
        return False
    try:
        original = json.loads(canonical_path.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise ValueError("原量化 JSON 顶层必须是对象")
        merged = dict(original)
        merged["medical_metrics_v2"] = to_incremental_document(
            json_safe(document)
        )
        canonical_path.write_text(
            json.dumps(merged, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in cleanup_json_paths:
            candidate = Path(path)
            if candidate != canonical_path:
                candidate.unlink(missing_ok=True)
        return True
    except Exception:
        logger.exception("医学 V2 合并到原量化 JSON 失败: %s", canonical_path)
        return False


# 旧调用名保留为兼容保护；新五项引擎应使用 write_medical_metrics。
def write_detailed_metrics(json_path: str | Path, csv_path: str | Path, payload: dict[str, Any]) -> None:
    safe = json_safe(payload)
    Path(json_path).write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with Path(csv_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["指标分类", "指标名称", "指标值"])
        for section, values in safe.items():
            if isinstance(values, dict):
                for name, value in values.items():
                    writer.writerow([section, name, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])


def build_detailed_payload(**kwargs: Any) -> dict[str, Any]:
    """兼容未迁移调用；不再作为五项正式医学报告入口。"""
    return {
        "检测项目": kwargs.get("algorithm_name", "工程量化"),
        "指标版本": "legacy_compatibility",
        "成像与单位说明": {},
        "总体指标": json_safe(kwargs.get("overall", {})),
        "分区指标": json_safe(kwargs.get("regions", {})),
        "目标明细": json_safe(kwargs.get("instances", [])),
        "质量控制": json_safe(kwargs.get("extra", {})),
        "医学局限性": [str(kwargs.get("disclaimer", ""))],
    }


def translate_fields(values: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    return {label: json_safe(values[key]) for key, label in labels.items() if key in values}


def translate_instances(instances: list[dict[str, Any]], labels: dict[str, str]) -> list[dict[str, Any]]:
    return [{label: json_safe(item[key]) for key, label in labels.items() if key in item} for item in instances or []]


def translate_region_statistics(statistics: dict[str, dict[str, Any]], labels: dict[str, str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for region, values in (statistics or {}).items():
        output[str(region)] = {label: json_safe(values[key]) for key, label in labels.items() if key in values}
    return output
