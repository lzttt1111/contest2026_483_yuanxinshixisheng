from __future__ import annotations

import math
from typing import Any


COMMON_KEYS = (
    "有效皮肤面积（像素）", "特征数量（个）", "单位面积密度（个/10万有效皮肤像素）",
    "特征总面积（像素）", "特征面积占比", "平均强度（0～1）", "P90强度（0～1）",
    "P95强度（0～1）", "主要集中区域",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _pick(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _numeric(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in keys:
        value = _number(source.get(key))
        if value is not None:
            result[key] = value
    return result


def _flatten_numeric(source: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a report tree without exposing paths, arrays or debug fields."""
    result: dict[str, float] = {}
    if not isinstance(source, dict):
        return result
    for key, value in source.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        number = _number(value)
        if number is not None:
            result[name] = number
            result.setdefault(str(key), number)
        elif isinstance(value, dict):
            result.update(_flatten_numeric(value, name))
    return result


def _first(source: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _present(**values: float | None) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if value is not None}


def _medical_v2_section(raw: dict[str, Any], item: str) -> dict[str, Any] | None:
    document = raw.get("medical_metrics_v2")
    if not isinstance(document, dict) and raw.get("metrics_version") == "medical_metrics_v2_20260728":
        document = raw
    if not isinstance(document, dict):
        return None
    overall = document.get("overall_metrics")
    if item in {"uv_spots", "porphyrin"} and isinstance(overall, dict):
        overall = overall.get(item)
    return overall if isinstance(overall, dict) else None


def _features_from_medical_v2(item: str, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]]] | None:
    section = _medical_v2_section(raw, item)
    if section is None:
        return None
    values = _flatten_numeric(section)
    legacy = _flatten_numeric(raw)
    # The medical node intentionally contains only additive metrics. Merge the
    # frozen legacy metrics for scoring, without changing either public JSON.
    values = {**legacy, **values}
    valid_area = _first(values, "valid_skin_area_px", "有效皮肤面积（像素）")
    max_continuous_area = _first(values, "max_continuous_region_area_px")
    if valid_area and max_continuous_area is not None:
        values.setdefault("max_continuous_region_area_ratio", max_continuous_area / valid_area)
    summary = {key: value for key, value in values.items() if "." not in key}

    if item == "redness":
        groups = {
            "泛红覆盖负担": _present(
                diffuse_red_area_ratio=_first(values, "diffuse_red_area_ratio", "red_area_ratio"),
                high_red_area_ratio=_first(values, "high_red_area_ratio", "high_intensity_area_ratio"),
            ),
            "泛红强度": _present(
                mean_redness=_first(values, "mean_redness", "mean_redness_in_red_area"),
                p90_redness=_first(values, "p90_redness", "redness_p90"),
            ),
            "背景红色连续性": _present(
                max_continuous_region_ratio=_first(values, "max_continuous_region_ratio", "max_continuous_region_area_ratio"),
                redness_continuity=_first(values, "redness_continuity"),
            ),
            "边界与均匀性": _present(
                boundary_gradient=_first(values, "redness_boundary_gradient", "boundary_gradient"),
                uniformity=_first(values, "diffuse_redness_uniformity", "redness_uniformity"),
            ),
        }
    elif item == "spots":
        groups = {
            "可见色斑范围": _present(
                density=_first(values, "feature_density_per_100k_skin_px"),
                area_ratio=_first(values, "spot_area_ratio", "feature_area_ratio"),
            ),
            "可见色斑对比": _present(p90_delta_e=_first(values, "p90_delta_e")),
            "可见色斑形态": _present(
                punctate_area_ratio=_first(values, "punctate_spot_area_ratio"),
                patch_area_ratio=_first(values, "patch_spot_area_ratio"),
                confluent_area_ratio=_first(values, "confluent_candidate_area_ratio"),
            ),
        }
    elif item == "brown":
        groups = {
            "Brown覆盖负担": _present(
                area_ratio=_first(values, "brown_spot_area_ratio", "feature_area_ratio"),
                continuous_coverage_ratio=_first(values, "continuous_brown_coverage_ratio"),
            ),
            "Brown强度": _present(p90_intensity=_first(values, "p90_intensity")),
            "高强度Brown区域": _present(high_intensity_area_ratio=_first(values, "high_intensity_area_ratio")),
            "Brown点片形态": _present(
                punctate_area_ratio=_first(values, "punctate_target_area_ratio"),
                patch_area_ratio=_first(values, "patch_target_area_ratio"),
            ),
        }
    elif item == "texture":
        groups = {
            "凸起样二维纹理": _present(
                density=_first(values, "feature_density_per_100k_skin_px"),
                area_ratio=_first(values, "raised_like_area_ratio"),
                p90_response=_first(values, "p90_intensity"),
            ),
            "凹陷样二维纹理": _present(area_ratio=_first(values, "depressed_like_area_ratio")),
            "二维纹理聚集": _present(
                cluster_count=_first(values, "cluster_region_count"),
                max_cluster_features=_first(values, "max_cluster_feature_count"),
            ),
        }
    elif item == "pores":
        groups = {
            "毛孔密度负担": _present(density=_first(values, "feature_density_per_100k_skin_px")),
            "毛孔面积负担": _present(area_ratio=_first(values, "feature_area_ratio")),
            "毛孔尺度负担": _present(
                p50_area=_first(values, "p50_instance_area_px"),
                p90_area=_first(values, "p90_instance_area_px"),
            ),
            "毛孔形态负担": _present(
                low_circularity_ratio=_first(values, "low_circularity_pore_ratio"),
                elongated_ratio=_first(values, "elongated_pore_ratio"),
            ),
        }
    elif item == "uv_spots":
        groups = {
            "UV样色素范围": _present(
                density=_first(values, "feature_density_per_100k_skin_px", "紫外线色斑单位面积密度（个/10万有效皮肤像素）"),
                area_ratio=_first(values, "continuous_abnormal_area_ratio", "punctate_target_area_ratio", "紫外线色斑特征面积占比"),
            ),
            "UV样色素强度": _present(
                p90_intensity=_first(values, "p90_intensity", "紫外线色斑P90强度（0～1）"),
                high_intensity_ratio=_first(values, "high_intensity_target_ratio"),
            ),
            "UV样点片形态": _present(
                punctate_area_ratio=_first(values, "punctate_target_area_ratio"),
                patch_area_ratio=_first(values, "patch_target_area_ratio"),
            ),
        }
    elif item == "porphyrin":
        groups = {
            "紫质目标范围": _present(
                density=_first(values, "feature_density_per_100k_skin_px", "紫质单位面积密度（个/10万有效皮肤像素）"),
                area_ratio=_first(values, "continuous_abnormal_area_ratio", "punctate_target_area_ratio", "紫质特征面积占比"),
            ),
            "紫质荧光代理强度": _present(
                p90_intensity=_first(values, "p90_intensity", "紫质P90强度（0～1）"),
                high_intensity_ratio=_first(values, "high_intensity_target_ratio"),
            ),
            "紫质聚集": _present(cluster_count=_first(values, "cluster_region_count")),
        }
    elif item == "acne":
        groups = {
            "候选范围负担": _present(
                density=_first(values, "feature_density_per_100k_skin_px"),
                area_ratio=_first(values, "candidate_box_area_ratio"),
                p90_area=_first(values, "p90_candidate_box_area_px"),
            ),
            "候选信号强度": _present(p90_confidence=_first(values, "p90_confidence")),
        }
    elif item == "wrinkle":
        groups = {
            "纹路密度": _present(
                density=_first(values, "wrinkle_pixel_density_per_10k_region_px", "density_per_10k_region_px", "单位面积密度（皱纹像素/1万分区像素）"),
            ),
            "纹路长度": _present(
                total_length=_first(values, "total_wrinkle_length_px"),
                p90_length=_first(values, "p90_segment_length_px_proxy"),
            ),
            "纹路宽度与面积": _present(
                width=_first(values, "mean_visible_width_px_proxy"),
                area_ratio=_first(values, "wrinkle_area_ratio", "皱纹面积占比"),
            ),
            "纹路视觉对比": _present(contrast=_first(values, "p90_visual_contrast_proxy")),
            "纹路线性连续性": _present(continuity=_first(values, "wrinkle_continuity")),
        }
    else:
        return None
    return summary, groups


def _derma_features(item: str, overall: dict[str, Any]) -> dict[str, dict[str, float]]:
    definitions = {
        "redness": {
            "红色面积及连续性": ("弥漫红区面积占比", "高强度区域面积占比", "红区连续性（0～1）"),
            "红色强度": ("平均强度（0～1）", "P90强度（0～1）", "P95强度（0～1）", "红度综合负担"),
            "边界渐变及背景均匀性": ("弥漫红区均匀度（0～1）",),
            "面部分布及左右差异": ("中央面部实例集中比例", "画面左右面颊面积占比差异", "画面左右面颊P90强度差异"),
        },
        "spots": {
            "色斑总面积及面积占比": ("特征面积占比", "P90单体面积（像素）"),
            "颜色明显程度": ("平均综合色差ΔE", "P90综合色差ΔE"),
            "色斑数量和密度": ("单位面积密度（个/10万有效皮肤像素）",),
            "点状片状及融合形态": ("片状斑点数量（个）", "融合样候选数量（个）"),
            "分布范围": ("画面左右面颊面积占比差异",),
        },
        "brown": {
            "Brown色素总面积及面积占比": ("连续棕色色素覆盖占比", "特征面积占比"),
            "Brown信号强度": ("平均强度（0～1）", "P90强度（0～1）", "棕色色素综合负担"),
            "高强度Brown区域": ("高强度区域面积占比",),
            "点状及片状形态": ("片状棕色目标数量（个）",),
            "分布范围": ("画面左右面颊面积占比差异",),
        },
        "texture": {
            "隆起样及凹陷样面积占比": ("凸起样面积占比", "凹陷样面积占比"),
            "特征密度": ("单位面积密度（个/10万有效皮肤像素）",),
            "二维响应强度": ("平均强度（0～1）", "P90强度（0～1）"),
            "聚集及分布": ("聚集区域数量（个）", "最大聚集区域特征数（个）"),
        },
        "pores": {
            "毛孔密度": ("单位面积密度（个/10万有效皮肤像素）",),
            "毛孔面积占比": ("特征面积占比",),
            "毛孔大小": ("P50等效直径（像素）", "P90等效直径（像素）"),
            "毛孔形态不规则度": ("低圆度毛孔比例", "拉长样毛孔比例"),
        },
    }
    return {group: _numeric(overall, keys) for group, keys in definitions[item].items()}


def _purple(item: str, overall: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    prefix = "紫外线色斑" if item == "uv_spots" else "紫质"
    keys = (
        "有效皮肤面积（像素）", f"{prefix}特征数量（个）",
        f"{prefix}单位面积密度（个/10万有效皮肤像素）", f"{prefix}特征总面积（像素）",
        f"{prefix}特征面积占比", f"{prefix}P50单体面积（像素）", f"{prefix}P90单体面积（像素）",
        f"{prefix}平均强度（0～1）", f"{prefix}P90强度（0～1）", f"{prefix}P95强度（0～1）",
        f"{prefix}实例P90强度（0～1）", f"{prefix}主要集中区域",
    )
    if item == "uv_spots":
        definitions = {
            "UV下更明显色素面积": (f"{prefix}特征面积占比",),
            "UV信号强度": (f"{prefix}平均强度（0～1）", f"{prefix}P90强度（0～1）", f"{prefix}P95强度（0～1）"),
            "数量和密度": (f"{prefix}单位面积密度（个/10万有效皮肤像素）",),
            "高强度UV区域": (f"{prefix}实例P90强度（0～1）",),
            "分布范围": (f"{prefix}画面左右面颊面积占比差异",),
        }
    else:
        definitions = {
            "卟啉目标密度": (f"{prefix}单位面积密度（个/10万有效皮肤像素）",),
            "卟啉面积占比": (f"{prefix}特征面积占比",),
            "卟啉荧光强度": (f"{prefix}平均强度（0～1）", f"{prefix}P90强度（0～1）", f"{prefix}P95强度（0～1）"),
            "高强度卟啉负担": (f"{prefix}实例P90强度（0～1）",),
            "毛囊定位表现": (f"{prefix}P50单体面积（像素）",),
        }
    return _pick(overall, *keys), {group: _numeric(overall, names) for group, names in definitions.items()}


def _wrinkle(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    regions = raw.get("region_metrics") or []
    segment_count = sum(float(x.get("segment_count") or 0) for x in regions)
    wrinkle_pixels = sum(float(x.get("wrinkle_pixels") or 0) for x in regions)
    area = sum(float(x.get("area_px") or 0) for x in regions)
    weighted = sum(float(x.get("relative_score") or 0) * float(x.get("wrinkle_pixels") or 0) for x in regions)
    weighted = weighted / wrinkle_pixels if wrinkle_pixels else 0.0
    max_length = max((float(x.get("max_segment_length") or 0) for x in regions), default=0.0)
    named_count = sum(float(x.get("segment_count") or 0) for x in regions if x.get("region_key") != "other")
    density = wrinkle_pixels / area if area else 0.0
    summary = {
        "纹路线段数量（段）": int(segment_count), "命名分区纹路线段数量（段）": int(named_count),
        "皱纹中心线像素（像素）": int(wrinkle_pixels), "分区面积（像素）": int(area),
        "皱纹像素面积占比": density, "最大线段长度（像素）": max_length,
        "按皱纹像素加权相对响应（0～100）": weighted,
        "建议皱纹区域占比": raw.get("stage2_recommended_ratio"),
        "主要分区": regions[0].get("region_name") if regions else "不可评估",
    }
    features = {
        "常见部位匹配": {"命名分区线段数量": named_count},
        "线性形态": {"最大线段长度": max_length, "中心线面积占比": density},
        "宽度和视觉强度": {"加权相对响应": weighted},
    }
    return summary, features


def _acne(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    detection = raw.get("detection") or {}
    detections = detection.get("detections") or []
    skin = _number((raw.get("preprocess") or {}).get("skin_pixels")) or 0.0
    count = float(detection.get("count") or 0)
    density = count * 100000.0 / skin if skin else 0.0
    confidences = sorted(float(x.get("confidence") or 0) for x in detections)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    p90_conf = confidences[min(len(confidences) - 1, int(0.9 * len(confidences)))] if confidences else 0.0
    grading = raw.get("grading") or {}
    summary = {
        "候选数量（个）": int(count), "单位面积密度（个/10万皮肤像素）": density,
        "平均置信度（0～1）": mean_conf, "P90置信度（0～1）": p90_conf,
        "检测范围": detection.get("detection_scope"), "Acne-LDS状态": grading.get("status"),
        "Acne-LDS等级": grading.get("severity_level"), "Acne-LDS等级概率": grading.get("severity_probabilities") or [],
    }
    features = {
        "模型候选数量密度及面积": {"候选密度": density},
        "候选置信度及聚集": {"平均置信度": mean_conf, "P90置信度": p90_conf},
    }
    severity = _number(grading.get("severity_level"))
    if severity is not None:
        features["Acne-LDS等级概率"] = {"预测等级": severity}
    return summary, features


def extract_item(item: str, raw: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    if not isinstance(raw, dict):
        return {}, {}
    medical = _features_from_medical_v2(item, raw)
    if medical is not None:
        return medical
    if raw.get("指标版本") in {"acne_review_metrics_v1", "wrinkle_review_metrics_v1"}:
        overall = raw.get("总体指标") or {}
        if item == "acne":
            return _acne_review(overall)
        if item == "wrinkle":
            return _wrinkle_review(overall)
        return overall, {}
    if item in {"redness", "spots", "brown", "texture", "pores"}:
        overall = raw.get("总体指标") or {}
        return _pick(overall, *COMMON_KEYS), _derma_features(item, overall)
    if item in {"uv_spots", "porphyrin"}:
        return _purple(item, raw.get("总体指标") or {})
    if item == "wrinkle":
        return _wrinkle(raw)
    if item == "acne":
        return _acne(raw)
    if item in {"surface_gloss", "vascular", "contour_firmness"}:
        metrics = _flatten_numeric(raw)
        summary = {key: value for key, value in metrics.items() if "." not in key}
        return summary, {"工程量化指标": summary}
    raise KeyError(item)


def _acne_review(overall: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    values = _flatten_numeric(overall)
    return overall, {
        "候选范围负担": _present(
            density=_first(values, "单位面积密度（个/10万有效皮肤像素）"),
            area_ratio=_first(values, "候选框面积占比"),
            p90_area=_first(values, "P90候选框面积（像素）"),
        ),
        "候选信号强度": _present(p90_confidence=_first(values, "P90置信度（0～1）")),
    }


def _wrinkle_review(overall: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    values = _flatten_numeric(overall)
    return overall, {
        "纹路密度": _present(density=_first(values, "单位面积密度（皱纹像素/1万分区像素）")),
        "纹路长度": _present(
            total_length=_first(values, "总纹路长度（中心线像素）", "皱纹中心线像素（像素）"),
            p90_length=_first(values, "P90线段长度（像素，分区均值代理）"),
        ),
        "纹路宽度与面积": _present(
            width=_first(values, "平均可见宽度（像素，面积/中心线代理）"),
            area_ratio=_first(values, "皱纹面积占比"),
        ),
        "纹路视觉对比": _present(contrast=_first(values, "P90视觉对比度（0～1，响应代理）")),
        "纹路线性连续性": _present(continuity=_first(values, "纹路连续性（0～1）")),
    }
