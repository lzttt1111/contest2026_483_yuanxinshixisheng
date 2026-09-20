from __future__ import annotations

"""Map current detector truth into the approved controlled-report table layout."""

from dataclasses import dataclass
from typing import TypeAlias

from .controlled_report_explanations import attach_missing_metric_explanations
from .controlled_report_redness import apply_current_redness as _apply_redness
from .controlled_report_pigmentation import apply_current_pigmentation
from .controlled_report_truth import (
    apply_complete_scoring,
    apply_oil_metric_layout,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
VASCULAR_REGION_LABELS = {
    "forehead": "额部",
    "nose_alar_nasal_side": "鼻翼及鼻侧",
    "left_periocular": "画面左眼周",
    "right_periocular": "画面右眼周",
    "left_zygoma": "画面左颧部",
    "right_zygoma": "画面右颧部",
    "left_cheek": "画面左面颊",
    "right_cheek": "画面右面颊",
    "left_jaw": "画面左下颌",
    "right_jaw": "画面右下颌",
}


def vascular_region_row(
    region_id: str,
    values: JsonObject,
) -> JsonObject:
    """Project vascular regions while keeping the forehead non-assessable."""

    label = VASCULAR_REGION_LABELS.get(region_id, region_id)
    forehead = region_id == "forehead" or label in {"额头", "额部"}
    valid_pixels = _number(
        values.get("valid_pixels", values.get("valid_skin_area_px"))
    )
    assessable = not forehead and valid_pixels > 0
    unavailable: JsonScalar = "不可评估"
    return {
        "分区名称": label,
        "状态": "检测完成" if assessable else (
            "不纳入评估" if forehead else "不可评估"
        ),
        "指标": {
            "目标数量": values.get("count", 0) if assessable else unavailable,
            "总长度（像素）": (
                values.get("total_length_px", 0.0) if assessable else unavailable
            ),
            "长度密度（像素/万有效像素）": (
                values.get("line_density_per_10k_px", 0.0)
                if assessable else unavailable
            ),
            "面积占比": values.get("area_ratio", 0.0) if assessable else unavailable,
        },
    }


@dataclass(frozen=True, slots=True)
class ControlledMetricLayoutError(RuntimeError):
    label: str

    def __str__(self) -> str:
        return f"正式报告指标布局缺失: {self.label}"


def _dictionary(value: JsonValue | None) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _nested(document: JsonObject, *keys: str) -> JsonValue:
    current: JsonValue = document
    for key in keys:
        current = _dictionary(current).get(key)
    return current


def _module(payload: JsonObject, module_id: str) -> JsonObject:
    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        raise ControlledMetricLayoutError("检测模块")
    for value in modules:
        if isinstance(value, dict) and value.get("模块编号") == module_id:
            return value
    raise ControlledMetricLayoutError(module_id)


def _metric(
    name: str,
    value: JsonValue,
    unit: str,
    display_type: str,
) -> JsonObject:
    return {
        "name": name,
        "medical_name": name,
        "value": value,
        "unit": unit,
        "display_type": display_type,
        "availability": "available" if value not in (None, "不可评估") else "unavailable",
    }


def _metrics_by_name(module: JsonObject) -> dict[str, JsonObject]:
    values: dict[str, JsonObject] = {}
    collections: list[JsonValue] = [module.get("核心指标"), module.get("医生结果分组")]
    for collection in collections:
        if not isinstance(collection, list):
            continue
        for row in collection:
            if not isinstance(row, dict):
                continue
            candidates = row.get("metrics") if "metrics" in row else [row]
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
                    values[str(candidate["name"])] = dict(candidate)
    return values


def _first(metrics: dict[str, JsonObject], *names: str) -> JsonObject:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return dict(value)
    return _metric(names[0], None, "", "TEXT")


def _number(value: JsonValue, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _replace_named_metric(
    values: JsonValue,
    names: set[str],
    replacement: JsonObject,
) -> list[JsonValue]:
    rows = list(values) if isinstance(values, list) else []
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("name") in names:
            rows[index] = replacement
            return rows
    rows.append(replacement)
    return rows


def _detector(complete: JsonObject, item_id: str) -> JsonObject:
    detectors = _dictionary(complete.get("detector_results"))
    return _dictionary(_dictionary(detectors.get(item_id)).get("metrics"))


def apply_brown_metric_layout(payload: JsonObject, raw: JsonObject) -> None:
    """Replace frozen Brown counts with the marker truth from this run."""

    module = _module(payload, "03")
    medical = _dictionary(raw.get("medical_metrics_v2"))
    overall = _dictionary(medical.get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    auxiliary = _dictionary(overall.get("auxiliary_metrics"))
    count = int(_number(raw.get("总计")))
    area_ratio = _nested(
        auxiliary,
        "scope_and_burden",
        "brown_spot_area_ratio",
    )
    mean_response = _nested(
        auxiliary,
        "signal_intensity",
        "mean_intensity",
    )
    p90_response = _nested(core, "signal_intensity", "p90_intensity")
    count_metric = _metric("棕色实例数量", count, "个", "COUNT")
    module["结果摘要"] = f"本次检出{count}个Brown综合色素目标。"
    module["核心指标"] = _replace_named_metric(
        module.get("核心指标"),
        {"棕色实例数量", "Brown综合色素目标数量"},
        count_metric,
    )
    groups = module.get("医生结果分组")
    if not isinstance(groups, list):
        raise ControlledMetricLayoutError("Brown综合色素")
    for group in groups:
        if isinstance(group, dict) and "Brown" in str(group.get("title")):
            group["summary"] = module["结果摘要"]
            group["metrics"] = [
                _metric("Brown综合色素目标数量", count, "个", "COUNT"),
                _metric("Brown综合色素面积占比", area_ratio, "%", "PERCENTAGE"),
                _metric("Brown综合色素平均响应", mean_response, "0～1", "INTENSITY_ENGINEERING"),
                _metric("Brown综合色素P90响应", p90_response, "0～1", "INTENSITY_ENGINEERING"),
            ]
            return
    raise ControlledMetricLayoutError("Brown综合色素")


def _apply_pigmentation(payload: JsonObject, complete: JsonObject) -> None:
    module = _module(payload, "03")
    apply_brown_metric_layout(payload, _detector(complete, "brown"))
    metrics = _metrics_by_name(module)
    uv = _dictionary(_detector(complete, "uv_spots").get("uv_spots"))
    total = _dictionary(_nested(uv, "总体指标"))
    core = _dictionary(total.get("核心指标"))
    auxiliary = _dictionary(total.get("辅助指标"))
    uv_count = _nested(auxiliary, "数量与密度", "特征数量（个）")
    uv_area = _nested(core, "范围与负担", "特征面积占比")
    uv_p90 = _nested(core, "信号强度", "P90强度（0～1）")
    module["核心指标"] = [
        _first(metrics, "可见斑点数量"),
        _first(metrics, "棕色实例数量", "Brown综合色素目标数量"),
        _first(metrics, "可见色斑主要集中区域"),
        _metric("UV色斑目标数量", uv_count, "个", "COUNT"),
        _metric("UV色斑面积占比", uv_area, "%", "PERCENTAGE"),
        _metric("P90 UV响应", uv_p90, "0～1", "INTENSITY_ENGINEERING"),
    ]


def apply_vascular_metric_layout(
    payload: JsonObject,
    raw: JsonObject,
) -> None:
    module = _module(payload, "05")
    if isinstance(raw.get("overall_metrics"), dict):
        core = _dictionary(_nested(raw, "overall_metrics", "core_metrics"))
        count = _dictionary(core.get("count_and_density"))
        scope = _dictionary(core.get("scope_and_morphology"))
        signal = _dictionary(core.get("signal_intensity"))
        auxiliary = _dictionary(core.get("auxiliary_statistics"))
        flat = {
            "vascular_count": count.get("vascular_count"),
            "vascular_line_density_per_10k_face_px": count.get(
                "vascular_line_density_per_10k_face_px"
            ),
            "branch_point_count": count.get("branch_point_count"),
            "vascular_total_length_px": scope.get("vascular_total_length_px"),
            "vascular_area_ratio": scope.get("vascular_area_ratio"),
            "p90_width_px": scope.get("p90_vascular_width_px"),
            "max_continuous_network_length_px": scope.get(
                "max_continuous_network_length_px"
            ),
            "p90_redness": signal.get("p90_redness_response"),
            "cp_rgb_visible_support_ratio": auxiliary.get(
                "rgb_visible_support_ratio"
            ),
        }
    else:
        flat = raw
    metrics = [
        _metric("血管样线状结构数量", flat.get("vascular_count"), "个", "COUNT"),
        _metric("总骨架长度", flat.get("vascular_total_length_px"), "像素", "PIXEL"),
        _metric("单位面积长度密度", flat.get("vascular_line_density_per_10k_face_px"), "像素/万有效像素", "DENSITY"),
        _metric("面积占比", flat.get("vascular_area_ratio"), "%", "PERCENTAGE"),
        _metric("P90宽度", flat.get("p90_width_px"), "像素", "PIXEL"),
        _metric("P90红色强度", flat.get("p90_redness"), "工程响应", "INTENSITY_ENGINEERING"),
        _metric("分支点数量", flat.get("branch_point_count"), "个", "COUNT"),
        _metric("最大连续网络长度", flat.get("max_continuous_network_length_px"), "像素", "PIXEL"),
        _metric("白光可见支持率", flat.get("cp_rgb_visible_support_ratio"), "%", "PERCENTAGE"),
    ]
    count_value = flat.get("vascular_count")
    length_value = flat.get("vascular_total_length_px")
    module["结果摘要"] = (
        f"本次检出{count_value if count_value is not None else '不可评估'}个"
        f"血管样线状结构，总长度约"
        f"{length_value if length_value is not None else '不可评估'}像素；"
        "结果用于正面二维结构观察。"
    )
    module["核心指标"] = metrics
    module["医生结果分组"] = [{
        "title": "血管样结构观察结果",
        "summary": module["结果摘要"],
        "metrics": [dict(value) for value in metrics],
    }]
    distribution = _dictionary(raw.get("region_distribution"))
    regions = [
        vascular_region_row(region_id, _dictionary(value))
        for region_id, value in distribution.items()
        if isinstance(value, dict)
    ]
    nested_regions = raw.get("region_metrics")
    if isinstance(nested_regions, list):
        regions = []
        for value in nested_regions:
            if not isinstance(value, dict):
                continue
            valid = _number(value.get("valid_skin_area_px"))
            region_core = _dictionary(value.get("core_metrics"))
            count = _dictionary(region_core.get("count_and_density"))
            scope = _dictionary(region_core.get("scope_and_morphology"))
            length = _number(scope.get("vascular_total_length_px"))
            area = _number(scope.get("vascular_area_px"))
            label = str(value.get("analysis_region", "未知分区"))
            region_id = "forehead" if label in {"额头", "额部"} else label
            regions.append(vascular_region_row(region_id, {
                "valid_skin_area_px": valid,
                "count": int(_number(count.get("vascular_count"))),
                "total_length_px": length,
                "line_density_per_10k_px": (
                    length * 10000.0 / valid if valid else 0.0
                ),
                "area_ratio": area / valid if valid else 0.0,
            }))
    module["分区指标"] = regions
    comparison = _dictionary(raw.get("left_right_comparison"))
    left = comparison.get("left_total_length_px")
    right = comparison.get("right_total_length_px")
    module["左右比较摘要"] = (
        f"画面左侧总长度约{left}像素，右侧总长度约{right}像素。"
        if left is not None and right is not None
        else "当前结果未形成可用的左右比较。"
    )


def _apply_vascular(payload: JsonObject, complete: JsonObject) -> None:
    apply_vascular_metric_layout(payload, _detector(complete, "vascular"))


def _apply_dry_lines(payload: JsonObject, complete: JsonObject) -> None:
    module = _module(payload, "07")
    raw = _detector(complete, "wrinkle")
    rows = raw.get("region_metrics")
    under_eye = [
        row for row in rows
        if isinstance(row, dict)
        and (
            row.get("region_key") in {"left_under_eye", "right_under_eye"}
            or row.get("region_name") in {"左眼下细纹", "右眼下细纹"}
            or row.get("analysis_region") in {"左眼下细纹", "右眼下细纹"}
        )
    ] if isinstance(rows, list) else []
    count = sum(int(row.get("segment_count", 0) or 0) for row in under_eye)
    length = sum(float(row.get("wrinkle_pixels", 0) or 0) for row in under_eye)
    area = sum(float(row.get("area_px", 0) or 0) for row in under_eye)
    means = [float(row.get("mean_segment_length", 0) or 0) for row in under_eye if row.get("segment_count")]
    maximum = max((float(row.get("max_segment_length", 0) or 0) for row in under_eye), default=0.0)
    primary = max(under_eye, key=lambda row: float(row.get("relative_score", 0) or 0), default={})
    metrics = [
        _metric("细纹线段数量", count, "段", "COUNT"),
        _metric("细纹中心线总长度", length, "像素", "PIXEL"),
        _metric("眼下有效皮肤面积", area, "像素", "PIXEL"),
        _metric("眼下单位面积纹路长度密度", length * 10000.0 / area if area else "不可评估", "像素/1万分区像素", "DENSITY"),
        _metric("平均线段长度", sum(means) / len(means) if means else 0.0, "像素", "PIXEL"),
        _metric("最大线段长度", maximum, "像素", "PIXEL"),
        _metric("主要集中区域", primary.get("region_name", "不可评估"), "", "TEXT"),
    ]
    module["核心指标"] = metrics
    module["医生结果分组"] = [{
        "title": "双眼下细纹表现",
        "summary": module.get("结果摘要", ""),
        "metrics": [dict(value) for value in metrics],
    }]


def _apply_texture(payload: JsonObject, complete: JsonObject) -> None:
    module = _module(payload, "10")
    existing_images = {
        str(group.get("title")): list(group.get("images") or [])
        for group in module.get("医生结果分组", []) or []
        if isinstance(group, dict)
    }
    raw = _detector(complete, "texture")
    count = _nested(raw, "medical_metrics_v2", "overall_metrics", "core_metrics", "scope_and_burden", "feature_count")
    area = _nested(raw, "medical_metrics_v2", "overall_metrics", "core_metrics", "scope_and_burden", "feature_area_ratio")
    p90 = _nested(raw, "medical_metrics_v2", "overall_metrics", "core_metrics", "signal_intensity", "p90_intensity")
    base = module.get("核心指标")
    base_metrics = [dict(value) for value in base if isinstance(value, dict)] if isinstance(base, list) else []
    irregular = [
        _metric("表面不规则目标数量", count, "个", "COUNT"),
        _metric("表面不规则面积占比", area, "%", "PERCENTAGE"),
        _metric("P90表面响应", p90, "0～1", "INTENSITY_ENGINEERING"),
    ]
    module["医生结果分组"] = [
        {
            "title": "表面纹理表现",
            "metrics": base_metrics,
            "images": existing_images.get("表面纹理表现", []),
        },
        {
            "title": "表面不规则表现",
            "metrics": irregular,
            "images": existing_images.get("表面不规则表现", []),
        },
    ]


def _apply_contour(payload: JsonObject, complete: JsonObject) -> None:
    module = _module(payload, "11")
    raw = _detector(complete, "contour_firmness")
    summary = _dictionary(raw.get("runtime_summary"))
    candidates = [
        ("下颌线连续比例", raw.get("jaw_continuity_ratio") if raw.get("jaw_continuity_ratio") is not None else summary.get("下颌缘连续性"), "0～1", "ENGINEERING_0_1"),
        ("下颌弧线左右差异", raw.get("jaw_arc_asymmetry_ratio") if raw.get("jaw_arc_asymmetry_ratio") is not None else summary.get("左右轮廓差异"), "比值", "RATIO"),
        ("下脸宽高比", raw.get("jaw_width_to_face_height_ratio"), "比值", "RATIO"),
        ("下颌曲率波动P90", raw.get("jaw_curve_p90"), "0～1", "ENGINEERING_0_1"),
        ("中面部曲面连续比例", raw.get("midface_surface_continuity_ratio") if raw.get("midface_surface_continuity_ratio") is not None else summary.get("中面部曲面连续性"), "0～1", "ENGINEERING_0_1"),
    ]
    metrics = [
        _metric(name, value, unit, display_type)
        for name, value, unit, display_type in candidates
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not metrics:
        module["结果摘要"] = "本次轮廓紧致度缺少可用的相对几何指标，标记为不可评估。"
        module["核心指标"] = []
        module["分区指标"] = []
        module["医生结果分组"] = []
        module["评估状态"] = "不可评估"
        module["用户评估状态"] = "不可评估"
        module["医生评估状态"] = "不可评估"
        return
    values = {str(metric["name"]): metric["value"] for metric in metrics}
    module["结果摘要"] = (
        "已完成当次相对几何量化；"
        + "，".join(f"{name}为{value}" for name, value in values.items())
        + "。"
    )
    module["核心指标"] = metrics
    module["分区指标"] = []
    module["医生结果分组"] = [{
        "title": "面部轮廓几何测量",
        "summary": module.get("结果摘要", ""),
        "metrics": [dict(value) for value in metrics],
    }]


def apply_controlled_metric_layout(
    payload: JsonObject,
    complete: JsonObject,
) -> None:
    """Apply the approved layout without using stale frozen metric values."""

    apply_oil_metric_layout(payload, complete)
    apply_current_pigmentation(payload, complete)
    _apply_redness(payload, complete)
    _apply_vascular(payload, complete)
    _apply_dry_lines(payload, complete)
    _apply_texture(payload, complete)
    _apply_contour(payload, complete)
    apply_complete_scoring(payload, complete)
    attach_missing_metric_explanations(payload)


__all__ = [
    "ControlledMetricLayoutError",
    "apply_brown_metric_layout",
    "apply_complete_scoring",
    "apply_controlled_metric_layout",
    "apply_oil_metric_layout",
    "apply_vascular_metric_layout",
    "vascular_region_row",
]
