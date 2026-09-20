from __future__ import annotations

"""Bind formal report oil/porphyrin values and scores to current truth."""

from dataclasses import dataclass
from typing import Any

from src.aisia_medical_report.word_scoring import (
    apply_word_scores,
    load_word_acne_2d_reference,
    load_word_population_profile,
    load_word_wrinkle_2d_references,
)


GROUP_LABELS = {
    "density_burden": "密度负担",
    "area_coverage_burden": "面积覆盖负担",
    "size_burden": "尺度负担",
    "large_pore_burden": "偏大毛孔负担",
    "shape_irregularity": "形态不规则度",
    "visible_spots": "可见色斑",
    "uv_spots": "UV色斑",
    "brown_pigment": "Brown综合色素",
    "coverage": "泛红覆盖",
    "intensity": "泛红强度",
    "continuity": "泛红连续性",
    "boundary_uniformity": "边界与均匀性",
}


@dataclass(frozen=True, slots=True)
class ControlledReportTruthError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        current = _dictionary(current).get(key)
    return current


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _module(payload: dict[str, Any], module_id: str) -> dict[str, Any]:
    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        raise ControlledReportTruthError("正式报告缺少检测模块")
    for module in modules:
        if isinstance(module, dict) and module.get("模块编号") == module_id:
            return module
    raise ControlledReportTruthError(f"正式报告缺少模块: {module_id}")


def _metric(name: str, value: Any, unit: str, display_type: str) -> dict[str, Any]:
    return {
        "name": name,
        "medical_name": name,
        "value": value,
        "unit": unit,
        "display_type": display_type,
        "availability": "available" if value not in (None, "不可评估") else "unavailable",
    }


def _detector(complete: dict[str, Any], item_id: str) -> dict[str, Any]:
    return _dictionary(
        _dictionary(_dictionary(complete.get("detector_results")).get(item_id)).get("metrics")
    )


def _gloss_values(raw: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doctor = _dictionary(raw.get("doctor_core_inputs"))
    if doctor:
        coverage = _dictionary(doctor.get("surface_gloss_coverage"))
        intensity = _dictionary(doctor.get("surface_gloss_intensity"))
        continuity = _dictionary(doctor.get("surface_gloss_continuity"))
        full = _dictionary(raw.get("full_face"))
        values = {
            **coverage,
            **intensity,
            **continuity,
            "gloss_patch_count": full.get("patch_count"),
        }
        regions = [
            {
                "name": str(row.get("label", key)),
                "status": str(row.get("status", "VALID")),
                "ratio": row.get("gloss_area_ratio"),
                "p90": row.get("p90_gloss_intensity"),
                "count": row.get("patch_count"),
            }
            for key, value in _dictionary(raw.get("region_metrics")).items()
            if (row := _dictionary(value))
        ]
        return values, regions
    overall = _dictionary(raw.get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    values = {
        **_dictionary(core.get("scope_and_morphology")),
        **_dictionary(core.get("signal_intensity")),
        **_dictionary(core.get("count_and_density")),
    }
    regions = []
    for value in raw.get("region_metrics", []) if isinstance(raw.get("region_metrics"), list) else []:
        row = _dictionary(value)
        region_core = _dictionary(row.get("core_metrics"))
        count = _nested(region_core, "count_and_density", "gloss_patch_count")
        p90 = _nested(region_core, "signal_intensity", "p90_gloss_intensity")
        if _number(count) == 0:
            p90 = "不可评估"
        regions.append({
            "name": str(row.get("analysis_region", "未知分区")),
            "status": str(row.get("evaluation_status", "ASSESSABLE")),
            "ratio": _nested(region_core, "scope_and_morphology", "gloss_area_ratio"),
            "p90": p90,
            "count": count,
        })
    return values, regions


def _porphyrin_values(raw: dict[str, Any]) -> dict[str, Any]:
    chinese = _dictionary(raw.get("porphyrin"))
    if chinese:
        total = _dictionary(chinese.get("总体指标"))
        core = _dictionary(total.get("核心指标"))
        auxiliary = _dictionary(total.get("辅助指标"))
        return {
            "count": _nested(auxiliary, "数量与密度", "特征数量（个）"),
            "density": _nested(core, "范围与负担", "单位面积密度（个/10万有效皮肤像素）"),
            "area_ratio": _nested(core, "范围与负担", "特征面积占比"),
            "p50": _nested(core, "信号强度", "实例P50强度（0～1）"),
            "p90": _nested(core, "信号强度", "实例P90强度（0～1）"),
        }
    overall = _dictionary(raw.get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    auxiliary = _dictionary(overall.get("auxiliary_metrics"))
    count = _nested(core, "count_and_density", "feature_count")
    valid = _number(_nested(auxiliary, "analysis_scope", "valid_skin_area_px"))
    density = count * 100000.0 / valid if _number(count) is not None and valid else None
    return {
        "count": count,
        "density": density,
        "area_ratio": _nested(core, "scope_and_morphology", "feature_area_ratio"),
        "p50": _nested(core, "signal_intensity", "p50_intensity"),
        "p90": _nested(core, "signal_intensity", "p90_intensity"),
    }


def _existing_images(module: dict[str, Any], token: str) -> list[Any]:
    for group in module.get("医生结果分组", []) or []:
        if isinstance(group, dict) and token in str(group.get("title")):
            images = group.get("images")
            return list(images) if isinstance(images, list) else []
    return []


def apply_oil_metric_layout(payload: dict[str, Any], complete: dict[str, Any]) -> None:
    module = _module(payload, "02")
    gloss, regions = _gloss_values(_detector(complete, "surface_gloss"))
    porphyrin = _porphyrin_values(_detector(complete, "porphyrin"))
    gloss_metrics = [
        _metric("表面油光面积占比", gloss.get("gloss_area_ratio"), "%", "PERCENTAGE"),
        _metric("高强度油光面积占比", gloss.get("high_gloss_area_ratio"), "%", "PERCENTAGE"),
        _metric("P50油光强度", gloss.get("p50_gloss_intensity"), "0～1", "INTENSITY_ENGINEERING"),
        _metric("P90油光强度", gloss.get("p90_gloss_intensity"), "0～1", "INTENSITY_ENGINEERING"),
        _metric("最大连续油光区域面积占比", gloss.get("largest_gloss_component_area_ratio"), "%", "PERCENTAGE"),
        _metric("油光区域数量", gloss.get("gloss_patch_count"), "个", "COUNT"),
    ]
    porphyrin_metrics = [
        _metric("紫质目标数量", porphyrin.get("count"), "个", "COUNT"),
        _metric("紫质单位面积密度", porphyrin.get("density"), "个/10万有效皮肤像素", "DENSITY"),
        _metric("紫质面积占比", porphyrin.get("area_ratio"), "%", "PERCENTAGE"),
        _metric("紫质P50强度", porphyrin.get("p50"), "0～1", "INTENSITY_ENGINEERING"),
        _metric("紫质P90强度", porphyrin.get("p90"), "0～1", "INTENSITY_ENGINEERING"),
    ]
    count = porphyrin.get("count")
    module["结果摘要"] = (
        f"表面油光面积占比为{float(gloss.get('gloss_area_ratio') or 0) * 100:.2f}%，"
        f"检测到紫质目标{count if count is not None else '不可评估'}个。"
    )
    module["核心指标"] = [*gloss_metrics, *porphyrin_metrics]
    module["医生结果分组"] = [
        {"title": "表面油光表现", "summary": module["结果摘要"], "metrics": gloss_metrics, "images": _existing_images(module, "油光")},
        {"title": "紫质表现", "summary": module["结果摘要"], "metrics": porphyrin_metrics, "images": _existing_images(module, "荧光") or _existing_images(module, "紫质")},
    ]
    module["分区指标"] = [
        {
            "分区名称": row["name"],
            "状态": "检测完成" if row["status"] in {"VALID", "ASSESSABLE"} else row["status"],
            "指标": {
                "油光面积占比": row["ratio"],
                "P90油光强度": row["p90"],
                "油光区域数量": row["count"],
            },
        }
        for row in regions
    ]


def apply_complete_scoring(payload: dict[str, Any], complete: dict[str, Any]) -> None:
    population_profile = (
        load_word_population_profile(
            capture_profile=str(
                (complete.get("provenance") or {}).get("capture_profile")
                or "consumer"
            )
        )
        if isinstance(complete.get("scoring_features"), dict)
        else None
    )
    wrinkle_references = (
        load_word_wrinkle_2d_references()
        if population_profile is not None
        else None
    )
    acne_reference = (
        load_word_acne_2d_reference()
        if population_profile is not None
        else None
    )
    apply_word_scores(
        payload,
        complete,
        population_profile,
        wrinkle_references,
        acne_reference,
    )


__all__ = ["apply_complete_scoring", "apply_oil_metric_layout"]
