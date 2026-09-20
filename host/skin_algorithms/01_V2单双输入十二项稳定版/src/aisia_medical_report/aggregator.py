from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import adapt_acne, adapt_dermavision, adapt_wrinkle
from .models import Metric, ModuleResult, sanitize_json
from src.wrinkle.report_region_overlays import region_name_in_group


MODULE_ORDER = [
    "pores",
    "oil",
    "pigmentation",
    "redness",
    "vascular",
    "acne",
    "dry_lines",
    "wrinkle",
    "grooves",
    "texture",
    "contour",
]


def _missing(module_id: str, title: str, reason: str, sources: list[str] | None = None) -> ModuleResult:
    return ModuleResult(
        module_id=module_id,
        title=title,
        status="本次未检测/待扩展",
        sources=sources or [],
        summary=reason,
        daily_note="该模块需相应成像条件或专用算法支持，本次报告不使用其他指标替代。",
        limitations=[reason],
    )


def build_report_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    sources = bundle.get("结果来源", {})
    if not isinstance(sources, dict):
        raise ValueError("结果来源必须为JSON对象")

    derma_modules, derma_raw = adapt_dermavision(sources.get("dermavision_result_dir", ""))
    redness_raw = derma_raw.get("redness") if isinstance(derma_raw, dict) else None
    acne = adapt_acne(sources.get("acne_summary_json"), redness_raw)
    wrinkle, grooves = adapt_wrinkle(sources.get("wrinkle_summary_json"))

    modules: dict[str, ModuleResult] = {
        **derma_modules,
        "oil": _oil_module(derma_raw),
        "vascular": _vascular_module(derma_modules.get("redness")),
        "acne": acne,
        "dry_lines": _dry_lines_module(wrinkle),
        "wrinkle": _stable_lines_module(wrinkle),
        "grooves": grooves,
        "contour": _contour_module(grooves, derma_modules.get("texture")),
    }

    ordered = [modules[name] for name in MODULE_ORDER]
    available = [m for m in ordered if m.status not in {"本次未检测", "本次未检测/待扩展", "检测失败/不可评估"}]
    unavailable = [m for m in ordered if m not in available]

    report_info = dict(bundle.get("报告信息", {})) if isinstance(bundle.get("报告信息"), dict) else {}
    report_id = str(report_info.get("报告编号") or f"AISIA-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    report_info.update(
        {
            "报告编号": report_id,
            "生成时间": str(report_info.get("生成时间") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "综合得分": None,
            "程度等级": None,
        }
    )

    task_status = bundle.get("任务状态", {}) if isinstance(bundle.get("任务状态"), dict) else {}
    quality = dict(bundle.get("采集与质量信息", {})) if isinstance(bundle.get("采集与质量信息"), dict) else {}
    quality.setdefault("左右说明", "画面左侧/画面右侧；未获得镜像方向时不表述为解剖学左右")
    quality.setdefault("成像说明", "普通白光RGB及各算法标准化图像；不等同于VISIA、UV或3D直接测量")
    quality.setdefault("任务状态", task_status)

    cross_summary = _cross_module_summary(modules)
    limitations = [
        "0～1强度为标准化图像指标，用于同条件下的面部特征比较。",
        "不同设备、曝光、白平衡、拍摄距离、面部表情、清洁状态和产品残留会影响结果。",
    ]

    payload = {
        "报告信息": report_info,
        "受检者信息": dict(bundle.get("受检者信息", {})) if isinstance(bundle.get("受检者信息"), dict) else {},
        "采集与质量信息": quality,
        "总体摘要": cross_summary,
        "检测模块": [m.to_dict() for m in ordered],
        "未检测模块": [
            {"模块编号": m.module_id, "模块名称": m.title, "状态": m.status, "原因": m.summary}
            for m in unavailable
        ],
        "医学局限性": limitations,
        "指标版本": "medical_report_v2",
        "模板版本": "AISIA_report_v2",
        "可用模块数量": len(available),
        "模块总数": len(ordered),
    }
    scoring_v011 = bundle.get("评分V011")
    if isinstance(scoring_v011, dict):
        _apply_v011_scoring(payload, scoring_v011)
    return sanitize_json(payload)


def _purple_overall(raw: dict[str, Any]) -> dict[str, Any]:
    purple = raw.get("purple")
    if not isinstance(purple, dict):
        return {}
    overall = purple.get("总体指标") or purple.get("overall_metrics") or {}
    if isinstance(overall, dict) and isinstance(overall.get("core_metrics"), dict):
        merged: dict[str, Any] = {}
        for section in (overall.get("core_metrics"), overall.get("auxiliary_metrics")):
            if isinstance(section, dict):
                for group in section.values():
                    if isinstance(group, dict):
                        merged.update(group)
        return merged
    return overall if isinstance(overall, dict) else {}


def _oil_module(raw: dict[str, Any]) -> ModuleResult:
    total = _purple_overall(raw)
    root = Path(str(raw.get("result_root") or "."))
    count = total.get("特征数量（个）", total.get("紫质特征数量（个）", total.get("porphyrin_total", 0)))
    density = total.get("单位面积密度（个/10万有效皮肤像素）", total.get("紫质单位面积密度（个/10万有效皮肤像素）", 0))
    ratio = total.get("特征面积占比", total.get("紫质特征面积占比", 0))
    intensity = total.get("实例P90强度（0～1）", total.get("P90强度（0～1）", total.get("紫质实例P90强度（0～1）", total.get("紫质P90强度（0～1）", 0))))
    return ModuleResult(
        module_id="02", title="油脂分泌倾向", status="可评估",
        sources=["紫质可见信号"],
        summary=f"检测到 {count} 个紫质特征，结合密度和可见强度展示面部油脂相关分布。",
        metrics=[Metric("紫质数量", count, "个"), Metric("紫质密度", density, "个/10万有效皮肤像素"), Metric("紫质面积占比", ratio, "比例"), Metric("紫质P90强度", intensity, "0～1")],
        images=[str(root / "七项检测/紫区/04_紫质检测结果.jpg")],
        daily_note="关注鼻部、额头及面颊的清洁与清爽度，复测时保持相同洁面状态。",
    )


def _vascular_module(redness: ModuleResult | None) -> ModuleResult:
    if redness is None:
        return _missing("05", "血管样结构", "缺少红区可见信号。")
    selected = [metric for metric in redness.metrics if any(key in metric.name for key in ("红度", "局灶", "连续"))][:6]
    return ModuleResult(
        module_id="05", title="血管样结构", status=redness.status,
        sources=["红区可见红度与局灶结构"],
        summary="展示面部可见红色结构的范围、强度和分区分布。",
        metrics=selected, regions=redness.regions, images=redness.images[:1],
        daily_note="建议在稳定室温和光线下复测，对比可见红色结构的分布变化。",
    )


def _region_number(region: Any, key: str) -> float:
    value = region.metrics.get(key, 0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _wrinkle_projection(
    wrinkle: ModuleResult,
    module_id: str,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Aggregate one chapter from exactly the regions drawn in its overlay."""

    regions = [
        region for region in wrinkle.regions
        if region_name_in_group(module_id, region.name)
    ]
    count = sum(int(_region_number(region, "纹路线段数量（段）")) for region in regions)
    length = sum(_region_number(region, "纹路中心线像素数") for region in regions)
    valid_area = sum(_region_number(region, "分区有效皮肤面积（像素）") for region in regions)
    if not regions or valid_area <= 0:
        return regions, None
    affected_area = sum(
        _region_number(region, "纹路面积占比")
        * _region_number(region, "分区有效皮肤面积（像素）")
        for region in regions
    )
    density = length * 10000.0 / valid_area if valid_area > 0 else 0.0
    area_ratio = affected_area / valid_area if valid_area > 0 else 0.0
    primary = max(
        regions,
        key=lambda region: (
            _region_number(region, "相对响应（0～100）"),
            _region_number(region, "纹路中心线像素数"),
            region.name,
        ),
        default=None,
    )
    weighted_continuity = (
        sum(
            _region_number(region, "纹路连续性（0～1）")
            * _region_number(region, "纹路中心线像素数")
            for region in regions
        ) / length
        if length > 0 else 0.0
    )
    return regions, {
        "count": count,
        "length": round(length, 3),
        "valid_area": int(round(valid_area)),
        "density": round(density, 6),
        "area_ratio": round(area_ratio, 6),
        "mean_length": round(length / count, 6) if count > 0 else 0.0,
        "max_length": round(max((_region_number(region, "最大线段长度（像素）") for region in regions), default=0.0), 3),
        "continuity": round(weighted_continuity, 6),
        "p90_contrast": round(max((_region_number(region, "视觉对比度（0～1）") for region in regions), default=0.0), 6),
        "primary": primary.name if primary is not None else "未检出",
    }


def _missing_wrinkle_projection(
    module_id: str,
    title: str,
    source: str,
) -> ModuleResult:
    reason = (
        f"缺少{source}的可识别分区或有效皮肤分母，本次不可评估；"
        "该状态不得解释为未检出。"
    )
    return ModuleResult(
        module_id=module_id,
        title=title,
        status="检测失败/不可评估",
        sources=[source],
        summary=reason,
        metrics=[],
        regions=[],
        images=[],
        daily_note="",
        limitations=[reason],
        evidence_status="UNAVAILABLE",
        module_status="NOT_ASSESSABLE",
        report_eligibility="NOT_REPORTABLE",
        missing_evidence=["WRINKLE_REGION_SUPPORT_OR_DENOMINATOR_UNAVAILABLE"],
        is_2d_supplement=True,
    )


def _dry_lines_module(wrinkle: ModuleResult) -> ModuleResult:
    regions, values = _wrinkle_projection(wrinkle, "07")
    if values is None:
        return _missing_wrinkle_projection("07", "干燥性细纹", "双眼下细纹分区")
    return ModuleResult(
        module_id="07", title="干燥性细纹", status=wrinkle.status,
        sources=["双眼下细纹分区"],
        summary=(
            f"双眼下分区共检测到 {values['count']} 段二维可见细纹，"
            f"中心线总长度为 {values['length']} 像素，主要集中于 {values['primary']}。"
        ),
        metrics=[
            Metric("细纹线段数量", values["count"], "段"),
            Metric("细纹中心线总长度", values["length"], "像素"),
            Metric("眼下有效皮肤面积", values["valid_area"], "像素"),
            Metric("眼下单位面积纹路长度密度", values["density"], "像素/1万分区像素"),
            Metric("平均线段长度", values["mean_length"], "像素"),
            Metric("最大线段长度", values["max_length"], "像素"),
            Metric("主要集中区域", values["primary"]),
        ],
        regions=regions, images=wrinkle.images[:1],
        daily_note="建议在相同表情、补光和清洁状态下复测，观察双眼下二维可见细纹变化。",
        limitations=[*wrinkle.limitations, "本章节仅统计双眼下分区，不包含鱼尾纹、额头纹或沟纹。"],
    )


def _stable_lines_module(wrinkle: ModuleResult) -> ModuleResult:
    regions, values = _wrinkle_projection(wrinkle, "08")
    if values is None:
        return _missing_wrinkle_projection(
            "08",
            "稳定线性皱纹",
            "额头、眉间及双侧鱼尾纹分区",
        )
    return ModuleResult(
        module_id="08", title="稳定线性皱纹", status=wrinkle.status,
        sources=["额头、眉间及双侧鱼尾纹分区"],
        summary=(
            f"额头、眉间和鱼尾纹分区共获得 {values['count']} 个二维可见纹路线段，"
            f"中心线总长度为 {values['length']} 像素，主要集中于 {values['primary']}。"
        ),
        metrics=[
            Metric("纹路线段数量", values["count"], "段"),
            Metric("中心线总长度", values["length"], "像素"),
            Metric("稳定区域有效皮肤面积", values["valid_area"], "像素"),
            Metric("皱纹面积占比", values["area_ratio"], "比例"),
            Metric("单位面积纹路长度密度", values["density"], "像素/1万分区像素"),
            Metric("平均线段长度", values["mean_length"], "像素"),
            Metric("最大线段长度", values["max_length"], "像素"),
            Metric("纹路连续性", values["continuity"], "0～1"),
            Metric("分区P90视觉对比度最大值", values["p90_contrast"], "0～1工程代理"),
            Metric("主要集中区域", values["primary"]),
        ],
        regions=regions, images=wrinkle.images[:1],
        daily_note="建议在相同表情、角度和光线下复测额头、眉间及鱼尾纹二维外观。",
        limitations=[*wrinkle.limitations, "本章节不包含眼下细纹、法令纹或木偶纹。"],
    )


def _contour_module(grooves: ModuleResult, texture: ModuleResult | None) -> ModuleResult:
    metrics = list(grooves.metrics[:3])
    if texture is not None:
        metrics.extend(texture.metrics[:2])
    return ModuleResult(
        module_id="11", title="面部轮廓紧致度", status="可评估",
        sources=["结构性沟纹与二维表面起伏"],
        summary="结合法令纹、木偶纹和表面起伏展示面部轮廓的二维可见特征。",
        metrics=metrics, regions=grooves.regions, images=grooves.images[:1],
        daily_note="建议保持相同角度、表情和拍摄距离进行周期复测。",
    )


def _apply_v011_scoring(payload: dict[str, Any], scoring: dict[str, Any]) -> None:
    """只把四项完整评分写入正式基线；七项不补零、不生成伪分。"""
    module_by_dimension = {
        "visible_pores": "01",
        "combined_pigmentation": "03",
        "diffuse_redness": "04",
        "surface_smoothness_decline": "10",
    }
    scores = {
        row.get("dimension_id"): row
        for row in scoring.get("formal_dimension_scores", [])
        if isinstance(row, dict)
    }
    for module in payload.get("检测模块", []):
        dimension_id = next(
            (key for key, module_id in module_by_dimension.items() if module_id == module.get("模块编号")),
            None,
        )
        row = scores.get(dimension_id) if dimension_id else None
        if row and row.get("status") == "formal" and isinstance(row.get("score"), (int, float)):
            module["综合得分"] = round(float(row["score"]), 2)
            module["程度等级"] = _burden_grade(float(row["score"]))
            module["评分追溯"] = row.get("groups", [])
        else:
            module["综合得分"] = None
            module["程度等级"] = None
    info = payload["报告信息"]
    info["评分模式"] = "v0.1.1_integrity"
    info["评分配置版本"] = scoring.get("scoring_profile_version")
    info["综合得分"] = None
    info["程度等级"] = None
    payload["评分追溯哈希"] = scoring.get("trace_sha256")
    payload["质量门禁"] = scoring.get("quality_gate")


def _burden_grade(score: float) -> str:
    if score <= 20:
        return "未见明显"
    if score <= 40:
        return "轻度"
    if score <= 60:
        return "中度"
    if score <= 80:
        return "较明显"
    return "显著"


def _cross_module_summary(modules: dict[str, ModuleResult]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for key in ("pores", "pigmentation", "redness", "acne", "wrinkle", "grooves", "texture"):
        module = modules.get(key)
        if not module or module.status in {"本次未检测", "本次未检测/待扩展", "检测失败/不可评估"}:
            continue
        output.append(
            {
                "模块": module.title,
                "状态": module.status,
                "重点": module.summary,
            }
        )
    if not output:
        output.append({"模块": "总体", "状态": "不可评估", "重点": "未收到可用于生成报告的同图算法结果。"})
    return output
