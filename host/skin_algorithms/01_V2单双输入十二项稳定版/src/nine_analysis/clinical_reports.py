from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ACNE_COLUMNS = (
    "检测范围", "评估状态", "有效皮肤面积（像素）", "疑似痤疮候选数量（个）",
    "单位面积密度（个/10万有效皮肤像素）", "候选框总面积（像素）",
    "候选框面积占比", "P50候选框面积（像素）", "P90候选框面积（像素）",
    "平均置信度（0～1）", "P90置信度（0～1）", "Acne-LDS评估状态",
    "Acne-LDS等级", "主要集中区域",
)

WRINKLE_COLUMNS = (
    "检测范围", "评估状态", "分区面积（像素）", "纹路线段数量（段）",
    "皱纹中心线像素（像素）", "皱纹面积占比",
    "单位面积密度（皱纹像素/1万分区像素）", "平均线段长度（像素）",
    "最大线段长度（像素）", "相对响应（0～100）",
    "占全部皱纹像素比例", "主要集中区域",
)

ACNE_REGION_NAMES = {
    "forehead": "额头",
    "subject_left_cheek": "受检者左面颊",
    "subject_right_cheek": "受检者右面颊",
    "nose": "鼻部",
    "chin": "下巴",
    "subject_left_jaw": "受检者左下颌",
    "subject_right_jaw": "受检者右下颌",
    "unassigned": "未分区可见区域",
}

REASON_NAMES = {
    "input_not_full_face": "非完整人脸，痤疮等级不可评估",
    "resolution_below_recommended": "分辨率低于建议值",
    "large_underexposed_area": "欠曝光区域较大",
}

MEDICAL_V2_VERSION = "medical_metrics_v2_20260728"


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rounded(value: Any, digits: int = 6) -> float:
    return round(_number(value), digits)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _grading_status(grading: dict[str, Any]) -> tuple[str, Any]:
    if grading.get("status") == "ok" and grading.get("severity_level") is not None:
        return "可评估", grading.get("severity_level")
    reason = str(grading.get("reason") or "")
    translated = [REASON_NAMES.get(part, part) for part in reason.split(",") if part]
    return "；".join(translated) or "不可评估", "不可评估"


def _acne_row(
    name: str,
    detections: list[dict[str, Any]],
    *,
    status: str,
    skin_pixels: int | str,
    grading_status: str,
    grading_level: Any,
    main_region: str,
) -> dict[str, Any]:
    areas = [_number(item.get("box_area")) for item in detections]
    confidences = [_number(item.get("confidence")) for item in detections]
    count = len(detections)
    numeric_skin = int(skin_pixels) if isinstance(skin_pixels, int) else 0
    skin_metric_status = (
        "仅全面部统计" if skin_pixels == "仅全面部统计" else "不可评估"
    )
    total_area = sum(areas)
    return {
        "检测范围": name,
        "评估状态": status,
        "有效皮肤面积（像素）": skin_pixels,
        "疑似痤疮候选数量（个）": count,
        "单位面积密度（个/10万有效皮肤像素）": (
            round(count * 100000.0 / numeric_skin, 6)
            if numeric_skin else skin_metric_status
        ),
        "候选框总面积（像素）": round(total_area, 4),
        "候选框面积占比": (
            round(total_area / numeric_skin, 6)
            if numeric_skin else skin_metric_status
        ),
        "P50候选框面积（像素）": round(_percentile(areas, 0.50), 4),
        "P90候选框面积（像素）": round(_percentile(areas, 0.90), 4),
        "平均置信度（0～1）": round(sum(confidences) / count, 6) if count else 0.0,
        "P90置信度（0～1）": round(_percentile(confidences, 0.90), 6),
        "Acne-LDS评估状态": grading_status,
        "Acne-LDS等级": grading_level,
        "主要集中区域": main_region,
    }


def build_acne_report(raw: dict[str, Any]) -> dict[str, Any]:
    detection = raw.get("detection") or {}
    preprocess = raw.get("preprocess") or {}
    grading = raw.get("grading") or {}
    detections = list(detection.get("detections") or [])
    skin_pixels = int(_number(preprocess.get("skin_pixels")))
    grading_text, grading_level = _grading_status(grading)
    region_counts = detection.get("region_counts") or {}
    if region_counts:
        main_key = max(region_counts, key=lambda key: region_counts[key])
        main_region = ACNE_REGION_NAMES.get(main_key, main_key)
    else:
        main_region = "局部可见区域" if detections else "未检出"
    scope = "全面部" if detection.get("input_mode") == "full_face" else "局部可见区域"
    overall = _acne_row(
        scope,
        detections,
        status="可评估" if detection.get("status") == "ok" else "不可评估",
        skin_pixels=skin_pixels,
        grading_status=grading_text,
        grading_level=grading_level,
        main_region=main_region,
    )
    regions: list[dict[str, Any]] = []
    if region_counts:
        ordered_keys = list(ACNE_REGION_NAMES)
        for key in ordered_keys:
            if key not in region_counts:
                continue
            selected = [item for item in detections if item.get("region") == key]
            regions.append(_acne_row(
                ACNE_REGION_NAMES[key],
                selected,
                status="可评估",
                skin_pixels="仅全面部统计",
                grading_status="仅全面部适用",
                grading_level="仅全面部适用",
                main_region="—",
            ))
    return {
        "检测项目": "痤疮",
        "指标版本": "acne_review_metrics_v1",
        "报告列": list(ACNE_COLUMNS),
        "总体指标": overall,
        "分区指标": regions,
        "质量控制": {
            "输入类型": "完整人脸" if detection.get("input_mode") == "full_face" else "局部人脸",
            "检测范围": "全脸分区" if detection.get("detection_scope") == "global" else "局部可见皮肤",
            "人脸数量": int(_number(preprocess.get("face_count"))),
            "分级不可评估原因": None if grading_text == "可评估" else grading_text,
        },
        "医学局限性": [
            "候选数量来自正式痤疮结果图中的模型候选，不等同于临床确诊病灶数量。",
            "候选框面积是二维图像包围框面积，不代表真实皮损面积或体积。",
            "Acne-LDS仅在图像质量和完整人脸条件满足时提供工程分级。",
        ],
    }


def _wrinkle_row(
    name: str,
    *,
    area: float,
    segments: float,
    pixels: float,
    mean_length: float,
    max_length: float,
    score: float,
    share: float,
    main_region: str,
    status: str = "可评估",
) -> dict[str, Any]:
    return {
        "检测范围": name,
        "评估状态": status,
        "分区面积（像素）": int(area),
        "纹路线段数量（段）": int(segments),
        "皱纹中心线像素（像素）": int(pixels),
        "皱纹面积占比": round(pixels / area, 6) if area else 0.0,
        "单位面积密度（皱纹像素/1万分区像素）": round(pixels * 10000.0 / area, 4) if area else 0.0,
        "平均线段长度（像素）": round(mean_length, 4),
        "最大线段长度（像素）": round(max_length, 4),
        "相对响应（0～100）": round(score, 4),
        "占全部皱纹像素比例": round(share, 6),
        "主要集中区域": main_region,
    }


def build_wrinkle_report(raw: dict[str, Any]) -> dict[str, Any]:
    source_regions = list(raw.get("region_metrics") or [])
    total_pixels = sum(_number(item.get("wrinkle_pixels")) for item in source_regions)
    total_area = sum(_number(item.get("area_px")) for item in source_regions)
    total_segments = sum(_number(item.get("segment_count")) for item in source_regions)
    max_length = max((_number(item.get("max_segment_length")) for item in source_regions), default=0.0)
    weighted_score = (
        sum(_number(item.get("relative_score")) * _number(item.get("wrinkle_pixels")) for item in source_regions)
        / total_pixels if total_pixels else 0.0
    )
    main_region = (
        max(source_regions, key=lambda item: _number(item.get("wrinkle_pixels"))).get("region_name")
        if source_regions else "不可评估"
    )
    overall = _wrinkle_row(
        "全面部命名分区",
        area=total_area,
        segments=total_segments,
        pixels=total_pixels,
        mean_length=total_pixels / total_segments if total_segments else 0.0,
        max_length=max_length,
        score=weighted_score,
        share=1.0 if total_pixels else 0.0,
        main_region=main_region,
        status="可评估" if raw.get("region_analysis_status") == "ok" else "不可评估",
    )
    region_mean_lengths = [
        _number(item.get("mean_segment_length")) for item in source_regions
        if _number(item.get("segment_count")) > 0
    ]
    region_scores = [
        _number(item.get("relative_score")) for item in source_regions
        if _number(item.get("segment_count")) > 0
    ]
    recommended_pixels = _number(raw.get("stage2_recommended_pixels"))
    overall.update({
        "总纹路长度（中心线像素）": int(total_pixels),
        "P50线段长度（像素，分区均值代理）": round(_percentile(region_mean_lengths, 0.50), 4),
        "P90线段长度（像素，分区均值代理）": round(_percentile(region_mean_lengths, 0.90), 4),
        "平均可见宽度（像素，面积/中心线代理）": round(recommended_pixels / total_pixels, 4) if total_pixels else 0.0,
        "P90视觉对比度（0～1，响应代理）": round(_percentile(region_scores, 0.90) / 100.0, 6),
        "纹路连续性（0～1）": round(max_length / total_pixels, 6) if total_pixels else 0.0,
    })
    regions = []
    for item in source_regions:
        pixels = _number(item.get("wrinkle_pixels"))
        region_row = _wrinkle_row(
            str(item.get("region_name") or "未命名区域"),
            area=_number(item.get("area_px")),
            segments=_number(item.get("segment_count")),
            pixels=pixels,
            mean_length=_number(item.get("mean_segment_length")),
            max_length=_number(item.get("max_segment_length")),
            score=_number(item.get("relative_score")),
            share=pixels / total_pixels if total_pixels else 0.0,
            main_region="—",
        )
        # 这些量可由现有分区结果直接复算，避免医生宽表出现无含义空白。
        # 长度分位和可见宽度缺少逐线段/逐区面积数据，明确标注适用边界，
        # 不使用均值或最大值伪造分位数。
        region_row.update({
            "总纹路长度（中心线像素）": int(pixels),
            "P50线段长度（像素，分区均值代理）": "仅全面部统计",
            "P90线段长度（像素，分区均值代理）": "仅全面部统计",
            "平均可见宽度（像素，面积/中心线代理）": "仅全面部统计",
            "P90视觉对比度（0～1，响应代理）": round(
                _number(item.get("relative_score")) / 100.0, 6
            ),
            "纹路连续性（0～1）": round(
                _number(item.get("max_segment_length")) / pixels, 6
            ) if pixels else 0.0,
        })
        regions.append(region_row)
    return {
        "检测项目": "皱纹",
        "指标版本": "wrinkle_review_metrics_v1",
        "报告列": list(WRINKLE_COLUMNS),
        "总体指标": overall,
        "分区指标": regions,
        "质量控制": {
            "运行模式": str(raw.get("run_preset") or "未知"),
            "成功推理次数": int(_number(raw.get("successful_runs"))),
            "失败推理次数": int(_number(raw.get("failed_runs"))),
            "人脸过滤状态": str(raw.get("face_filter_status") or "未知"),
            "语义皮肤分割状态": str(raw.get("semantic_skin_status") or "未知"),
        },
        "医学局限性": [
            "结果为普通白光二维图像中的可见纹路代理，不代表皱纹真实深度或皮肤弹性。",
            "中心线像素、线段长度和分区响应仅适用于相同预处理尺度下的工程比较。",
            "分区左右沿用现有皱纹引擎定义。",
        ],
    }


def write_report_csv(report: dict[str, Any], csv_path: Path) -> None:
    columns = list(report["报告列"])
    rows = [report["总体指标"], *report["分区指标"]]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    write_report_csv(report, csv_path)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


V2_CORE_COLUMNS = {
    "acne": (
        "单位面积密度（个/10万有效皮肤像素）", "候选框面积占比",
        "P50候选框面积（像素）", "P90候选框面积（像素）",
        "P90置信度（0～1）", "Acne-LDS评估状态", "Acne-LDS等级",
    ),
    "wrinkle": (
        "单位面积密度（皱纹像素/1万分区像素）", "总纹路长度（中心线像素）",
        "P50线段长度（像素，分区均值代理）", "P90线段长度（像素，分区均值代理）",
        "平均可见宽度（像素，面积/中心线代理）", "皱纹面积占比",
        "P90视觉对比度（0～1，响应代理）", "纹路连续性（0～1）",
    ),
}


def _v2_dimension(name: str) -> str:
    if any(token in name for token in ("面积", "长度", "宽度", "连续")):
        return "范围与形态"
    if any(token in name for token in ("密度", "数量", "线段")):
        return "数量与密度"
    if any(token in name for token in ("置信", "对比", "响应")):
        return "信号强度"
    if "Acne-LDS" in name:
        return "模型分级"
    return "辅助统计"


def _group_v2_row(row: dict[str, Any], project: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ignored = {"检测范围", "评估状态", "有效皮肤面积（像素）", "分区面积（像素）"}
    core_names = set(V2_CORE_COLUMNS[project])
    core: dict[str, dict[str, Any]] = {}
    auxiliary: dict[str, dict[str, Any]] = {
        "分析范围": {
            "检测范围": row.get("检测范围", "不可评估"),
            "评估状态": row.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": row.get(
                "有效皮肤面积（像素）", row.get("分区面积（像素）", "不可评估")
            ),
        }
    }
    for name, value in row.items():
        if name in ignored:
            continue
        target = core if name in core_names else auxiliary
        target.setdefault(_v2_dimension(name), {})[name] = value
    return core, auxiliary


def build_medical_v2_report(report: dict[str, Any], project: str) -> dict[str, Any]:
    overall_core, overall_aux = _group_v2_row(report["总体指标"], project)
    regions = []
    for row in report["分区指标"]:
        core, auxiliary = _group_v2_row(row, project)
        regions.append({
            "检测范围": row.get("检测范围", "不可评估"),
            "评估状态": row.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": row.get(
                "有效皮肤面积（像素）", row.get("分区面积（像素）", "不可评估")
            ),
            "核心指标": core,
            "辅助指标": auxiliary,
        })
    limitation = list(report.get("医学局限性") or [])
    if project == "wrinkle":
        limitation.append("线段长度分位数由现有分区均值汇总，是二维代理，不是真实逐纹路长度分布。")
    return {
        "检测项目": report["检测项目"],
        "指标版本": MEDICAL_V2_VERSION,
        "评分状态": "uncalibrated",
        "成像与单位说明": {
            "成像类型": "标准化普通白光RGB图像",
            "面积单位": "标准化图像像素",
            "强度说明": "工程归一化或模型响应值，不代表真实高度、深度或诊断概率",
        },
        "总体指标": {"核心指标": overall_core, "辅助指标": overall_aux},
        "分区指标": regions,
        "左右比较": {},
        "质量控制": report.get("质量控制", {}),
        "医学局限性": limitation,
    }


def _flatten_v2(groups: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}-{name}": value
        for values in groups.values() if isinstance(values, dict)
        for name, value in values.items()
        if name not in {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    }


def write_medical_v2_report(
    report: dict[str, Any], project: str, csv_path: Path, json_path: Path | None
) -> dict[str, Any]:
    document = build_medical_v2_report(report, project)
    overall = report["总体指标"]
    source_rows = [{
        "检测范围": document["总体指标"]["辅助指标"]["分析范围"]["检测范围"],
        "评估状态": document["总体指标"]["辅助指标"]["分析范围"]["评估状态"],
        "有效皮肤面积（像素）": document["总体指标"]["辅助指标"]["分析范围"]["有效皮肤面积（像素）"],
        **document["总体指标"],
    }, *document["分区指标"]]
    flat_rows = [{
        "检测范围": row["检测范围"],
        "评估状态": row["评估状态"],
        "有效皮肤面积（像素）": row["有效皮肤面积（像素）"],
        **_flatten_v2(row.get("核心指标", {}), "核心"),
        **_flatten_v2(row.get("辅助指标", {}), "辅助"),
    } for row in source_rows]
    columns = list(flat_rows[0])
    for row in flat_rows[1:]:
        columns.extend(name for name in row if name not in columns)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    if json_path is not None:
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return document
