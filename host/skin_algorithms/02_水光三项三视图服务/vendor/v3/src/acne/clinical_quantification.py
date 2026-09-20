from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.acne.medical_v2_schema import to_incremental_document


REPORT_COLUMNS = (
    "检测范围", "评估状态", "有效皮肤面积（像素）", "疑似痤疮候选数量（个）",
    "单位面积密度（个/10万有效皮肤像素）", "候选框总面积（像素）",
    "候选框面积占比", "P50候选框面积（像素）", "P90候选框面积（像素）",
    "平均置信度（0～1）", "P90置信度（0～1）", "Acne-LDS评估状态",
    "Acne-LDS等级", "主要集中区域",
)

REGION_NAMES = {
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
    "large_overexposed_area": "过曝光区域较大",
    "possible_blur": "图像可能模糊",
    "possible_underexposure": "图像可能欠曝光",
    "possible_overexposure": "图像可能过曝光",
    "landmark_quality_not_reliable": "人脸关键点质量不足",
    "empty_skin_mask": "有效皮肤区域为空",
}

MEDICAL_V2_VERSION = "medical_metrics_v2_20260728"
logger = logging.getLogger(__name__)


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


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


def _report_row(
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


def build_report(summary: dict[str, Any]) -> dict[str, Any]:
    detection = summary.get("detection") or {}
    preprocess = summary.get("preprocess") or {}
    grading = summary.get("grading") or {}
    detections = list(detection.get("detections") or [])
    skin_pixels = int(_number(preprocess.get("skin_pixels")))
    grading_text, grading_level = _grading_status(grading)
    region_counts = detection.get("region_counts") or {}
    if region_counts:
        main_key = max(region_counts, key=lambda key: region_counts[key])
        main_region = REGION_NAMES.get(main_key, main_key)
    else:
        main_region = "局部可见区域" if detections else "未检出"
    scope = "全面部" if detection.get("input_mode") == "full_face" else "局部可见区域"
    overall = _report_row(
        scope,
        detections,
        status="可评估" if detection.get("status") == "ok" else "不可评估",
        skin_pixels=skin_pixels,
        grading_status=grading_text,
        grading_level=grading_level,
        main_region=main_region,
    )
    regions: list[dict[str, Any]] = []
    for key, display_name in REGION_NAMES.items():
        if key not in region_counts:
            continue
        selected = [item for item in detections if item.get("region") == key]
        regions.append(_report_row(
            display_name,
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
        "报告列": list(REPORT_COLUMNS),
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


def write_report(summary: dict[str, Any], csv_path: Path, json_path: Path) -> dict[str, Any]:
    report = build_report(summary)
    rows = [report["总体指标"], *report["分区指标"]]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report["报告列"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


V2_CORE_COLUMNS = {
    "单位面积密度（个/10万有效皮肤像素）",
    "候选框面积占比",
    "P50候选框面积（像素）",
    "P90候选框面积（像素）",
    "P90置信度（0～1）",
    "Acne-LDS评估状态",
    "Acne-LDS等级",
}


def _v2_dimension(name: str) -> str:
    if any(token in name for token in ("面积", "长度", "宽度")):
        return "范围与形态"
    if any(token in name for token in ("密度", "数量")):
        return "数量与密度"
    if any(token in name for token in ("置信", "对比", "响应")):
        return "信号强度"
    if "Acne-LDS" in name:
        return "模型分级"
    return "辅助统计"


def _group_v2(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ignored = {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    core: dict[str, dict[str, Any]] = {}
    auxiliary: dict[str, dict[str, Any]] = {
        "分析范围": {
            "检测范围": row.get("检测范围", "不可评估"),
            "评估状态": row.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": row.get("有效皮肤面积（像素）", "不可评估"),
        }
    }
    for name, value in row.items():
        if name in ignored:
            continue
        target = core if name in V2_CORE_COLUMNS else auxiliary
        target.setdefault(_v2_dimension(name), {})[name] = value
    return core, auxiliary


def build_medical_v2_report(report: dict[str, Any]) -> dict[str, Any]:
    core, auxiliary = _group_v2(report["总体指标"])
    regions = []
    for row in report["分区指标"]:
        region_core, region_auxiliary = _group_v2(row)
        regions.append({
            "检测范围": row.get("检测范围", "不可评估"),
            "评估状态": row.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": row.get("有效皮肤面积（像素）", "不可评估"),
            "核心指标": region_core,
            "辅助指标": region_auxiliary,
        })
    return {
        "检测项目": "痤疮",
        "指标版本": MEDICAL_V2_VERSION,
        "评分状态": "uncalibrated",
        "成像与单位说明": {
            "成像类型": "标准化普通白光RGB图像",
            "面积单位": "标准化图像像素",
            "强度说明": "模型工程响应值，不等同于临床诊断概率",
        },
        "总体指标": {"核心指标": core, "辅助指标": auxiliary},
        "分区指标": regions,
        "左右比较": {},
        "质量控制": report.get("质量控制", {}),
        "医学局限性": list(report.get("医学局限性") or []),
    }


def _flatten_groups(groups: dict[str, Any], prefix: str) -> dict[str, Any]:
    ignored = {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    return {
        f"{prefix}-{name}": value
        for values in groups.values() if isinstance(values, dict)
        for name, value in values.items() if name not in ignored
    }


def write_medical_v2_report(
    report: dict[str, Any], csv_path: Path, json_path: Path | None
) -> dict[str, Any]:
    document = build_medical_v2_report(report)
    analysis = document["总体指标"]["辅助指标"]["分析范围"]
    source_rows = [{
        "检测范围": analysis["检测范围"],
        "评估状态": analysis["评估状态"],
        "有效皮肤面积（像素）": analysis["有效皮肤面积（像素）"],
        **document["总体指标"],
    }, *document["分区指标"]]
    rows = [{
        "检测范围": row["检测范围"],
        "评估状态": row["评估状态"],
        "有效皮肤面积（像素）": row["有效皮肤面积（像素）"],
        **_flatten_groups(row.get("核心指标", {}), "核心"),
        **_flatten_groups(row.get("辅助指标", {}), "辅助"),
    } for row in source_rows]
    columns = list(rows[0])
    for row in rows[1:]:
        columns.extend(name for name in row if name not in columns)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if json_path is not None:
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return document


def try_write_medical_v2_report(
    report: dict[str, Any], csv_path: Path, json_path: Path
) -> dict[str, Any] | None:
    try:
        return write_medical_v2_report(report, csv_path, json_path)
    except Exception:
        logger.exception("痤疮医学 V2 旁路生成失败")
        csv_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return None


def try_merge_medical_v2_report(
    report: dict[str, Any], csv_path: Path, canonical_json_path: Path
) -> dict[str, Any] | None:
    """把医学 V2 嵌入原痤疮量化 JSON，原字段只增不改。"""
    try:
        document = write_medical_v2_report(report, csv_path, None)
        original = json.loads(canonical_json_path.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise ValueError("原痤疮量化 JSON 顶层必须是对象")
        merged = dict(original)
        public_document = to_incremental_document(document, "acne")
        merged["medical_metrics_v2"] = public_document
        canonical_json_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return public_document
    except Exception:
        logger.exception("痤疮医学 V2 合并到原量化 JSON 失败")
        csv_path.unlink(missing_ok=True)
        return None
