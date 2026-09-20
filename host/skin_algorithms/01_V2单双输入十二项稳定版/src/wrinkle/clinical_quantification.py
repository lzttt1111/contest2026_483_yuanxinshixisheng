from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.wrinkle.medical_v2_schema import to_incremental_document


REPORT_COLUMNS = (
    "检测范围", "评估状态", "分区面积（像素）", "纹路线段数量（段）",
    "皱纹中心线像素（像素）", "皱纹面积占比",
    "单位面积密度（皱纹像素/1万分区像素）", "平均线段长度（像素）",
    "最大线段长度（像素）", "相对响应（0～100）",
    "占全部皱纹像素比例", "主要集中区域",
)

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


def _report_row(
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


def build_report(summary: dict[str, Any]) -> dict[str, Any]:
    source_regions = list(summary.get("region_metrics") or [])
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
    overall = _report_row(
        "全面部命名分区",
        area=total_area,
        segments=total_segments,
        pixels=total_pixels,
        mean_length=total_pixels / total_segments if total_segments else 0.0,
        max_length=max_length,
        score=weighted_score,
        share=1.0 if total_pixels else 0.0,
        main_region=str(main_region),
        status="可评估" if summary.get("region_analysis_status") == "ok" else "不可评估",
    )
    region_mean_lengths = [
        _number(item.get("mean_segment_length")) for item in source_regions
        if _number(item.get("segment_count")) > 0
    ]
    region_scores = [
        _number(item.get("relative_score")) for item in source_regions
        if _number(item.get("segment_count")) > 0
    ]
    recommended_pixels = _number(summary.get("stage2_recommended_pixels"))
    overall.update({
        "总纹路长度（中心线像素）": int(total_pixels),
        "P50线段长度（像素，分区均值代理）": round(
            _percentile(region_mean_lengths, 0.50), 4
        ),
        "P90线段长度（像素，分区均值代理）": round(
            _percentile(region_mean_lengths, 0.90), 4
        ),
        "平均可见宽度（像素，面积/中心线代理）": round(
            recommended_pixels / total_pixels, 4
        ) if total_pixels else 0.0,
        "P90视觉对比度（0～1，响应代理）": round(
            _percentile(region_scores, 0.90) / 100.0, 6
        ),
        "纹路连续性（0～1）": round(
            max_length / total_pixels, 6
        ) if total_pixels else 0.0,
    })
    regions = []
    for item in source_regions:
        pixels = _number(item.get("wrinkle_pixels"))
        region_row = _report_row(
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
        "报告列": list(REPORT_COLUMNS),
        "总体指标": overall,
        "分区指标": regions,
        "质量控制": {
            "运行模式": str(summary.get("run_preset") or "未知"),
            "成功推理次数": int(_number(summary.get("successful_runs"))),
            "失败推理次数": int(_number(summary.get("failed_runs"))),
            "人脸过滤状态": str(summary.get("face_filter_status") or "未知"),
            "语义皮肤分割状态": str(summary.get("semantic_skin_status") or "未知"),
        },
        "医学局限性": [
            "结果为普通白光二维图像中的可见纹路代理，不代表皱纹真实深度或皮肤弹性。",
            "中心线像素、线段长度和分区响应仅适用于相同预处理尺度下的工程比较。",
            "分区左右沿用现有皱纹引擎定义。",
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
    "总纹路长度（中心线像素）",
    "皱纹面积占比",
    "单位面积密度（皱纹像素/1万分区像素）",
    "P50线段长度（像素，分区均值代理）",
    "P90线段长度（像素，分区均值代理）",
    "平均可见宽度（像素，面积/中心线代理）",
    "P90视觉对比度（0～1，响应代理）",
    "纹路连续性（0～1）",
}


def _v2_dimension(name: str) -> str:
    if any(token in name for token in ("面积", "长度", "宽度", "连续")):
        return "范围与形态"
    if any(token in name for token in ("密度", "数量", "线段", "像素")):
        return "数量与密度"
    if any(token in name for token in ("响应", "对比")):
        return "信号强度"
    return "辅助统计"


def _group_v2(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ignored = {"检测范围", "评估状态", "分区面积（像素）"}
    core: dict[str, dict[str, Any]] = {}
    auxiliary: dict[str, dict[str, Any]] = {
        "分析范围": {
            "检测范围": row.get("检测范围", "不可评估"),
            "评估状态": row.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": row.get("分区面积（像素）", "不可评估"),
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
            "有效皮肤面积（像素）": row.get("分区面积（像素）", "不可评估"),
            "核心指标": region_core,
            "辅助指标": region_auxiliary,
        })
    limitations = list(report.get("医学局限性") or [])
    limitations.append("当前长度分位与可见宽度仍待逐纹路级稳定统计，不输出真实深度。")
    return {
        "检测项目": "皱纹",
        "指标版本": MEDICAL_V2_VERSION,
        "评分状态": "uncalibrated",
        "成像与单位说明": {
            "成像类型": "标准化普通白光RGB图像",
            "面积单位": "标准化图像像素",
            "强度说明": "二维可见纹路响应，不代表真实皱纹深度或皮肤弹性",
        },
        "总体指标": {"核心指标": core, "辅助指标": auxiliary},
        "分区指标": regions,
        "左右比较": {},
        "质量控制": report.get("质量控制", {}),
        "医学局限性": limitations,
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
        logger.exception("皱纹医学 V2 旁路生成失败")
        csv_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return None


def try_merge_medical_v2_report(
    report: dict[str, Any], csv_path: Path, canonical_json_path: Path
) -> dict[str, Any] | None:
    """把医学 V2 嵌入原皱纹量化 JSON，旧字段和值保持不变。"""
    try:
        document = write_medical_v2_report(report, csv_path, None)
        original = json.loads(canonical_json_path.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise ValueError("原皱纹量化 JSON 顶层必须是对象")
        merged = dict(original)
        public_document = to_incremental_document(document, "wrinkle")
        merged["medical_metrics_v2"] = public_document
        canonical_json_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return public_document
    except Exception:
        logger.exception("皱纹医学 V2 合并到原量化 JSON 失败")
        csv_path.unlink(missing_ok=True)
        return None
