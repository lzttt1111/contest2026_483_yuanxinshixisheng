from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from ..io_utils import existing_files, load_json
from ..models import Metric, ModuleResult, RegionResult
from src.wrinkle.report_region_overlays import region_name_in_group


GROOVE_KEYS = {"left_nasolabial", "right_nasolabial", "left_marionette", "right_marionette"}


def _image(summary_path: Path, raw: dict[str, Any], relative: str | None) -> str | None:
    if not relative:
        return None
    output_dir = Path(str(raw.get("output_dir", summary_path.parent))).expanduser()
    path = Path(relative).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    return str(path.resolve()) if path.is_file() else None


def adapt_wrinkle(summary_json: str | Path | None) -> tuple[ModuleResult, ModuleResult]:
    if not summary_json:
        return _missing("08", "稳定线性皱纹", "本次报告未收到同一受检者的皱纹检测结果。"), _missing(
            "09", "结构性沟纹", "本次报告未收到同一受检者的沟纹检测结果。"
        )
    path = Path(summary_json).expanduser().resolve()
    if not path.is_file():
        reason = f"皱纹结果文件不存在: {path}"
        return _missing("08", "稳定线性皱纹", reason, failed=True), _missing("09", "结构性沟纹", reason, failed=True)

    raw = load_json(path)
    if isinstance(raw.get("overall_metrics"), dict):
        return _adapt_medical_v2(path, raw)
    if isinstance(raw.get("总体指标"), dict):
        return _adapt_review_metrics(path, raw)
    rows = raw.get("region_metrics", []) if isinstance(raw.get("region_metrics"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    all_regions = [_region(row) for row in rows]
    total_segments = sum(int(row.get("segment_count", 0) or 0) for row in rows)
    total_length = sum(float(row.get("wrinkle_pixels", 0) or 0) for row in rows)
    lengths = [float(row.get("mean_segment_length", 0) or 0) for row in rows if row.get("segment_count", 0)]
    max_length = max((float(row.get("max_segment_length", 0) or 0) for row in rows), default=0.0)
    total_area = sum(float(row.get("area_px", 0) or 0) for row in rows)
    density = round(total_length * 10000.0 / total_area, 6) if total_area > 0 else "不可评估"
    primary = max(rows, key=lambda row: float(row.get("relative_score", 0) or 0), default={})

    output_end = raw.get("output_end", {}) if isinstance(raw.get("output_end"), dict) else {}
    region_overlay = _image(path, raw, output_end.get("region_overlay"))
    final_image = _image(path, raw, "08_最终皱纹检测图.jpg")

    wrinkle = ModuleResult(
        module_id="08",
        title="稳定线性皱纹",
        status="可评估" if raw.get("region_analysis_status") == "ok" else "部分支持",
        sources=["皱纹balanced多尺度分割", "皱纹中心线和分区统计"],
        summary=(
            f"共获得 {total_segments} 个二维可见纹路线段，主要集中于 {primary.get('region_name', '不可评估')}。"
            "该结果描述单次标准化白光图像中的稳定线性纹路代理。"
        ),
        metrics=[
            Metric("纹路线段数量", total_segments, "段"),
            Metric("分区中心线总长度", round(total_length, 3), "像素"),
            Metric("单位面积纹路长度密度", density, "像素/1万分区像素"),
            Metric("平均线段长度", round(statistics.fmean(lengths), 6) if lengths else 0.0, "像素"),
            Metric("P50分区平均线段长度", _percentile(lengths, 0.5), "像素"),
            Metric("P90分区平均线段长度", _percentile(lengths, 0.9), "像素"),
            Metric("最大线段长度", round(max_length, 3), "像素"),
            Metric("可见宽度", "不可评估", note="当前summary未输出中心线对应宽度"),
            Metric("视觉对比度", "不可评估", note="当前summary未输出线内外对比度"),
            Metric("主要方向", "不可评估", note="当前summary未输出方向直方图"),
            Metric("连续性", "不可评估", note="当前summary未输出跨视图线段匹配"),
            Metric("主要集中区域", primary.get("region_name", "不可评估")),
        ],
        regions=all_regions,
        images=existing_files([x for x in [final_image, region_overlay] if x]),
        daily_note="建议在相同表情、角度和光线下复测，观察二维可见纹路的长期趋势。",
        limitations=["单次二维白光图像代理", "不输出真实深度或体积", "未做多次拍摄的同一纹路匹配，不能宣称真实稳定识别比例"],
    )

    groove_rows = [row for row in rows if str(row.get("region_key")) in GROOVE_KEYS]
    groove_segments = sum(int(row.get("segment_count", 0) or 0) for row in groove_rows)
    groove_length = sum(float(row.get("wrinkle_pixels", 0) or 0) for row in groove_rows)
    groove_primary = max(groove_rows, key=lambda row: float(row.get("relative_score", 0) or 0), default={})
    groove = ModuleResult(
        module_id="09",
        title="结构性沟纹",
        status="部分支持" if groove_rows else "本次未检测",
        sources=["皱纹检测中的法令纹/木偶纹二维区域"] if groove_rows else [],
        summary=(
            f"法令纹和木偶纹区域共获得 {groove_segments} 个二维可见线段，"
            f"中心线总长度为 {round(groove_length, 3)} 像素，主要区域为 {groove_primary.get('region_name', '不可评估')}。"
        ) if groove_rows else "现有结果未包含可评估的法令纹或木偶纹区域。",
        metrics=[
            Metric("二维沟纹线段数量", groove_segments, "段"),
            Metric("二维沟纹中心线总长度", round(groove_length, 3), "像素"),
            Metric("主要沟纹区域", groove_primary.get("region_name", "不可评估")),
            Metric("平均相对深度", "不可评估"),
            Metric("最大相对深度", "不可评估"),
            Metric("沟纹体积", "不可评估"),
        ],
        regions=[_region(row) for row in groove_rows],
        images=existing_files([x for x in [region_overlay] if x]),
        daily_note="表情、光线、体重和拍摄角度均可能改变法令纹和木偶纹外观，应在相同条件下复测。",
        limitations=["仅为二维可见沟纹代理", "不输出真实深度、宽度或体积", "不能替代3D面部测量"],
    )
    return wrinkle, groove


def _medical_sections(value: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(value, dict):
        return output
    for section in (value.get("core_metrics"), value.get("auxiliary_metrics")):
        if not isinstance(section, dict):
            continue
        for group in section.values():
            if isinstance(group, dict):
                output.update(group)
    return output


def _adapt_medical_v2(path: Path, raw: dict[str, Any]) -> tuple[ModuleResult, ModuleResult]:
    total = _medical_sections(raw.get("overall_metrics"))
    rows = [row for row in raw.get("region_metrics", []) if isinstance(row, dict)]
    regions: list[RegionResult] = []
    normalized_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        values = _medical_sections(row)
        name = str(row.get("analysis_region", values.get("analysis_region", "未命名分区")))
        normalized_rows.append((row, values))
        regions.append(RegionResult(
            name=name,
            status=str(row.get("evaluation_status", values.get("evaluation_status", "可评估"))),
            metrics={
                "纹路线段数量（段）": values.get("wrinkle_segment_count", 0),
                "纹路中心线像素数": values.get("wrinkle_centerline_length_px", values.get("total_wrinkle_length_px", 0)),
                "纹路面积占比": values.get("wrinkle_area_ratio", 0),
                "单位面积长度密度（像素/1万分区像素）": values.get("wrinkle_pixel_density_per_10k_region_px", 0),
                "平均线段长度（像素）": values.get("mean_segment_length_px", 0),
                "最大线段长度（像素）": values.get("max_segment_length_px", 0),
                "视觉对比度（0～1）": values.get("p90_visual_contrast_proxy", 0),
                "纹路连续性（0～1）": values.get("wrinkle_continuity", 0),
                "分区有效皮肤面积（像素）": values.get("valid_skin_area_px", row.get("valid_skin_area_px", 0)),
                "相对响应（0～100）": values.get("relative_response_0_100", 0),
            },
        ))
    primary = total.get("primary_concentration_region", "—")
    image = path.parent / "01_皱纹检测结果图.jpg"
    wrinkle = ModuleResult(
        module_id="08", title="稳定线性皱纹", status=str(total.get("evaluation_status", "可评估")),
        sources=["皱纹balanced多尺度分割", "皱纹中心线和分区统计"],
        summary=f"共获得 {int(total.get('wrinkle_segment_count', 0) or 0)} 个二维可见纹路线段，主要集中于 {primary}。",
        metrics=[
            Metric("纹路线段数量", total.get("wrinkle_segment_count", 0), "段"),
            Metric("中心线总长度", total.get("total_wrinkle_length_px", total.get("wrinkle_centerline_length_px", 0)), "像素"),
            Metric("皱纹面积占比", total.get("wrinkle_area_ratio", "不可评估"), "比例"),
            Metric("单位面积纹路长度密度", total.get("wrinkle_pixel_density_per_10k_region_px", "不可评估"), "像素/1万分区像素"),
            Metric("P50线段长度", total.get("p50_segment_length_px_proxy", "不可评估"), "像素"),
            Metric("P90线段长度", total.get("p90_segment_length_px_proxy", "不可评估"), "像素"),
            Metric("平均可见宽度", total.get("mean_visible_width_px_proxy", "不可评估"), "像素"),
            Metric("纹路连续性", total.get("wrinkle_continuity", "不可评估"), "0～1"),
            Metric("P90视觉对比度", total.get("p90_visual_contrast_proxy", "不可评估"), "0～1"),
            Metric("平均线段长度", total.get("mean_segment_length_px", "不可评估"), "像素"),
            Metric("最大线段长度", total.get("max_segment_length_px", "不可评估"), "像素"),
            Metric("主要集中区域", primary),
        ],
        regions=regions, images=existing_files([image]),
        daily_note="建议在相同表情、角度和光线下复测，观察二维可见纹路的变化趋势。",
        limitations=list(raw.get("medical_limitations", [])),
    )
    groove_rows = [
        (row, values)
        for row, values in normalized_rows
        if region_name_in_group("09", str(row.get("analysis_region", "")))
    ]
    groove_segments = sum(int(values.get("wrinkle_segment_count", 0) or 0) for _, values in groove_rows)
    groove_length = sum(float(values.get("wrinkle_centerline_length_px", values.get("total_wrinkle_length_px", 0)) or 0) for _, values in groove_rows)
    groove_primary = max(groove_rows, key=lambda item: float(item[1].get("relative_response_0_100", 0) or 0), default=({}, {}))
    groove_name = groove_primary[0].get("analysis_region", "—") if groove_rows else "—"
    groove = ModuleResult(
        module_id="09", title="结构性沟纹", status="可评估",
        sources=["法令纹、木偶纹及沟纹分区"],
        summary=f"沟纹分区共获得 {groove_segments} 个二维可见线段，中心线总长度为 {round(groove_length, 3)} 像素，主要集中于 {groove_name}。",
        metrics=[
            Metric("二维沟纹线段数量", groove_segments, "段"),
            Metric("二维沟纹中心线总长度", round(groove_length, 3), "像素"),
            Metric("主要沟纹区域", groove_name),
        ],
        regions=[regions[rows.index(row)] for row, _ in groove_rows], images=existing_files([image]),
        daily_note="建议保持相同表情、光线和拍摄角度进行周期复测。",
        limitations=list(raw.get("medical_limitations", [])),
    )
    return wrinkle, groove


def _adapt_review_metrics(path: Path, raw: dict[str, Any]) -> tuple[ModuleResult, ModuleResult]:
    total = raw.get("总体指标", {})
    rows = [row for row in raw.get("分区指标", []) if isinstance(row, dict)]
    regions = [_review_region(row) for row in rows]
    image = path.parent / "01_皱纹检测结果图.jpg"
    status = str(total.get("评估状态", "可评估"))
    primary = total.get("主要集中区域", "不可评估")
    wrinkle = ModuleResult(
        module_id="08",
        title="稳定线性皱纹",
        status=status,
        sources=["皱纹balanced多尺度分割", "皱纹中心线和分区统计"],
        summary=f"共获得 {total.get('纹路线段数量（段）', 0)} 个二维可见纹路线段，主要集中于 {primary}。",
        metrics=[
            Metric("纹路线段数量", total.get("纹路线段数量（段）", 0), "段"),
            Metric("分区中心线总长度", total.get("皱纹中心线像素（像素）", 0), "像素"),
            Metric("皱纹面积占比", total.get("皱纹面积占比", "不可评估"), "比例"),
            Metric("单位面积纹路长度密度", total.get("单位面积密度（皱纹像素/1万分区像素）", "不可评估"), "像素/1万分区像素"),
            Metric("平均线段长度", total.get("平均线段长度（像素）", "不可评估"), "像素"),
            Metric("最大线段长度", total.get("最大线段长度（像素）", "不可评估"), "像素"),
            Metric("相对响应", total.get("相对响应（0～100）", "不可评估"), "0～100工程相对值"),
            Metric("主要集中区域", primary),
        ],
        regions=regions,
        images=existing_files([image]),
        daily_note="建议在相同表情、角度和光线下复测，观察二维可见纹路的长期趋势。",
        limitations=list(raw.get("医学局限性", [])) or ["单次二维白光图像代理，不输出真实深度或体积"],
    )

    groove_rows = [
        row
        for row in rows
        if region_name_in_group("09", str(row.get("检测范围", "")))
    ]
    groove_segments = sum(int(row.get("纹路线段数量（段）", 0) or 0) for row in groove_rows)
    groove_length = sum(float(row.get("皱纹中心线像素（像素）", 0) or 0) for row in groove_rows)
    groove_primary = max(
        groove_rows,
        key=lambda row: float(row.get("相对响应（0～100）", 0) or 0),
        default={},
    )
    groove = ModuleResult(
        module_id="09",
        title="结构性沟纹",
        status="部分支持" if groove_rows else "本次未检测",
        sources=["皱纹检测中的法令纹/木偶纹二维区域"] if groove_rows else [],
        summary=(
            f"法令纹和木偶纹区域共获得 {groove_segments} 个二维可见线段，"
            f"中心线总长度为 {round(groove_length, 3)} 像素，主要区域为 {groove_primary.get('检测范围', '不可评估')}。"
        ) if groove_rows else "现有结果未包含可评估的法令纹或木偶纹区域。",
        metrics=[
            Metric("二维沟纹线段数量", groove_segments, "段"),
            Metric("二维沟纹中心线总长度", round(groove_length, 3), "像素"),
            Metric("主要沟纹区域", groove_primary.get("检测范围", "不可评估")),
            Metric("平均相对深度", "不可评估"),
            Metric("最大相对深度", "不可评估"),
            Metric("沟纹体积", "不可评估"),
        ],
        regions=[_review_region(row) for row in groove_rows],
        images=existing_files([image]),
        daily_note="表情、光线、体重和拍摄角度均可能改变法令纹和木偶纹外观，应在相同条件下复测。",
        limitations=["仅为二维可见沟纹代理", "不输出真实深度、宽度或体积", "不能替代3D面部测量"],
    )
    return wrinkle, groove


def _review_region(row: dict[str, Any]) -> RegionResult:
    return RegionResult(
        name=str(row.get("检测范围", "未命名分区")),
        status=str(row.get("评估状态", "不可评估")),
        metrics={key: value for key, value in row.items() if key not in {"检测范围", "评估状态"}},
    )


def _region(row: dict[str, Any]) -> RegionResult:
    return RegionResult(
        name=str(row.get("region_name", row.get("region_key", "未命名分区"))),
        status="可评估",
        metrics={
            "纹路线段数量（段）": row.get("segment_count", 0),
            "纹路中心线像素数": row.get("wrinkle_pixels", 0),
            "平均线段长度（像素）": row.get("mean_segment_length", 0),
            "最大线段长度（像素）": row.get("max_segment_length", 0),
            "单位面积长度密度（像素/1万分区像素）": row.get("density_per_10k", 0),
            "全脸纹路占比（%）": row.get("share_pct", 0),
            "分区面积（像素）": row.get("area_px", 0),
            "相对显示分数（非医学分）": row.get("relative_score", 0),
        },
    )


def _missing(module_id: str, title: str, reason: str, failed: bool = False) -> ModuleResult:
    return ModuleResult(
        module_id=module_id,
        title=title,
        status="检测失败/不可评估" if failed else "本次未检测",
        sources=[],
        summary=reason,
        limitations=[reason],
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    w = pos - lo
    return round(ordered[lo] * (1.0 - w) + ordered[hi] * w, 6)
