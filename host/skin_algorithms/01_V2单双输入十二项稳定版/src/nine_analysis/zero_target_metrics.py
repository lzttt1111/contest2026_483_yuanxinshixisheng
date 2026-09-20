"""Public medical artifact semantics for rows without detected targets."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Final, Mapping


UNAVAILABLE: Final = "不可评估"
COUNT_COLUMNS: Final = frozenset({
    "特征数量（个）",
    "疑似痤疮候选数量（个）",
    "目标数量（个）",
})
DISTRIBUTION_COLUMNS: Final = frozenset({
    "P50单体面积（像素）",
    "P90单体面积（像素）",
    "实例P50强度（0～1）",
    "实例P90强度（0～1）",
    "P50综合色差ΔE",
    "P90综合色差ΔE",
    "边界清晰度中位数（0～1）",
    "颜色均匀性中位数（0～1）",
    "P50等效直径（像素）",
    "P90等效直径（像素）",
    "P50中心—环带视觉对比度（0～1）",
    "P90中心—环带视觉对比度（0～1）",
    "P50圆度（0～1）",
    "P50长宽比",
    "P50候选框面积（像素）",
    "P90候选框面积（像素）",
    "平均置信度（0～1）",
    "P90置信度（0～1）",
    "P50宽度（像素）",
    "P90宽度（像素）",
    "P50红色响应",
    "P90红色响应",
})
ACNE_DISTRIBUTION_COLUMNS: Final = frozenset({
    "P50候选框面积（像素）",
    "P90候选框面积（像素）",
    "平均置信度（0～1）",
    "P90置信度（0～1）",
})


def _metric_name(column: str) -> str:
    for prefix in ("核心-", "辅助-"):
        if column.startswith(prefix):
            return column.removeprefix(prefix)
    return column


def _zero_count_row(row: Mapping[str, Any]) -> bool:
    for column, value in row.items():
        if _metric_name(column) not in COUNT_COLUMNS:
            continue
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False
    return False


def normalize_acne_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a copy whose zero-candidate distributions are unavailable."""
    normalized = deepcopy(report)
    rows = [normalized["总体指标"], *normalized["分区指标"]]
    for row in rows:
        if not _zero_count_row(row):
            continue
        for name in ACNE_DISTRIBUTION_COLUMNS:
            if name in row:
                row[name] = UNAVAILABLE
        if "主要集中区域" in row:
            row["主要集中区域"] = "—"
    return normalized


def normalize_zero_target_medical_csv(path: Path) -> int:
    """Rewrite only instance-distribution cells on proven zero-target rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    changed = 0
    for row in rows:
        if not _zero_count_row(row):
            continue
        for column in columns:
            if _metric_name(column) not in DISTRIBUTION_COLUMNS:
                continue
            if row.get(column) != UNAVAILABLE:
                row[column] = UNAVAILABLE
                changed += 1
    if changed:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return changed


def normalize_zero_target_detector_results(
    detector_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only distribution metrics with explicit zero-target proof."""
    normalized = deepcopy(dict(detector_results))
    gloss = normalized.get("surface_gloss", {}).get("metrics", {})
    for row in gloss.get("region_metrics", {}).values():
        if row.get("patch_count") != 0:
            continue
        for key in ("p50_gloss_intensity", "p90_gloss_intensity"):
            if key in row:
                row[key] = UNAVAILABLE
    vascular = normalized.get("vascular", {}).get("metrics", {})
    if vascular.get("vascular_count") == 0:
        for key in (
            "p50_width_px",
            "p90_width_px",
            "p50_redness",
            "p90_redness",
        ):
            if key in vascular:
                vascular[key] = UNAVAILABLE
    for row in vascular.get("region_distribution", {}).values():
        if row.get("count") != 0:
            continue
        for key in ("p50_redness", "p90_redness"):
            if key in row:
                row[key] = UNAVAILABLE
    return normalized


__all__ = [
    "normalize_acne_report",
    "normalize_zero_target_detector_results",
    "normalize_zero_target_medical_csv",
]
