from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PUBLIC_REGIONS = ("total", "forehead", "left_cheek", "right_cheek", "nose", "chin")
PUBLIC_LABELS = {
    "total": "总计", "forehead": "额头", "left_cheek": "左脸颊",
    "right_cheek": "右脸颊", "nose": "鼻部", "chin": "下巴",
}


@dataclass(frozen=True, slots=True)
class PublicMetricFiles:
    metrics_json: Path
    compact_csv: Path
    medical_csv_v2: Path
    full_metrics_json: Path


def _write_json(path: Path, metrics: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(metrics), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_compact(
    path: Path,
    rows: list[tuple[str, str, Mapping[str, int | float]]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "指标",
            "单位",
            *[PUBLIC_LABELS[name] for name in PUBLIC_REGIONS],
        ])
        for label, unit, values in rows:
            writer.writerow([
                label,
                unit,
                *[values[name] for name in PUBLIC_REGIONS],
            ])


def _public_regions(values: Mapping[str, int | float]) -> dict[str, int | float]:
    return {PUBLIC_LABELS[name]: values[name] for name in PUBLIC_REGIONS}


def _write_wide(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sum_region(
    regions: Mapping[str, Mapping[str, int | float | str | None]],
    names: tuple[str, ...],
    field: str,
) -> float:
    return float(sum(float(regions.get(name, {}).get(field) or 0) for name in names))


def write_gloss_metrics(result, root: Path) -> PublicMetricFiles:
    groups = {
        "forehead": ("forehead",),
        "left_cheek": ("left_nasal_side", "left_inner_cheek", "left_outer_cheek"),
        "right_cheek": ("right_nasal_side", "right_inner_cheek", "right_outer_cheek"),
        "nose": ("nose",),
        "chin": ("chin",),
    }
    areas = {"total": int(result.valid_skin_area_px)}
    ratios = {"total": float(result.gloss_area_ratio)}
    counts = {"total": int(result.patch_count)}
    for public, names in groups.items():
        valid = _sum_region(result.region_metrics, names, "valid_area_px")
        gloss = _sum_region(result.region_metrics, names, "gloss_area_px")
        areas[public] = int(valid)
        ratios[public] = float(gloss / valid) if valid else 0.0
        counts[public] = int(_sum_region(result.region_metrics, names, "patch_count"))
    rounded_ratios = {
        name: round(ratios[name], 8) for name in PUBLIC_REGIONS
    }
    metrics = {
        "油光面积占比": _public_regions(rounded_ratios),
        "油光区域数量": _public_regions(counts),
    }
    paths = PublicMetricFiles(root / "表面油光量化指标.json", root / "表面油光量化指标.csv", root / "表面油光医学量化指标_V2.csv", root / "表面油光完整指标.json")
    _write_json(paths.metrics_json, metrics)
    full_metrics = (
        result.metrics()
        if callable(getattr(result, "metrics", None))
        else {
            "full_face": {
                "valid_skin_area_px": int(result.valid_skin_area_px),
                "gloss_area_ratio": float(result.gloss_area_ratio),
                "patch_count": int(result.patch_count),
            },
            "region_metrics": result.region_metrics,
        }
    )
    _write_json(paths.full_metrics_json, full_metrics)
    _write_compact(paths.compact_csv, [
        ("油光面积占比", "比例（0～1）", rounded_ratios),
        ("油光区域数量", "个", counts),
    ])
    rows = [{
        "检测范围": PUBLIC_LABELS[name], "评估状态": "可评估",
        "有效皮肤面积（像素）": areas[name],
        "核心-油光面积占比": round(ratios[name], 8),
        "核心-油光区域数量（个）": counts[name],
    } for name in PUBLIC_REGIONS]
    _write_wide(paths.medical_csv_v2, rows)
    return paths


def _vascular_public_regions(regions: Mapping[str, Mapping[str, int | float | str]]) -> tuple[dict[str, int], dict[str, float], dict[str, int]]:
    groups = {
        "forehead": ("forehead",),
        "left_cheek": ("left_periocular", "left_zygoma", "left_cheek"),
        "right_cheek": ("right_periocular", "right_zygoma", "right_cheek"),
        "nose": ("nose_alar_nasal_side",),
        "chin": ("left_jaw", "right_jaw"),
    }
    counts: dict[str, int] = {}
    lengths: dict[str, float] = {}
    valid: dict[str, int] = {}
    for public, names in groups.items():
        counts[public] = int(_sum_region(regions, names, "count"))
        lengths[public] = _sum_region(regions, names, "total_length_px")
        valid[public] = int(_sum_region(regions, names, "valid_pixels"))
    return counts, lengths, valid


def write_vascular_metrics(raw: Mapping[str, object], root: Path) -> PublicMetricFiles:
    regions = raw.get("region_distribution")
    region_map = regions if isinstance(regions, dict) else {}
    counts, lengths, valid = _vascular_public_regions(region_map)
    counts["total"] = int(raw.get("vascular_count") or 0)
    lengths["total"] = float(raw.get("vascular_total_length_px") or 0.0)
    valid["total"] = int(raw.get("valid_face_pixels") or 0)
    rounded_lengths = {
        name: round(lengths[name], 3) for name in PUBLIC_REGIONS
    }
    metrics = {
        "血管样结构数量": _public_regions(counts),
        "血管样结构总长度": _public_regions(rounded_lengths),
    }
    paths = PublicMetricFiles(root / "血管样结构量化指标.json", root / "血管样结构量化指标.csv", root / "血管样结构医学量化指标_V2.csv", root / "血管样结构完整指标.json")
    _write_json(paths.metrics_json, metrics)
    _write_json(paths.full_metrics_json, raw)
    _write_compact(paths.compact_csv, [
        ("血管样结构数量", "个", counts),
        ("血管样结构总长度", "标准化图像像素", rounded_lengths),
    ])
    rows = [{
        "检测范围": "全面部", "评估状态": "可评估" if raw.get("qc_passed") else "不可评估",
        "有效皮肤面积（像素）": valid["total"], "核心-目标数量（个）": counts["total"],
        "核心-总长度（像素）": round(lengths["total"], 3),
        "辅助-P90宽度（像素）": round(float(raw.get("p90_width_px") or 0.0), 4),
        "辅助-分支点数量（个）": int(raw.get("branch_point_count") or 0),
    }]
    for name, row in region_map.items():
        rows.append({
            "检测范围": str(row.get("region_id", name)), "评估状态": "可评估",
            "有效皮肤面积（像素）": int(row.get("valid_pixels") or 0),
            "核心-目标数量（个）": int(row.get("count") or 0),
            "核心-总长度（像素）": round(float(row.get("total_length_px") or 0.0), 3),
        })
    _write_wide(paths.medical_csv_v2, rows)
    return paths


def write_contour_metrics(metrics: Mapping[str, float], valid_area: int, root: Path) -> PublicMetricFiles:
    paths = PublicMetricFiles(root / "轮廓紧致度量化指标.json", root / "轮廓紧致度量化指标.csv", root / "轮廓紧致度医学量化指标_V2.csv", root / "轮廓紧致度完整指标.json")
    compact = {
        "中面部曲面连续性": round(
            float(metrics.get("midface_surface_continuity_ratio", 0.0)), 6
        ),
        "下颌缘连续性": round(
            float(metrics.get("jaw_continuity_ratio", 0.0)), 6
        ),
        "左右轮廓差异": round(
            float(metrics.get("jaw_arc_asymmetry_ratio", 0.0)), 6
        ),
    }
    _write_json(paths.metrics_json, compact)
    _write_json(paths.full_metrics_json, metrics)
    with paths.compact_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("指标", "单位", "数值"))
        for name, value in compact.items():
            writer.writerow((name, "相对代理值（0～1）", value))
    _write_wide(paths.medical_csv_v2, [{
        "检测范围": "全面部", "评估状态": "可评估",
        "有效皮肤面积（像素）": valid_area,
        **{f"核心-{name}": value for name, value in metrics.items()},
    }])
    return paths


__all__ = ["PublicMetricFiles", "write_contour_metrics", "write_gloss_metrics", "write_vascular_metrics"]
