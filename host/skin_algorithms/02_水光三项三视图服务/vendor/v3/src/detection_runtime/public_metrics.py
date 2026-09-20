from __future__ import annotations

"""Stable public CSV/JSON projections for split and newly added detections."""

import csv
import json
from pathlib import Path
from typing import Any

from typing_extensions import assert_never

from src.detection_runtime.contracts import RuntimeRoute
from src.nine_analysis.medical_v2_schema import to_english_document


REGION_ORDER = ("forehead", "left_cheek", "right_cheek", "nose", "chin")
REGION_LABELS = {
    "forehead": "额头", "left_cheek": "左脸颊", "right_cheek": "右脸颊",
    "nose": "鼻部", "chin": "下巴", "left_nasal_side": "左鼻旁",
    "right_nasal_side": "右鼻旁", "left_inner_cheek": "左内侧面颊",
    "right_inner_cheek": "右内侧面颊", "left_outer_cheek": "左外侧面颊",
    "right_outer_cheek": "右外侧面颊", "left_periocular": "左眼周",
    "right_periocular": "右眼周", "left_zygoma": "左颧部",
    "right_zygoma": "右颧部", "left_jaw": "左下颌", "right_jaw": "右下颌",
    "nose_alar_nasal_side": "鼻翼及鼻旁",
}


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, header: tuple[str, ...] | list[str], rows: list[tuple[Any, ...] | list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _metric_groups(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in metrics.items():
        if any(token in name for token in ("数量", "密度", "分支", "次数")):
            dimension = "数量与密度"
        elif any(token in name for token in ("面积", "长度", "宽度", "比例", "曲率", "连续")):
            dimension = "范围与形态"
        elif any(token in name for token in ("强度", "响应", "起伏")):
            dimension = "信号强度"
        else:
            dimension = "辅助统计"
        grouped.setdefault(dimension, {})[name] = value
    return grouped


def _document(
    *, project: str, version: str, imaging: str, valid_area: int,
    overall: dict[str, Any], region_rows: list[dict[str, Any]],
    laterality: dict[str, Any], limitations: list[str], assessable: bool = True,
) -> dict[str, Any]:
    status = "ASSESSABLE" if assessable else "UNASSESSABLE"
    scope = {
        "检测范围": "全面部", "评估状态": status,
        "有效皮肤面积（像素）": int(valid_area),
    }
    return {
        "检测项目": project,
        "指标版本": version,
        "评分状态": "uncalibrated",
        "成像与单位说明": {
            "成像类型": imaging,
            "面积单位": "标准化图像像素",
            "强度单位": "0～1工程归一化响应",
            "强度说明": "工程图像响应，不代表真实浓度、深度或诊断概率",
            "密度单位": "按有效分析像素标准化",
            "左右定义": "以画面左右为准",
        },
        "总体指标": {
            "核心指标": _metric_groups(overall),
            "辅助指标": {"分析范围": scope},
        },
        "分区指标": region_rows,
        "左右比较": laterality,
        "质量控制": {},
        "医学局限性": limitations,
    }


def _region_row(label: str, valid_area: int, metrics: dict[str, Any], *, assessable: bool = True) -> dict[str, Any]:
    return {
        "检测范围": label,
        "评估状态": "ASSESSABLE" if assessable else "UNASSESSABLE",
        "有效皮肤面积（像素）": int(valid_area),
        "核心指标": _metric_groups(metrics),
        "辅助指标": {},
    }


def _wide_csv(path: Path, document: dict[str, Any]) -> Path:
    scope = document["总体指标"]["辅助指标"]["分析范围"]
    rows = [{
        "检测范围": scope["检测范围"],
        "评估状态": "可评估" if scope["评估状态"] == "ASSESSABLE" else "不可评估",
        "有效皮肤面积（像素）": scope["有效皮肤面积（像素）"],
        **{f"核心-{name}": value for group in document["总体指标"]["核心指标"].values() for name, value in group.items()},
    }]
    for region in document["分区指标"]:
        row = {
            "检测范围": region["检测范围"],
            "评估状态": "可评估" if region["评估状态"] == "ASSESSABLE" else "不可评估",
            "有效皮肤面积（像素）": region["有效皮肤面积（像素）"],
        }
        for group in region.get("核心指标", {}).values():
            row.update({f"核心-{name}": value for name, value in group.items()})
        rows.append(row)
    columns = list(rows[0])
    for row in rows[1:]:
        columns.extend(name for name in row if name not in columns)
    return _write_csv(path, columns, [[row.get(name, "") for name in columns] for row in rows])


def _recursive_number(value: Any, names: tuple[str, ...], default: Any = 0) -> Any:
    if isinstance(value, dict):
        for name in names:
            number = value.get(name)
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                return number
        for child in value.values():
            number = _recursive_number(child, names, None)
            if number is not None:
                return number
    elif isinstance(value, list):
        for child in value:
            number = _recursive_number(child, names, None)
            if number is not None:
                return number
    return default


def _valid_region_area(value: dict[str, Any]) -> int:
    if value.get("valid_area_px") is not None:
        return int(value["valid_area_px"])
    area = float(value.get("area") or 0)
    ratio = float(value.get("area_ratio") or 0)
    return int(round(area / ratio)) if ratio > 0 else 0


def write_uv_or_porphyrin_public_metrics(
    *, item_id: str, raw: dict[str, Any], json_path: Path, csv_path: Path,
    v2_path: Path, route: RuntimeRoute,
) -> tuple[Path, Path, Path]:
    if item_id not in {"uv_spots", "porphyrin"}:
        raise ValueError(item_id)
    label = "UV色斑" if item_id == "uv_spots" else "卟啉"
    match route:
        case RuntimeRoute.CONSUMER_RGB:
            version = {"uv_spots": "UVSpots-RGBProxy-V1", "porphyrin": "RGBProxyPorphyrin-Frozen-V1"}[item_id]
            imaging = "单RGB正面图像（工程代理）"
            limitations = ["仅表示单RGB正面图像推断的工程代理候选，不代表真实UV/365成像或荧光信号，不等同于临床诊断。"]
        case RuntimeRoute.CLINIC_FOUR_LIGHT:
            version = {"uv_spots": "UV365Spots-V1", "porphyrin": "FluorescencePorphyrin-Real365-HighRecall-V1"}[item_id]
            imaging = "真实365nm正面图像"
            limitations = ["仅表示二维UV图像候选，不等同于临床诊断。"]
        case unreachable:
            assert_never(unreachable)
    regions = raw.get("regions") or raw.get("region_distribution") or {
        name: {"label": REGION_LABELS[name], "count": raw.get(f"{item_id}_{name}", 0)}
        for name in REGION_ORDER
    }
    count = int(raw.get("instance_count", _recursive_number(raw, (f"{item_id}_total", "feature_count"))))
    area = int(raw.get("area_px", _recursive_number(raw, ("total_feature_area_px",))))
    valid = int(raw.get("valid_area_px", _recursive_number(raw, ("valid_skin_area_px",))))
    ratio = float(raw.get("area_ratio", _recursive_number(raw, ("feature_area_ratio",))))
    p50 = float(raw.get("p50_intensity", _recursive_number(raw, ("p50_intensity",))))
    p90 = float(raw.get("p90_intensity", _recursive_number(raw, ("p90_intensity",))))
    counts: list[int] = []
    region_rows: list[dict[str, Any]] = []
    for name in REGION_ORDER:
        value = regions.get(name) or {}
        local_count = int(value.get("count") or 0)
        counts.append(local_count)
        region_rows.append(_region_row(
            str(value.get("label") or REGION_LABELS[name]),
            _valid_region_area(value),
            {
                "特征数量（个）": local_count,
                "特征总面积（像素）": int(value.get("area") or 0),
                "特征面积占比": float(value.get("area_ratio") or 0),
                "平均强度（0～1）": float(value.get("mean_score") or 0),
            },
        ))
    document = _document(
        project=label, version=version, imaging=imaging, valid_area=valid,
        overall={"特征数量（个）": count, "特征总面积（像素）": area, "特征面积占比": ratio, "P50强度（0～1）": p50, "P90强度（0～1）": p90},
        region_rows=region_rows, laterality={},
        limitations=limitations,
    )
    _write_json(json_path, to_english_document(document))
    _write_csv(csv_path, ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"), [(count, *counts)])
    _wide_csv(v2_path, document)
    return json_path, csv_path, v2_path


CONTOUR_FIELDS = (
    ("下颌线连续比例", "jaw_continuity_ratio"), ("下颌曲率波动P90", "jaw_curve_p90"),
    ("下颌曲率方向变化次数", "jaw_curve_direction_changes"), ("下颌弧长左右差异", "jaw_arc_asymmetry_ratio"),
    ("下脸宽高比", "jaw_width_to_face_height_ratio"), ("中面部曲面连续比例", "midface_surface_continuity_ratio"),
    ("中面部凸凹转折次数", "midface_convex_concave_turns"), ("中面部相对起伏P90", "midface_relative_relief_p90"),
)


def write_added_item_public_metrics(
    *, item_id: str, raw: dict[str, Any], json_path: Path, csv_path: Path, v2_path: Path,
) -> tuple[Path, Path, Path]:
    if item_id == "surface_gloss":
        project, version, imaging = "油光", "SurfaceGloss-V3.0", "平行偏振光或标准白光正面图像"
        full, source = raw.get("full_face") or {}, raw.get("region_metrics") or {}
        overall = {
            "油光面积（像素）": int(full.get("gloss_area_px") or 0),
            "油光面积占比": float(full.get("gloss_area_ratio") or 0),
            "高强度油光面积占比": float(full.get("high_gloss_area_ratio") or 0),
            "P50油光强度（0～1）": float(full.get("p50_gloss_intensity") or 0),
            "P90油光强度（0～1）": float(full.get("p90_gloss_intensity") or 0),
            "油光区域数量（个）": int(full.get("patch_count") or 0),
        }
        regions = [_region_row(str(value.get("label") or REGION_LABELS.get(name, name)), int(value.get("valid_area_px") or 0), {"油光面积占比": value.get("gloss_area_ratio"), "P90油光强度（0～1）": value.get("p90_gloss_intensity"), "油光区域数量（个）": int(value.get("patch_count") or 0)}, assessable=value.get("status") in {None, "VALID"}) for name, value in source.items()]
        counts = [int((source.get(name) or {}).get("patch_count") or 0) for name in REGION_ORDER]
        ratios = [float((source.get(name) or {}).get("gloss_area_ratio") or 0) for name in REGION_ORDER]
        compact_header = ("指标", "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
        compact_rows = [("油光面积占比", overall["油光面积占比"], *ratios), ("油光区域数量", overall["油光区域数量（个）"], *counts)]
        valid, laterality = int(full.get("valid_skin_area_px") or 0), {}
        limitations = ["表示二维图像中的表面油光表型，不等同于绝对皮脂分泌量。"]
    elif item_id == "vascular":
        project = "血管样结构"
        version = str(raw.get("algorithm") or "VascularStructure-V3-Clinic")
        imaging = {
            "RGB_M": "单RGB正面图像（同图代理）",
            "CP_M": "交叉偏振光正面图像",
        }.get(str(raw.get("measurement_channel")), "交叉偏振光正面图像")
        fields = (
            ("血管样目标数量（个）", "vascular_count"), ("血管样总骨架长度（像素）", "vascular_total_length_px"),
            ("血管样单位面积长度密度（像素/1万有效像素）", "vascular_line_density_per_10k_face_px"),
            ("血管样总面积（像素）", "vascular_area_px"), ("血管样面积占比", "vascular_area_ratio"),
            ("P50血管样宽度（像素）", "p50_width_px"), ("P90血管样宽度（像素）", "p90_width_px"),
            ("P50红色响应", "p50_redness"), ("P90红色响应", "p90_redness"),
            ("分支点数量（个）", "branch_point_count"), ("最大连续网络长度（像素）", "max_continuous_network_length_px"),
            ("白光可见支持率", "cp_rgb_visible_support_ratio"),
        )
        overall = {label: raw.get(key) for label, key in fields}
        source = raw.get("region_distribution") or {}
        regions = [_region_row(REGION_LABELS.get(name, name), int(value.get("valid_pixels") or 0), {"血管样目标数量（个）": int(value.get("count") or 0), "血管样总骨架长度（像素）": float(value.get("total_length_px") or 0), "血管样总面积（像素）": int(value.get("total_area_px") or 0), "P90红色响应": float(value.get("p90_redness") or 0)}) for name, value in source.items()]
        counts = [int((source.get(name) or {}).get("count") or 0) for name in REGION_ORDER]
        lengths = [float((source.get(name) or {}).get("total_length_px") or 0) for name in REGION_ORDER]
        compact_header = ("指标", "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
        compact_rows = [("目标数量", raw.get("vascular_count"), *counts), ("总长度（像素）", raw.get("vascular_total_length_px"), *lengths)]
        valid = int(raw.get("valid_face_pixels") or 0)
        lr = raw.get("left_right_summary") or {}
        laterality = {"左侧总长度（像素）": float(lr.get("left_total_length_px") or 0), "右侧总长度（像素）": float(lr.get("right_total_length_px") or 0), "长度不对称比例": float(lr.get("length_asymmetry_ratio") or 0), "红色响应差异": float(lr.get("redness_difference") or 0)}
        limitations = list(raw.get("limitations") or ["仅表示二维血管样线状结构，不用于疾病诊断。"])
    elif item_id == "contour_firmness":
        project, version, imaging = "轮廓紧致度", "RelativeFaceMeshGeometryProxy-V4", "标准白光正面相对2.5D几何"
        overall = {label: raw.get(key) for label, key in CONTOUR_FIELDS}
        regions, laterality = [], {"下颌弧长左右差异": overall["下颌弧长左右差异"]}
        compact_header = tuple(label for label, _ in CONTOUR_FIELDS)
        compact_rows = [tuple(overall[label] for label, _ in CONTOUR_FIELDS)]
        valid = int(raw.get("face_scope_area_px") or raw.get("valid_face_pixels") or 0)
        limitations = ["仅为正面相对2.5D面部几何代理，不是毫米级真实深度。", "当前不提供紧致度0～100评分。"]
    else:
        raise ValueError(item_id)
    document = _document(project=project, version=version, imaging=imaging, valid_area=valid, overall=overall, region_rows=regions, laterality=laterality, limitations=limitations, assessable=bool(raw.get("qc_passed", True)))
    _write_json(json_path, to_english_document(document))
    _write_csv(csv_path, compact_header, compact_rows)
    _wide_csv(v2_path, document)
    return json_path, csv_path, v2_path


__all__ = ["write_added_item_public_metrics", "write_uv_or_porphyrin_public_metrics"]
