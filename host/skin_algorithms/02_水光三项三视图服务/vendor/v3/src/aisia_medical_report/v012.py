from __future__ import annotations

import copy
import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .aggregator import build_report_payload, _burden_grade
from .integration import DEFAULT_TEMPLATE, _load_successful_result, _scoring_features, _source_image
from .report import render_doctor_docx, render_user_docx
from .formal_view import build_formal_report_view
from src.engines.visia_regions import (
    VISIA_BOUNDARY_COLOR,
    VisiaRegionSet,
    build_visia_regions,
    draw_region_boundaries,
)
from src.preprocess.image_preprocessor import ImagePreprocessor, PreprocessResultV2
from src.scoring_calibration.v011.quality import evaluate_image_path
from src.scoring_calibration.v011.scoring import score_observation
from src.scoring_calibration.v012.explanation import attach_score_explanations
from src.scoring_calibration.v012.pigmentation import score_combined_pigmentation_shadow
from src.aisia_medical_report.controlled_evidence import (
    ControlledEvidencePaths,
    bind_controlled_evidence,
)
from src.aisia_medical_report.controlled_report_metrics import (
    apply_controlled_metric_layout,
    vascular_region_row,
)
from src.aisia_medical_report.twelve_delivery import (
    bind_twelve_report_images,
)


OFFICIAL_VERSION = "aisia_scoring_v0.1.1_integrity_full_reference_20260807"
SHADOW_VERSION = "aisia_scoring_v0.1.2_front5_shadow_20260807"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_profile(path: Path, version: str, role: str | None = None) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("scoring_profile_version") != version or document.get("profile_ready") is not True:
        raise RuntimeError(f"评分配置无效: {path}")
    if role is not None and document.get("profile_role") != role:
        raise RuntimeError(f"评分配置角色不是{role}: {path}")
    return document


def _uv_detail(root: Path) -> dict[str, float]:
    path = root / "七项检测" / "紫区" / "紫区医学量化指标_V2.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("检测项目") == "标准化UV紫外线色斑工程代理" and row.get("检测范围") == "全面部":
                def number(key: str) -> float:
                    return float(row[key])
                return {
                    "density": number("核心-单位面积密度（个/10万有效皮肤像素）"),
                    "area_ratio": number("核心-特征面积占比"),
                    "p90_intensity": number("核心-P90强度（0～1）"),
                    "high_intensity_ratio": number("核心-高强度目标比例"),
                    "high_intensity_area_ratio": number("辅助-高强度区域面积占比"),
                }
    raise RuntimeError(f"紫区医学CSV缺少全面部UV指标: {path}")


def _enriched_features(root: Path) -> dict[str, Any]:
    features = _scoring_features(root)
    uv = _uv_detail(root)
    features["uv_spots"].setdefault("UV样色素范围", {}).update({
        "density": uv["density"], "area_ratio": uv["area_ratio"],
    })
    features["uv_spots"].setdefault("UV样色素强度", {}).update({
        "p90_intensity": uv["p90_intensity"],
        "high_intensity_ratio": uv["high_intensity_ratio"],
    })
    return features


def _report_features(
    root: Path,
    complete_document: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    provenance = (
        complete_document.get("provenance")
        if isinstance(complete_document, dict)
        else None
    )
    if isinstance(provenance, dict) and provenance.get("capture_profile") == "institution":
        return _scoring_features(root), False
    return _enriched_features(root), True


DISPLAY_TYPES = {
    "PERCENTAGE", "ENGINEERING_0_1", "COUNT", "PIXEL", "DENSITY",
    "INTENSITY_ENGINEERING", "SCORE", "TEXT",
}


def _metric(name: str, value: Any, unit: str = "", display_type: str = "TEXT", note: str = "") -> dict[str, Any]:
    if display_type not in DISPLAY_TYPES:
        raise ValueError(f"未知报告展示类型: {display_type}")
    result = {"name": name, "value": value, "unit": unit, "display_type": display_type}
    if note:
        result["note"] = note
    return result


def _set_structured_module_state(
    module: dict[str, Any],
    *,
    evidence_status: str,
    module_status: str,
    report_eligibility: str,
    evidence_name: str = "",
    enforce: bool = False,
) -> None:
    """Attach engineering state without changing the frozen formal payload.

    The default path is additive metadata only. ``enforce=True`` is reserved
    for an explicitly selected engineering-debug projection; the V0.1.2
    formal builder does not call this helper.
    """
    status_text = {
        "NOT_ASSESSABLE": "不可评估（证据不足）",
        "PARTIAL": "内部工程证据（完整V2证据不足）",
        "ENGINEERING_READY": "工程就绪（未医学通过）",
        "MEDICAL_READY": "医学就绪",
    }[module_status]
    engineering_state = {
        "评估状态": status_text,
        "证据状态": evidence_status,
        "模块状态": module_status,
        "报告资格": report_eligibility,
        "医学构念满足": module_status == "MEDICAL_READY",
    }
    if evidence_name:
        engineering_state["证据名称"] = evidence_name
    module["内部工程调试状态"] = engineering_state
    if not enforce:
        return
    module.update({
        "评估状态": status_text,
        "用户评估状态": status_text,
        "医生评估状态": status_text,
        "证据状态": evidence_status,
        "模块状态": module_status,
        "报告资格": report_eligibility,
        "医学构念满足": module_status == "MEDICAL_READY",
    })
    if evidence_name:
        module["证据名称"] = evidence_name
    if module_status == "NOT_ASSESSABLE" or report_eligibility == "NOT_REPORTABLE":
        internal_summary = module.get("结果摘要")
        if internal_summary:
            module["内部证据摘要"] = internal_summary
        module["结果摘要"] = "证据不足，当前不可评估；候选证据仅限内部调试，不得解释为未发现问题。"
        module["综合得分"] = None
        module["程度等级"] = None
        module["日常管理提示"] = ""


def _apply_wrinkle_report_images(
    payload: dict[str, Any],
    images: dict[str, str | Path] | None,
) -> None:
    """Replace only the report images for modules 07-09.

    The module metrics, summaries, regions, scores, and source algorithm data
    remain the legacy report values.  This hook is deliberately report-only.
    """

    if images is None:
        return
    expected = {"07", "08", "09"}
    if set(images) != expected:
        raise ValueError(f"07～09报告图必须完整提供，收到: {sorted(images)}")

    resolved: dict[str, str] = {}
    image_sizes: set[tuple[int, int]] = set()
    for module_id, value in images.items():
        path = Path(value).expanduser().resolve()
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if not path.is_file() or decoded is None:
            raise FileNotFoundError(f"报告分区图不存在或无法解码: {path}")
        image_sizes.add((int(decoded.shape[1]), int(decoded.shape[0])))
        resolved[module_id] = str(path)
    if len(image_sizes) != 1:
        raise ValueError(f"07～09报告图尺寸不一致: {sorted(image_sizes)}")

    found: set[str] = set()
    for module in payload.get("检测模块", []) or []:
        module_id = str(module.get("模块编号", ""))
        if module_id in resolved:
            module["结果图"] = [resolved[module_id]]
            found.add(module_id)
    if found != expected:
        raise ValueError(f"报告缺少07～09模块，实际找到: {sorted(found)}")


def _nested(document: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = document
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _medical_v2(root: Path, algorithm: str) -> dict[str, Any]:
    path = root / "七项检测" / algorithm / f"{algorithm}量化指标.json"
    # The current results use algorithm-specific file names; retain a bounded discovery
    # here because this is report-only input discovery, never a Worker contract change.
    candidates = list((root / "七项检测" / algorithm).glob("*量化指标.json"))
    for candidate in candidates:
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and isinstance(document.get("medical_metrics_v2"), dict):
            return document["medical_metrics_v2"]
    return {}


def _csv_row(path: Path, project: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("检测项目") == project and row.get("检测范围") == "全面部":
                return row
    return {}


def _num(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, "", "不可评估"):
        return None
    return float(raw)


def _vascular_result_summary(data: dict[str, Any]) -> str:
    count = data.get("vascular_count")
    length = data.get("vascular_total_length_px")
    count_text = (
        f"检测到{count}个红色线状血管样候选结构"
        if isinstance(count, (int, float))
        else "红色线状血管样候选结构数量不可评估"
    )
    length_text = (
        f"总可见长度约{length:.0f}像素"
        if isinstance(length, (int, float))
        else "总可见长度不可评估"
    )
    return f"{count_text}，{length_text}。"


def _build_report_visia_regions(
    preprocess_result: PreprocessResultV2,
) -> VisiaRegionSet:
    """Reuse the same VISIA geometry that formal detector overlays use."""
    regions = build_visia_regions(
        np.asarray(preprocess_result.analysis_image),
        np.asarray(preprocess_result.skin_mask),
        np.asarray(preprocess_result.landmarks),
        preprocess_result.quality_flags,
        include_chin=True,
        mode="full",
    )
    if regions.display_contour is None or regions.scope_mask is None:
        raise RuntimeError("报告附加图缺少统一VISIA线框")
    if np.count_nonzero(regions.scope_mask) < 500:
        raise RuntimeError("报告附加图的统一VISIA有效范围过小")
    return regions


def _draw_report_visia_contour(
    image: np.ndarray,
    regions: VisiaRegionSet,
) -> np.ndarray:
    return draw_region_boundaries(
        image,
        regions.display_regions,
        VISIA_BOUNDARY_COLOR,
        thickness=3,
        partial_face=regions.partial_face,
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )


def _report_mask_overlay(
    analysis_image: np.ndarray,
    mask: Path,
    target: Path,
    color: tuple[int, int, int],
    *,
    regions: VisiaRegionSet,
) -> str | None:
    """Create a report-only visualization from existing candidate masks."""
    binary = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        return None
    image = np.asarray(analysis_image)
    if binary.shape != image.shape[:2]:
        raise ValueError(
            f"报告Mask尺寸不一致: mask={binary.shape} image={image.shape[:2]}"
        )
    if regions.scope_mask is None or regions.scope_mask.shape != image.shape[:2]:
        raise ValueError("统一VISIA范围与报告底图尺寸不一致")
    active = (binary > 0) & (regions.scope_mask > 0)
    result = image.copy()
    result[active] = np.clip(0.45 * result[active] + 0.55 * np.asarray(color), 0, 255).astype(np.uint8)
    result = _draw_report_visia_contour(result, regions)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), result)
    return str(target)


def _copy_report_image(
    source: Path,
    target: Path,
    *,
    analysis_image: np.ndarray,
    regions: VisiaRegionSet,
) -> str | None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        return None
    background = np.asarray(analysis_image)
    if image.shape[:2] != background.shape[:2]:
        raise ValueError(
            f"报告融合图尺寸不一致: source={image.shape[:2]} image={background.shape[:2]}"
        )
    if regions.scope_mask is None or regions.scope_mask.shape != background.shape[:2]:
        raise ValueError("统一VISIA范围与报告底图尺寸不一致")
    inside = regions.scope_mask > 0
    result = background.copy()
    result[inside] = image[inside]
    result = _draw_report_visia_contour(result, regions)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), result)
    return str(target)


def _module(payload: dict[str, Any], module_id: str) -> dict[str, Any]:
    return next(item for item in payload["检测模块"] if item.get("模块编号") == module_id)


def _apply_front5_display_metadata(payload: dict[str, Any]) -> None:
    """Attach explicit semantic display metadata without changing JSON values."""
    percentage_names = {
        "可见斑点面积占比", "棕色实例面积占比", "连续棕色色素覆盖占比",
        "UV-like色斑面积占比", "红棕重叠比例", "疑似纯红区域面积占比",
        "弥漫红区面积占比", "高红度区域面积占比", "最大连续红区面积占比",
        "红区连续性", "红区均匀度", "边界渐变", "弥漫红区均匀度",
    }
    engineering_names = {
        "平均红度", "P90红度", "P95红度", "P90综合色差ΔE", "P90棕色强度",
        "P90 UV-like 信号", "P90 UV-like强度", "圆度", "均匀度",
    }
    for module in payload.get("检测模块", []):
        if module.get("模块编号") not in {"01", "02", "03", "04", "05"}:
            continue
        for collection_key in ("核心指标",):
            for metric in module.get(collection_key, []) or []:
                if metric.get("display_type", "TEXT") != "TEXT":
                    continue
                name = str(metric.get("name", ""))
                if name in percentage_names or "面积占比" in name or "比例" in name:
                    metric["display_type"] = "PERCENTAGE"
                elif name in engineering_names or "强度" in name or "红度" in name or "圆度" in name:
                    metric["display_type"] = "INTENSITY_ENGINEERING"
                elif "数量" in name or "数" in name:
                    metric["display_type"] = "COUNT"


def _official_row(scoring: dict[str, Any], dimension_id: str) -> dict[str, Any]:
    return next(row for row in scoring["formal_dimension_scores"] if row.get("dimension_id") == dimension_id)


def _decorate_front5(
    payload: dict[str, Any],
    root: Path,
    candidate_root: Path,
    official: dict[str, Any],
    shadow_pigment: dict[str, Any] | None,
    preprocess_result: PreprocessResultV2,
    report_assets_dir: Path,
) -> None:
    # 01 毛孔：正式分不变；明确区分当前工程口径和医生目标口径。
    pores = _module(payload, "01")
    official_pores = _official_row(official, "visible_pores")
    pores["综合得分"] = round(float(official_pores["score"]), 6)
    pores["程度等级"] = _burden_grade(float(official_pores["score"]))
    pores["当前评分口径"] = "V0.1.1全量历史ECDF五组评分口径"
    pores["评分解释"] = official_pores.get("score_explanation")
    pore_group_names = {
        "density": "毛孔密度（当前V0.1.1评分口径）",
        "coverage": "毛孔面积覆盖（当前V0.1.1评分口径）",
        "size": "典型毛孔面积P50（当前V0.1.1评分口径）",
        "large_pores": "偏大毛孔面积P90代理（当前V0.1.1评分口径）",
        "shape": "毛孔形态不规则度（当前V0.1.1评分口径）",
    }
    for group in pores.get("评分追溯") or []:
        group_id = str(group.get("group_id") or "")
        if group_id in pore_group_names:
            group["group_name"] = pore_group_names[group_id]
    pores["医生目标GAP"] = [
        "当前P90单体面积仅作为偏大毛孔面积代理，不等于大毛孔比例。",
        "当前未生成医生目标的大毛孔面积占比和九医学分区加权总分。",
        "后续参考人群分区P75阈值建立后再形成医生目标口径。",
    ]
    pores["评估状态"] = pores["用户评估状态"] = pores["医生评估状态"] = "检测完成"
    pore_medical = _medical_v2(root, "毛孔")
    pore_core = _nested(pore_medical, "overall_metrics", "core_metrics", default={})
    pore_aux = _nested(pore_medical, "overall_metrics", "auxiliary_metrics", default={})
    pore_public_path = root / "七项检测" / "毛孔" / "毛孔量化指标.json"
    pore_public = json.loads(pore_public_path.read_text(encoding="utf-8")) if pore_public_path.is_file() else {}
    pores["核心指标"] = [
        _metric("可见毛孔数量", pore_public.get("总计"), "个", "COUNT"),
        _metric("单位面积密度", _nested(pore_core, "scope_and_burden", "feature_density_per_100k_skin_px"), "个/10万有效皮肤像素", "DENSITY"),
        _metric("毛孔面积占比", _nested(pore_core, "scope_and_burden", "feature_area_ratio"), "", "PERCENTAGE"),
        _metric("P50单体面积", _nested(pore_core, "scope_and_burden", "p50_instance_area_px"), "像素", "PIXEL"),
        _metric("P90单体面积", _nested(pore_core, "scope_and_burden", "p90_instance_area_px"), "像素", "PIXEL", "当前V0.1.1偏大毛孔面积代理。"),
        _metric("最大单体面积", _nested(pore_aux, "scope_and_burden", "max_instance_area_px"), "像素", "PIXEL"),
        _metric("P50圆度", _nested(pore_core, "morphology", "p50_circularity"), "0～1工程值", "ENGINEERING_0_1"),
        _metric("P50长宽比", _nested(pore_aux, "morphology", "p50_aspect_ratio"), "比值", "ENGINEERING_0_1"),
        _metric("低圆度毛孔比例", _nested(pore_aux, "morphology", "low_circularity_pore_ratio"), "", "PERCENTAGE"),
        _metric("拉长样毛孔比例", _nested(pore_core, "auxiliary_statistics", "elongated_pore_ratio"), "", "PERCENTAGE"),
    ]
    pores["分区指标"] = []
    pores["medical_9_region_status"] = "not_ready"
    pores["medical_summary"] = {
        "overall_finding": "基于当前V0.1.1全量历史ECDF口径完成可见毛孔问题负担评分。",
        "dominant_feature": "数量、覆盖、尺寸、偏大面积代理与形态五组共同构成当前评分。",
        "secondary_feature": "P50/P90单体面积与圆度用于描述当前实例尺度和形态。",
        "main_regions": _nested(pore_aux, "auxiliary_statistics", "distribution", "primary_concentration_region", default="当前版本尚未形成可靠量化"),
        "left_right_finding": "当前仅保留画面左右面颊工程统计，尚未形成医生九分区左右加权结论。",
        "evidence_limit": "当前P90面积为偏大毛孔代理；大毛孔比例、大毛孔面积负担和九医学分区加权结果尚未形成可靠量化。",
    }

    sample_candidate = candidate_root / root.name
    gloss_json = next((sample_candidate / "02_油脂分泌倾向").glob("*/表面油光量化指标.json"))
    gloss = json.loads(gloss_json.read_text(encoding="utf-8"))
    full = gloss["full_face"]
    porphyrin = (json.loads((root / "九项核心量化指标.json").read_text(encoding="utf-8"))["九项"]["porphyrin"])
    oil = _module(payload, "02")
    porphyrin_row = _csv_row(root / "七项检测" / "紫区" / "紫区医学量化指标_V2.csv", "标准化荧光UV紫质工程代理")
    porphyrin_p50 = _num(porphyrin_row, "核心-实例P50强度（0～1）")
    porphyrin_p90 = _num(porphyrin_row, "核心-实例P90强度（0～1）")
    if porphyrin_p50 is None or porphyrin_p90 is None or not np.isfinite([porphyrin_p50, porphyrin_p90]).all() or porphyrin_p50 > porphyrin_p90:
        raise RuntimeError(f"紫质实例强度字段无效: {root.name} P50={porphyrin_p50} P90={porphyrin_p90}")
    oil.update({
        "综合得分": None, "程度等级": None,
        "结果摘要": (
            f"表面油光覆盖{full['gloss_area_ratio']*100:.2f}%，高强度油光覆盖"
            f"{full['high_gloss_area_ratio']*100:.2f}%；检测到紫质目标{porphyrin['核心总体指标'].get('porphyrin_total', 0)}个。"
        ),
        "评估状态": "检测结果", "用户评估状态": "检测结果", "医生评估状态": "检测结果",
        "核心指标": [
            _metric("表面油光面积占比", full["gloss_area_ratio"], "", "PERCENTAGE"),
            _metric("高强度油光面积占比", full["high_gloss_area_ratio"], "", "PERCENTAGE"),
            _metric("P50油光强度", full["p50_gloss_intensity"], "0～1工程值", "INTENSITY_ENGINEERING"),
            _metric("P90油光强度", full["p90_gloss_intensity"], "0～1工程值", "INTENSITY_ENGINEERING"),
            _metric("最大连续油光区域面积占比", full["largest_gloss_component_area_ratio"], "", "PERCENTAGE"),
        ],
        "结果图": [str(gloss_json.parent / "01_表面油光检测结果图.jpg"), str(root / "七项检测/紫区/04_紫质检测结果.jpg")],
        "评分状态说明": "Surface Gloss历史参考分布尚未建立，本次只展示候选检测结果。",
    })
    oil["医生结果分组"] = [
        {"title": "表面油光：覆盖、强度与连续性", "metrics": oil["核心指标"]},
        {"title": "紫质：现有量化结果", "metrics": [
            _metric("紫质目标数量", porphyrin["核心总体指标"].get("porphyrin_total", 0), "个", "COUNT"),
            _metric("紫质单位面积密度", _num(porphyrin_row, "核心-单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素", "DENSITY"),
            _metric("紫质面积占比", _num(porphyrin_row, "核心-特征面积占比"), "", "PERCENTAGE"),
            _metric("紫质P50强度", porphyrin_p50, "0～1相对强度", "INTENSITY_ENGINEERING"),
            _metric("紫质P90强度", porphyrin_p90, "0～1相对强度", "INTENSITY_ENGINEERING"),
            _metric("紫质高强度目标比例", _num(porphyrin_row, "核心-高强度目标比例"), "", "PERCENTAGE"),
            _metric("紫质高强度区域面积占比", _num(porphyrin_row, "辅助-高强度区域面积占比"), "", "PERCENTAGE"),
        ]},
    ]
    oil["分区指标"] = [
        {"分区名称": "T区", "状态": gloss["t_zone_summary"].get("status", "检测完成"), "指标": {"油光面积占比": f"{gloss['t_zone_summary']['gloss_area_ratio'] * 100:.2f}%", "P90油光强度": gloss["t_zone_summary"]["p90_gloss_intensity"]}},
        {"分区名称": "面颊区", "状态": gloss["cheek_zone_summary"].get("status", "检测完成"), "指标": {"油光面积占比": f"{gloss['cheek_zone_summary']['gloss_area_ratio'] * 100:.2f}%", "P90油光强度": gloss["cheek_zone_summary"]["p90_gloss_intensity"]}},
    ]
    for values in gloss.get("region_metrics", {}).values():
        oil["分区指标"].append({
            "分区名称": values.get("label", "当前分区"),
            "状态": values.get("status", "检测完成"),
            "指标": {
                "油光面积占比": f"{float(values.get('gloss_area_ratio', 0)) * 100:.2f}%",
                "高强度油光面积占比": f"{float(values.get('high_gloss_area_ratio', 0)) * 100:.2f}%",
                "P90油光强度": values.get("p90_gloss_intensity"),
                "最大连续油光区域面积占比": f"{float(values.get('largest_gloss_component_area_ratio', 0)) * 100:.2f}%",
            },
        })
    oil["左右比较"] = gloss.get("left_right_summary", {})
    lr_gloss = gloss.get("left_right_summary", {})
    lr_parts = []
    for label, values in (("鼻旁", lr_gloss.get("nasal_side", {})), ("内侧面颊", lr_gloss.get("inner_cheek", {})), ("外侧面颊", lr_gloss.get("outer_cheek", {}))):
        if values:
            lr_parts.append(
                f"{label}画面左减右油光面积占比差{float(values.get('gloss_area_ratio_difference', 0)) * 100:.2f}个百分点，"
                f"P90强度差{float(values.get('p90_intensity_difference', 0)):.4f}"
            )
    oil["左右比较摘要"] = "；".join(lr_parts)
    oil["medical_summary"] = {
        "overall_finding": "表面油光与紫质结果并列展示；当前未建立油脂专用历史参考分布。",
        "dominant_feature": "以表面油光覆盖、强度、连续区域及紫质现有指标描述。",
        "secondary_feature": "T区与面颊区结果分别列示，左右差异仅保留可重复的工程比较。",
        "main_regions": "T区与面颊区见分区表。",
        "left_right_finding": "当前按分区并列展示，不输出主观一致性结论。",
        "evidence_limit": "当前版本尚未形成可靠的油脂人群评分与严重度等级。",
    }

    pigmentation = _module(payload, "03")
    official_pigment = _official_row(official, "combined_pigmentation")
    # 用户默认继续读取V0.1.1 official分。shadow只放在医生/内部对照节点。
    official_pigment_score = official_pigment.get("score")
    if isinstance(official_pigment_score, (int, float)) and np.isfinite(official_pigment_score):
        pigmentation["综合得分"] = round(float(official_pigment_score), 6)
        pigmentation["程度等级"] = _burden_grade(float(official_pigment_score))
        pigmentation["评分解释"] = official_pigment.get("score_explanation")
    else:
        # 统一线框作为空间门禁后，极少数样本可能因有效实例不足而
        # 不再满足既有评分稳定性门槛。这是合法的“暂不评分”，不能补零
        # 或恢复线外目标来伪造正式分。
        pigmentation["综合得分"] = None
        pigmentation["程度等级"] = None
        pigmentation["评分解释"] = None
    if shadow_pigment is not None:
        pigmentation["V0.1.2_shadow对照"] = {
            "score_status": "shadow_v012",
            "score": shadow_pigment["score"],
            "grade": _burden_grade(float(shadow_pigment["score"])),
            "formula": shadow_pigment["formal_formula"],
            "groups": shadow_pigment["groups"],
            "score_explanation": shadow_pigment["score_explanation"],
            "uv_internal_weight_source": "engineering_default_equal_v1",
            "promotion_status": "not_promoted",
        }
    fusion = sample_candidate / "03_综合色素/fusion/综合色素融合指标.json"
    if fusion.is_file():
        fusion_data = json.loads(fusion.read_text(encoding="utf-8"))
        pigmentation["空间融合证据"] = fusion_data
        report_regions = _build_report_visia_regions(preprocess_result)
        analysis_image = np.asarray(preprocess_result.analysis_image)
        red_brown_overlay = _report_mask_overlay(
            analysis_image, sample_candidate / "03_综合色素/fusion/05_red_brown_mixed.png",
            report_assets_dir / "03_红褐混合印记结果图.jpg", (40, 30, 230),
            regions=report_regions,
        )
        priority_overlay = _copy_report_image(
            sample_candidate / "03_综合色素/fusion/07_综合色素融合结果图.jpg",
            report_assets_dir / "04_重点色斑区域结果图.jpg",
            analysis_image=analysis_image,
            regions=report_regions,
        )
        pigmentation["结果图"] = [
            str(sample_candidate / "03_综合色素/fusion/07_综合色素融合结果图.jpg"),
            str(root / "七项检测/斑点/01_Spots斑点结果图.jpg"),
            str(root / "七项检测/棕区/02_VISIA棕色斑实例图.jpg"),
        ]
        fusion_metrics = fusion_data.get("metrics") or {}
        mixed = fusion_metrics.get("red_brown_mixed") or {}
        pure_red = fusion_metrics.get("pure_red_suspected") or {}
        visible_fusion = fusion_metrics.get("visible_pigment") or {}
        mixed_to_visible_ratio = float(mixed.get("area_px", 0)) / max(float(visible_fusion.get("area_px", 0)), 1.0)
        for metric in pigmentation.get("核心指标") or []:
            if metric.get("name") == "红棕重叠比例":
                metric.update({
                    "value": mixed.get("area_ratio", 0.0),
                    "unit": "比例",
                    "note": "Red仅用于辅助空间核查，不参与综合色素评分。",
                })
                break
        pigmentation.setdefault("核心指标", []).append(
            _metric("疑似纯红区域面积占比", pure_red.get("area_ratio", 0.0), "比例")
        )
        pigmentation["医学局限性"] = [
            "综合色素由Visible、UV与Brown三类图像证据综合分析。",
            "Red仅用于红褐混合与疑似纯红区域的辅助空间核查，不参与当前综合色素评分。",
            "纵向复测应保持设备、光照、拍摄角度及面部状态一致。",
        ]
        dominant = (fusion_data.get("priority_pigment_regions") or [{}])[0].get("region_name", "全面部")
        drivers = (
            shadow_pigment["score_explanation"]["drivers"]
            if shadow_pigment is not None
            else []
        )
        if pigmentation["综合得分"] is None:
            pigmentation["结果摘要"] = f"已完成Visible、UV与Brown综合色素检测，重点区域为{dominant}。"
            pigmentation["评估状态"] = pigmentation["用户评估状态"] = pigmentation["医生评估状态"] = "检测结果"
        elif shadow_pigment is not None and drivers:
            pigmentation["结果摘要"] = (
                f"当前正式V0.1.1分为{official_pigment_score:.2f}；V0.1.2验收shadow分为"
                f"{shadow_pigment['score']:.2f}，主要由{drivers[0]['name']}驱动，重点区域为{dominant}。"
            )
            pigmentation["评估状态"] = pigmentation["用户评估状态"] = pigmentation["医生评估状态"] = "检测完成"
        else:
            pigmentation["结果摘要"] = (
                f"当前正式V0.1.1分为{official_pigment_score:.2f}，重点区域为{dominant}。"
            )
            pigmentation["评估状态"] = pigmentation["用户评估状态"] = pigmentation["医生评估状态"] = "检测完成"
        core = {metric.get("name"): metric for metric in pigmentation.get("核心指标", [])}
        pigmentation["医生结果分组"] = [
            {"title": "A. Visible可见色斑", "metrics": [
                core.get("可见斑点数量", _metric("可见色斑结果", None)),
                core.get("可见斑点密度", _metric("可见色斑密度", None)),
                core.get("可见斑点面积占比", _metric("可见色斑面积占比", None)),
                core.get("P90综合色差ΔE", _metric("P90综合色差ΔE", None)),
                core.get("点状斑点数量", _metric("点状色斑数量", None)),
                core.get("片状斑点数量", _metric("片状色斑数量", None)),
            ]},
            {"title": "B. UV下更明显的色斑", "summary": "当前由Visible与UV空间比较生成UV下相对更突出的区域；UV正式评分仍使用全部UV四指标。", "metrics": [
                _metric("UV全部目标面积占比", fusion_metrics.get("uv_all_pigment", {}).get("area_ratio"), "", "PERCENTAGE"),
                _metric("UV下更明显区域面积占比", fusion_metrics.get("uv_enhanced_pigment", {}).get("area_ratio"), "", "PERCENTAGE"),
                _metric("UV下更明显区域数量", fusion_metrics.get("uv_enhanced_pigment", {}).get("component_count"), "个", "COUNT"),
            ]},
            {"title": "C. Brown综合色素", "metrics": [
                core.get("棕色实例数量", _metric("Brown结果", None)),
                core.get("棕色实例面积占比", _metric("Brown面积占比", None)),
                core.get("P90棕色强度", _metric("P90Brown强度", None)),
                core.get("连续棕色色素覆盖占比", _metric("连续Brown覆盖", None)),
            ]},
            {"title": "D. 红褐混合印记（辅助空间结果）", "metrics": [
                _metric("红褐混合目标数量", mixed.get("component_count"), "个", "COUNT"),
                _metric("红褐混合总面积占比", mixed.get("area_ratio"), "", "PERCENTAGE"),
                _metric("红褐混合面积占可见色斑面积比例", mixed_to_visible_ratio, "", "PERCENTAGE"),
                _metric("疑似纯红区域面积占比", pure_red.get("area_ratio"), "", "PERCENTAGE"),
            ]},
            {"title": "E. 重点色斑区域（辅助空间结果）", "metrics": [
                _metric("综合色素联合区域数量", sum(int(row.get("component_count", 0)) for row in fusion_data.get("region_distribution", []) if row.get("available")), "个", "COUNT"),
                _metric("综合色素联合区域面积占比", sum(float(row.get("feature_area_px", 0)) for row in fusion_data.get("region_distribution", []) if row.get("available")) / max(sum(float(row.get("valid_area_px", 0)) for row in fusion_data.get("region_distribution", []) if row.get("available")), 1), "", "PERCENTAGE"),
            ]},
        ]
        pigmentation["技术附录结果图"] = [path for path in (red_brown_overlay, priority_overlay, str(sample_candidate / "03_综合色素/fusion/02_uv_all_pigment.png")) if path]
        pigmentation["medical_10_region_status"] = "partial"
        pigmentation["医生目标GAP"] = list(pigmentation.get("医生目标GAP") or []) + [
            "当前空间融合已生成稳定分区结果，但尚未把眼周与颧部分离为医生目标十医学分区，故十医学分区尚未完整形成。",
            "红褐混合印记当前可报告数量与面积关系，尚未形成独立可靠的混合印记强度量化。",
        ]
        official_drivers = (
            (official_pigment.get("score_explanation") or {}).get("drivers")
            or []
        )
        primary_driver = (
            drivers[0].get("name")
            if drivers
            else (
                official_drivers[0].get("name")
                if official_drivers
                else "Visible、UV与Brown综合证据"
            )
        )
        pigmentation["medical_summary"] = {
            "overall_finding": "正式评分由Visible、UV与Brown三类证据构成；红色结果仅作空间辅助核查。",
            "dominant_feature": f"当前正式分的主要驱动为{primary_driver}。",
            "secondary_feature": "UV下更明显区域、红褐混合与重点区域均为辅助结果，不重复计分。",
            "main_regions": dominant,
            "left_right_finding": "十医学分区见分区表；画面左右仅按当前图像坐标解释。",
            "evidence_limit": (
                "V0.1.2为验收Shadow对照，尚未晋升为正式生产评分。"
                if shadow_pigment is not None
                else "机构端仅展示当前正式V0.1.1评分，不生成未标定的候选评分。"
            ),
        }
    else:
        core = {metric.get("name"): metric for metric in pigmentation.get("核心指标", [])}
        pigmentation["医生结果分组"] = [
            {"title": "A. Visible可见色斑", "metrics": [
                core.get("可见斑点数量", _metric("可见色斑结果", None)),
                core.get("可见斑点密度", _metric("可见色斑密度", None)),
                core.get("可见斑点面积占比", _metric("可见色斑面积占比", None)),
                core.get("P90综合色差ΔE", _metric("P90综合色差ΔE", None)),
            ]},
            {"title": "B. UV色斑", "summary": "当前单RGB仅展示UV工程代理结果。", "metrics": [
                core.get("UV色斑数量", _metric("UV色斑数量", None)),
                core.get("UV色斑密度", _metric("UV色斑密度", None)),
                core.get("UV色斑面积占比", _metric("UV色斑面积占比", None)),
                core.get("UV色斑P90强度", _metric("UV色斑P90强度", None)),
            ]},
            {"title": "C. Brown综合色素", "metrics": [
                core.get("棕色实例数量", _metric("Brown结果", None)),
                core.get("棕色实例面积占比", _metric("Brown面积占比", None)),
                core.get("P90棕色强度", _metric("P90Brown强度", None)),
                core.get("连续棕色色素覆盖占比", _metric("连续Brown覆盖", None)),
            ]},
            {"title": "D. 红褐混合印记（本轮未输出空间融合Mask）", "metrics": [
                _metric("红褐混合目标数量", None, "个", "COUNT"),
                _metric("红褐混合总面积占比", None, "", "PERCENTAGE"),
            ]},
            {"title": "E. 重点色斑区域（本轮未输出空间融合Mask）", "metrics": [
                _metric("综合色素联合区域数量", None, "个", "COUNT"),
                _metric("综合色素联合区域面积占比", None, "", "PERCENTAGE"),
            ]},
        ]

    redness = _module(payload, "04")
    official_redness = _official_row(official, "diffuse_redness")
    redness["综合得分"] = round(float(official_redness["score"]), 6)
    redness["程度等级"] = _burden_grade(float(official_redness["score"]))
    redness["评分解释"] = official_redness.get("score_explanation")
    redness["候选血管不变性说明"] = "本次候选血管Mask未参与V0.1.1弥漫性泛红正式评分。"
    redness["评估状态"] = redness["用户评估状态"] = redness["医生评估状态"] = "检测完成"
    red_core = json.loads((root / "九项核心量化指标.json").read_text(encoding="utf-8"))["九项"]["redness"]["核心总体指标"]
    red_drivers = (official_redness.get("score_explanation") or {}).get("drivers") or []
    dominant_red = red_drivers[0].get("name", "覆盖、强度、连续性和边界均匀性") if red_drivers else "覆盖、强度、连续性和边界均匀性"
    redness["结果摘要"] = (
        f"弥漫红区覆盖{float(red_core.get('diffuse_red_area_ratio', 0)) * 100:.2f}%，"
        f"当前正式问题负担分为{float(official_redness['score']):.2f}分；"
        f"总分由覆盖、强度、连续性及边界均匀性共同决定，当前主要评分驱动为{dominant_red}。"
    )
    redness["医生结果分组"] = [
        {"title": "泛红覆盖", "metrics": [
            _metric("弥漫红区面积占比", red_core.get("diffuse_red_area_ratio"), "", "PERCENTAGE"),
            _metric("高红度区域面积占比", red_core.get("high_red_area_ratio"), "", "PERCENTAGE"),
        ]},
        {"title": "泛红强度", "metrics": [
            _metric("平均红度", red_core.get("mean_redness"), "0～1工程值", "INTENSITY_ENGINEERING"),
            _metric("P90红度", red_core.get("p90_redness"), "0～1工程值", "INTENSITY_ENGINEERING"),
        ]},
        {"title": "连续性", "metrics": [
            _metric("最大连续红区面积占比", red_core.get("max_continuous_region_area_ratio"), "", "PERCENTAGE"),
            _metric("红区连续性", red_core.get("redness_continuity"), "0～1工程值", "ENGINEERING_0_1"),
        ]},
        {"title": "边界与均匀性", "metrics": [
            _metric("边界渐变", red_core.get("redness_boundary_gradient"), "0～1工程值", "ENGINEERING_0_1"),
            _metric("弥漫红区均匀度", red_core.get("diffuse_redness_uniformity"), "0～1工程值", "ENGINEERING_0_1"),
        ]},
    ]
    redness["排除状态说明"] = {
        "已实际参与当前正式计算": "当前有效皮肤统计Mask已排除五官、鼻孔、唇部、脸部边界及毛发等无效区域；局灶红色实例Mask已从弥漫红区连续Mask中扣除。",
        "尚未正式参与当前正式计算": "Vascular V2.1候选Mask、痤疮候选Mask未回灌至V0.1.1正式弥漫性泛红评分。",
    }
    redness["medical_summary"] = {
        "overall_finding": "当前正式评分由覆盖、强度、连续性、边界与均匀性四组构成。",
        "dominant_feature": "评分原因说明列出各评分组的加权贡献，可解释覆盖率与总分不完全同步的情况。",
        "secondary_feature": "局灶红色实例保留为辅助观察，不单独改变正式弥漫性泛红分。",
        "main_regions": "中央面部集中度与画面左右面颊差异见核心指标和分区结果。",
        "left_right_finding": "画面左右差异按图像坐标计算。",
        "evidence_limit": "候选血管结构尚未参与正式扣除链，正式结果仍保留该项排除缺口。",
    }

    vascular_path = sample_candidate / "05_血管样结构/血管样结构量化指标.json"
    vascular_data = json.loads(vascular_path.read_text(encoding="utf-8"))
    vascular = _module(payload, "05")
    vascular_region_labels = {
        "forehead": "额部", "nose_alar_nasal_side": "鼻翼及鼻侧",
        "left_periocular_zygoma": "画面左眼周/颧区", "right_periocular_zygoma": "画面右眼周/颧区",
        "left_cheek": "画面左面颊", "right_cheek": "画面右面颊",
        "left_jaw": "画面左下颌", "right_jaw": "画面右下颌",
    }
    vascular.update({
        "综合得分": None, "程度等级": None,
        "评估状态": "候选检测结果", "用户评估状态": "血管样结构观察结果", "医生评估状态": "候选检测结果",
        "数据来源": ["VascularStructure-V2.1候选算法"],
        "结果摘要": _vascular_result_summary(vascular_data),
        "核心指标": [
            _metric("血管样候选结构数量", vascular_data["vascular_count"], "个", "COUNT"),
            _metric("候选线状结构总长度", vascular_data["vascular_total_length_px"], "像素", "PIXEL"),
            _metric("候选结构面积占比", vascular_data["vascular_area_ratio"], "", "PERCENTAGE"),
            _metric("P90候选宽度", vascular_data["p90_width_px"], "像素", "PIXEL"),
            _metric("P90局部红度工程响应", vascular_data["p90_redness"], "工程响应", "INTENSITY_ENGINEERING"),
            _metric("候选分支点数量", vascular_data["branch_point_count"], "个", "COUNT"),
        ],
        "分区指标": [
            vascular_region_row(key, values)
            for key, values in vascular_data.get("region_distribution", {}).items()
        ],
        "结果图": [
            str(sample_candidate / "05_血管样结构/05_血管样结构检测结果图.jpg"),
            str(sample_candidate / "05_血管样结构/02_血管样结构热力图.jpg"),
            str(sample_candidate / "05_血管样结构/04_血管骨架Mask.png"),
        ],
        "评分状态说明": "候选算法已接入；专用历史参考分布尚未建立，本次不生成0～100分和程度等级。",
        "候选算法追溯": vascular_data,
    })
    vascular["医生结果分组"] = [
        {"title": "A. 候选数量与长度", "metrics": [
            _metric("血管样候选结构数量", vascular_data.get("vascular_count"), "个", "COUNT"),
            _metric("候选线长度密度", vascular_data.get("vascular_line_density"), "像素/万有效皮肤像素", "DENSITY"),
        ]},
        {"title": "B. 候选覆盖与宽度", "metrics": [
            _metric("候选结构面积占比", vascular_data.get("vascular_area_ratio"), "", "PERCENTAGE"),
            _metric("P50候选宽度", vascular_data.get("p50_width_px"), "像素", "PIXEL"),
            _metric("P90候选宽度", vascular_data.get("p90_width_px"), "像素", "PIXEL"),
        ]},
        {"title": "C. 局部红度工程响应", "metrics": [
            _metric("P50局部红度工程响应", vascular_data.get("p50_redness"), "工程响应", "INTENSITY_ENGINEERING"),
            _metric("P90局部红度工程响应", vascular_data.get("p90_redness"), "工程响应", "INTENSITY_ENGINEERING"),
        ]},
        {"title": "D. 候选分支与网络形态", "metrics": [
            _metric("候选分支点数量", vascular_data.get("branch_point_count"), "个", "COUNT"),
            _metric("候选分支点密度", vascular_data.get("branch_point_density"), "个/万有效皮肤像素", "DENSITY"),
            _metric("候选网络比例", vascular_data.get("network_ratio"), "", "PERCENTAGE"),
        ]},
    ]
    vascular_lr = vascular_data.get("left_right_summary", {})
    dominant_side = {"left": "画面左侧", "right": "画面右侧"}.get(vascular_lr.get("dominant_side"), "未形成明确侧别")
    vascular["左右比较摘要"] = (
        f"画面左侧候选{int(_nested(vascular_lr, 'left', 'count', default=0))}个、总长度"
        f"{float(_nested(vascular_lr, 'left', 'total_length_px', default=0)):.0f}像素；"
        f"画面右侧候选{int(_nested(vascular_lr, 'right', 'count', default=0))}个、总长度"
        f"{float(_nested(vascular_lr, 'right', 'total_length_px', default=0)):.0f}像素；候选长度主要分布于{dominant_side}。"
    )
    vascular["技术附录结果图"] = [
        str(sample_candidate / "05_血管样结构/02_血管样结构热力图.jpg"),
        str(sample_candidate / "05_血管样结构/03_血管样结构Mask.png"),
        str(sample_candidate / "05_血管样结构/04_血管骨架Mask.png"),
    ]
    vascular["medical_summary"] = {
        "overall_finding": "本章节展示血管样候选结构观察结果，不生成正式严重度评分。",
        "dominant_feature": "候选数量、长度、覆盖、宽度、局部红度工程响应及分支网络结果分别列示。",
        "secondary_feature": "分区与画面左右差异来源于同一候选实例集合。",
        "main_regions": "候选分布见分区表。",
        "left_right_finding": "按画面左右候选总长度汇总，见候选算法追溯。",
        "evidence_limit": "当前版本尚未形成可靠的血管样结构人群参考分布与正式评分。",
    }


def generate_front5_v012_reports(
    result_dir: str | Path,
    official_profile_path: str | Path,
    shadow_profile_path: str | Path,
    candidate_root: str | Path,
    output_dir: str | Path,
    *,
    report_id: str,
    subject_id: str,
    collection_date: date | str | None = None,
    wrinkle_report_images: dict[str, str | Path] | None = None,
    controlled_evidence: ControlledEvidencePaths | None = None,
    formal_images: dict[str, Path] | None = None,
    complete_document: dict[str, Any] | None = None,
) -> dict[str, str]:
    root = Path(result_dir).resolve()
    _load_successful_result(root)
    source = _source_image(root)
    official_path = Path(official_profile_path).resolve()
    shadow_path = Path(shadow_profile_path).resolve()
    official_profile = _load_profile(official_path, OFFICIAL_VERSION)
    shadow_profile = _load_profile(shadow_path, SHADOW_VERSION, "shadow")
    preprocessor = ImagePreprocessor()
    try:
        gate = evaluate_image_path(source, preprocessor=preprocessor)
        if gate.get("status") != "PASS":
            raise RuntimeError(f"输入未通过评分门禁: {root.name} {gate}")
        preprocess_result = preprocessor.last_result
        if preprocess_result is None:
            raise RuntimeError(f"输入缺少可复用的预处理结果: {root.name}")
    finally:
        preprocessor.close()
    features, use_shadow = _report_features(root, complete_document)
    official = score_observation(
        {"status": "success", "input_quality_gate": gate, "quality": {"status": "PASS"}, "features": features},
        official_profile["references"], normalization_profile_sha256=_sha(official_path),
        scoring_profile_version=OFFICIAL_VERSION,
    )
    official = attach_score_explanations(official)
    shadow_pigment = (
        score_combined_pigmentation_shadow(features, shadow_profile)
        if use_shadow
        else None
    )
    bundle = {
        "报告信息": {"报告编号": report_id, "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "受检者信息": {"姓名或编号": subject_id},
        "采集与质量信息": {
            "源图": str(source), "评分门禁": gate,
            "成像说明": "标准化白光图像与UV图像；纵向复测应保持采集条件一致。",
        },
        "任务状态": {},
        "结果来源": {
            "dermavision_result_dir": str(root),
            "acne_summary_json": str(root / "痤疮/痤疮量化指标.json"),
            "wrinkle_summary_json": str(root / "皱纹/皱纹量化指标.json"),
        },
        "评分V011": official,
    }
    payload = build_report_payload(bundle)
    payload["报告数据版本"] = "report_payload_front5_v012_validation"
    payload["评分配置摘要"] = {
        "official_profile_version": OFFICIAL_VERSION,
        "official_profile_sha256": _sha(official_path),
        "shadow_profile_version": SHADOW_VERSION,
        "shadow_profile_sha256": _sha(shadow_path),
        "shadow_promotion_status": (
            "not_promoted" if use_shadow else "not_available_for_institution"
        ),
        "uv_internal_weight_source": "engineering_default_equal_v1",
        "registry_sha256": official.get("registry_sha256"),
    }
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    _decorate_front5(
        payload,
        root,
        Path(candidate_root),
        official,
        shadow_pigment,
        preprocess_result,
        target / "报告层派生图",
    )
    _apply_wrinkle_report_images(payload, wrinkle_report_images)
    _apply_front5_display_metadata(payload)
    structured = target / f"AISIA_面部多指标检测报告_{report_id}_结构化数据.json"
    trace = target / f"AISIA_面部多指标检测报告_{report_id}_评分追溯.json"
    user = target / f"AISIA_面部多指标检测报告_{report_id}_用户精简版.docx"
    doctor = target / f"AISIA_面部多指标检测报告_{report_id}_医生详细版.docx"
    trace.write_text(json.dumps({
        "official_v011": official,
        "shadow_v012_combined_pigmentation": shadow_pigment,
        "invariance": {"red_weight": 0.0, "candidate_vascular_changes_official_redness": False},
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    formal_payload = build_formal_report_view(payload, collected_on=collection_date)
    structured_payload = payload
    if complete_document is not None:
        apply_controlled_metric_layout(formal_payload, complete_document)
        structured_payload = formal_payload
    if formal_images is not None:
        bind_twelve_report_images(formal_payload, formal_images, source)
        structured_payload = formal_payload
    if controlled_evidence is not None:
        technical = [
            Path(value)
            for value in _module(payload, "03").get("技术附录结果图", [])
        ]
        approved = ControlledEvidencePaths(
            red_brown_mixed=next(
                path for path in technical if "红褐混合" in path.name
            ),
            priority_pigment=next(
                path for path in technical if "重点色斑" in path.name
            ),
            surface_irregularity=controlled_evidence.surface_irregularity,
            pigmentation_metrics=controlled_evidence.pigmentation_metrics,
        )
        bind_controlled_evidence(formal_payload, approved)
        structured_payload = formal_payload
    structured.write_text(
        json.dumps(
            structured_payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    render_user_docx(formal_payload, DEFAULT_TEMPLATE, user)
    render_doctor_docx(formal_payload, DEFAULT_TEMPLATE, doctor)
    return {"user_docx": str(user), "doctor_docx": str(doctor), "structured_json": str(structured), "trace_json": str(trace)}


__all__ = ["generate_front5_v012_reports"]
