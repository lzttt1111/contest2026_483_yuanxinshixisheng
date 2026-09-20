"""Bind every compact CSV metric/region/value to its public JSON truth."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from pydantic import JsonValue
from typing_extensions import assert_never


REGIONS = {
    "总计": "total",
    "额头": "forehead",
    "左脸颊": "left_cheek",
    "右脸颊": "right_cheek",
    "鼻部": "nose",
    "下巴": "chin",
}
CLINICAL_FIELDS = {
    "检测范围": "analysis_region",
    "评估状态": "evaluation_status",
    "分区面积（像素）": "valid_skin_area_px",
    "有效皮肤面积（像素）": "valid_skin_area_px",
    "纹路线段数量（段）": "wrinkle_segment_count",
    "皱纹中心线像素（像素）": "wrinkle_centerline_length_px",
    "皱纹面积占比": "wrinkle_area_ratio",
    "单位面积密度（皱纹像素/1万分区像素）": "wrinkle_pixel_density_per_10k_region_px",
    "平均线段长度（像素）": "mean_segment_length_px",
    "最大线段长度（像素）": "max_segment_length_px",
    "相对响应（0～100）": "relative_response_0_100",
    "占全部皱纹像素比例": "share_of_total_wrinkle_pixels",
    "疑似痤疮候选数量（个）": "suspected_acne_candidate_count",
    "单位面积密度（个/10万有效皮肤像素）": "feature_density_per_100k_skin_px",
    "候选框总面积（像素）": "candidate_box_total_area_px",
    "候选框面积占比": "candidate_box_area_ratio",
    "P50候选框面积（像素）": "p50_candidate_box_area_px",
    "P90候选框面积（像素）": "p90_candidate_box_area_px",
    "平均置信度（0～1）": "mean_confidence",
    "P90置信度（0～1）": "p90_confidence",
    "Acne-LDS评估状态": "acne_lds_evaluation_status",
    "Acne-LDS等级": "acne_lds_level",
    "主要集中区域": "primary_concentration_region",
}


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _same(csv_value: str, truth: JsonValue | None) -> bool:
    if isinstance(truth, bool) or isinstance(truth, (dict, list)):
        return False
    text = csv_value.strip()
    try:
        return Decimal(text) == Decimal(str(truth))
    except (InvalidOperation, ValueError):
        return text == str(truth)


def _all_match(values: dict[str, str], truth: dict[str, JsonValue]) -> bool:
    return all(name in truth and _same(value, truth[name]) for name, value in values.items())


def _clinical_row(public: dict[str, JsonValue], region: str) -> dict[str, JsonValue]:
    candidates: list[JsonValue] = []
    overall = public.get("overall_metrics")
    if isinstance(overall, dict):
        candidates.append(overall)
    regions = public.get("region_metrics")
    if isinstance(regions, list):
        candidates.extend(regions)
    for candidate in candidates:
        leaves = _leaf_values(candidate)
        if any(_same(region, value) for value in leaves.get("analysis_region", [])):
            return candidate if isinstance(candidate, dict) else {}
    return {}


def _leaf_values(value: JsonValue | None) -> dict[str, list[JsonValue]]:
    rows: dict[str, list[JsonValue]] = {}

    def visit(current: JsonValue | None) -> None:
        match current:
            case dict():
                for key, child in current.items():
                    if not isinstance(child, (dict, list)):
                        rows.setdefault(key, []).append(child)
                    visit(child)
            case list():
                for child in current:
                    visit(child)
            case None | bool() | int() | float() | str():
                pass
            case unreachable:
                assert_never(unreachable)

    visit(value)
    return rows


def _clinical_passed(rows: list[dict[str, str]], public: dict[str, JsonValue]) -> bool:
    for row in rows:
        region = row.get("检测范围", "")
        leaves = _leaf_values(_clinical_row(public, region))
        for heading, value in row.items():
            if not value:
                continue
            field = CLINICAL_FIELDS.get(heading)
            if field is None or not any(_same(value, item) for item in leaves.get(field, [])):
                return False
    return True


def _legacy_count_passed(
    item_id: str,
    row: dict[str, str],
    public: dict[str, JsonValue],
) -> bool:
    if item_id == "redness":
        total = public.get("red_feature_count")
        distribution = _mapping(public.get("red_feature_region_distribution"))
    else:
        total = public.get("spot_count")
        distribution = _mapping(public.get("region_distribution"))
    for label, region in REGIONS.items():
        expected = total if region == "total" else _mapping(distribution.get(region)).get("count")
        if label not in row or not _same(row[label], expected):
            return False
    return True


def compact_truth_passed(
    item_id: str,
    rows: list[dict[str, str]],
    public: dict[str, JsonValue],
) -> bool:
    if not rows:
        return False
    if item_id in {"wrinkle", "acne"}:
        return _clinical_passed(rows, public)
    if item_id in {"redness", "spots"}:
        return len(rows) == 1 and _legacy_count_passed(item_id, rows[0], public)
    if item_id in {"brown", "texture", "pores"}:
        return len(rows) == 1 and _all_match(rows[0], public)
    if item_id in {"uv_spots", "porphyrin"}:
        prefix = "uv_spots" if item_id == "uv_spots" else "porphyrin"
        project = "紫外线色斑" if item_id == "uv_spots" else "紫质"
        if len(rows) != 1 or rows[0].get("检测项目") != project:
            return False
        return all(
            _same(rows[0].get(label, ""), public.get(f"{prefix}_{region}"))
            for label, region in REGIONS.items()
        )
    if item_id in {"surface_gloss", "vascular"}:
        return all(
            isinstance(metric := row.get("指标"), str)
            and _all_match(
                {label: row.get(label, "") for label in REGIONS},
                _mapping(public.get(metric)),
            )
            for row in rows
        )
    if item_id == "contour_firmness" or all("值" in row for row in rows):
        value_key = "数值" if item_id == "contour_firmness" else "值"
        return all(
            isinstance(metric := row.get("指标"), str)
            and _same(row.get(value_key, ""), public.get(metric))
            for row in rows
        )
    return False
