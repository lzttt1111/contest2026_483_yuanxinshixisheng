from __future__ import annotations

import copy
from typing import Any


PORE_GROUP_NAMES = {
    "density": "毛孔密度（当前V0.1.1评分口径）",
    "coverage": "毛孔面积覆盖（当前V0.1.1评分口径）",
    "size": "典型毛孔面积P50（当前V0.1.1评分口径）",
    "large_pores": "偏大毛孔面积P90代理（当前V0.1.1评分口径）",
    "shape": "毛孔形态不规则度（当前V0.1.1评分口径）",
}


def _evidence_summary(group: dict[str, Any]) -> str:
    rows: list[str] = []
    for metric in group.get("metrics", []):
        raw = metric.get("raw_value")
        unit = metric.get("unit") or ""
        rows.append(f"{metric.get('metric_id')}={raw}{unit}")
    return "；".join(rows)


def _contrastive_text(dimension_id: str, drivers: list[dict[str, Any]]) -> str:
    if not drivers:
        return "当前没有足够的评分组用于解释。"
    high = drivers[0]
    low = drivers[-1]
    if dimension_id == "diffuse_redness":
        return (
            f"最终分同时考虑覆盖、强度、连续性和边界均匀性；"
            f"其中{high['name']}贡献最高，{low['name']}贡献相对最低，"
            "因此不能仅根据红区面积判断最终程度。"
        )
    if dimension_id == "combined_pigmentation":
        low_name = str(low["name"])
        low_phrase = f"{low_name}相对较轻" if low_name.endswith("负担") else f"{low_name}负担相对较轻"
        return (
            f"综合色素由Visible、UV和Brown三类信号共同形成；"
            f"本次主要由{high['name']}驱动，{low_phrase}。"
        )
    if dimension_id == "visible_pores":
        return (
            f"当前V0.1.1毛孔分主要由{high['name']}驱动；"
            "P90面积仅为偏大毛孔代理，不等同于医生目标口径中的大毛孔比例或大毛孔面积占比。"
        )
    return f"本次主要由{high['name']}驱动，{low['name']}贡献相对最低。"


def build_score_explanation(dimension: dict[str, Any]) -> dict[str, Any] | None:
    score = dimension.get("score")
    if not isinstance(score, (int, float)):
        return None
    dimension_id = str(dimension.get("dimension_id") or "")
    drivers: list[dict[str, Any]] = []
    for group in dimension.get("groups", []):
        group_score = group.get("score")
        weight = group.get("group_weight")
        if not isinstance(group_score, (int, float)) or not isinstance(weight, (int, float)):
            continue
        name = str(group.get("group_name") or group.get("group_id") or "")
        if dimension_id == "visible_pores":
            name = PORE_GROUP_NAMES.get(str(group.get("group_id")), name)
        drivers.append({
            "id": group.get("group_id"),
            "name": name,
            "score": float(group_score),
            "weight": float(weight),
            "weighted_contribution": float(group_score) * float(weight),
            "evidence_summary": _evidence_summary(group),
        })
    drivers.sort(key=lambda row: row["weighted_contribution"], reverse=True)
    contribution = sum(row["weighted_contribution"] for row in drivers)
    if abs(contribution - float(score)) > 1e-6:
        raise ValueError(
            f"评分组贡献与维度分不一致: {dimension_id} {contribution} != {score}"
        )
    return {
        "dominant_driver": drivers[0] if drivers else None,
        "secondary_driver": drivers[1] if len(drivers) > 1 else None,
        "lowest_driver": drivers[-1] if drivers else None,
        "drivers": drivers,
        "contrastive_explanation": _contrastive_text(dimension_id, drivers),
        "summary_text": (
            f"综合问题负担分为{float(score):.2f}分。"
            + _contrastive_text(dimension_id, drivers)
        ),
    }


def attach_score_explanations(scoring: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(scoring)
    for dimension in output.get("formal_dimension_scores", []):
        explanation = build_score_explanation(dimension)
        if explanation is not None:
            dimension["score_explanation"] = explanation
            dimension["score_status"] = "official_v011_full_reference"
            if dimension.get("dimension_id") == "visible_pores":
                dimension["doctor_alignment"] = {
                    "status": "partial",
                    "current_scoring_scope": "V0.1.1全脸五组工程口径",
                    "known_gaps": [
                        "毛孔大小组当前只使用P50面积",
                        "P90面积仅为偏大毛孔代理，不是大毛孔比例或大毛孔面积占比",
                        "尚未建立九医学分区独立ECDF及分区加权汇总",
                    ],
                }
    return output
