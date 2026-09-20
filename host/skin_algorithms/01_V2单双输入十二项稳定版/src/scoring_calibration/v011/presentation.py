from __future__ import annotations

from typing import Any

from .registry import DISPLAY_ONLY_DIMENSIONS


DISPLAY_NAMES = {
    "oiliness_tendency": "油脂分泌倾向",
    "vascular_like_structures": "血管样结构",
    "follicular_inflammation_acne": "毛囊炎症及痤疮样活动",
    "dry_fine_lines": "干燥性细纹",
    "stable_linear_wrinkles": "稳定性线性皱纹",
    "structural_grooves": "结构性沟纹",
    "contour_firmness_decline": "面部轮廓紧致度下降",
}


def report_scoring_payload(score_result: dict[str, Any], raw_results: dict[str, Any]) -> dict[str, Any]:
    """供用户/医生报告共用；不生成11维总分或伪造七项正式分。"""
    formal = [
        {
            "dimension_id": item["dimension_id"],
            "dimension_name": item["dimension_name"],
            "score": item["score"],
            "status": item["status"],
            "groups": item["groups"],
        }
        for item in score_result.get("formal_dimension_scores", [])
    ]
    display_only = [
        {
            "dimension_id": dimension_id,
            "dimension_name": DISPLAY_NAMES[dimension_id],
            "score": None,
            "result": raw_results.get(dimension_id),
        }
        for dimension_id in DISPLAY_ONLY_DIMENSIONS
    ]
    return {
        "scoring_profile_version": score_result.get("scoring_profile_version"),
        "score_direction": score_result.get("score_direction"),
        "overall_score": None,
        "overall_grade": None,
        "formal_scores": formal,
        "display_only_results": display_only,
        "quality_gate": score_result.get("quality_gate"),
        "registry_sha256": score_result.get("registry_sha256"),
        "normalization_profile_sha256": score_result.get("normalization_profile_sha256"),
        "trace_sha256": score_result.get("trace_sha256"),
    }
