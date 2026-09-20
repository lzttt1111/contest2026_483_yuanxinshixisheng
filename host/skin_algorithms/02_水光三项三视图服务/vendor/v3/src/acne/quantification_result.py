from __future__ import annotations

from typing import Any


SEVERITY_LABELS = {
    1: "较轻",
    2: "偏轻",
    3: "中等",
    4: "较重",
}

SEVERITY_SCALE_NOTE = (
    "痤疮严重程度共4级：1级较轻，2级偏轻，3级中等，4级较重；"
    "等级数字越大表示越严重。"
)

GRADING_REASON_CN = {
    "input_not_full_face": "当前图片不满足完整正脸评级条件，未进行严重程度评估。",
    "possible_blur": "当前图片可能模糊或质量不足，未进行严重程度评估。",
    "disabled_by_worker_config": "当前运行未启用严重程度评估。",
    "input_not_eligible_for_grading": "当前图片不满足完整正脸评级条件，未进行严重程度评估。",
    "checkpoint_not_configured": "严重程度模型未正确配置，未进行严重程度评估。",
}


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _build_circle_count(detection: dict[str, Any]) -> int | None:
    if str(detection.get("status", "unknown")) != "ok":
        return None
    count = _integer_or_none(detection.get("count", detection.get("filtered_count")))
    if count is None or count < 0:
        return None
    return count


def _build_severity(grading: dict[str, Any]) -> dict[str, int | str | None]:
    if str(grading.get("status", "unknown")) == "ok":
        level = _integer_or_none(grading.get("severity_level"))
        if level in SEVERITY_LABELS:
            return {
                "等级": level,
                "注释": f"本次为{level}级（{SEVERITY_LABELS[level]}）。{SEVERITY_SCALE_NOTE}",
            }
        return {
            "等级": None,
            "注释": f"严重程度评估结果无效，未返回等级。{SEVERITY_SCALE_NOTE}",
        }

    reason = str(grading.get("reason", "") or "")
    reason_text = GRADING_REASON_CN.get(reason, "当前图片未完成严重程度评估。")
    return {
        "等级": None,
        "注释": f"{reason_text}{SEVERITY_SCALE_NOTE}",
    }


def build_quantification_result(summary: dict[str, Any]) -> dict[str, Any]:
    """构建供前端直接读取的两项中文量化结果。"""

    detection = summary.get("detection", {}) or {}
    grading = summary.get("grading", {}) or {}
    return {
        "疑似痤疮圈选数量": _build_circle_count(detection),
        "痤疮严重程度等级": _build_severity(grading),
    }
