from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REGION_NAME_CN = {
    "forehead": "额头",
    "subject_left_cheek": "左脸颊",
    "subject_right_cheek": "右脸颊",
    "nose": "鼻部",
    "chin": "下颌",
    "subject_left_jaw": "左下颌/下颚侧",
    "subject_right_jaw": "右下颌/下颚侧",
    "unassigned": "未分配区域",
}

SEVERITY_LABELS = {
    1: "较轻",
    2: "偏轻",
    3: "中等",
    4: "较重",
}

GRADING_REASON_MAP = {
    "input_not_full_face": "当前图片不是完整正脸，因此不做整脸严重程度评估。",
    "possible_blur": "图片可能存在模糊或质量不足，无法可靠进行整脸严重程度评估。",
    "disabled_by_worker_config": "当前运行未启用严重程度评估。",
    "input_not_eligible_for_grading": "当前图片不满足整脸评级条件。",
    "checkpoint_not_configured": "严重程度模型未正确配置，因此无法评估。",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any, default: str = "未知") -> str:
    number = _safe_float(value)
    return f"{number:.4f}" if number is not None else default


def _fmt_int_like(value: Any, default: str = "未知") -> str:
    number = _safe_float(value)
    return str(int(round(number))) if number is not None else default


def _join_reason(reason: Any) -> str:
    if reason is None:
        return ""
    if isinstance(reason, str):
        return reason
    if isinstance(reason, (list, tuple)):
        return "，".join(str(item) for item in reason if str(item))
    return str(reason)


def _region_distribution(region_counts: dict[str, Any]) -> tuple[str, str]:
    if not region_counts:
        return "未见可用于区域统计的疑似病灶坐标映射。", "未分配区域"
    items = [(name, int(count)) for name, count in region_counts.items() if int(count or 0) > 0]
    if not items:
        return "疑似病灶未映射到具体区域。", "未分配区域"
    ordered = sorted(items, key=lambda item: item[1], reverse=True)
    top_name, top_count = ordered[0]
    details = "，".join(f"{REGION_NAME_CN.get(name, name)} {count} 个" for name, count in ordered)
    return (
        f"主要分布在 {REGION_NAME_CN.get(top_name, top_name)}（{top_count} 个）。区域明细：{details}。",
        REGION_NAME_CN.get(top_name, top_name),
    )


def _top_detections(detections: list[dict[str, Any]], csv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [
        item
        for item in detections
        if isinstance(item, dict) and _safe_float(item.get("confidence")) is not None
    ]
    ranked = sorted(ranked, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)[:3]
    if ranked:
        return ranked

    fallback: list[dict[str, Any]] = []
    for row in csv_rows:
        conf = _safe_float(row.get("confidence"))
        if conf is None:
            continue
        fallback.append(
            {
                "confidence": conf,
                "region": row.get("region", "unassigned"),
                "source": row.get("source", "yolo"),
            }
        )
        if len(fallback) >= 3:
            break
    return fallback


def _grading_lines(grading: dict[str, Any]) -> tuple[list[str], str]:
    if not grading:
        return [
            "本次未找到整脸严重程度评估结果，因此仅展示逐颗圈选结果。"
        ], "missing"

    status = str(grading.get("status", "unknown"))
    if status == "ok":
        level = int(_safe_float(grading.get("severity_level")) or 0)
        level_label = SEVERITY_LABELS.get(level, "未知")
        count_text = _fmt_int_like(grading.get("predicted_count"))
        probs = grading.get("severity_probabilities", [])
        if isinstance(probs, Sequence) and probs:
            prob_text = "，".join(
                f"{idx}级 {SEVERITY_LABELS.get(idx, '未知')} {_fmt_num(prob, '-')}"
                for idx, prob in enumerate(probs, start=1)
            )
        else:
            prob_text = "未输出"
        return [
            "此阶段使用 Acne-LDS，对整张完整正脸进行严重程度估计，不负责在图片上圈出单颗病灶。",
            f"本次评级结果：{level}级（{level_label}）。",
            "等级说明：1级较轻，2级偏轻，3级中等，4级较重。",
            "说明：数字越大，表示整脸痤疮严重程度越高。",
            f"预测整脸疑似痤疮数量约：{count_text} 个。",
            f"各等级概率参考：{prob_text}。",
        ], "ok"

    reason = _join_reason(grading.get("reason"))
    reason_text = GRADING_REASON_MAP.get(reason, reason or "未提供原因。")
    if not reason_text.endswith("。"):
        reason_text += "。"
    return [
        "此阶段使用 Acne-LDS，对整张完整正脸进行严重程度估计，不负责在图片上圈出单颗病灶。",
        f"本次未执行整脸严重程度评估：{reason_text}",
        "因此当前报告只提供逐颗疑似病灶圈选结果。",
    ], status


def _detection_lines(compact: dict[str, Any], csv_rows: list[dict[str, Any]]) -> tuple[list[str], str]:
    detection = compact.get("detection", {}) or {}
    detections = detection.get("detections", [])
    if not isinstance(detections, list):
        detections = []

    status = str(detection.get("status", "unknown"))
    raw_count = _fmt_int_like(detection.get("raw_count"))
    filtered_count = _fmt_int_like(detection.get("filtered_count", detection.get("count")))
    distribution_text, top_region = _region_distribution(detection.get("region_counts", {}) or {})

    lines = [
        "此阶段使用我们训练的检测模型，在图片上圈出疑似痤疮病灶候选。",
        f"圈选状态：{status}。",
    ]

    if status == "skipped":
        reason = _join_reason(detection.get("reason")) or "未提供原因"
        lines.append(f"本次未执行逐颗圈选：{reason}。")
        return lines, top_region

    if str(filtered_count) == "0":
        lines.extend(
            [
                f"模型原始候选数：{raw_count} 个；过滤后保留：0 个。",
                "本次未圈出明确的疑似痤疮病灶；这不等于完全没有痤疮，也可能与照片角度、清晰度或当前阈值有关。",
            ]
        )
        return lines, top_region

    lines.extend(
        [
            f"模型原始候选数：{raw_count} 个；过滤后保留：{filtered_count} 个。",
            distribution_text,
        ]
    )

    top_candidates = _top_detections(detections, csv_rows)
    if top_candidates:
        lines.append("高置信候选参考：")
        for idx, item in enumerate(top_candidates, start=1):
            conf = _fmt_num(item.get("confidence"), "未知")
            region = REGION_NAME_CN.get(str(item.get("region", "unassigned")), str(item.get("region", "unassigned")))
            source = item.get("source", "yolo")
            lines.append(f"{idx}. 置信度 {conf}，位置 {region}，来源 {source}。")
    else:
        lines.append("本次没有可展示的高置信候选明细。")

    return lines, top_region


def build_user_report_from_payloads(
    *,
    compact: dict[str, Any],
    grading: dict[str, Any],
    csv_rows: list[dict[str, Any]],
) -> str:
    image_name = Path(str(compact.get("input_image", ""))).name or "unknown"
    input_mode = compact.get("input_mode", "unknown")
    detection_scope = compact.get("detection_scope", "unknown")
    elapsed_text = _fmt_num(compact.get("elapsed_seconds"), "未知")
    profile = compact.get("model_profile", {}) or {}

    grading_lines, grading_status = _grading_lines(grading)
    detection_lines, top_region = _detection_lines(compact, csv_rows)
    acne_presence = ((compact.get("detection", {}) or {}).get("acne_presence", {}) or {})
    acne_presence_message = str(acne_presence.get("message", "") or "")
    acne_presence_status = str(acne_presence.get("status", "unknown") or "unknown")

    filtered_text = _fmt_int_like((compact.get("detection", {}) or {}).get("filtered_count", (compact.get("detection", {}) or {}).get("count")))
    severity_level = _fmt_int_like(grading.get("severity_level"), "未评估")
    severity_label = SEVERITY_LABELS.get(int(_safe_float(grading.get("severity_level")) or 0), "未评估")

    summary_line = (
        "本次已完成痤疮图像分析。系统分别给出了整脸严重程度参考，以及图中疑似痤疮病灶的圈选结果。"
    )
    if acne_presence_status == "not_detected":
        conclusion = (
            "总结：本次未检测到痤疮。"
            "当前未圈出明确的疑似痤疮候选，但这不等于完全没有痤疮，仍可能受到照片角度、清晰度或当前阈值影响。"
        )
    elif grading_status == "ok":
        conclusion = (
            f"总结：本图当前圈出 {filtered_text} 个疑似痤疮病灶候选，"
            f"主要关注区域为 {top_region}；整脸严重程度参考为 {severity_level}级（{severity_label}）。"
        )
    elif grading_status == "skipped":
        conclusion = (
            f"总结：本图当前圈出 {filtered_text} 个疑似痤疮病灶候选；"
            "由于不满足整脸评级条件，本次未提供整脸严重程度。"
        )
    else:
        conclusion = f"总结：本图当前圈出 {filtered_text} 个疑似痤疮病灶候选。"

    lines = [
        "【用户版痤疮检测报告】",
        f"图片名称：{image_name}",
        f"输入模式：{input_mode}",
        f"检测范围：{detection_scope}",
        f"运行耗时：{elapsed_text} 秒",
        f"检测模型配置：{profile.get('name', 'unknown')} / {Path(str(profile.get('weights', 'unknown'))).name}",
        f"痤疮检出提醒：{acne_presence_message or '未知'}",
        "",
        summary_line,
        "",
        "【阶段一：全脸严重程度评估】",
    ]
    lines.extend(grading_lines)
    lines.extend(
        [
            "",
            "【阶段二：逐颗疑似病灶圈选】",
        ]
    )
    lines.extend(detection_lines)
    lines.extend(
        [
            "",
            conclusion,
            "",
            "风险提示：本结果仅供皮肤状态观察与研发辅助参考，不作为医学诊断或治疗依据。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_user_report_from_paths(compact_path: Path, grading_path: Path, csv_path: Path) -> str:
    compact = _load_json(compact_path)
    grading = _load_json(grading_path)
    csv_rows = _read_csv_rows(csv_path)
    return build_user_report_from_payloads(compact=compact, grading=grading, csv_rows=csv_rows)
