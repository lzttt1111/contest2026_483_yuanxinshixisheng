from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.acne.quantification_result import build_quantification_result
from src.acne.user_detection_report import build_user_report_from_paths
from src.acne.clinical_quantification import (
    try_merge_medical_v2_report,
    write_report as write_clinical_quantification,
)


DISPLAY_FILENAMES = {
    "original_image": "01_原始照片.jpg",
    "overlay_original": "02_痤疮圈选结果_原图.jpg",
    "overlay_standardized": "03_痤疮圈选结果_标准化人脸.jpg",
    "detections_csv": "04_检测结果表.csv",
    "summary_json": "05_检测结果摘要.json",
    "grading_json": "06_严重程度评估.json",
    "readme_md": "07_输出说明.md",
    "user_report_txt": "08_用户检测报告.txt",
    "quantification_csv": "痤疮量化指标.csv",
    "quantification_json": "痤疮量化指标.json",
    "detections_detail_csv": "04_检测结果表.csv",
    "summary_detail_json": "05_检测结果摘要.json",
}

MEDICAL_V2_CSV = "痤疮医学量化指标_V2.csv"

EMPTY_CSV_HEADERS = [
    "confidence",
    "class_id",
    "label",
    "label_zh",
    "region",
    "bbox_standardized_xyxy",
    "bbox_original_xyxy",
    "center_standardized_xy",
    "center_original_xy",
    "source",
    "input_mode",
    "detection_scope",
]

DETECTION_REASON_CN = {
    "local_detection_not_allowed": "当前图片不满足痤疮检测条件",
    "empty_skin_mask": "当前图片缺少可用于分析的皮肤区域",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _copy_with_fallback(primary: Path, fallback: Path | None, target: Path) -> str | None:
    source = primary if primary.exists() else fallback
    if source is None or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_empty_csv(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EMPTY_CSV_HEADERS)
        writer.writeheader()
    return str(path)


def build_acne_presence(detection: dict[str, Any]) -> dict[str, str]:
    status = str(detection.get("status", "unknown"))
    filtered_count_raw = detection.get("filtered_count", detection.get("count", 0))
    try:
        filtered_count = int(round(float(filtered_count_raw or 0)))
    except (TypeError, ValueError):
        filtered_count = 0

    if status == "ok":
        if filtered_count > 0:
            return {
                "status": "detected",
                "message": "已检测到疑似痤疮",
            }
        return {
            "status": "not_detected",
            "message": "未检测到痤疮",
        }

    raw_reason = detection.get("reason")
    reason = str(raw_reason) if raw_reason is not None else ""
    message = DETECTION_REASON_CN.get(reason)
    if not message:
        message = "当前图片未完成痤疮检测"
    return {
        "status": "not_applicable",
        "message": message,
    }


def build_display_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    preprocess = summary.get("preprocess", {}) or {}
    profile = summary.get("profile", {}) or {}
    detection = summary.get("detection", {}) or {}
    grading = summary.get("grading", {}) or {}
    acne_presence = build_acne_presence(detection)

    region_counts_raw = detection.get("region_counts", {})
    detector_status = detection.get("status")
    if detector_status == "ok" and isinstance(region_counts_raw, dict):
        region_counts = [{"region": k, "count": v} for k, v in region_counts_raw.items()]
    else:
        region_counts = []

    return {
        "input_mode": preprocess.get("input_mode"),
        "detection_scope": preprocess.get("detection_scope"),
        "profile_name": profile.get("name"),
        "profile_weights": profile.get("weights"),
        "device": profile.get("device"),
        "detector_status": detection.get("status"),
        "detector_reason": detection.get("reason"),
        "acne_presence": acne_presence,
        "acne_count": detection.get("count"),
        "region_counts": region_counts,
        "grading_status": grading.get("status"),
        "grading_reason": grading.get("reason"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }


def build_compact_summary(
    summary: dict[str, Any],
    display_files: dict[str, str],
    input_image: str,
) -> dict[str, Any]:
    detection = summary.get("detection", {}) or {}
    preprocess = summary.get("preprocess", {}) or {}
    grading = summary.get("grading", {}) or {}
    profile = summary.get("profile", {}) or {}
    max_recall = summary.get("max_recall", {}) or {}
    acne_presence = build_acne_presence(detection)
    return {
        "status": summary.get("status", "unknown"),
        "量化结果": build_quantification_result(summary),
        "input_image": input_image,
        "input_mode": preprocess.get("input_mode"),
        "detection_scope": preprocess.get("detection_scope"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "model_profile": {
            "name": profile.get("name"),
            "weights": profile.get("weights"),
            "imgsz": profile.get("imgsz"),
            "conf": profile.get("conf"),
            "device": profile.get("device"),
        },
        "detection": {
            "status": detection.get("status"),
            "reason": detection.get("reason"),
            "acne_presence": acne_presence,
            "raw_count": detection.get("raw_count"),
            "filtered_count": detection.get("count"),
            "region_counts": detection.get("region_counts", {}),
            "region_analysis": detection.get("region_analysis", {}),
            "empty_detections_are_valid": detection.get("empty_detections_are_valid", True),
            "detections": detection.get("detections", []),
        },
        "grading": grading,
        "max_recall_debug_summary": {
            "status": max_recall.get("status"),
            "unsupervised_focal_count": max_recall.get("unsupervised_focal_count"),
            "diffuse_erythema_region_count": max_recall.get("diffuse_erythema_region_count"),
            "combined_count": max_recall.get("combined_count"),
            "note": "无监督候选仅作辅助候选池，不等于真实痤疮数量。",
        },
        "display_files": display_files,
        "warnings": [
            "AI 辅助结果，不代替专业诊断。",
            "当前类别统一为 acne_candidate，即疑似痤疮病灶。",
        ],
    }


def _write_readme(target: Path, compact: dict[str, Any]) -> str:
    detection = compact.get("detection", {}) or {}
    grading = compact.get("grading", {}) or {}
    profile = compact.get("model_profile", {}) or {}
    lines = [
        "# 输出说明",
        "",
        "本目录是正式展示版结果，供业务系统或用户查看。",
        "",
        "## 本次运行概览",
        "",
        f"- 输入图片：`{Path(str(compact.get('input_image', ''))).name}`",
        f"- 输入模式：`{compact.get('input_mode', 'unknown')}`",
        f"- 检测范围：`{compact.get('detection_scope', 'unknown')}`",
        f"- 模型 profile：`{profile.get('name', 'unknown')}`",
        f"- 模型权重：`{profile.get('weights', 'unknown')}`",
        f"- 输入尺寸：`{profile.get('imgsz', 'unknown')}`",
        f"- 检测阈值：`{profile.get('conf', 'unknown')}`",
        f"- 运行设备：`{profile.get('device', 'unknown')}`",
        f"- 原始候选数量：`{detection.get('raw_count', 'unknown')}`",
        f"- 过滤后疑似痤疮病灶数量：`{detection.get('filtered_count', 'unknown')}`",
        f"- 痤疮检出提醒：`{(detection.get('acne_presence', {}) or {}).get('message', '未知')}`",
        f"- 严重程度评估状态：`{grading.get('status', 'unknown')}`",
        f"- Acne-LDS 数量估计：`{grading.get('predicted_count', None)}`",
        f"- Acne-LDS 严重程度等级：`{grading.get('severity_level', None)}`",
        f"- 总耗时：`{compact.get('elapsed_seconds', 'unknown')}` 秒",
        "",
        "## 文件含义",
        "",
        "- `01_原始照片.jpg`：输入照片副本。",
        "- `02_痤疮圈选结果_原图.jpg`：疑似痤疮病灶映射回原始照片的结果图。",
        "- `03_痤疮圈选结果_标准化人脸.jpg`：疑似痤疮病灶在标准化人脸图上的结果图。",
        "- `04_检测结果表.csv`：检测结果表格，包含坐标、置信度和类别。",
        "- `05_检测结果摘要.json`：精简结构化结果。",
        "- `06_严重程度评估.json`：Acne-LDS 全脸数量估计和严重程度输出。",
        "- `07_输出说明.md`：当前说明文件。",
        "- `08_用户检测报告.txt`：面向普通用户的文字版检测报告。",
        "",
        "## 说明",
        "",
        "- 若整脸严重程度不适用，`06_严重程度评估.json` 会返回 skipped 与原因，而不是缺文件。",
        "- 若本次检测未圈出明确候选，`02/03/04/05/08` 仍会存在，只是内容会显示 0 个候选。",
        "- 本结果是医美评估辅助研发结果，不是临床诊断结论。",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return str(target)


def build_display_outputs(
    *,
    raw_dir: Path,
    input_image: str | Path,
    analysis_summary: dict[str, Any] | None = None,
    raw_result_files: dict[str, str] | None = None,
    target_dir: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    summary = analysis_summary or _read_json(raw_dir / "summary.json")
    files = raw_result_files or {}
    target = target_dir or raw_dir
    target.mkdir(parents=True, exist_ok=True)

    original_src = Path(files.get("original_image", raw_dir / "01_original.jpg"))
    standardized_src = Path(files.get("acne_standardized", raw_dir / "02_standardized.jpg"))
    overlay_original_src = Path(files.get("original_yolo_circles", raw_dir / "24_original_yolo_circles.jpg"))
    overlay_standardized_src = Path(files.get("acne_circles", raw_dir / "17_acne_candidate_circles.jpg"))
    detections_csv_src = Path(files.get("acne_detections", raw_dir / "18_detections.csv"))
    grading_src = raw_dir / "13_grading_result.json"

    display_files: dict[str, str] = {}

    copied = _copy_with_fallback(original_src, None, target / DISPLAY_FILENAMES["original_image"])
    if copied:
        display_files["original_image"] = copied

    copied = _copy_with_fallback(
        overlay_original_src,
        original_src if original_src.exists() else None,
        target / DISPLAY_FILENAMES["overlay_original"],
    )
    if copied:
        display_files["overlay_original"] = copied

    copied = _copy_with_fallback(
        overlay_standardized_src,
        standardized_src if standardized_src.exists() else None,
        target / DISPLAY_FILENAMES["overlay_standardized"],
    )
    if copied:
        display_files["overlay_standardized"] = copied

    if detections_csv_src.exists():
        copied = _copy_with_fallback(detections_csv_src, None, target / DISPLAY_FILENAMES["detections_csv"])
        if copied:
            display_files["detections_csv"] = copied
    else:
        display_files["detections_csv"] = _write_empty_csv(target / DISPLAY_FILENAMES["detections_csv"])

    grading_target = target / DISPLAY_FILENAMES["grading_json"]
    if grading_src.exists():
        copied = _copy_with_fallback(grading_src, None, grading_target)
        if copied:
            display_files["grading_json"] = copied
    else:
        display_files["grading_json"] = _write_json(grading_target, summary.get("grading", {}) or {})

    compact = build_compact_summary(summary, display_files, str(input_image))
    compact_path = target / DISPLAY_FILENAMES["summary_json"]
    display_files["summary_json"] = _write_json(compact_path, compact)

    readme_path = target / DISPLAY_FILENAMES["readme_md"]
    display_files["readme_md"] = _write_readme(readme_path, compact)

    report_path = target / DISPLAY_FILENAMES["user_report_txt"]
    report_path.write_text(
        build_user_report_from_paths(
            compact_path,
            target / DISPLAY_FILENAMES["grading_json"],
            target / DISPLAY_FILENAMES["detections_csv"],
        ),
        encoding="utf-8",
    )
    display_files["user_report_txt"] = str(report_path)

    compact["display_files"] = display_files
    compact_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")

    quantification_csv = target / DISPLAY_FILENAMES["quantification_csv"]
    quantification_json = target / DISPLAY_FILENAMES["quantification_json"]
    clinical_report = write_clinical_quantification(
        summary, quantification_csv, quantification_json
    )
    try_merge_medical_v2_report(
        clinical_report,
        target / MEDICAL_V2_CSV,
        quantification_json,
    )
    # 旧 display_result 键继续存在，但正式指向医生可读且相互对应的量化文件。
    # 原始逐候选明细和精简摘要不删除，以新增 detail 键继续提供调试能力。
    display_files["detections_detail_csv"] = display_files["detections_csv"]
    display_files["summary_detail_json"] = display_files["summary_json"]
    display_files["detections_csv"] = str(quantification_csv)
    display_files["summary_json"] = str(quantification_json)
    display_files["quantification_csv"] = str(quantification_csv)
    display_files["quantification_json"] = str(quantification_json)

    metadata = build_display_metadata(summary)
    return display_files, compact, metadata
