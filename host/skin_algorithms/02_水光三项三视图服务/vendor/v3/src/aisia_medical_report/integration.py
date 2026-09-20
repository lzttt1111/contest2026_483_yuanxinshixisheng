from __future__ import annotations

import csv
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from .aggregator import build_report_payload
from .report import render_docx, render_doctor_docx, render_user_docx
from src.scoring_calibration.v011.quality import evaluate_image_path
from src.scoring_calibration.v011.scoring import score_observation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "AISIA_面部多指标检测汇总模板_V2.docx"
SCORING_PROFILE_VERSION = "aisia_scoring_v0.1.1_integrity_test_20260806"


def _load_successful_result(root: Path) -> tuple[dict[str, Any], Path]:
    index_path = root / "九项检测结果索引.json"
    if not root.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"九项人工验收结果目录无效: {root}")
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if index.get("状态") != "success" or int(index.get("成功项目数", 0)) != 9:
        raise RuntimeError(f"双版报告仅接受九项全部成功的结果目录: {root}")
    return index, index_path


def _source_image(root: Path) -> Path:
    candidates = sorted(path for path in root.glob("00_输入图片.*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"结果目录缺少输入图片: {root}")
    return candidates[0]


def _scoring_features(root: Path) -> dict[str, Any]:
    path = root / "九项核心量化指标.json"
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    items = document.get("九项") or {}
    features = {
        key: value.get("评分输入") or {}
        for key, value in items.items()
        if isinstance(value, dict)
    }
    expected = {"redness", "spots", "brown", "texture", "pores", "uv_spots", "porphyrin", "wrinkle", "acne"}
    if set(features) != expected:
        raise RuntimeError(f"九项评分输入不完整: missing={sorted(expected - set(features))}")
    # 紫区前端合同只保留精简整数计数；医生评分所需的单位面积密度
    # 从同次检测的医学详细CSV读取。该适配只存在于独立报告层，
    # 不修改Worker raw_result.metrics或已经与前端对齐的字段。
    uv_range = features["uv_spots"].setdefault("UV样色素范围", {})
    if "density" not in uv_range:
        csv_path = root / "七项检测" / "紫区" / "紫区医学量化指标_V2.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            density_column = "核心-单位面积密度（个/10万有效皮肤像素）"
            for row in rows:
                if row.get("检测项目") == "标准化UV紫外线色斑工程代理" and row.get("检测范围") == "全面部":
                    uv_range["density"] = float(row[density_column])
                    break
        if "density" not in uv_range:
            raise RuntimeError(f"紫区医学CSV缺少全面部UV色斑密度: {csv_path}")
    return features


def _load_ready_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("profile_ready") is not True:
        raise RuntimeError("拒绝生成正式双版报告：评分常模 profile_ready 不是 true")
    if profile.get("scoring_profile_version") != SCORING_PROFILE_VERSION:
        raise RuntimeError(
            "评分常模版本不匹配: "
            f"{profile.get('scoring_profile_version')} != {SCORING_PROFILE_VERSION}"
        )
    return profile, hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def generate_dual_reports_from_review_result(
    result_dir: str | Path,
    scoring_profile_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    report_id: str | None = None,
    subject_id: str = "匿名受检者",
    template_path: str | Path | None = None,
) -> dict[str, str]:
    """Generate user/doctor DOCX files from one immutable nine-analysis result."""
    root = Path(result_dir).expanduser().resolve()
    index, index_path = _load_successful_result(root)
    source_image = _source_image(root)
    profile_path = Path(scoring_profile_path).expanduser().resolve()
    profile, profile_sha256 = _load_ready_profile(profile_path)
    gate = evaluate_image_path(source_image)
    if gate.get("status") != "PASS":
        raise RuntimeError(f"输入图片未通过报告评分门禁: {gate.get('status')} {gate.get('reason_codes')}")
    scoring = score_observation(
        {
            "status": "success",
            "input_quality_gate": gate,
            "quality": {"status": "PASS"},
            "features": _scoring_features(root),
        },
        profile.get("references") or {},
        normalization_profile_sha256=profile_sha256,
        scoring_profile_version=profile.get("scoring_profile_version"),
    )
    if any(row.get("status") != "formal" for row in scoring.get("formal_dimension_scores", [])):
        missing = {
            row.get("dimension_id"): row.get("missing_required_groups")
            for row in scoring.get("formal_dimension_scores", [])
            if row.get("status") != "formal"
        }
        raise RuntimeError(f"四项正式评分证据不完整: {missing}")

    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in root.name)
    resolved_report_id = report_id or f"AISIA-{safe_stem}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    bundle: dict[str, Any] = {
        "报告信息": {"报告编号": resolved_report_id},
        "受检者信息": {"姓名或编号": subject_id},
        "采集与质量信息": {"源图": str(source_image), "评分门禁": gate},
        "任务状态": {key: value.get("状态", "unknown") for key, value in index.get("九项结果", {}).items()},
        "结果来源": {
            "dermavision_result_dir": str(root),
            "acne_summary_json": str(root / "痤疮" / "痤疮量化指标.json"),
            "wrinkle_summary_json": str(root / "皱纹" / "皱纹量化指标.json"),
        },
        "评分V011": scoring,
    }
    payload = build_report_payload(bundle)
    payload["报告数据版本"] = "report_payload_v011"
    payload["评分配置摘要"] = {
        "scoring_profile_version": profile.get("scoring_profile_version"),
        "profile_sha256": profile_sha256,
        "registry_sha256": scoring.get("registry_sha256"),
        "algorithm_version": profile.get("algorithm_version", "nine_analysis_dev_97feba3"),
        "metrics_schema_version": profile.get("metrics_schema_version", "scoring_features_v1"),
        "reference_sample_counts": profile.get("reference_sample_counts"),
    }
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "医学报告"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in resolved_report_id)
    structured_path = target_dir / f"AISIA_面部多指标检测报告_{safe_id}_结构化数据.json"
    trace_path = target_dir / f"AISIA_面部多指标检测报告_{safe_id}_评分追溯.json"
    user_path = target_dir / f"AISIA_面部多指标检测报告_{safe_id}_用户精简版.docx"
    doctor_path = target_dir / f"AISIA_面部多指标检测报告_{safe_id}_医生详细版.docx"
    structured_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    trace_path.write_text(json.dumps(scoring, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    template = Path(template_path).expanduser().resolve() if template_path else DEFAULT_TEMPLATE
    render_user_docx(payload, template, user_path)
    render_doctor_docx(payload, template, doctor_path)

    index["医学报告"] = {
        "状态": "success",
        "汇总JSON": _relative_or_name(structured_path, root),
        "DOCX": _relative_or_name(doctor_path, root),
    }
    index["双版报告"] = {
        "状态": "success",
        "用户精简版": _relative_or_name(user_path, root),
        "医生详细版": _relative_or_name(doctor_path, root),
        "结构化数据": _relative_or_name(structured_path, root),
        "评分追溯": _relative_or_name(trace_path, root),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "json": str(structured_path), "docx": str(doctor_path),
        "user_docx": str(user_path), "doctor_docx": str(doctor_path),
        "structured_json": str(structured_path), "scoring_trace_json": str(trace_path),
    }


def generate_report_from_review_result(
    result_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    report_id: str | None = None,
    subject_id: str = "匿名受检者",
    template_path: str | Path | None = None,
) -> dict[str, str]:
    """Generate the optional medical report from one nine-analysis review directory."""
    root = Path(result_dir).expanduser().resolve()
    index_path = root / "九项检测结果索引.json"
    if not root.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"九项人工验收结果目录无效: {root}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("状态") != "success" or int(index.get("成功项目数", 0)) != 9:
        raise RuntimeError(f"医学报告仅接受九项全部成功的结果目录: {root}")

    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in root.name)
    resolved_report_id = report_id or f"AISIA-{safe_stem}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    acne_json = root / "痤疮" / "痤疮量化指标.json"
    wrinkle_json = root / "皱纹" / "皱纹量化指标.json"
    source_image = root / "00_输入图片.png"
    bundle: dict[str, Any] = {
        "报告信息": {"报告编号": resolved_report_id},
        "受检者信息": {"姓名或编号": subject_id},
        "采集与质量信息": {"源图": str(source_image) if source_image.is_file() else "未提供"},
        "任务状态": {key: value.get("状态", "unknown") for key, value in index.get("九项结果", {}).items()},
        "结果来源": {
            "dermavision_result_dir": str(root),
            "acne_summary_json": str(acne_json) if acne_json.is_file() else None,
            "wrinkle_summary_json": str(wrinkle_json) if wrinkle_json.is_file() else None,
        },
    }
    payload = build_report_payload(bundle)
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "医学报告"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in resolved_report_id)
    json_path = target_dir / f"AISIA_面部多指标检测报告_{safe_id}.json"
    docx_path = target_dir / f"AISIA_面部多指标检测报告_{safe_id}.docx"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    template = Path(template_path).expanduser().resolve() if template_path else DEFAULT_TEMPLATE
    render_docx(payload, template, docx_path)

    index["医学报告"] = {
        "状态": "success",
        "汇总JSON": str(json_path.relative_to(root)),
        "DOCX": str(docx_path.relative_to(root)),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"json": str(json_path), "docx": str(docx_path)}
