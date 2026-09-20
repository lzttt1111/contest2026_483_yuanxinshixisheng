from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.scoring_bridge.medical_review_selection import (
    MedicalReviewCandidate,
    select_medical_review_pack,
)


def write_medical_review_packs(
    *,
    output_dir: Path,
    population: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    pair_by_subject = {str(pair["subject_id"]): pair for pair in pairs}
    written: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for module_id, result in population.items():
        if result.status != "promoted":
            continue
        candidates: list[MedicalReviewCandidate] = []
        for subject_id, score in zip(
            result.scored_subject_ids,
            result.scores,
            strict=True,
        ):
            pair = pair_by_subject[subject_id]
            relative_path = str(pair["source_relative_path"])
            observation = observations.get(relative_path) or {}
            quality = observation.get("quality") or {}
            candidates.append(MedicalReviewCandidate(
                subject_id=subject_id,
                score=score,
                qc_status=str(quality.get("status") or "UNKNOWN"),
                pose_stratum="unknown",
                relative_path=relative_path,
            ))
        selection = select_medical_review_pack(tuple(candidates), per_anchor=6)
        path = output_dir / f"{module_id}_medical_review_30.jsonl"
        path.write_text("".join(
            json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in selection
        ), encoding="utf-8")
        written[module_id] = path.name
    return written


def write_confirmation_summary_md(
    *,
    path: Path,
    attempted_count: int,
    compatibility: Mapping[str, Any],
    population: Mapping[str, Any],
    report_sha: str,
) -> None:
    lines = [
        "# DermaVision Hybrid V2 确认集验收报告",
        "",
        f"- 冻结确认队列：{attempted_count}个主体",
        f"- 机器报告SHA256：`{report_sha}`",
        "- 确认集只apply冻结候选，未重新拟合或调参。",
        "- QC拒绝计入流失；空证据保持null，不写0。",
        "",
        "## 旧四项逐维确认",
        "",
        "| 维度 | 状态 | 有效/流失 | MAE | P95 | Spearman | 跨两档 | 失败规则 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for dimension_id, result in sorted(compatibility.items()):
        diagnostic = result.candidate_diagnostics
        lines.append(
            f"| {dimension_id} | {result.status} | "
            f"{result.evaluable_count}/{result.attrition_count} | "
            f"{diagnostic.mean_absolute_error:.4f} | "
            f"{diagnostic.absolute_error_p95:.4f} | "
            f"{diagnostic.spearman:.4f} | "
            f"{diagnostic.multi_grade_crossing_count} | "
            f"{','.join(result.decision.failed_rules) or '-'} |"
        )
    lines.extend([
        "",
        "## 新七项技术门",
        "",
        "| 模块 | 状态 | 有效/流失 | 0分率 | 100分率 | 中位数 | IQR | 失败规则 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for module_id, result in sorted(population.items()):
        decision = result.decision
        lines.append(
            f"| {module_id} | {result.status} | "
            f"{result.evaluable_count}/{result.attrition_count} | "
            f"{decision.zero_rate:.4f} | {decision.hundred_rate:.4f} | "
            f"{decision.median if decision.median is not None else 'null'} | "
            f"{decision.iqr if decision.iqr is not None else 'null'} | "
            f"{','.join(decision.failed_rules) or '-'} |"
        )
    lines.extend([
        "",
        "## 发布边界",
        "",
        "- 各旧维度独立晋级；blocked维度继续使用V0.1.1。",
        "- 新模块只进入内部JSON；医生批准前不进入Word。",
        "- 本报告不修改正式配置、不更新allowlist、不触发部署。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["write_confirmation_summary_md", "write_medical_review_packs"]
