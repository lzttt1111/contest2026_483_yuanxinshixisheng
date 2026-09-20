from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from src.capture_profile import CaptureProfile
from src.nine_analysis.public_sanitize import sanitize_public_document


STATE_VERSION: Final = "nine_batch_state_v3"
SUMMARY_CSV_FIELDS: Final = (
    "序号", "相对路径", "输出批次", "输出目录",
    "状态", "成功项目数", "失败分类", "失败项目",
    "建议重试", "耗时秒", "错误",
)


@dataclass
class BatchStatistics:
    """批处理常量内存统计，不保留逐图 record。"""

    processed: int = 0
    duration_count: int = 0
    duration_total: float = 0.0
    statuses: Counter[str] = field(default_factory=Counter)
    failure_categories: Counter[str] = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        status = str(record.get("status", "unknown"))
        self.processed += 1
        self.statuses[status] += 1
        if status != "claimed_elsewhere":
            self.duration_count += 1
            self.duration_total += float(record.get("seconds", 0.0))
        if status in {"partial_success", "failed"}:
            self.failure_categories[
                str(record.get("failure_category", "unknown"))
            ] += 1

    @property
    def all_successful(self) -> bool:
        return self.processed > 0 and not any(
            self.statuses.get(status, 0)
            for status in ("partial_success", "failed")
        )


def summary_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    """将一条状态记录压缩为人工可读的批处理汇总行。"""

    output_value = str(record.get("output_dir", ""))
    output_path = Path(output_value)
    if output_path.is_absolute():
        output_value = (
            Path(str(record.get("output_group", ""))) / output_path.name
        ).as_posix()
    return {
        "序号": record["index"],
        "相对路径": record["signature"]["relative_path"],
        "输出批次": record.get("output_group", ""),
        "输出目录": output_value,
        "状态": record["status"],
        "成功项目数": record.get("success_items", 0),
        "失败分类": record.get("failure_category", ""),
        "失败项目": ",".join(record.get("failed_items", [])),
        "建议重试": record.get("retry_recommended", ""),
        "耗时秒": record.get("seconds", 0.0),
        "错误": record.get("error", ""),
    }


def write_summary(
    summary_dir: Path,
    *,
    job_name: str,
    input_root: Path,
    output_root: Path,
    images_per_group: int,
    worker_id: str,
    selected: int,
    skipped: int,
    statistics: BatchStatistics,
    cold_start: dict[str, Any] | None,
    capture_profile: CaptureProfile = CaptureProfile.INSTITUTION,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_version": STATE_VERSION,
        "job_name": job_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": input_root.name,
        "output_root": output_root.name,
        "images_per_group": images_per_group,
        "worker_id": worker_id,
        "capture_profile": capture_profile.value,
        "selected_images": selected,
        "resume_skipped": skipped,
        "processed_this_run": statistics.processed,
        "success": statistics.statuses.get("success", 0),
        "partial_success": statistics.statuses.get("partial_success", 0),
        "failed": statistics.statuses.get("failed", 0),
        "claimed_elsewhere": statistics.statuses.get("claimed_elsewhere", 0),
        "average_seconds": (
            round(statistics.duration_total / statistics.duration_count, 4)
            if statistics.duration_count
            else 0.0
        ),
        "cold_start": sanitize_public_document(cold_start or {}),
        "failure_categories": dict(statistics.failure_categories),
        "records_file": f"{job_name}.{worker_id}.summary.csv",
        "state_file": f"state.{worker_id}.jsonl",
    }
    summary_stem = f"{job_name}.{worker_id}"
    temporary = summary_dir / f".{summary_stem}.summary.tmp"
    target = summary_dir / f"{summary_stem}.summary.json"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


__all__ = [
    "BatchStatistics",
    "STATE_VERSION",
    "SUMMARY_CSV_FIELDS",
    "summary_csv_row",
    "write_summary",
]
