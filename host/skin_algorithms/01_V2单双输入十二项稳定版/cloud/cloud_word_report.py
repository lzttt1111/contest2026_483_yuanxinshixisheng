from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.aisia_medical_report.identity import resolve_subject_id


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CloudWordGenerationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _identity_bound_input_name(
    input_path: Path,
    subject_id: str | None,
) -> str:
    """Build a stable, non-PII filename from the input and report subject."""

    digest = hashlib.sha256()
    digest.update(input_path.read_bytes())
    digest.update(b"\0")
    digest.update((subject_id or "").encode("utf-8"))
    suffix = input_path.suffix.lower() or ".jpg"
    return f"report-{digest.hexdigest()[:16]}{suffix}"


def generate_cloud_review_reports(
    input_path: Path,
    output_root: Path,
    subject_id: str | None = None,
) -> tuple[Path, Path]:
    """Generate the optional two Word reports through the validated consumer CLI."""
    input_root = output_root / ".medical-report-input"
    run_root = output_root / ".medical-report-run"
    runtime_root = output_root / ".medical-report-runtime"
    cache_root = output_root / ".medical-report-cache"
    log_path = output_root / ".medical-report-generation.log"
    input_root.mkdir()
    resolved_subject_id = resolve_subject_id(subject_id, source=input_path)
    staged_name = _identity_bound_input_name(input_path, resolved_subject_id)
    shutil.copy2(input_path, input_root / staged_name)
    environment = os.environ.copy()
    environment["DERMAVISION_CAPTURE_PROFILE"] = "consumer"
    environment["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run.py"),
        "--input-dir",
        str(input_root),
        "--output-dir",
        str(run_root),
        "--runtime-dir",
        str(runtime_root),
        "--capture-profile",
        "consumer",
        "--generate-medical-report",
        "--limit",
        "1",
        "--no-resume",
        "--job-name",
        "cloud-word",
    ]
    command.extend(("--report-subject-id", resolved_subject_id))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise CloudWordGenerationError(
            f"云端Word聚合生成失败: exit={completed.returncode}; "
            f"日志={log_path.name}"
        )
    reports = tuple(sorted(run_root.rglob("*.docx")))
    if len(reports) != 2:
        raise CloudWordGenerationError(
            f"云端Word聚合应生成两份DOCX，实际={len(reports)}"
        )
    role_matches = {
        "user": [path for path in reports if "用户精简版" in path.name],
        "doctor": [path for path in reports if "医生详细版" in path.name],
    }
    if any(len(paths) != 1 for paths in role_matches.values()):
        raise CloudWordGenerationError("云端Word聚合报告角色不完整")
    published = tuple(output_root / path.name for path in reports)
    if any(path.exists() for path in published):
        raise CloudWordGenerationError("云端样本根已存在Word报告")
    for source, target in zip(reports, published, strict=True):
        shutil.copy2(source, target)
    for path in (input_root, run_root, runtime_root, cache_root):
        shutil.rmtree(path, ignore_errors=True)
    log_path.unlink(missing_ok=True)
    return published


__all__ = ["CloudWordGenerationError", "generate_cloud_review_reports"]
