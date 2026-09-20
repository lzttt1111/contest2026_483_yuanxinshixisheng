from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, TextIO

from src.capture_profile import CaptureProfile
from src.nine_analysis.batch_runtime.claims import (
    WorkClaim,
    preserve_terminal_claim,
    release_work_claim,
)
from src.nine_analysis.batch_runtime.summary import STATE_VERSION


def write_batch_metadata(
    result_root: Path,
    *,
    signature: dict[str, Any],
    output_group: str,
) -> None:
    payload = {
        "state_version": STATE_VERSION,
        "signature": signature,
        "output_group": output_group,
    }
    receipt = result_root / "运行回执.json"
    if receipt.is_file():
        document = json.loads(receipt.read_text(encoding="utf-8"))
        document["batch"] = payload
        temporary = result_root / ".运行回执.tmp"
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, receipt)
        return
    target = result_root / "批处理元数据.json"
    temporary = result_root / ".批处理元数据.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def signature_key(signature: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(signature["relative_path"]),
        int(signature["size"]),
        int(signature["mtime_ns"]),
        str(signature.get("capture_profile", CaptureProfile.INSTITUTION.value)) + (
            "|" + str(signature["clinical_spec"]) + "|pdf=" + str(bool(signature.get("v3_pdf_requested")))
            + "|ref=" + str(signature.get("v3_reference","uncalibrated"))
            + "|captured=" + str(signature.get("v3_captured_at"))
            + "|versions=" + json.dumps(signature.get("v3_versions",{}),sort_keys=True)
            + "|output=" + str(signature.get("v3_output_profile","review"))
            + "|subject=" + json.dumps(signature.get("subject_id"),ensure_ascii=False)
            + "|input=" + json.dumps(signature.get("input_sha256",{}),sort_keys=True)
            if signature.get("clinical_spec") else ""
        ),
    )


def result_matches_signature(result_root: Path, signature: dict[str, Any]) -> bool:
    try:
        if signature.get("clinical_spec"):
            if signature.get("v3_versions"):
                from src.doctor_v3.stage1_resume import matches
                return matches(result_root,signature)
            else:
                from src.doctor_v3.bundle import load
                load(result_root / "doctor_v3")
        if (result_root / "十二项检测结果索引.json").is_file():
            payload = json.loads(
                (result_root / "运行回执.json").read_text(encoding="utf-8")
            )
            stored = payload["batch"]["signature"]
        elif (result_root / "九项检测结果索引.json").is_file():
            payload = json.loads(
                (result_root / "批处理元数据.json").read_text(encoding="utf-8")
            )
            stored = payload["signature"]
        else:
            return False
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError):
        return False
    return signature_key(stored) == signature_key(signature)


def file_signature(path: Path, input_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(input_root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def should_skip_terminal(
    signature: tuple[str, int, int, str],
    completed: set[tuple[str, int, int, str]],
    *,
    output_profile: str,
    skip_recorded_failures: bool,
    result_index_exists: bool,
) -> bool:
    if signature not in completed:
        return False
    return skip_recorded_failures or (output_profile == "scoring" and "doctor-v3-stage1" not in str(signature[3])) or result_index_exists


def shard_number(path: Path, input_root: Path, shard_count: int) -> int:
    relative = path.relative_to(input_root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def load_completed(state_path: Path) -> set[tuple[str, int, int, str]]:
    completed: set[tuple[str, int, int, str]] = set()
    if not state_path.is_file():
        return completed
    with state_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = record.get("signature")
            if record.get("status") == "success" and isinstance(signature, dict):
                completed.add(signature_key(signature))
    return completed


def load_completed_states(state_dir: Path) -> set[tuple[str, int, int, str]]:
    completed: set[tuple[str, int, int, str]] = set()
    for state_path in sorted(state_dir.glob("state*.jsonl")):
        completed.update(load_completed(state_path))
    return completed


def load_terminal_states(
    state_dir: Path,
    *,
    include_failures: bool,
) -> set[tuple[str, int, int, str]]:
    statuses = {"success"}
    if include_failures:
        statuses.update(("partial_success", "failed"))
    terminal: set[tuple[str, int, int, str]] = set()
    for state_path in sorted(state_dir.glob("state*.jsonl")):
        with state_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                signature = record.get("signature")
                if record.get("status") in statuses and isinstance(signature, dict):
                    terminal.add(signature_key(signature))
    return terminal


def _job_state_directories(batch_root: Path, job_name: str) -> tuple[Path, ...]:
    if not batch_root.is_dir():
        return ()
    shard_pattern = re.compile(
        rf"^{re.escape(job_name)}-shard-\d{{4}}-of-\d{{4}}$"
    )
    return tuple(
        candidate
        for candidate in sorted(batch_root.iterdir())
        if candidate.is_dir()
        and (candidate.name == job_name or shard_pattern.fullmatch(candidate.name))
    )


def load_completed_job_states(
    batch_root: Path,
    job_name: str,
) -> set[tuple[str, int, int, str]]:
    completed: set[tuple[str, int, int, str]] = set()
    for candidate in _job_state_directories(batch_root, job_name):
        completed.update(load_completed_states(candidate))
    return completed


def load_terminal_job_states(
    batch_root: Path,
    job_name: str,
    *,
    include_failures: bool,
) -> set[tuple[str, int, int, str]]:
    terminal: set[tuple[str, int, int, str]] = set()
    for candidate in _job_state_directories(batch_root, job_name):
        terminal.update(
            load_terminal_states(candidate, include_failures=include_failures)
        )
    return terminal


def append_state(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def commit_terminal_record(
    handle: TextIO,
    record: dict[str, Any],
    claim: WorkClaim,
    *,
    preserve_claim: bool = False,
) -> None:
    append_state(handle, record)
    if preserve_claim:
        preserve_terminal_claim(claim, record)
    else:
        release_work_claim(claim)


def write_health_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "append_state",
    "commit_terminal_record",
    "file_signature",
    "load_completed",
    "load_completed_job_states",
    "load_completed_states",
    "load_terminal_job_states",
    "load_terminal_states",
    "result_matches_signature",
    "shard_number",
    "should_skip_terminal",
    "signature_key",
    "write_batch_metadata",
    "write_health_snapshot",
]
