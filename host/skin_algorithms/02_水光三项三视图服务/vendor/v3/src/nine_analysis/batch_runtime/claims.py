from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from src.nine_analysis.batch_runtime.identifiers import claim_key


DEFAULT_CLAIM_TIMEOUT_SECONDS: Final = 6 * 60 * 60


@dataclass(frozen=True, slots=True)
class WorkClaim:
    path: Path
    token: str


def acquire_work_claim(
    claim_root: Path,
    signature: dict[str, Any],
    *,
    worker_id: str,
    stale_after_seconds: int = DEFAULT_CLAIM_TIMEOUT_SECONDS,
    retry_failed_terminal: bool = False,
    retry_failed_terminal_before: datetime | None = None,
) -> WorkClaim | None:
    """在共享输出目录用原子 mkdir 抢占图片，避免多机重复计算。"""

    if stale_after_seconds < 1:
        raise ValueError("stale_after_seconds 必须 >= 1")
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_root / f"{claim_key(signature)}.lock"
    done_path = claim_path.with_suffix(".done")
    if done_path.is_dir():
        if not retry_failed_terminal:
            return None
        try:
            terminal = json.loads(
                (done_path / "terminal.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if terminal.get("status") not in {"partial_success", "failed"}:
            return None
        if retry_failed_terminal_before is not None:
            try:
                finished_at = datetime.fromisoformat(str(terminal["finished_at"]))
            except (KeyError, TypeError, ValueError):
                finished_at = datetime.fromtimestamp(done_path.stat().st_mtime)
            if finished_at >= retry_failed_terminal_before:
                return None
        retry_history = claim_root / f".{done_path.name}.retry-{uuid.uuid4().hex}"
        try:
            os.replace(done_path, retry_history)
        except FileNotFoundError:
            pass
    for _ in range(3):
        token = uuid.uuid4().hex
        try:
            claim_path.mkdir()
        except FileExistsError:
            try:
                age_seconds = max(0.0, time.time() - claim_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if age_seconds < stale_after_seconds:
                return None
            stale_path = claim_root / f".{claim_path.name}.stale-{uuid.uuid4().hex}"
            try:
                os.replace(claim_path, stale_path)
            except FileNotFoundError:
                continue
            shutil.rmtree(stale_path, ignore_errors=True)
            continue
        payload = {
            "token": token,
            "worker_id": worker_id,
            "claimed_at": datetime.now().isoformat(timespec="seconds"),
            "signature": signature,
        }
        try:
            (claim_path / "claim.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            shutil.rmtree(claim_path, ignore_errors=True)
            raise
        return WorkClaim(path=claim_path, token=token)
    return None


def release_work_claim(claim: WorkClaim) -> bool:
    """只释放属于当前执行者的锁，避免误删后来抢占者的锁。"""

    try:
        payload = json.loads((claim.path / "claim.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if payload.get("token") != claim.token:
        return False
    shutil.rmtree(claim.path, ignore_errors=True)
    return True


def preserve_terminal_claim(claim: WorkClaim, record: dict[str, Any]) -> bool:
    """Atomically convert one owned in-flight claim into a durable done marker."""

    try:
        payload = json.loads((claim.path / "claim.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if payload.get("token") != claim.token:
        return False
    (claim.path / "terminal.json").write_text(
        json.dumps(
            {
                "status": record.get("status"),
                "finished_at": record.get("finished_at"),
                "signature": record.get("signature"),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    done_path = claim.path.with_suffix(".done")
    try:
        os.replace(claim.path, done_path)
    except (FileExistsError, FileNotFoundError, OSError):
        return False
    return True


__all__ = [
    "DEFAULT_CLAIM_TIMEOUT_SECONDS",
    "WorkClaim",
    "acquire_work_claim",
    "preserve_terminal_claim",
    "release_work_claim",
]
