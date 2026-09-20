"""Immutable baseline Git proof before and after acceptance."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.acceptance.acceptance_io import AcceptanceInputError


BASELINE_SHA = "3cbd57e2405f70790f1ae988db32f194db96227e"
ALLOWED_REQUIREMENTS_ROOT = "归档文件夹/面部检测需求md_V2_20260728/"


def prove_baseline(project_root: Path) -> None:
    head = subprocess.run(
        ("git", "-C", str(project_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ignored = subprocess.run(
        (
            "git", "-c", "core.quotePath=false", "-C", str(project_root),
            "ls-files", "--others", "--ignored", "--exclude-standard",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ignored_paths = tuple(line for line in ignored.splitlines() if line)
    requirements = tuple(
        path for path in ignored_paths
        if path.startswith(ALLOWED_REQUIREMENTS_ROOT)
    )
    unexpected = tuple(path for path in ignored_paths if path not in requirements)
    requirements_invalid = bool(requirements) and (
        len(requirements) != 11
        or any(not path.endswith(".md") for path in requirements)
    )
    if head != BASELINE_SHA or status or unexpected or requirements_invalid:
        raise AcceptanceInputError(detail="baseline freeze proof failed")
