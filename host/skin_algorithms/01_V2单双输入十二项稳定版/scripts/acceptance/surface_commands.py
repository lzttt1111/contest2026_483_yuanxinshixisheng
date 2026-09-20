"""Branch- and surface-specific acceptance command construction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


Profile = Literal["baseline", "merged"]
Surface = Literal["run", "cloud"]


def build_surface_command(
    profile: Profile,
    surface: Surface,
    project_root: Path,
    tool_root: Path,
    python: Path,
    inputs_manifest: Path,
    results: Path,
    evidence: Path,
) -> tuple[str, ...]:
    if surface == "cloud":
        return (
            str(python),
            "-B",
            str(tool_root / "scripts" / "acceptance" / "cloud_batch_coordinator.py"),
            "--project-root",
            str(project_root),
            "--tool-root",
            str(tool_root),
            "--python",
            str(python),
            "--inputs-manifest",
            str(inputs_manifest),
            "--results",
            str(results),
            "--evidence",
            str(evidence),
            "--profile",
            profile,
        )
    if profile == "baseline":
        return (
            str(python),
            "-B",
            str(project_root / "scripts" / "review" / "run_nine_analysis.py"),
            "--input-dir",
            str(inputs_manifest.parent),
            "--output",
            str(results),
            "--layout",
            "review",
            "--repeat",
            "1",
            "--cuda-device",
            "0",
        )
    return (
        str(python),
        "-B",
        str(tool_root / "scripts" / "acceptance" / "ordered_local_entry.py"),
        "--project-root",
        str(project_root),
        "--inputs-manifest",
        str(inputs_manifest),
        "--output",
        str(results),
        "--runtime",
        str(evidence / "runtime"),
    )
