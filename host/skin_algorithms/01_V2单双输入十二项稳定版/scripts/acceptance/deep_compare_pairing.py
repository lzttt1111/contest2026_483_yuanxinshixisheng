"""Deterministic file pairing with explicit unmatched and ambiguous evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from scripts.acceptance.acceptance_types import JsonValue


@dataclass(frozen=True, slots=True)
class FilePair:
    baseline: Path
    merged: Path
    match_kind: str


@dataclass(frozen=True, slots=True)
class PairingResult:
    pairs: tuple[FilePair, ...]
    unmatched_baseline: tuple[str, ...]
    unmatched_merged: tuple[str, ...]
    ambiguous: tuple[JsonValue, ...]


_CASE_PATTERN = re.compile(r"clinic28-\d+")


def _sample_basename(relative: str) -> str | None:
    match = _CASE_PATTERN.search(relative)
    return f"{match.group(0)}::{Path(relative).name}" if match else None


def pair_file_trees(
    baseline_root: Path,
    merged_root: Path,
    suffixes: set[str],
) -> PairingResult:
    baseline = sorted(
        path for path in baseline_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    merged = sorted(
        path for path in merged_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    baseline_by_relative = {
        path.relative_to(baseline_root).as_posix(): path for path in baseline
    }
    merged_by_relative = {
        path.relative_to(merged_root).as_posix(): path for path in merged
    }
    paired_baseline: set[str] = set()
    paired_merged: set[str] = set()
    pairs: list[FilePair] = []
    for relative in sorted(set(baseline_by_relative) & set(merged_by_relative)):
        pairs.append(
            FilePair(
                baseline=baseline_by_relative[relative],
                merged=merged_by_relative[relative],
                match_kind="relative_path",
            )
        )
        paired_baseline.add(relative)
        paired_merged.add(relative)
    baseline_names: dict[str, list[str]] = {}
    merged_names: dict[str, list[str]] = {}
    baseline_samples: dict[str, list[str]] = {}
    merged_samples: dict[str, list[str]] = {}
    for relative in sorted(set(baseline_by_relative) - paired_baseline):
        sample_key = _sample_basename(relative)
        if sample_key is not None:
            baseline_samples.setdefault(sample_key, []).append(relative)
        baseline_names.setdefault(Path(relative).name, []).append(relative)
    for relative in sorted(set(merged_by_relative) - paired_merged):
        sample_key = _sample_basename(relative)
        if sample_key is not None:
            merged_samples.setdefault(sample_key, []).append(relative)
        merged_names.setdefault(Path(relative).name, []).append(relative)
    for sample_key in sorted(set(baseline_samples) & set(merged_samples)):
        left = baseline_samples[sample_key]
        right = merged_samples[sample_key]
        if len(left) != 1 or len(right) != 1:
            continue
        pairs.append(
            FilePair(
                baseline=baseline_by_relative[left[0]],
                merged=merged_by_relative[right[0]],
                match_kind="sample_basename",
            )
        )
        paired_baseline.add(left[0])
        paired_merged.add(right[0])
    baseline_names = {}
    merged_names = {}
    for relative in sorted(set(baseline_by_relative) - paired_baseline):
        baseline_names.setdefault(Path(relative).name, []).append(relative)
    for relative in sorted(set(merged_by_relative) - paired_merged):
        merged_names.setdefault(Path(relative).name, []).append(relative)
    ambiguous: list[JsonValue] = []
    for basename in sorted(set(baseline_names) & set(merged_names)):
        left = baseline_names[basename]
        right = merged_names[basename]
        if len(left) == 1 and len(right) == 1:
            pairs.append(
                FilePair(
                    baseline=baseline_by_relative[left[0]],
                    merged=merged_by_relative[right[0]],
                    match_kind="unique_basename",
                )
            )
            paired_baseline.add(left[0])
            paired_merged.add(right[0])
        else:
            ambiguous.append(
                {"basename": basename, "baseline": left, "merged": right}
            )
    return PairingResult(
        pairs=tuple(pairs),
        unmatched_baseline=tuple(sorted(set(baseline_by_relative) - paired_baseline)),
        unmatched_merged=tuple(sorted(set(merged_by_relative) - paired_merged)),
        ambiguous=tuple(ambiguous),
    )
