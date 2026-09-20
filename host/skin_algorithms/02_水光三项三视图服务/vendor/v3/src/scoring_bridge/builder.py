from __future__ import annotations

import json
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.scoring_bridge.fitting import (
    GRADE_THRESHOLDS,
    apply_isotonic,
    fit_isotonic,
    mapping_diagnostics,
)
from src.scoring_bridge.legacy_scoring import (
    legacy_dimension_scores_from_features,
    prepare_references,
)
from src.scoring_bridge.manifest import DIMENSION_IDS


def load_latest_observations(
    paths: Iterable[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        (path.resolve() for path in paths if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    latest_terminal: dict[str, dict[str, Any]] = {}
    for path in ordered:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                observation = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = observation.get("signature") or {}
            relative = str(signature.get("relative_path") or "")
            if not relative:
                continue
            observation["_source_jsonl"] = str(path)
            observation["_source_line"] = line_number
            status = observation.get("status")
            if status in {"success", "partial_success", "failed"}:
                latest_terminal[relative] = observation
    successes = {
        relative: observation
        for relative, observation in latest_terminal.items()
        if (
            observation.get("status") == "success"
            and int(observation.get("success_items", 0)) == 12
        )
        or (
            observation.get("status") == "partial_success"
            and observation.get("failure_category") != "input_quality_reject"
            and (observation.get("quality") or {}).get("status") != "REJECT"
            and bool(observation.get("scoring_features"))
        )
    }
    rejected = [
        observation
        for relative, observation in latest_terminal.items()
        if relative not in successes
    ]
    return successes, rejected


def load_legacy_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def v011_features_from_observation(
    observation: dict[str, Any],
) -> dict[str, Any]:
    features = deepcopy(observation.get("scoring_features") or {})
    uv_density = (
        (observation.get("v2_scoring_features") or {})
        .get("pigmentation", {})
        .get("uv_spots", {})
        .get("uv_spot_density_per_100k_px")
    )
    if not isinstance(uv_density, bool) and isinstance(uv_density, (int, float)):
        features.setdefault("uv_spots", {}).setdefault(
            "UV样色素范围", {}
        )["density"] = float(uv_density)
    return features


def build_paired_score_rows(
    observations: dict[str, dict[str, Any]],
    legacy_pairs: Iterable[dict[str, Any]],
    references: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prepared = prepare_references(references)
    paired: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for pair in legacy_pairs:
        relative = str(pair["source_relative_path"])
        observation = observations.get(relative)
        if observation is None:
            unavailable.append({"relative_path": relative, "reason": "current_result_missing"})
            continue
        scores = legacy_dimension_scores_from_features(
            v011_features_from_observation(observation),
            prepared,
        )
        legacy = pair.get("legacy_scores") or {}
        for dimension in DIMENSION_IDS:
            current_score = scores.get(dimension)
            if current_score is None:
                unavailable.append({
                    "relative_path": relative,
                    "reason": f"current_{dimension}_unavailable",
                })
                continue
            target = legacy.get(dimension)
            if isinstance(target, bool) or not isinstance(target, (int, float)):
                unavailable.append({"relative_path": relative, "reason": f"legacy_{dimension}_missing"})
                continue
            paired.append({
                "relative_path": relative,
                "subject_id": pair.get("subject_id"),
                "split": pair.get("split"),
                "dimension_id": dimension,
                "current_score": float(current_score),
                "legacy_score": float(target),
            })
    return paired, unavailable


def _grade(value: float) -> int:
    return int(np.searchsorted(GRADE_THRESHOLDS, value, side="right"))


def fit_dimension_bridges(
    score_rows: Iterable[dict[str, Any]],
    *,
    minimum_validation: int = 150,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in score_rows]
    mappings: dict[str, Any] = {}
    for dimension in sorted({str(row["dimension_id"]) for row in rows}):
        train = [
            row for row in rows
            if row["dimension_id"] == dimension and row["split"] == "train"
        ]
        mappings[dimension] = fit_isotonic(
            [row["current_score"] for row in train],
            [row["legacy_score"] for row in train],
        )
    for row in rows:
        mapped = apply_isotonic(
            mappings[str(row["dimension_id"])],
            float(row["current_score"]),
        )
        row["mapped_score"] = round(mapped, 8)
        row["raw_error"] = round(float(row["current_score"] - row["legacy_score"]), 8)
        row["mapped_error"] = round(mapped - float(row["legacy_score"]), 8)
        row["legacy_grade"] = _grade(float(row["legacy_score"]))
        row["current_grade"] = _grade(float(row["current_score"]))
        row["mapped_grade"] = _grade(mapped)
    by_dimension: dict[str, Any] = {}
    enough_samples = bool(mappings)
    promotion_allowed = bool(mappings)
    for dimension in mappings:
        splits: dict[str, Any] = {}
        for split in ("train", "validation"):
            subset = [
                row for row in rows
                if row["dimension_id"] == dimension and row["split"] == split
            ]
            if not subset:
                continue
            target = [row["legacy_score"] for row in subset]
            current = [row["current_score"] for row in subset]
            mapped = [row["mapped_score"] for row in subset]
            splits[split] = {
                "before_mapping": mapping_diagnostics(target, current),
                "after_mapping": mapping_diagnostics(target, mapped),
            }
        validation = splits.get("validation", {}).get("after_mapping", {})
        enough_samples &= int(validation.get("count", 0)) >= minimum_validation
        before = splits.get("validation", {}).get("before_mapping", {})
        promotion_allowed &= bool(
            int(validation.get("severe_reversal_count", 1)) == 0
            and float(validation.get("mean_absolute_error", float("inf")))
            <= float(before.get("mean_absolute_error", float("-inf")))
        )
        by_dimension[dimension] = splits
    return mappings, rows, {
        "status": "candidate" if enough_samples else "insufficient_samples",
        "promotion_gate": "pass" if enough_samples and promotion_allowed else "blocked",
        "minimum_validation_per_dimension": minimum_validation,
        "row_count": len(rows),
        "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        "dimensions": by_dimension,
    }


__all__ = [
    "build_paired_score_rows",
    "fit_dimension_bridges",
    "load_legacy_pairs",
    "load_latest_observations",
    "v011_features_from_observation",
]
