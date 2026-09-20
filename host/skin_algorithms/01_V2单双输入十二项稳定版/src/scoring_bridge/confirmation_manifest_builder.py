from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.scoring_bridge.confirmation_selection import (
    ConfirmationCandidate,
    select_confirmation_pool,
)
from src.scoring_bridge.manifest import (
    DIMENSION_IDS,
    LegacyCandidate,
    attach_legacy_scores,
    deduplicate_subjects,
    discover_candidates,
)


GRADE_THRESHOLDS = (20.0, 40.0, 60.0, 80.0)


@dataclass(frozen=True, slots=True)
class ConfirmationPoolReceipt:
    manifest_path: Path
    pairs_path: Path
    metadata_path: Path
    manifest_sha256: str
    pairs_sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_image(sample_dir: Path) -> Path:
    images = tuple(
        path for path in sample_dir.iterdir()
        if path.is_file() and path.name.startswith("00_输入图片.")
    )
    if len(images) != 1:
        raise ValueError(f"sample input image count is invalid: {sample_dir.name}")
    return images[0]


def _quality_band(metrics_path: Path) -> str:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = (
        (payload.get("九项") or {})
        .get("redness", {})
        .get("核心总体指标", {})
        .get("quality_score")
    )
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "unknown"
    if score >= 80:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _grade(score: float) -> int:
    return sum(score >= threshold for threshold in GRADE_THRESHOLDS)


def _candidate_row(candidate: LegacyCandidate) -> ConfirmationCandidate:
    if candidate.scores is None:
        raise ValueError("legacy candidate has no scores")
    metrics = candidate.sample_dir / "九项核心量化指标.json"
    image = _input_image(candidate.sample_dir)
    return ConfirmationCandidate(
        subject_id=candidate.subject_id,
        relative_path=image.as_posix(),
        batch_name=candidate.batch_name,
        grades=tuple(_grade(score) for score in candidate.scores),
        quality_band=_quality_band(metrics),
        image_suffix=image.suffix.lower(),
        rank_key=candidate.rank_key,
    )


def _jsonl(path: Path, rows: Sequence[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def build_confirmation_pool_from_legacy(
    *,
    legacy_root: Path,
    development_subjects: set[str],
    references: dict[str, list[float]],
    output_dir: Path,
    code_sha: str,
    seed: str = "20260827-confirmation-v1",
    batch_start: int = 20,
    batch_end: int = 30,
    count: int = 250,
    minimum_per_grade: int = 15,
) -> ConfirmationPoolReceipt:
    root = legacy_root.resolve(strict=True)
    discovered = deduplicate_subjects(discover_candidates(
        root,
        batch_start=batch_start,
        batch_end=batch_end,
        seed=seed,
    ))
    eligible = tuple(
        candidate for candidate in discovered
        if candidate.subject_id not in development_subjects
    )
    scored = attach_legacy_scores(
        eligible,
        references=references,
        profile_version="v011",
    )
    candidate_rows = tuple(_candidate_row(candidate) for candidate in scored)
    selected = select_confirmation_pool(
        candidate_rows,
        development_subjects=development_subjects,
        count=count,
        minimum_per_grade=minimum_per_grade,
        seed=seed,
    )
    by_subject = {candidate.subject_id: candidate for candidate in scored}
    manifest_rows: list[dict] = []
    pair_rows: list[dict] = []
    audit_rows: list[dict] = []
    for order, row in enumerate(selected):
        source = by_subject[row.subject_id]
        image = _input_image(source.sample_dir)
        metrics = source.sample_dir / "九项核心量化指标.json"
        stat = image.stat()
        metrics_sha = _sha(metrics)
        relative_image = image.relative_to(root).as_posix()
        relative_metrics = metrics.relative_to(root).as_posix()
        manifest_rows.append({
            "relative_path": relative_image,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "subject_id": row.subject_id,
            "legacy_record_sha256": metrics_sha,
            "split": "reserve",
        })
        pair_rows.append({
            "subject_id": row.subject_id,
            "sample_id": source.sample_name,
            "batch_name": source.batch_name,
            "source_relative_path": relative_image,
            "legacy_metrics_relative_path": relative_metrics,
            "legacy_record_sha256": metrics_sha,
            "legacy_scores": dict(zip(DIMENSION_IDS, source.scores, strict=True)),
            "split": "confirmation",
        })
        audit_rows.append({
            "order": order,
            "subject_id": row.subject_id,
            "grades": row.grades,
            "batch_name": row.batch_name,
            "quality_band": row.quality_band,
            "image_suffix": row.image_suffix,
            "pose_stratum": "unknown",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "confirmation_pool_250_v1.jsonl"
    pairs_path = output_dir / "confirmation_pairs_250_v1.jsonl"
    manifest_sha = _jsonl(manifest_path, manifest_rows)
    pairs_sha = _jsonl(pairs_path, pair_rows)
    metadata_path = output_dir / "confirmation_pool_250_v1.metadata.json"
    grade_counts = {
        dimension: {
            str(grade): sum(row.grades[index] == grade for row in selected)
            for grade in range(5)
        }
        for index, dimension in enumerate(DIMENSION_IDS)
    }
    metadata = {
        "schema_version": "aisia_confirmation_pool_v1",
        "status": "frozen_before_run",
        "seed": seed,
        "candidate_count": len(selected),
        "development_subject_count": len(development_subjects),
        "manifest_sha256": manifest_sha,
        "pairs_sha256": pairs_sha,
        "code_sha": code_sha,
        "batch_range": [batch_start, batch_end],
        "grade_counts": grade_counts,
        "batch_counts": dict(Counter(row.batch_name for row in selected)),
        "quality_band_counts": dict(Counter(row.quality_band for row in selected)),
        "format_counts": dict(Counter(row.image_suffix for row in selected)),
        "pose_stratum": "unknown_in_legacy_source",
        "members": audit_rows,
    }
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)
    return ConfirmationPoolReceipt(
        manifest_path,
        pairs_path,
        metadata_path,
        manifest_sha,
        pairs_sha,
    )


__all__ = ["ConfirmationPoolReceipt", "build_confirmation_pool_from_legacy"]
