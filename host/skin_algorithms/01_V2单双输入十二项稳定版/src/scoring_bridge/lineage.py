from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.manifest import DIMENSION_IDS


class ConfirmationManifestRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    relative_path: str
    size: int
    mtime_ns: int
    subject_id: str
    legacy_record_sha256: str


class ConfirmationPairRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    source_relative_path: str
    legacy_metrics_relative_path: str
    legacy_record_sha256: str


class DevelopmentRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    source_relative_path: str


class FoldRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    relative_path: str


class FoldMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_count: int
    active_pairs_sha256: str
    manifest_sha256: str


class ConfirmationFreezeMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    candidate_count: int
    manifest_sha256: str
    pairs_sha256: str
    grade_counts: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class ConfirmationLineageReceipt:
    subject_count: int
    subject_ids: frozenset[str]
    manifest_sha256: str
    pairs_sha256: str


@dataclass(frozen=True, slots=True)
class DevelopmentFoldLineageReceipt:
    subject_count: int
    subject_ids: frozenset[str]
    development_manifest_sha256: str
    fold_manifest_sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("confirmation lineage path escapes the approved root") from exc
    return path


def _rows(path: Path, row_type: type[BaseModel]) -> tuple[BaseModel, ...]:
    return tuple(
        row_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def validate_confirmation_lineage(
    *,
    manifest_path: Path,
    pairs_path: Path,
    legacy_root: Path,
    expected_count: int = 250,
) -> ConfirmationLineageReceipt:
    root = legacy_root.resolve(strict=True)
    manifest_rows = tuple(
        ConfirmationManifestRow.model_validate(row)
        for row in _rows(manifest_path, ConfirmationManifestRow)
    )
    pair_rows = tuple(
        ConfirmationPairRow.model_validate(row)
        for row in _rows(pairs_path, ConfirmationPairRow)
    )
    if len(manifest_rows) != expected_count:
        raise ValueError("confirmation subject count does not match frozen contract")
    if len(manifest_rows) != len(pair_rows):
        raise ValueError("confirmation manifest and pairs counts do not match")
    manifest_by_subject = {row.subject_id: row for row in manifest_rows}
    pairs_by_subject = {row.subject_id: row for row in pair_rows}
    if len(manifest_by_subject) != len(manifest_rows):
        raise ValueError("confirmation manifest contains duplicate subjects")
    if len(pairs_by_subject) != len(pair_rows):
        raise ValueError("confirmation pairs contain duplicate subjects")
    if set(manifest_by_subject) != set(pairs_by_subject):
        raise ValueError("confirmation manifest and pairs subjects do not match")
    for subject_id, manifest in manifest_by_subject.items():
        pair = pairs_by_subject[subject_id]
        if manifest.relative_path != pair.source_relative_path:
            raise ValueError(f"confirmation input path mismatch: {subject_id}")
        if manifest.legacy_record_sha256 != pair.legacy_record_sha256:
            raise ValueError(f"confirmation legacy record SHA mismatch: {subject_id}")
        image = _inside(root, manifest.relative_path)
        metrics = _inside(root, pair.legacy_metrics_relative_path)
        stat = image.stat()
        if stat.st_size != manifest.size or stat.st_mtime_ns != manifest.mtime_ns:
            raise ValueError(f"confirmation input signature mismatch: {subject_id}")
        if _sha(metrics) != manifest.legacy_record_sha256:
            raise ValueError(f"confirmation legacy record SHA mismatch: {subject_id}")
    return ConfirmationLineageReceipt(
        subject_count=len(manifest_rows),
        subject_ids=frozenset(manifest_by_subject),
        manifest_sha256=_sha(manifest_path),
        pairs_sha256=_sha(pairs_path),
    )


def validate_development_fold_lineage(
    *,
    development_manifest_path: Path,
    fold_manifest_path: Path,
    fold_metadata_path: Path,
    expected_count: int = 1000,
) -> DevelopmentFoldLineageReceipt:
    development_rows = tuple(
        DevelopmentRow.model_validate(row)
        for row in _rows(development_manifest_path, DevelopmentRow)
    )
    fold_rows = tuple(
        FoldRow.model_validate(row)
        for row in _rows(fold_manifest_path, FoldRow)
    )
    metadata = FoldMetadata.model_validate(json.loads(
        fold_metadata_path.read_text(encoding="utf-8")
    ))
    development_by_subject = {
        row.subject_id: row.source_relative_path for row in development_rows
    }
    fold_by_subject = {row.subject_id: row.relative_path for row in fold_rows}
    if len(development_rows) != expected_count:
        raise ValueError("development subject count does not match frozen contract")
    if len(development_by_subject) != len(development_rows):
        raise ValueError("development manifest contains duplicate subjects")
    if len(fold_by_subject) != len(fold_rows):
        raise ValueError("fold manifest contains duplicate subjects")
    if development_by_subject != fold_by_subject:
        raise ValueError("development and fold subjects or paths do not match")
    development_sha = _sha(development_manifest_path)
    fold_sha = _sha(fold_manifest_path)
    if metadata.subject_count != len(development_rows):
        raise ValueError("fold metadata subject count mismatch")
    if metadata.active_pairs_sha256 != development_sha:
        raise ValueError("fold metadata development manifest SHA mismatch")
    if metadata.manifest_sha256 != fold_sha:
        raise ValueError("fold metadata manifest SHA mismatch")
    return DevelopmentFoldLineageReceipt(
        subject_count=len(development_rows),
        subject_ids=frozenset(development_by_subject),
        development_manifest_sha256=development_sha,
        fold_manifest_sha256=fold_sha,
    )


def validate_confirmation_freeze_metadata(
    *,
    metadata_path: Path,
    lineage: ConfirmationLineageReceipt,
    minimum_per_grade: int = 15,
) -> None:
    metadata = ConfirmationFreezeMetadata.model_validate(json.loads(
        metadata_path.read_text(encoding="utf-8")
    ))
    if metadata.candidate_count != lineage.subject_count:
        raise ValueError("confirmation metadata subject count mismatch")
    if metadata.manifest_sha256 != lineage.manifest_sha256:
        raise ValueError("confirmation metadata manifest SHA mismatch")
    if metadata.pairs_sha256 != lineage.pairs_sha256:
        raise ValueError("confirmation metadata pairs SHA mismatch")
    if set(metadata.grade_counts) != set(DIMENSION_IDS):
        raise ValueError("confirmation metadata dimensions mismatch")
    for dimension_id in DIMENSION_IDS:
        counts = metadata.grade_counts[dimension_id]
        if sum(counts.get(str(grade), 0) for grade in range(5)) != lineage.subject_count:
            raise ValueError(
                f"confirmation grade count total mismatch: {dimension_id}"
            )
        if any(counts.get(str(grade), 0) < minimum_per_grade for grade in range(5)):
            raise ValueError(
                f"confirmation grade coverage is insufficient: {dimension_id}"
            )


__all__ = [
    "ConfirmationLineageReceipt",
    "DevelopmentFoldLineageReceipt",
    "validate_confirmation_lineage",
    "validate_confirmation_freeze_metadata",
    "validate_development_fold_lineage",
]
