from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.builder import load_latest_observations
from src.scoring_bridge.manifest import DIMENSION_IDS
from src.scoring_bridge.split_contract import (
    FoldSample,
    build_nested_fold_assignments,
)


GRADE_THRESHOLDS: Final = (20.0, 40.0, 60.0, 80.0)
POSE_FLAGS: Final = frozenset({
    "POSE_YAW",
    "POSE_PITCH",
    "POSE_ROLL",
    "EXTREME_POSE",
    "PARTIAL_FACE",
})


class ActivePair(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    source_relative_path: str
    batch_name: str
    legacy_scores: dict[str, float]


class ObservationSignature(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    relative_path: str


class ObservationQuality(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: str = "UNKNOWN"
    flags: tuple[str, ...] = ()


class FoldObservation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    signature: ObservationSignature
    quality: ObservationQuality = ObservationQuality()


@dataclass(frozen=True, slots=True)
class FoldManifestReceipt:
    manifest_path: Path
    metadata_path: Path
    manifest_sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grade(value: float) -> int:
    return sum(value >= threshold for threshold in GRADE_THRESHOLDS)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _stratum(pair: ActivePair, observation: FoldObservation) -> str:
    grades = "".join(
        str(_grade(pair.legacy_scores[dimension]))
        for dimension in DIMENSION_IDS
    )
    pose = (
        "pose_warning"
        if POSE_FLAGS.intersection(observation.quality.flags)
        else "pose_regular"
    )
    suffix = Path(pair.source_relative_path).suffix.lower() or "none"
    return "|".join((
        pair.batch_name,
        grades,
        observation.quality.status,
        suffix,
        pose,
    ))


def build_development_fold_manifest(
    *,
    active_pairs_path: Path,
    observation_paths: Sequence[Path],
    output_dir: Path,
    seed: str,
) -> FoldManifestReceipt:
    pairs = tuple(
        ActivePair.model_validate_json(line)
        for line in active_pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    observations, _ = load_latest_observations(observation_paths)
    samples: list[FoldSample] = []
    for pair in pairs:
        raw = observations.get(pair.source_relative_path)
        if raw is None:
            raise ValueError(
                f"active pair has no eligible observation: {pair.source_relative_path}"
            )
        observation = FoldObservation.model_validate(raw)
        samples.append(FoldSample(
            subject_id=pair.subject_id,
            relative_path=pair.source_relative_path,
            stratum=_stratum(pair, observation),
        ))
    assignments = build_nested_fold_assignments(tuple(samples), seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "development_nested_folds_1000.jsonl"
    manifest_payload = "".join(
        json.dumps(
            asdict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        for row in assignments
    ).encode()
    _atomic_write(manifest_path, manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    metadata_path = output_dir / "development_nested_folds_1000.metadata.json"
    metadata = {
        "schema_version": "aisia_scoring_nested_folds_v1",
        "seed": seed,
        "subject_count": len(assignments),
        "active_pairs_sha256": _sha(active_pairs_path),
        "observation_sources": [
            {"name": path.name, "sha256": _sha(path)}
            for path in sorted(observation_paths, key=lambda item: item.name)
        ],
        "manifest_sha256": manifest_sha,
    }
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return FoldManifestReceipt(
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        manifest_sha256=manifest_sha,
    )


__all__ = ["FoldManifestReceipt", "build_development_fold_manifest"]
