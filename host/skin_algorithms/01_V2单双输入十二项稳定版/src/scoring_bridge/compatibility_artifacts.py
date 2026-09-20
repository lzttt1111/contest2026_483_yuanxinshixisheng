from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.builder import (
    load_latest_observations,
    v011_features_from_observation,
)
from src.scoring_bridge.compatibility_dataset import (
    compatibility_rows_from_score_documents,
)
from src.scoring_bridge.compatibility_models import CompatibilityRow
from src.scoring_bridge.legacy_scoring import features_from_metrics
from src.scoring_calibration.v011.registry import REGISTRY
from src.scoring_calibration.v011.scoring import score_observation


class ActiveCompatibilityPair(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    source_relative_path: str
    legacy_metrics_relative_path: str
    legacy_record_sha256: str | None = None


class OfficialReferences(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    references: dict[str, list[float]]


@dataclass(frozen=True, slots=True)
class CompatibilityDevelopmentData:
    rows_by_dimension: Mapping[str, tuple[CompatibilityRow, ...]]
    group_weights_by_dimension: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class CompatibilityConfirmationData:
    rows_by_dimension: Mapping[str, tuple[CompatibilityRow, ...]]
    group_weights_by_dimension: Mapping[str, Mapping[str, float]]
    unavailable_subjects: tuple[str, ...]


def _score(features: Mapping[str, object], references: dict[str, list[float]]) -> dict:
    return score_observation(
        {
            "status": "success",
            "features": features,
            "input_quality_gate": {"status": "PASS", "reason_codes": []},
        },
        references,
    )


def _legacy_metrics_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("legacy metrics path escapes the approved root") from exc
    return candidate


def load_compatibility_development_data(
    *,
    active_pairs_path: Path,
    observation_paths: Sequence[Path],
    legacy_root: Path,
    official_profile_path: Path,
) -> CompatibilityDevelopmentData:
    root = legacy_root.resolve(strict=True)
    pairs = tuple(
        ActiveCompatibilityPair.model_validate_json(line)
        for line in active_pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    observations, _ = load_latest_observations(observation_paths)
    official = OfficialReferences.model_validate_json(
        official_profile_path.read_text(encoding="utf-8")
    )
    rows: dict[str, list[CompatibilityRow]] = {
        dimension.id: [] for dimension in REGISTRY
    }
    for pair in pairs:
        observation = observations.get(pair.source_relative_path)
        if observation is None:
            raise ValueError(
                f"active pair has no eligible observation: {pair.source_relative_path}"
            )
        legacy_features = features_from_metrics(
            _legacy_metrics_path(root, pair.legacy_metrics_relative_path)
        )
        current_features = v011_features_from_observation(observation)
        paired = compatibility_rows_from_score_documents(
            subject_id=pair.subject_id,
            current=_score(current_features, official.references),
            legacy=_score(legacy_features, official.references),
        )
        for dimension_id, row in paired.items():
            rows[dimension_id].append(row)
    weights = {
        dimension.id: {
            group.id: group.weight for group in dimension.groups
        }
        for dimension in REGISTRY
    }
    return CompatibilityDevelopmentData(
        rows_by_dimension={
            dimension_id: tuple(dimension_rows)
            for dimension_id, dimension_rows in rows.items()
        },
        group_weights_by_dimension=weights,
    )


def load_compatibility_confirmation_data(
    *,
    confirmation_pairs_path: Path,
    observation_paths: Sequence[Path],
    legacy_root: Path,
    official_profile_path: Path,
) -> CompatibilityConfirmationData:
    root = legacy_root.resolve(strict=True)
    pairs = tuple(
        ActiveCompatibilityPair.model_validate_json(line)
        for line in confirmation_pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    observations, _ = load_latest_observations(observation_paths)
    official = OfficialReferences.model_validate_json(
        official_profile_path.read_text(encoding="utf-8")
    )
    rows: dict[str, list[CompatibilityRow]] = {
        dimension.id: [] for dimension in REGISTRY
    }
    unavailable: list[str] = []
    for pair in pairs:
        observation = observations.get(pair.source_relative_path)
        if observation is None:
            unavailable.append(pair.subject_id)
            continue
        legacy_path = _legacy_metrics_path(root, pair.legacy_metrics_relative_path)
        if pair.legacy_record_sha256 is None:
            raise ValueError(f"confirmation legacy record SHA is missing: {pair.subject_id}")
        if hashlib.sha256(legacy_path.read_bytes()).hexdigest() != pair.legacy_record_sha256:
            raise ValueError(f"confirmation legacy record SHA mismatch: {pair.subject_id}")
        legacy_features = features_from_metrics(legacy_path)
        current_features = v011_features_from_observation(observation)
        paired = compatibility_rows_from_score_documents(
            subject_id=pair.subject_id,
            current=_score(current_features, official.references),
            legacy=_score(legacy_features, official.references),
        )
        for dimension_id, row in paired.items():
            rows[dimension_id].append(row)
    weights = {
        dimension.id: {
            group.id: group.weight for group in dimension.groups
        }
        for dimension in REGISTRY
    }
    return CompatibilityConfirmationData(
        rows_by_dimension={
            dimension_id: tuple(dimension_rows)
            for dimension_id, dimension_rows in rows.items()
        },
        group_weights_by_dimension=weights,
        unavailable_subjects=tuple(sorted(unavailable)),
    )


__all__ = [
    "CompatibilityConfirmationData",
    "CompatibilityDevelopmentData",
    "load_compatibility_confirmation_data",
    "load_compatibility_development_data",
]
