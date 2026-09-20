from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, RootModel

from src.scoring_bridge.builder import load_latest_observations
from src.scoring_bridge.population_profile import (
    NumericFeatures,
    PopulationMetricSpec,
    build_population_profile,
)
from src.scoring_bridge.population_serialization import (
    population_profile_document,
)


class ActiveSubject(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    source_relative_path: str


class FormulaMetric(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    module_id: str
    group_id: str
    metric_id: str
    direction: str
    unit: str
    group_weight: float
    metric_weight: float


class FormulaRegistry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    formula_version: str
    metric_formulas: dict[str, FormulaMetric]


class NumericFeaturesDocument(RootModel[dict[str, dict[str, dict[str, int | float | None]]]]):
    pass


@dataclass(frozen=True, slots=True)
class PopulationProfileReceipt:
    profile_path: Path
    profile_sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build_population_profile_from_artifacts(
    *,
    active_pairs_path: Path,
    observation_paths: Sequence[Path],
    formula_registry_path: Path,
    output_dir: Path,
    module_ids: Sequence[str],
    code_sha: str,
) -> PopulationProfileReceipt:
    active_subjects = tuple(
        ActiveSubject.model_validate_json(line)
        for line in active_pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    observations, _ = load_latest_observations(observation_paths)
    features: list[NumericFeatures] = []
    for subject in active_subjects:
        observation = observations.get(subject.source_relative_path)
        if observation is None:
            raise ValueError(
                f"active subject has no eligible observation: {subject.source_relative_path}"
            )
        parsed = NumericFeaturesDocument.model_validate(
            observation.get("v2_scoring_features") or {}
        )
        features.append(parsed.root)
    registry = FormulaRegistry.model_validate_json(
        formula_registry_path.read_text(encoding="utf-8")
    )
    selected_modules = set(module_ids)
    metric_specs = tuple(
        PopulationMetricSpec(
            module_id=metric.module_id,
            group_id=metric.group_id,
            metric_id=metric.metric_id,
            direction=metric.direction,
            unit=metric.unit,
            group_weight=metric.group_weight,
            metric_weight=metric.metric_weight,
        )
        for metric in registry.metric_formulas.values()
        if metric.module_id in selected_modules
    )
    profile = build_population_profile(
        observations=tuple(features),
        metric_specs=metric_specs,
        module_ids=tuple(module_ids),
    )
    profile_document = population_profile_document(profile)
    body = {
        "schema_version": "aisia_population_ecdf_profile_v1_candidate_1",
        **profile_document,
        "provenance": {
            "active_pairs_sha256": _sha(active_pairs_path),
            "formula_registry_sha256": _sha(formula_registry_path),
            "formula_version": registry.formula_version,
            "code_sha": code_sha,
            "observation_sources": [
                {"name": path.name, "sha256": _sha(path)}
                for path in sorted(observation_paths, key=lambda item: item.name)
            ],
        },
        "medical_boundary": (
            "engineering population-relative burden; not clinical calibration"
        ),
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    profile_sha = hashlib.sha256(canonical).hexdigest()
    document = {**body, "profile_sha256": profile_sha}
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "population_ecdf_hybrid_v1_candidate.json"
    _atomic_write(
        profile_path,
        (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return PopulationProfileReceipt(profile_path, profile_sha)


__all__ = [
    "PopulationProfileReceipt",
    "build_population_profile_from_artifacts",
]
