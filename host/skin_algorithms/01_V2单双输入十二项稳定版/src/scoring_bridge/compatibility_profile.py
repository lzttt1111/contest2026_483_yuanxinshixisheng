from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.scoring_bridge.compatibility_cv import select_nested_compatibility
from src.scoring_bridge.compatibility_models import CompatibilityRow
from src.scoring_bridge.compatibility_serialization import (
    compatibility_model_document,
)
from src.scoring_bridge.split_contract import (
    FoldAssignment,
    fold_assignments_sha256,
)


@dataclass(frozen=True, slots=True)
class CompatibilityProfileProvenance:
    development_manifest_sha256: str
    fold_manifest_sha256: str
    official_v011_profile_sha256: str
    registry_sha256: str
    formula_registry_sha256: str
    code_sha: str


@dataclass(frozen=True, slots=True)
class CompatibilityProfileReceipt:
    profile_path: Path
    profile_sha256: str


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build_compatibility_profile(
    *,
    rows_by_dimension: Mapping[str, Sequence[CompatibilityRow]],
    assignments: Mapping[str, FoldAssignment],
    group_weights_by_dimension: Mapping[str, Mapping[str, float]],
    provenance: CompatibilityProfileProvenance,
    output_dir: Path,
    seed: int,
) -> CompatibilityProfileReceipt:
    if (
        fold_assignments_sha256(tuple(assignments.values()))
        != provenance.fold_manifest_sha256
    ):
        raise ValueError("compatibility assignments do not match fold manifest SHA")
    dimensions: dict[str, dict] = {}
    for dimension_id, rows in sorted(rows_by_dimension.items()):
        selection = select_nested_compatibility(
            rows,
            assignments=assignments,
            group_weights=group_weights_by_dimension[dimension_id],
            seed=seed,
        )
        dimensions[dimension_id] = {
            "status": "candidate",
            "method": selection.method.value,
            "regularization": selection.regularization,
            "model": compatibility_model_document(selection.model),
            "development_count": len(rows),
            "identity_diagnostics": asdict(selection.identity_diagnostics),
            "selected_diagnostics": asdict(selection.selected_diagnostics),
        }
    body = {
        "schema_version": "aisia_compatibility_profile_v2_candidate_1",
        "profile_id": "consumer_rgb_v011_compatibility_v2_candidate",
        "status": "candidate",
        "score_direction": "higher_is_more_burden",
        "provenance": asdict(provenance),
        "dimensions": dimensions,
        "fallback_profile": "aisia_scoring_v0.1.1_integrity_20260806",
        "medical_boundary": "engineering compatibility candidate; not clinical calibration",
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
    path = output_dir / "consumer_rgb_v011_compatibility_v2_candidate.json"
    _atomic_write(
        path,
        (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return CompatibilityProfileReceipt(path, profile_sha)


__all__ = [
    "CompatibilityProfileProvenance",
    "CompatibilityProfileReceipt",
    "build_compatibility_profile",
]
