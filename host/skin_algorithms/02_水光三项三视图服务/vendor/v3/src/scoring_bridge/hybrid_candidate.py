from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.scoring_bridge.hybrid_profile import (
    finalize_hybrid_profile_document,
    verify_hybrid_profile_document,
)
from src.scoring_bridge.confirmation_report import verify_confirmation_report


@dataclass(frozen=True, slots=True)
class HybridCandidateReceipt:
    path: Path
    sha256: str


def _component_sha(document: Mapping[str, Any]) -> str:
    expected = document.get("profile_sha256")
    body = {
        key: value for key, value in document.items()
        if key != "profile_sha256"
    }
    actual = hashlib.sha256(json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    if expected != actual:
        raise ValueError("component profile SHA mismatch")
    return actual


def _subject_set_sha(subject_ids: frozenset[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(subject_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _technical_status(
    confirmation: Mapping[str, Any],
    section: str,
    item_id: str,
) -> str:
    items = confirmation.get(section)
    if not isinstance(items, dict):
        raise ValueError(f"confirmation section is missing: {section}")
    item = items.get(item_id)
    if not isinstance(item, dict):
        raise ValueError(f"confirmation item is missing: {item_id}")
    status = item.get("status")
    if status not in {"promoted", "blocked"}:
        raise ValueError(f"confirmation status is invalid: {item_id}")
    return str(status)


def _provenance(document: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    value = document.get("provenance")
    if not isinstance(value, dict):
        raise ValueError(f"{label} provenance is missing")
    return value


def _bound_sha(label: str, *values: Any) -> str:
    if any(not isinstance(value, str) or len(value) != 64 for value in values):
        raise ValueError(f"{label} SHA is missing")
    if len(set(values)) != 1:
        raise ValueError(f"{label} SHA binding mismatch")
    return str(values[0])


def _validate_asset_binding(
    *,
    compatibility_profile: Mapping[str, Any],
    population_profile: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    provided: Mapping[str, Any],
    compatibility_sha: str,
    population_sha: str,
) -> None:
    compatibility = _provenance(compatibility_profile, "compatibility profile")
    population = _provenance(population_profile, "population profile")
    confirmation = _provenance(confirmation_report, "confirmation report")
    _bound_sha(
        "development manifest",
        provided.get("development_manifest_sha256"),
        compatibility.get("development_manifest_sha256"),
        population.get("active_pairs_sha256"),
    )
    _bound_sha(
        "fold manifest",
        provided.get("fold_manifest_sha256"),
        compatibility.get("fold_manifest_sha256"),
    )
    _bound_sha(
        "confirmation manifest",
        provided.get("confirmation_manifest_sha256"),
        confirmation.get("confirmation_manifest_sha256"),
    )
    _bound_sha(
        "confirmation pairs",
        provided.get("confirmation_pairs_sha256"),
        confirmation.get("confirmation_pairs_sha256"),
    )
    _bound_sha(
        "official V0.1.1 profile",
        provided.get("official_v011_profile_sha256"),
        compatibility.get("official_v011_profile_sha256"),
        confirmation.get("official_v011_profile_sha256"),
    )
    _bound_sha(
        "V0.1.1 registry",
        provided.get("registry_sha256"),
        compatibility.get("registry_sha256"),
    )
    _bound_sha(
        "formula registry",
        provided.get("formula_registry_sha256"),
        compatibility.get("formula_registry_sha256"),
        population.get("formula_registry_sha256"),
    )
    _bound_sha(
        "compatibility profile",
        compatibility_sha,
        confirmation.get("compatibility_profile_sha256"),
    )
    _bound_sha(
        "population profile",
        population_sha,
        confirmation.get("population_profile_sha256"),
    )


def build_hybrid_candidate(
    *,
    compatibility_profile: Mapping[str, Any],
    population_profile: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    development_subject_ids: frozenset[str],
    confirmation_subject_ids: frozenset[str],
    output_path: Path,
    provenance: Mapping[str, Any],
) -> HybridCandidateReceipt:
    if len(development_subject_ids) != 1000 or len(confirmation_subject_ids) != 250:
        raise ValueError("hybrid candidate subject count does not match frozen contract")
    if development_subject_ids.intersection(confirmation_subject_ids):
        raise ValueError("development and confirmation subjects overlap")
    if confirmation_report.get("attempted_count") != len(confirmation_subject_ids):
        raise ValueError("confirmation report subject count mismatch")
    compatibility_sha = _component_sha(compatibility_profile)
    population_sha = _component_sha(population_profile)
    confirmation_sha = verify_confirmation_report(confirmation_report)
    _validate_asset_binding(
        compatibility_profile=compatibility_profile,
        population_profile=population_profile,
        confirmation_report=confirmation_report,
        provided=provenance,
        compatibility_sha=compatibility_sha,
        population_sha=population_sha,
    )
    dimensions = compatibility_profile.get("dimensions")
    modules = population_profile.get("modules")
    if not isinstance(dimensions, dict) or not isinstance(modules, dict):
        raise ValueError("candidate component sections are invalid")
    legacy_dimensions: dict[str, dict[str, Any]] = {}
    technical_pass = False
    for dimension_id, dimension in sorted(dimensions.items()):
        if not isinstance(dimension, dict) or not isinstance(dimension.get("model"), dict):
            raise ValueError(f"compatibility model is invalid: {dimension_id}")
        technical = _technical_status(
            confirmation_report,
            "compatibility_dimensions",
            dimension_id,
        )
        technical_pass |= technical == "promoted"
        legacy_dimensions[dimension_id] = {
            "status": "candidate" if technical == "promoted" else "blocked",
            "technical_status": technical,
            "route": "compatibility_v2" if technical == "promoted" else "v011_fallback",
            "fallback_profile": "aisia_scoring_v0.1.1_integrity_20260806",
            "model": dimension["model"],
        }
    population_modules: dict[str, dict[str, Any]] = {}
    for module_id in sorted(modules):
        technical = _technical_status(
            confirmation_report,
            "population_modules",
            module_id,
        )
        technical_pass |= technical == "promoted"
        population_modules[module_id] = {
            "status": "internal_candidate" if technical == "promoted" else "blocked",
            "technical_status": technical,
            "doctor_approval": "pending" if technical == "promoted" else "not_applicable",
            "user_report_exposure": False,
        }
    numerical_core_sha = hashlib.sha256(
        f"{compatibility_sha}:{population_sha}".encode()
    ).hexdigest()
    body = {
        "schema_version": "aisia_hybrid_scoring_profile_v1",
        "profile_id": "consumer_rgb_hybrid_v2_candidate",
        "status": "candidate" if technical_pass else "blocked",
        "capture_profile": "consumer",
        "institution_temporary_alias_id": "institution_hybrid_v2_temporary_alias",
        "score_direction": "higher_is_more_burden",
        "numerical_core_sha256": numerical_core_sha,
        "provenance": {
            **dict(provenance),
            "development_subject_count": len(development_subject_ids),
            "development_subjects_sha256": _subject_set_sha(development_subject_ids),
            "confirmation_subject_count": len(confirmation_subject_ids),
            "confirmation_subjects_sha256": _subject_set_sha(confirmation_subject_ids),
            "compatibility_profile_sha256": compatibility_sha,
            "population_profile_sha256": population_sha,
            "confirmation_report_sha256": confirmation_sha,
        },
        "legacy_dimensions": legacy_dimensions,
        "population_modules": population_modules,
        "population_profile": {
            key: population_profile[key]
            for key in ("profile_id", "status", "development_count", "modules")
        },
        "medical_boundary": (
            "engineering compatibility and population-relative burden candidate; "
            "not clinical calibration"
        ),
    }
    document = finalize_hybrid_profile_document(body)
    profile_sha = verify_hybrid_profile_document(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return HybridCandidateReceipt(output_path, profile_sha)


__all__ = ["HybridCandidateReceipt", "build_hybrid_candidate"]
