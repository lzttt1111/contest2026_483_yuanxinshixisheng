from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.scoring_bridge as scoring_bridge


def _profile(status: str, dimension_status: str = "promoted") -> dict:
    finalize = getattr(scoring_bridge, "finalize_hybrid_profile_document", None)
    assert callable(finalize)
    return finalize({
        "schema_version": "aisia_hybrid_scoring_profile_v1",
        "profile_id": "consumer_rgb_hybrid_v2_candidate",
        "status": status,
        "capture_profile": "consumer",
        "institution_temporary_alias_id": "institution_hybrid_v2_temporary_alias",
        "score_direction": "higher_is_more_burden",
        "numerical_core_sha256": "1" * 64,
        "provenance": {
            "development_manifest_sha256": "2" * 64,
            "confirmation_manifest_sha256": "3" * 64,
            "registry_sha256": "4" * 64,
            "formula_registry_sha256": "5" * 64,
            "code_sha": "6" * 40,
        },
        "legacy_dimensions": {
            "visible_pores": {
                "status": dimension_status,
                "route": "compatibility_v2",
                "fallback_profile": "aisia_scoring_v0.1.1_integrity_20260806",
                "model": {"method": "identity"},
            }
        },
        "population_profile": {"profile_id": "population_ecdf_hybrid_v1"},
        "medical_boundary": "engineering relative burden only",
    })


def test_candidate_profile_is_refused_by_production_loader(tmp_path: Path) -> None:
    load = getattr(scoring_bridge, "load_promoted_hybrid_profile", None)
    assert callable(load)
    document = _profile("candidate")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="not promoted"):
        load(path, allowlisted_sha256={document["profile_sha256"]})


def test_tampered_profile_fails_closed(tmp_path: Path) -> None:
    load = getattr(scoring_bridge, "load_promoted_hybrid_profile")
    document = _profile("promoted")
    original_sha = document["profile_sha256"]
    document["medical_boundary"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA"):
        load(path, allowlisted_sha256={original_sha})


def test_promoted_profile_uses_dimension_fallback_independently(
    tmp_path: Path,
) -> None:
    load = getattr(scoring_bridge, "load_promoted_hybrid_profile")
    resolve = getattr(scoring_bridge, "resolve_legacy_dimension_route", None)
    identity = getattr(scoring_bridge, "capture_profile_identity", None)
    assert callable(resolve)
    assert callable(identity)
    document = _profile("promoted", dimension_status="blocked")
    path = tmp_path / "promoted.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load(path, allowlisted_sha256={document["profile_sha256"]})

    assert resolve(loaded, "visible_pores") == "v011_fallback"
    assert identity(loaded, "consumer")[0] == loaded["profile_id"]
    assert identity(loaded, "institution")[0] == loaded["institution_temporary_alias_id"]
    assert identity(loaded, "institution")[1] == loaded["numerical_core_sha256"]
