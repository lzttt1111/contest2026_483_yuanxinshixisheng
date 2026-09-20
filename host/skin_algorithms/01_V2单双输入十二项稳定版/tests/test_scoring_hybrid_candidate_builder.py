from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.scoring_bridge as scoring_bridge


def _component(body: dict) -> dict:
    import hashlib

    canonical = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return {**body, "profile_sha256": hashlib.sha256(canonical).hexdigest()}


def _bound_inputs(tmp_path: Path):
    provenance = {
        "development_manifest_sha256": "1" * 64,
        "fold_manifest_sha256": "2" * 64,
        "confirmation_manifest_sha256": "3" * 64,
        "confirmation_pairs_sha256": "4" * 64,
        "official_v011_profile_sha256": "5" * 64,
        "registry_sha256": "6" * 64,
        "formula_registry_sha256": "7" * 64,
        "code_sha": "8" * 40,
    }
    compatibility = _component({
        "profile_id": "compatibility-candidate",
        "status": "candidate",
        "provenance": {
            key: provenance[key]
            for key in (
                "development_manifest_sha256",
                "fold_manifest_sha256",
                "official_v011_profile_sha256",
                "registry_sha256",
                "formula_registry_sha256",
                "code_sha",
            )
        },
        "dimensions": {
            "visible_pores": {"model": {"method": "identity"}},
            "combined_pigmentation": {"model": {"method": "scalar_pava"}},
        },
    })
    population = _component({
        "profile_id": "population-candidate",
        "status": "candidate",
        "development_count": 1000,
        "provenance": {
            "active_pairs_sha256": provenance["development_manifest_sha256"],
            "formula_registry_sha256": provenance["formula_registry_sha256"],
            "code_sha": provenance["code_sha"],
        },
        "modules": {
            "acne_activity": {"status": "candidate"},
            "oil_tendency": {"status": "blocked"},
        },
    })
    write_confirmation = getattr(scoring_bridge, "write_confirmation_report")
    receipt = write_confirmation(
        output_path=tmp_path / "confirmation.json",
        attempted_count=250,
        compatibility_results={
            "visible_pores": {"status": "promoted"},
            "combined_pigmentation": {"status": "blocked"},
        },
        population_results={
            "acne_activity": {"status": "promoted"},
            "oil_tendency": {"status": "blocked"},
        },
        provenance={
            "confirmation_manifest_sha256": provenance["confirmation_manifest_sha256"],
            "confirmation_pairs_sha256": provenance["confirmation_pairs_sha256"],
            "official_v011_profile_sha256": provenance["official_v011_profile_sha256"],
            "compatibility_profile_sha256": compatibility["profile_sha256"],
            "population_profile_sha256": population["profile_sha256"],
            "code_sha": provenance["code_sha"],
        },
    )
    confirmation = json.loads(receipt.path.read_text(encoding="utf-8"))
    development_subjects = frozenset(f"development-{index}" for index in range(1000))
    confirmation_subjects = frozenset(f"confirmation-{index}" for index in range(250))
    return (
        compatibility,
        population,
        confirmation,
        provenance,
        development_subjects,
        confirmation_subjects,
    )


def test_hybrid_candidate_keeps_technical_routes_separate_from_approval(
    tmp_path: Path,
) -> None:
    build = getattr(scoring_bridge, "build_hybrid_candidate", None)
    assert callable(build)
    (
        compatibility,
        population,
        confirmation,
        provenance,
        development_subjects,
        confirmation_subjects,
    ) = _bound_inputs(tmp_path)

    receipt = build(
        compatibility_profile=compatibility,
        population_profile=population,
        confirmation_report=confirmation,
        development_subject_ids=development_subjects,
        confirmation_subject_ids=confirmation_subjects,
        output_path=tmp_path / "hybrid-candidate.json",
        provenance=provenance,
    )

    profile = json.loads(receipt.path.read_text(encoding="utf-8"))
    assert profile["status"] == "candidate"
    assert profile["legacy_dimensions"]["visible_pores"]["status"] == "candidate"
    assert profile["legacy_dimensions"]["visible_pores"]["route"] == "compatibility_v2"
    assert profile["legacy_dimensions"]["combined_pigmentation"]["status"] == "blocked"
    assert profile["legacy_dimensions"]["combined_pigmentation"]["route"] == "v011_fallback"
    assert profile["population_modules"]["acne_activity"]["status"] == "internal_candidate"
    assert profile["population_modules"]["oil_tendency"]["status"] == "blocked"
    assert profile["profile_sha256"] == receipt.sha256
    assert "/tmp/" not in json.dumps(profile)


def test_hybrid_candidate_rejects_tampered_confirmation_report(
    tmp_path: Path,
) -> None:
    build = getattr(scoring_bridge, "build_hybrid_candidate")
    (
        compatibility,
        population,
        confirmation,
        provenance,
        development_subjects,
        confirmation_subjects,
    ) = _bound_inputs(tmp_path)
    confirmation["attempted_count"] = 249

    try:
        build(
            compatibility_profile=compatibility,
            population_profile=population,
            confirmation_report=confirmation,
            development_subject_ids=development_subjects,
            confirmation_subject_ids=confirmation_subjects,
            output_path=tmp_path / "hybrid.json",
            provenance=provenance,
        )
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("tampered confirmation report was accepted")


def test_hybrid_candidate_rejects_cross_asset_sha_mismatch(tmp_path: Path) -> None:
    build = getattr(scoring_bridge, "build_hybrid_candidate")
    (
        compatibility,
        population,
        confirmation,
        provenance,
        development_subjects,
        confirmation_subjects,
    ) = _bound_inputs(tmp_path)
    mismatched = {**provenance, "confirmation_manifest_sha256": "9" * 64}

    try:
        build(
            compatibility_profile=compatibility,
            population_profile=population,
            confirmation_report=confirmation,
            development_subject_ids=development_subjects,
            confirmation_subject_ids=confirmation_subjects,
            output_path=tmp_path / "hybrid.json",
            provenance=mismatched,
        )
    except ValueError as error:
        assert "confirmation manifest" in str(error)
    else:
        raise AssertionError("cross-asset SHA mismatch was accepted")


@pytest.mark.parametrize(("key", "message"), (
    ("development_manifest_sha256", "development manifest"),
    ("fold_manifest_sha256", "fold manifest"),
    ("confirmation_pairs_sha256", "confirmation pairs"),
    ("official_v011_profile_sha256", "official V0.1.1 profile"),
    ("registry_sha256", "V0.1.1 registry"),
    ("formula_registry_sha256", "formula registry"),
))
def test_hybrid_candidate_rejects_each_frozen_asset_mismatch(
    tmp_path: Path,
    key: str,
    message: str,
) -> None:
    build = getattr(scoring_bridge, "build_hybrid_candidate")
    (
        compatibility,
        population,
        confirmation,
        provenance,
        development_subjects,
        confirmation_subjects,
    ) = _bound_inputs(tmp_path)
    mismatched = {**provenance, key: "9" * 64}

    with pytest.raises(ValueError, match=message):
        build(
            compatibility_profile=compatibility,
            population_profile=population,
            confirmation_report=confirmation,
            development_subject_ids=development_subjects,
            confirmation_subject_ids=confirmation_subjects,
            output_path=tmp_path / "hybrid.json",
            provenance=mismatched,
        )


def test_hybrid_candidate_rejects_development_confirmation_overlap(
    tmp_path: Path,
) -> None:
    build = getattr(scoring_bridge, "build_hybrid_candidate")
    (
        compatibility,
        population,
        confirmation,
        provenance,
        development_subjects,
        confirmation_subjects,
    ) = _bound_inputs(tmp_path)
    overlapping_rows = sorted(confirmation_subjects)
    overlapping_rows[0] = next(iter(development_subjects))
    overlapping = frozenset(overlapping_rows)

    with pytest.raises(ValueError, match="overlap"):
        build(
            compatibility_profile=compatibility,
            population_profile=population,
            confirmation_report=confirmation,
            development_subject_ids=development_subjects,
            confirmation_subject_ids=overlapping,
            output_path=tmp_path / "hybrid.json",
            provenance=provenance,
        )


@pytest.mark.parametrize(("development_count", "confirmation_count"), (
    (999, 250),
    (1000, 249),
))
def test_hybrid_candidate_rejects_wrong_subject_counts(
    tmp_path: Path,
    development_count: int,
    confirmation_count: int,
) -> None:
    build = getattr(scoring_bridge, "build_hybrid_candidate")
    compatibility, population, confirmation, provenance, _, _ = _bound_inputs(tmp_path)

    with pytest.raises(ValueError, match="subject count"):
        build(
            compatibility_profile=compatibility,
            population_profile=population,
            confirmation_report=confirmation,
            development_subject_ids=frozenset(
                f"development-{index}" for index in range(development_count)
            ),
            confirmation_subject_ids=frozenset(
                f"confirmation-{index}" for index in range(confirmation_count)
            ),
            output_path=tmp_path / "hybrid.json",
            provenance=provenance,
        )
