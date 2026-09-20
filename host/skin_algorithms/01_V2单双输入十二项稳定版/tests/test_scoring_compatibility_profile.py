from __future__ import annotations

import json
from pathlib import Path

import src.scoring_bridge as scoring_bridge


def test_compatibility_profile_builder_writes_candidate_without_absolute_paths(
    tmp_path: Path,
) -> None:
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    fold_sample = getattr(scoring_bridge, "FoldSample")
    build_folds = getattr(scoring_bridge, "build_nested_fold_assignments")
    provenance_type = getattr(scoring_bridge, "CompatibilityProfileProvenance", None)
    fold_sha = getattr(scoring_bridge, "fold_assignments_sha256")
    build_profile = getattr(scoring_bridge, "build_compatibility_profile", None)
    assert callable(provenance_type)
    assert callable(build_profile)
    rows = tuple(
        row_type(
            subject_id=f"subject-{index}",
            current_score=float(index),
            legacy_score=float(index),
            current_groups=(group_type("density", float(index)),),
            legacy_groups=(group_type("density", float(index)),),
        )
        for index in range(50)
    )
    folds = build_folds(tuple(
        fold_sample(row.subject_id, f"{row.subject_id}.jpg", "fixture")
        for row in rows
    ), seed="fixture")
    provenance = provenance_type(
        development_manifest_sha256="a" * 64,
        fold_manifest_sha256=fold_sha(folds),
        official_v011_profile_sha256="c" * 64,
        registry_sha256="d" * 64,
        formula_registry_sha256="e" * 64,
        code_sha="f" * 40,
    )

    receipt = build_profile(
        rows_by_dimension={"visible_pores": rows},
        assignments={row.subject_id: row for row in folds},
        group_weights_by_dimension={"visible_pores": {"density": 1.0}},
        provenance=provenance,
        output_dir=tmp_path / "output",
        seed=20260827,
    )

    profile = json.loads(receipt.profile_path.read_text(encoding="utf-8"))
    assert profile["status"] == "candidate"
    assert profile["dimensions"]["visible_pores"]["method"] == "identity"
    assert profile["profile_sha256"] == receipt.profile_sha256
    assert "/tmp/" not in json.dumps(profile)


def test_compatibility_profile_rejects_wrong_fold_assignment_sha(
    tmp_path: Path,
) -> None:
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    fold_sample = getattr(scoring_bridge, "FoldSample")
    build_folds = getattr(scoring_bridge, "build_nested_fold_assignments")
    provenance_type = getattr(scoring_bridge, "CompatibilityProfileProvenance")
    build_profile = getattr(scoring_bridge, "build_compatibility_profile")
    rows = tuple(
        row_type(
            f"subject-{index}",
            float(index),
            float(index),
            (group_type("density", float(index)),),
            (group_type("density", float(index)),),
        )
        for index in range(20)
    )
    folds = build_folds(tuple(
        fold_sample(row.subject_id, f"{row.subject_id}.jpg", "fixture")
        for row in rows
    ), seed="fixture")
    provenance = provenance_type(
        development_manifest_sha256="a" * 64,
        fold_manifest_sha256="b" * 64,
        official_v011_profile_sha256="c" * 64,
        registry_sha256="d" * 64,
        formula_registry_sha256="e" * 64,
        code_sha="f" * 40,
    )

    try:
        build_profile(
            rows_by_dimension={"visible_pores": rows},
            assignments={row.subject_id: row for row in folds},
            group_weights_by_dimension={"visible_pores": {"density": 1.0}},
            provenance=provenance,
            output_dir=tmp_path,
            seed=1,
        )
    except ValueError as error:
        assert "fold manifest SHA" in str(error)
    else:
        raise AssertionError("wrong fold assignment SHA was accepted")
