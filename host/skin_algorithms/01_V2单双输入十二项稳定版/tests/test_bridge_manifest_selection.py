from __future__ import annotations

from pathlib import Path

import pytest

import src.scoring_bridge as scoring_bridge
from src.scoring_bridge.manifest import (
    LegacyCandidate,
    deduplicate_subjects,
    proportional_quotas,
    split_selection,
    subject_id_from_sample_name,
)
from src.scoring_bridge.legacy_scoring import (
    legacy_dimension_scores_from_features,
    legacy_scores_from_features,
    prepare_references,
)
from src.scoring_calibration.v011.registry import REGISTRY
from src.scoring_calibration.v011.scoring import score_observation


def _candidate(index: int) -> LegacyCandidate:
    batch = f"batch_{20 + index % 3:06d}"
    return LegacyCandidate(
        batch_name=batch,
        sample_name=f"{index}_1.jpg",
        subject_id=str(index),
        sample_dir=Path(batch) / f"{index}_1.jpg",
        rank_key=f"{index:064x}",
        scores=(float(index % 100),) * 4,
        score_decile=index % 10,
    )


def test_subject_id_removes_view_suffix_and_extension() -> None:
    assert subject_id_from_sample_name("481468_1.jpg") == "481468"
    assert subject_id_from_sample_name("clinic28-25.png") == "clinic28-25"


def test_proportional_quotas_use_largest_remainder_and_exact_total() -> None:
    quotas = proportional_quotas({"a": 1950, "b": 1950, "c": 556}, 200)
    assert sum(quotas.values()) == 200
    assert quotas["c"] < quotas["a"]


def test_subject_deduplication_keeps_one_view_deterministically() -> None:
    rows = (
        LegacyCandidate("batch", "1_0.jpg", "1", Path("a"), "b"),
        LegacyCandidate("batch", "1_1.jpg", "1", Path("b"), "a"),
    )
    assert deduplicate_subjects(rows)[0].sample_name == "1_1.jpg"


def test_split_selection_is_disjoint_and_exact() -> None:
    selection = split_selection(
        tuple(_candidate(index) for index in range(300)),
        primary_count=100,
        validation_count=20,
        reserve_count=20,
        seed="20260827",
    )
    train = {candidate.subject_id for candidate in selection.train}
    validation = {candidate.subject_id for candidate in selection.validation}
    reserve = {candidate.subject_id for candidate in selection.reserve}
    assert len(train) == 80
    assert len(validation) == 20
    assert len(reserve) == 20
    assert train.isdisjoint(validation | reserve)
    assert validation.isdisjoint(reserve)


def test_nested_fold_manifest_is_deterministic_and_subject_disjoint() -> None:
    fold_sample = getattr(scoring_bridge, "FoldSample", None)
    build_folds = getattr(scoring_bridge, "build_nested_fold_assignments", None)
    assert callable(fold_sample)
    assert callable(build_folds)
    samples = tuple(
        fold_sample(
            subject_id=f"subject-{index}",
            relative_path=f"batch/sample-{index}.jpg",
            stratum=f"batch-{index % 5}|grade-{index % 5}",
        )
        for index in range(100)
    )

    first = build_folds(samples, seed="20260827-scoring-v2")
    second = build_folds(samples, seed="20260827-scoring-v2")

    assert first == second
    assert len({row.subject_id for row in first}) == 100
    assert {row.outer_fold for row in first} == {0, 1, 2, 3, 4}
    assert all(set(row.inner_fold_by_outer) == {0, 1, 2, 3, 4} for row in first)
    assert all(row.inner_fold_by_outer[row.outer_fold] is None for row in first)
    assert all(
        inner is None or 0 <= inner < 4
        for row in first
        for inner in row.inner_fold_by_outer.values()
    )


def test_nested_folds_distribute_singleton_strata_by_stable_offset() -> None:
    fold_sample = getattr(scoring_bridge, "FoldSample")
    build_folds = getattr(scoring_bridge, "build_nested_fold_assignments")
    samples = tuple(
        fold_sample(
            subject_id=f"subject-{index}",
            relative_path=f"batch/sample-{index}.jpg",
            stratum=f"unique-{index}",
        )
        for index in range(100)
    )

    assignments = build_folds(samples, seed="20260827-scoring-v2")

    counts = {
        fold: sum(row.outer_fold == fold for row in assignments)
        for fold in range(5)
    }
    assert max(counts.values()) - min(counts.values()) <= 10


def test_confirmation_contract_rejects_development_overlap_and_grade_gaps() -> None:
    confirmation_member = getattr(scoring_bridge, "ConfirmationMember", None)
    validate_contract = getattr(scoring_bridge, "validate_confirmation_contract", None)
    contract_error = getattr(scoring_bridge, "ConfirmationContractError", None)
    assert callable(confirmation_member)
    assert callable(validate_contract)
    assert isinstance(contract_error, type)
    members = tuple(
        confirmation_member(
            subject_id=f"subject-{index}",
            grades=(index % 5, index % 5, index % 5, index % 5),
        )
        for index in range(25)
    )

    with pytest.raises(contract_error):
        validate_contract(
            development_subjects={"subject-1"},
            confirmation_members=members,
            minimum_subjects=20,
            minimum_per_grade=5,
        )

    with pytest.raises(contract_error):
        validate_contract(
            development_subjects=set(),
            confirmation_members=members[:10],
            minimum_subjects=20,
            minimum_per_grade=5,
        )


def test_prepared_reference_fast_path_matches_official_v011_scorer() -> None:
    features: dict[str, dict] = {}
    references: dict[str, list[float]] = {}
    for dimension in REGISTRY:
        for group in dimension.groups:
            for metric in group.metrics:
                item, group_name, metric_name = metric.source.split(".")
                features.setdefault(item, {}).setdefault(group_name, {})[metric_name] = 1.0
                references[metric.id] = [0.0, 1.0, 2.0]
    official = score_observation(
        {
            "status": "success",
            "features": features,
            "input_quality_gate": {"status": "PASS", "reason_codes": []},
        },
        references,
    )
    expected = tuple(
        row["score"]
        for row in official["formal_dimension_scores"]
    )
    actual = legacy_scores_from_features(features, prepare_references(references))
    assert actual == expected


def test_missing_one_dimension_keeps_other_dimension_scores() -> None:
    features: dict[str, dict] = {}
    references: dict[str, list[float]] = {}
    for dimension in REGISTRY:
        for group in dimension.groups:
            for metric in group.metrics:
                item, group_name, metric_name = metric.source.split(".")
                features.setdefault(item, {}).setdefault(group_name, {})[
                    metric_name
                ] = 1.0
                references[metric.id] = [0.0, 1.0, 2.0]
    features.pop("pores")

    scores = legacy_dimension_scores_from_features(
        features,
        prepare_references(references),
    )

    assert scores["visible_pores"] is None
    assert scores["combined_pigmentation"] is not None
    assert scores["diffuse_redness"] is not None
    assert scores["surface_smoothness_decline"] is not None
