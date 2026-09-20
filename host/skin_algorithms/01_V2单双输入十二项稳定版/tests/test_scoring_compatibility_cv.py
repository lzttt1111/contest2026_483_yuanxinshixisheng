from __future__ import annotations

from dataclasses import replace

import src.scoring_bridge as scoring_bridge


def _score_document(total: float, group_a: float, group_b: float) -> dict:
    return {
        "formal_dimension_scores": [{
            "dimension_id": "visible_pores",
            "status": "formal",
            "score": total,
            "groups": [
                {"group_id": "a", "status": "scored", "score": group_a},
                {"group_id": "b", "status": "scored", "score": group_b},
            ],
        }]
    }


def test_score_documents_build_same_group_compatibility_rows() -> None:
    build_rows = getattr(scoring_bridge, "compatibility_rows_from_score_documents", None)
    assert callable(build_rows)

    rows = build_rows(
        subject_id="subject",
        current=_score_document(30.0, 20.0, 40.0),
        legacy=_score_document(35.0, 25.0, 45.0),
    )

    row = rows["visible_pores"]
    assert row.current_score == 30.0
    assert row.legacy_score == 35.0
    assert [group.group_id for group in row.current_groups] == ["a", "b"]
    assert [group.group_id for group in row.legacy_groups] == ["a", "b"]


def test_outer_scalar_predictions_do_not_use_outer_validation_targets() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    assignment_type = getattr(scoring_bridge, "FoldAssignment")
    predict_outer = getattr(scoring_bridge, "cross_validated_predictions", None)
    assert callable(predict_outer)
    rows = tuple(
        row_type(
            subject_id=f"subject-{index}",
            current_score=float(index * 5 + 5),
            legacy_score=float(index * 5 + 10),
            current_groups=(group_type("a", float(index * 5 + 5)),),
            legacy_groups=(group_type("a", float(index * 5 + 10)),),
        )
        for index in range(20)
    )
    assignments = {
        row.subject_id: assignment_type(
            subject_id=row.subject_id,
            relative_path=f"{row.subject_id}.jpg",
            stratum="fixture",
            outer_fold=index % 5,
            inner_fold_by_outer={
                outer: None if index % 5 == outer else index % 4
                for outer in range(5)
            },
        )
        for index, row in enumerate(rows)
    }
    changed = tuple(
        replace(row, legacy_score=100.0)
        if assignments[row.subject_id].outer_fold == 0
        else row
        for row in rows
    )

    original_predictions = predict_outer(
        method.SCALAR_PAVA,
        rows,
        assignments=assignments,
        group_weights={"a": 1.0},
    )
    changed_predictions = predict_outer(
        method.SCALAR_PAVA,
        changed,
        assignments=assignments,
        group_weights={"a": 1.0},
    )

    original_by_subject = {row.subject_id: row.predicted for row in original_predictions}
    changed_by_subject = {row.subject_id: row.predicted for row in changed_predictions}
    assert all(
        original_by_subject[subject_id] == changed_by_subject[subject_id]
        for subject_id, assignment in assignments.items()
        if assignment.outer_fold == 0
    )


def test_nested_selector_prefers_identity_when_no_mapping_improves_it() -> None:
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    fold_sample = getattr(scoring_bridge, "FoldSample")
    build_folds = getattr(scoring_bridge, "build_nested_fold_assignments")
    select = getattr(scoring_bridge, "select_nested_compatibility", None)
    method = getattr(scoring_bridge, "CompatibilityMethod")
    assert callable(select)
    rows = tuple(
        row_type(
            subject_id=f"subject-{index}",
            current_score=float(index),
            legacy_score=float(index),
            current_groups=(group_type("a", float(index)),),
            legacy_groups=(group_type("a", float(index)),),
        )
        for index in range(50)
    )
    folds = build_folds(tuple(
        fold_sample(row.subject_id, f"{row.subject_id}.jpg", "fixture")
        for row in rows
    ), seed="fixture")

    selection = select(
        rows,
        assignments={row.subject_id: row for row in folds},
        group_weights={"a": 1.0},
        seed=20260827,
    )

    assert selection.method is method.IDENTITY
