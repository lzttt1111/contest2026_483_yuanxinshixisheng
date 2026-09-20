from __future__ import annotations

import numpy as np

import src.scoring_bridge as scoring_bridge


def _rows() -> tuple:
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    values = (
        (10.0, 20.0, 18.0, 28.0),
        (20.0, 30.0, 25.0, 35.0),
        (30.0, 40.0, 38.0, 48.0),
        (40.0, 50.0, 50.0, 60.0),
        (50.0, 60.0, 62.0, 72.0),
    )
    return tuple(
        row_type(
            subject_id=f"subject-{index}",
            current_score=0.4 * current_a + 0.6 * current_b,
            legacy_score=0.4 * legacy_a + 0.6 * legacy_b,
            current_groups=(
                group_type(group_id="a", score=current_a),
                group_type(group_id="b", score=current_b),
            ),
            legacy_groups=(
                group_type(group_id="a", score=legacy_a),
                group_type(group_id="b", score=legacy_b),
            ),
        )
        for index, (current_a, current_b, legacy_a, legacy_b) in enumerate(values)
    )


def test_compatibility_identity_and_scalar_pava_are_bounded() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    predict = getattr(scoring_bridge, "predict_compatibility")
    rows = _rows()

    identity = fit(method.IDENTITY, rows, group_weights={"a": 0.4, "b": 0.6})
    scalar = fit(method.SCALAR_PAVA, rows, group_weights={"a": 0.4, "b": 0.6})

    assert [predict(identity, row) for row in rows] == [
        row.current_score for row in rows
    ]
    scalar_scores = [predict(scalar, row) for row in rows]
    assert all(0.0 <= score <= 100.0 for score in scalar_scores)
    assert np.all(np.diff(scalar_scores) >= 0)


def test_fixed_group_pava_maps_same_named_groups_with_registry_weights() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    predict = getattr(scoring_bridge, "predict_compatibility")
    rows = _rows()

    model = fit(
        method.FIXED_GROUP_PAVA,
        rows,
        group_weights={"a": 0.4, "b": 0.6},
    )
    scores = [predict(model, row) for row in rows]

    assert np.mean(np.abs(np.asarray(scores) - [row.legacy_score for row in rows])) < 1.0
    assert model.group_ids == ("a", "b")
    assert model.group_weights == (0.4, 0.6)


def test_additive_compatibility_uses_nonnegative_weights_and_clips_output() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    predict = getattr(scoring_bridge, "predict_compatibility")
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    rows = _rows()
    model = fit(
        method.ADDITIVE,
        rows,
        group_weights={"a": 0.4, "b": 0.6},
        regularization=1.0,
    )
    extreme = row_type(
        subject_id="extreme",
        current_score=1000.0,
        legacy_score=100.0,
        current_groups=(
            group_type(group_id="a", score=1000.0),
            group_type(group_id="b", score=1000.0),
        ),
        legacy_groups=(
            group_type(group_id="a", score=100.0),
            group_type(group_id="b", score=100.0),
        ),
    )

    assert all(weight >= 0.0 for weight in model.group_weights)
    assert abs(sum(model.group_weights) - 1.0) < 1e-8
    assert predict(model, extreme) == 100.0


def test_compatibility_model_json_roundtrip_preserves_predictions() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    predict = getattr(scoring_bridge, "predict_compatibility")
    to_document = getattr(scoring_bridge, "compatibility_model_document", None)
    from_document = getattr(scoring_bridge, "compatibility_model_from_document", None)
    assert callable(to_document)
    assert callable(from_document)
    rows = _rows()
    model = fit(
        method.FIXED_GROUP_PAVA,
        rows,
        group_weights={"a": 0.4, "b": 0.6},
    )

    restored = from_document(to_document(model))

    assert [predict(restored, row) for row in rows] == [
        predict(model, row) for row in rows
    ]
