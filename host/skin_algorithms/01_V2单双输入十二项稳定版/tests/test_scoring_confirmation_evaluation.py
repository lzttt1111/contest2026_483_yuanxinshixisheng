from __future__ import annotations

import src.scoring_bridge as scoring_bridge


def _compatibility_rows(count: int, *, offset: float = 0.0):
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    return tuple(
        row_type(
            subject_id=f"subject-{index}",
            current_score=float(index % 101),
            legacy_score=float(index % 101) + offset,
            current_groups=(group_type("burden", float(index % 101)),),
            legacy_groups=(group_type("burden", float(index % 101) + offset),),
        )
        for index in range(count)
    )


def test_confirmation_promotes_dimensions_independently() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    evaluate = getattr(scoring_bridge, "evaluate_compatibility_confirmation", None)
    assert callable(evaluate)
    good_rows = _compatibility_rows(220)
    bad_rows = _compatibility_rows(220, offset=30.0)
    good_model = fit(method.IDENTITY, good_rows, group_weights={"burden": 1.0})
    bad_model = fit(method.IDENTITY, bad_rows, group_weights={"burden": 1.0})
    attempted = tuple(row.subject_id for row in good_rows)

    results = evaluate(
        rows_by_dimension={"good": good_rows, "bad": bad_rows},
        models_by_dimension={"good": good_model, "bad": bad_model},
        attempted_subject_ids=attempted,
        bootstrap_repetitions=20,
        seed=20260827,
    )

    assert results["good"].status == "promoted"
    assert results["good"].decision.passed is True
    assert results["bad"].status == "blocked"
    assert results["bad"].decision.passed is False
    assert "mean_absolute_error" in results["bad"].decision.failed_rules


def test_confirmation_counts_attrition_without_writing_zero() -> None:
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    evaluate = getattr(scoring_bridge, "evaluate_compatibility_confirmation")
    rows = _compatibility_rows(199)
    model = fit(method.IDENTITY, rows, group_weights={"burden": 1.0})
    attempted = tuple(f"subject-{index}" for index in range(250))

    result = evaluate(
        rows_by_dimension={"visible_pores": rows},
        models_by_dimension={"visible_pores": model},
        attempted_subject_ids=attempted,
        bootstrap_repetitions=10,
        seed=1,
    )["visible_pores"]

    assert result.evaluable_count == 199
    assert result.attrition_count == 51
    assert result.status == "blocked"
    assert "count" in result.decision.failed_rules


def test_population_confirmation_keeps_missing_evidence_out_of_scores() -> None:
    spec_type = getattr(scoring_bridge, "PopulationMetricSpec")
    build = getattr(scoring_bridge, "build_population_profile")
    evaluate = getattr(scoring_bridge, "evaluate_population_confirmation", None)
    assert callable(evaluate)
    development = tuple(
        {"oil_tendency": {"coverage": {"ratio": index / 1000.0}}}
        for index in range(1000)
    )
    profile = build(
        observations=development,
        metric_specs=(spec_type(
            module_id="oil_tendency",
            group_id="coverage",
            metric_id="ratio",
            direction="higher_burden",
            unit="ratio_0_1",
            group_weight=1.0,
            metric_weight=1.0,
        ),),
        module_ids=("oil_tendency",),
    )
    attempted = tuple(f"subject-{index}" for index in range(250))
    features = {
        subject_id: (
            {"oil_tendency": {"coverage": {}}}
            if index == 0
            else {"oil_tendency": {"coverage": {"ratio": index / 250.0}}}
        )
        for index, subject_id in enumerate(attempted)
    }

    result = evaluate(
        profile=profile,
        features_by_subject=features,
        attempted_subject_ids=attempted,
    )["oil_tendency"]

    assert result.evaluable_count == 249
    assert result.attrition_count == 1
    assert len(result.scores) == 249
    assert len(result.scored_subject_ids) == 249
    assert "subject-0" not in result.scored_subject_ids
    assert result.decision.availability_rate == 249 / 250
    assert result.direction_perturbation_passed is True
