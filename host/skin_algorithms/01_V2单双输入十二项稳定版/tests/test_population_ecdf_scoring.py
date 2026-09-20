from __future__ import annotations

import src.scoring_bridge as scoring_bridge


def _spec(metric_id: str, metric_weight: float = 1.0):
    spec_type = getattr(scoring_bridge, "PopulationMetricSpec")
    return spec_type(
        module_id="oil_tendency",
        group_id="coverage",
        metric_id=metric_id,
        direction="higher_burden",
        unit="ratio_0_1",
        group_weight=1.0,
        metric_weight=metric_weight,
    )


def test_population_profile_marks_constant_metric_unusable() -> None:
    build = getattr(scoring_bridge, "build_population_profile", None)
    assert callable(build)
    observations = tuple(
        {"oil_tendency": {"coverage": {"variable": index / 100.0, "constant": 0.0}}}
        for index in range(1000)
    )

    profile = build(
        observations=observations,
        metric_specs=(_spec("variable", 0.8), _spec("constant", 0.2)),
        module_ids=("oil_tendency",),
    )

    metrics = profile.modules["oil_tendency"].groups["coverage"].metrics
    assert metrics["variable"].usable is True
    assert metrics["constant"].usable is False
    assert metrics["constant"].unusable_reason == "insufficient_unique_values"
    assert profile.modules["oil_tendency"].status == "candidate"
    assert metrics["variable"].effective_weight == 1.0


def test_population_runtime_keeps_zero_and_missing_evidence_distinct() -> None:
    build = getattr(scoring_bridge, "build_population_profile")
    score = getattr(scoring_bridge, "score_population_modules", None)
    assert callable(score)
    observations = tuple(
        {"oil_tendency": {"coverage": {"variable": 0.0 if index < 100 else index / 1000.0}}}
        for index in range(1000)
    )
    profile = build(
        observations=observations,
        metric_specs=(_spec("variable"),),
        module_ids=("oil_tendency",),
    )

    zero = score(
        features={"oil_tendency": {"coverage": {"variable": 0.0}}},
        profile=profile,
    )["oil_tendency"]
    missing = score(
        features={"oil_tendency": {"coverage": {}}},
        profile=profile,
    )["oil_tendency"]

    assert zero.score == 0.0
    assert zero.score_valid is True
    assert missing.score is None
    assert missing.score_valid is False
    assert missing.grade == "不可评估"


def test_population_profile_json_roundtrip_preserves_scores() -> None:
    build = getattr(scoring_bridge, "build_population_profile")
    score = getattr(scoring_bridge, "score_population_modules")
    to_document = getattr(scoring_bridge, "population_profile_document", None)
    from_document = getattr(scoring_bridge, "population_profile_from_document", None)
    assert callable(to_document)
    assert callable(from_document)
    observations = tuple(
        {"oil_tendency": {"coverage": {"variable": index / 1000.0}}}
        for index in range(1000)
    )
    profile = build(
        observations=observations,
        metric_specs=(_spec("variable"),),
        module_ids=("oil_tendency",),
    )
    features = {"oil_tendency": {"coverage": {"variable": 0.5}}}

    restored = from_document(to_document(profile))

    assert score(features=features, profile=restored) == score(
        features=features,
        profile=profile,
    )


def test_population_score_is_monotonic_for_higher_burden_metric() -> None:
    build = getattr(scoring_bridge, "build_population_profile")
    score = getattr(scoring_bridge, "score_population_modules")
    observations = tuple(
        {"oil_tendency": {"coverage": {"variable": index / 1000.0}}}
        for index in range(1000)
    )
    profile = build(
        observations=observations,
        metric_specs=(_spec("variable"),),
        module_ids=("oil_tendency",),
    )

    lower = score(
        features={"oil_tendency": {"coverage": {"variable": 0.3}}},
        profile=profile,
    )["oil_tendency"]
    higher = score(
        features={"oil_tendency": {"coverage": {"variable": 0.8}}},
        profile=profile,
    )["oil_tendency"]

    assert lower.score is not None and higher.score is not None
    assert higher.score >= lower.score


def test_population_module_gate_blocks_endpoint_saturation() -> None:
    evaluate = getattr(scoring_bridge, "evaluate_population_module_gate", None)
    assert callable(evaluate)

    passed = evaluate(
        scores=tuple(float(index) for index in range(1, 100)),
        attempted_count=100,
        direction_perturbation_passed=True,
    )
    saturated = evaluate(
        scores=(0.0,) * 50 + (100.0,) * 50,
        attempted_count=100,
        direction_perturbation_passed=True,
    )

    assert passed.passed is True
    assert saturated.passed is False
    assert "endpoint_rate" in saturated.failed_rules
