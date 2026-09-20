from __future__ import annotations

from src.scoring_bridge.population_profile import (
    PopulationMetricSpec,
    build_population_profile,
)
from src.scoring_bridge.population_runtime import score_population_modules
from src.scoring_bridge.word_population_profile import build_word_population_profile


def test_word_profile_freezes_remaining_group_weights_when_constant_group_removed() -> None:
    observations = tuple({
        "oil_tendency": {
            "usable": {"variable": index / 1000.0},
            "constant": {"fixed": 1.0},
        }
    } for index in range(1000))
    strict = build_population_profile(
        observations=observations,
        metric_specs=(
            PopulationMetricSpec(
                "oil_tendency", "usable", "variable", "higher_burden",
                "ratio", 0.7, 1.0,
            ),
            PopulationMetricSpec(
                "oil_tendency", "constant", "fixed", "higher_burden",
                "ratio", 0.3, 1.0,
            ),
        ),
        module_ids=("oil_tendency",),
    )
    assert strict.modules["oil_tendency"].status == "blocked"

    profile = build_word_population_profile(
        strict_profile=strict,
        observations=observations,
        module_ids=("oil_tendency",),
        minimum_retained_group_weight=0.6,
    )

    module = profile.modules["oil_tendency"]
    assert module.status == "candidate"
    assert tuple(module.groups) == ("usable",)
    assert module.groups["usable"].group_weight == 1.0
    result = score_population_modules(
        features={"oil_tendency": {"usable": {"variable": 0.8}}},
        profile=profile,
    )["oil_tendency"]
    assert result.score_valid is True
    assert result.score is not None


def test_word_profile_blocks_module_when_retained_group_weight_is_below_sixty_percent() -> None:
    observations = tuple({
        "stable_wrinkles": {
            "usable": {"variable": index / 1000.0},
            "constant": {"fixed": 1.0},
        }
    } for index in range(1000))
    strict = build_population_profile(
        observations=observations,
        metric_specs=(
            PopulationMetricSpec(
                "stable_wrinkles", "usable", "variable", "higher_burden",
                "ratio", 0.55, 1.0,
            ),
            PopulationMetricSpec(
                "stable_wrinkles", "constant", "fixed", "higher_burden",
                "ratio", 0.45, 1.0,
            ),
        ),
        module_ids=("stable_wrinkles",),
    )

    profile = build_word_population_profile(
        strict_profile=strict,
        observations=observations,
        module_ids=("stable_wrinkles",),
        minimum_retained_group_weight=0.6,
    )

    assert profile.modules["stable_wrinkles"].status == "blocked"


def test_word_profile_freezes_metric_allowlist_for_institution_alias() -> None:
    observations = tuple({
        "vascular": {
            "network": {
                "available": index / 1000.0,
                "missing_in_institution": (index % 97) / 97.0,
            },
            "excluded_group": {"other": (index % 83) / 83.0},
        }
    } for index in range(1000))
    strict = build_population_profile(
        observations=observations,
        metric_specs=(
            PopulationMetricSpec(
                "vascular", "network", "available", "higher_burden",
                "ratio", 1.0, 0.5,
            ),
            PopulationMetricSpec(
                "vascular", "network", "missing_in_institution", "higher_burden",
                "ratio", 1.0, 0.5,
            ),
            PopulationMetricSpec(
                "vascular", "excluded_group", "other", "higher_burden",
                "ratio", 0.5, 1.0,
            ),
        ),
        module_ids=("vascular",),
    )

    profile = build_word_population_profile(
        strict_profile=strict,
        observations=observations,
        module_ids=("vascular",),
        minimum_retained_group_weight=0.2,
        metric_allowlist={"vascular": {"network": frozenset({"available"})}},
        profile_id="word_population_reference_1000_institution_alias",
    )

    metrics = profile.modules["vascular"].groups["network"].metrics
    assert tuple(profile.modules["vascular"].groups) == ("network",)
    assert tuple(metrics) == ("available",)
    assert metrics["available"].effective_weight == 1.0
