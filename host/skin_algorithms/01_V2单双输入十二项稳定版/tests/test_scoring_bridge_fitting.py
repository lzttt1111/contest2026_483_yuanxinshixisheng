from __future__ import annotations

import numpy as np
import json
import os
from pathlib import Path

import src.scoring_bridge as scoring_bridge

from src.scoring_bridge.fitting import (
    apply_isotonic,
    build_ecdf_profile,
    fit_isotonic,
    mapping_diagnostics,
)
from src.scoring_bridge.builder import (
    fit_dimension_bridges,
    load_latest_observations,
    v011_features_from_observation,
)


def test_isotonic_mapping_is_monotonic_and_reduces_training_error() -> None:
    source = [10.0, 20.0, 30.0, 40.0, 50.0]
    target = [18.0, 36.0, 32.0, 72.0, 85.0]

    mapping = fit_isotonic(source, target)
    predicted = [apply_isotonic(mapping, value) for value in source]

    assert np.all(np.diff(predicted) >= 0)
    assert np.mean(np.abs(np.asarray(predicted) - target)) < np.mean(
        np.abs(np.asarray(source) - target)
    )
    assert apply_isotonic(mapping, -100.0) == mapping["y"][0]
    assert apply_isotonic(mapping, 1000.0) == mapping["y"][-1]


def test_mapping_diagnostics_reports_no_severe_rank_reversal() -> None:
    target = [5.0, 25.0, 50.0, 75.0, 95.0]
    predicted = [10.0, 20.0, 55.0, 70.0, 90.0]

    diagnostics = mapping_diagnostics(target, predicted)

    assert diagnostics["spearman"] > 0.99
    assert diagnostics["pair_concordance"] == 1.0
    assert diagnostics["severe_reversal_count"] == 0
    assert diagnostics["absolute_error_p95"] >= diagnostics["absolute_error_p50"]


def test_diagnostics_v2_reports_pair_denominator_rate_and_cluster_bootstrap() -> None:
    diagnostics_v2 = getattr(scoring_bridge, "mapping_diagnostics_v2", None)
    assert callable(diagnostics_v2)

    result = diagnostics_v2(
        subject_ids=("a", "b", "c", "d", "e"),
        target=(5.0, 25.0, 65.0, 85.0, 95.0),
        predicted=(5.0, 70.0, 20.0, 85.0, 95.0),
        bootstrap_repetitions=100,
        seed=20260827,
    )

    assert result.severe_comparable_pair_count > 0
    assert result.severe_reversal_count > 0
    assert 0.0 < result.severe_reversal_rate <= 1.0
    assert result.severe_reversal_rate_ci95_upper >= result.severe_reversal_rate


def test_diagnostics_v2_marks_empty_severe_pair_denominator_unavailable() -> None:
    diagnostics_v2 = getattr(scoring_bridge, "mapping_diagnostics_v2", None)
    assert callable(diagnostics_v2)

    result = diagnostics_v2(
        subject_ids=("a", "b", "c"),
        target=(41.0, 42.0, 43.0),
        predicted=(40.0, 41.0, 42.0),
        bootstrap_repetitions=20,
        seed=20260827,
    )

    assert result.severe_comparable_pair_count == 0
    assert result.severe_reversal_rate is None
    assert result.severe_reversal_rate_ci95_upper is None


def test_promotion_gate_v2_passes_balanced_thresholds_and_blocks_empty_pairs() -> None:
    diagnostics_type = getattr(scoring_bridge, "MappingDiagnosticsV2", None)
    evaluate_gate = getattr(scoring_bridge, "evaluate_promotion_gate_v2", None)
    assert callable(diagnostics_type)
    assert callable(evaluate_gate)
    identity = diagnostics_type(
        count=220,
        spearman=0.92,
        pair_concordance=0.91,
        mean_absolute_error=5.5,
        absolute_error_p95=14.0,
        grade_crossing_count=30,
        multi_grade_crossing_count=0,
        severe_comparable_pair_count=1500,
        severe_reversal_count=8,
        severe_reversal_rate=8 / 1500,
        severe_reversal_rate_ci95_upper=0.008,
    )
    candidate = diagnostics_type(
        count=220,
        spearman=0.93,
        pair_concordance=0.92,
        mean_absolute_error=5.0,
        absolute_error_p95=13.0,
        grade_crossing_count=25,
        multi_grade_crossing_count=0,
        severe_comparable_pair_count=1500,
        severe_reversal_count=7,
        severe_reversal_rate=7 / 1500,
        severe_reversal_rate_ci95_upper=0.009,
    )

    assert evaluate_gate(candidate=candidate, identity=identity).passed is True

    blocked = diagnostics_type(
        count=220,
        spearman=0.99,
        pair_concordance=0.99,
        mean_absolute_error=1.0,
        absolute_error_p95=2.0,
        grade_crossing_count=0,
        multi_grade_crossing_count=0,
        severe_comparable_pair_count=0,
        severe_reversal_count=0,
        severe_reversal_rate=None,
        severe_reversal_rate_ci95_upper=None,
    )
    decision = evaluate_gate(candidate=blocked, identity=identity)
    assert decision.passed is False
    assert "severe_comparable_pair_count" in decision.failed_rules


def test_ecdf_profile_preserves_zero_and_unavailable_rates() -> None:
    observations = [
        {"pores": {"density": {"value": 0.0}}},
        {"pores": {"density": {"value": 2.0}}},
        {"pores": {"density": {"value": None}}},
    ]

    profile = build_ecdf_profile(observations)
    value = profile["pores"]["density.value"]

    assert value["count"] == 2
    assert value["zero_rate"] == 0.5
    assert value["unevaluable_rate"] == 1 / 3
    assert value["p50"] == 1.0


def test_latest_success_observation_prefers_newer_jsonl(tmp_path: Path) -> None:
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.write_text(json.dumps({
        "signature": {"relative_path": "a.jpg"},
        "status": "success",
        "success_items": 12,
        "scoring_features": {"version": 1},
    }) + "\n", encoding="utf-8")
    newer.write_text(json.dumps({
        "signature": {"relative_path": "a.jpg"},
        "status": "success",
        "success_items": 12,
        "scoring_features": {"version": 2},
    }) + "\n", encoding="utf-8")
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))

    observations, _ = load_latest_observations([older, newer])

    assert observations["a.jpg"]["scoring_features"]["version"] == 2


def test_latest_partial_supersedes_older_success_for_bridge(tmp_path: Path) -> None:
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.write_text(json.dumps({
        "signature": {"relative_path": "a.jpg"},
        "status": "success",
        "success_items": 12,
        "scoring_features": {"version": 1},
    }) + "\n", encoding="utf-8")
    newer.write_text(json.dumps({
        "signature": {"relative_path": "a.jpg"},
        "status": "partial_success",
        "success_items": 2,
        "failure_category": "input_quality_reject",
    }) + "\n", encoding="utf-8")
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))

    observations, rejected = load_latest_observations([older, newer])

    assert "a.jpg" not in observations
    assert [row["signature"]["relative_path"] for row in rejected] == ["a.jpg"]


def test_algorithm_partial_keeps_available_modules_for_bridge(tmp_path: Path) -> None:
    result = tmp_path / "algorithm-partial.jsonl"
    result.write_text(json.dumps({
        "signature": {"relative_path": "a.jpg"},
        "status": "partial_success",
        "success_items": 11,
        "failure_category": "algorithm_error",
        "failed_items": ["acne"],
        "quality": {"status": "WARNING"},
        "scoring_features": {"pores": {"version": 2}},
    }) + "\n", encoding="utf-8")

    observations, rejected = load_latest_observations([result])

    assert observations["a.jpg"]["failed_items"] == ["acne"]
    assert rejected == []


def test_dimension_bridge_fits_train_and_reports_validation() -> None:
    rows = []
    for dimension in ("visible_pores", "combined_pigmentation"):
        for index, (current, legacy) in enumerate(
            ((10, 15), (20, 28), (30, 40), (40, 52), (50, 65), (60, 75))
        ):
            rows.append({
                "relative_path": f"{dimension}-{index}.jpg",
                "dimension_id": dimension,
                "split": "train" if index < 4 else "validation",
                "current_score": float(current),
                "legacy_score": float(legacy),
            })

    mappings, evaluated, diagnostics = fit_dimension_bridges(
        rows,
        minimum_validation=2,
    )

    assert set(mappings) == {"visible_pores", "combined_pigmentation"}
    assert len(evaluated) == len(rows)
    assert diagnostics["status"] == "candidate"
    assert diagnostics["promotion_gate"] == "blocked"
    assert all(
        row["mapped_score"] >= 0.0 and row["mapped_score"] <= 100.0
        for row in evaluated
    )


def test_current_v2_uv_density_enriches_v011_feature_contract() -> None:
    observation = {
        "scoring_features": {"uv_spots": {"UV下更明显色素面积": {}}},
        "v2_scoring_features": {
            "pigmentation": {
                "uv_spots": {"uv_spot_density_per_100k_px": 12.5}
            }
        },
    }

    features = v011_features_from_observation(observation)

    assert features["uv_spots"]["UV样色素范围"]["density"] == 12.5
