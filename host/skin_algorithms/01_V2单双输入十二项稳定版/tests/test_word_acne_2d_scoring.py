from __future__ import annotations

from src.scoring_bridge.word_acne_2d import (
    Acne2DMetrics,
    Acne2DReferences,
    extract_acne_2d_metrics,
    score_acne_2d,
)


def _references() -> Acne2DReferences:
    return Acne2DReferences(
        tuple(float(value) for value in range(1, 11)),
        tuple(float(value) for value in range(1, 11)),
        tuple(value / 100.0 for value in range(1, 11)),
        tuple(value / 10.0 for value in range(1, 11)),
    )


def test_zero_formal_candidates_always_score_zero() -> None:
    assert score_acne_2d(Acne2DMetrics(0.0, 0.0, 0.0, 0.0), _references()) == 0.0


def test_one_candidate_cannot_be_pushed_to_extreme_by_size_alone() -> None:
    score = score_acne_2d(Acne2DMetrics(1.0, 1.0, 0.1, 1.0), _references())
    assert 17.0 < score < 23.0


def test_two_candidates_remain_in_a_narrow_intuitive_band() -> None:
    low = score_acne_2d(Acne2DMetrics(2.0, 1.0, 0.01, 0.1), _references())
    high = score_acne_2d(Acne2DMetrics(2.0, 10.0, 0.1, 1.0), _references())
    assert 31.0 < low < high < 37.0


def test_extract_acne_2d_uses_formal_detection_and_valid_area() -> None:
    result = {
        "metrics": {
            "detection": {
                "count": 2,
                "detections": [
                    {"box_area": 100, "confidence": 0.2},
                    {"box_area": 300, "confidence": 0.4},
                ],
            },
        },
        "public_metrics": {
            "overall_metrics": {
                "auxiliary_metrics": {
                    "analysis_scope": {"valid_skin_area_px": 100_000},
                },
            },
        },
    }
    metrics = extract_acne_2d_metrics(result)
    assert metrics.candidate_count == 2.0
    assert metrics.density_per_100k_px == 2.0
    assert metrics.candidate_area_ratio == 0.004
    assert 0.37 < metrics.p90_confidence < 0.39
