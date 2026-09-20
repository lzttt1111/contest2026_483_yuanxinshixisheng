from __future__ import annotations

from src.scoring_bridge.word_wrinkle_2d import (
    Wrinkle2DMetrics,
    Wrinkle2DReferences,
    extract_wrinkle_2d_metrics,
    score_wrinkle_2d,
)


def test_wrinkle_2d_score_cannot_be_driven_high_by_one_long_segment() -> None:
    references = Wrinkle2DReferences(
        segment_count=tuple(float(index) for index in range(1, 101)),
        total_length_px=tuple(float(index * 10) for index in range(1, 101)),
        max_segment_length_px=tuple(float(index * 2) for index in range(1, 101)),
    )

    score = score_wrinkle_2d(
        Wrinkle2DMetrics(1.0, 130.0, 130.0),
        references,
    )

    assert score < 30.0


def test_wrinkle_2d_score_is_zero_when_no_segments_are_detected() -> None:
    references = Wrinkle2DReferences((1.0, 2.0), (10.0, 20.0), (5.0, 10.0))

    assert score_wrinkle_2d(Wrinkle2DMetrics(0.0, 0.0, 0.0), references) == 0.0


def test_wrinkle_2d_extraction_uses_only_requested_report_regions() -> None:
    metrics = {
        "region_metrics": [
            {
                "analysis_region": "额头纹",
                "core_metrics": {"scope_and_morphology": {"total_wrinkle_length_px": 100}},
                "auxiliary_metrics": {
                    "count_and_density": {"wrinkle_segment_count": 2},
                    "scope_and_morphology": {"max_segment_length_px": 60},
                },
            },
            {
                "analysis_region": "左法令纹",
                "core_metrics": {"scope_and_morphology": {"total_wrinkle_length_px": 300}},
                "auxiliary_metrics": {
                    "count_and_density": {"wrinkle_segment_count": 1},
                    "scope_and_morphology": {"max_segment_length_px": 300},
                },
            },
        ]
    }

    stable = extract_wrinkle_2d_metrics(metrics, "stable_wrinkles")
    grooves = extract_wrinkle_2d_metrics(metrics, "structural_grooves")

    assert stable == Wrinkle2DMetrics(2.0, 100.0, 60.0)
    assert grooves == Wrinkle2DMetrics(1.0, 300.0, 300.0)


def test_wrinkle_2d_extraction_accepts_compact_complete_json_rows() -> None:
    metrics = {
        "region_metrics": [
            {
                "region_name": "额头纹",
                "segment_count": 3,
                "wrinkle_pixels": 339,
                "max_segment_length": 112,
            },
            {
                "region_name": "左法令纹",
                "segment_count": 1,
                "wrinkle_pixels": 130,
                "max_segment_length": 130,
            },
        ]
    }

    assert extract_wrinkle_2d_metrics(metrics, "stable_wrinkles") == Wrinkle2DMetrics(
        3.0, 339.0, 112.0
    )
    assert extract_wrinkle_2d_metrics(metrics, "structural_grooves") == Wrinkle2DMetrics(
        1.0, 130.0, 130.0
    )
