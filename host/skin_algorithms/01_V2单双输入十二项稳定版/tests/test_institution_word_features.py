from __future__ import annotations

from src.aisia_medical_report.institution_word_features import (
    project_institution_word_features,
)


def test_institution_word_projection_uses_existing_four_light_metrics() -> None:
    complete = {
        "scoring_features": {},
        "detector_results": {
            "surface_gloss": {"metrics": {"overall_metrics": {
                "core_metrics": {
                    "scope_and_morphology": {
                        "gloss_area_ratio": 0.04,
                        "high_gloss_area_ratio": 0.01,
                    },
                    "signal_intensity": {
                        "p50_gloss_intensity": 0.4,
                        "p90_gloss_intensity": 0.7,
                    },
                }
            }}},
            "porphyrin": {"metrics": {"overall_metrics": {
                "core_metrics": {
                    "count_and_density": {"feature_count": 100},
                    "scope_and_morphology": {"feature_area_ratio": 0.02},
                    "signal_intensity": {"p50_intensity": 0.5, "p90_intensity": 0.8},
                },
                "auxiliary_metrics": {"analysis_scope": {"valid_skin_area_px": 200000}},
            }}},
            "vascular": {"metrics": {"overall_metrics": {
                "core_metrics": {
                    "count_and_density": {"vascular_count": 50, "branch_point_count": 10},
                    "scope_and_morphology": {"vascular_total_length_px": 1000},
                    "signal_intensity": {"p50_redness_response": 2.0, "p90_redness_response": 5.0},
                },
                "auxiliary_metrics": {"analysis_scope": {"valid_skin_area_px": 250000}},
            }}},
            "contour_firmness": {"metrics": {
                "中面部曲面连续性": 0.3,
                "下颌缘连续性": 0.8,
            }},
        },
    }

    features = project_institution_word_features(complete)

    assert features["oil_tendency"]["porphyrin_density_area"]["porphyrin_density_per_100k_px"] == 50.0
    assert features["vascular"]["count_length"]["vascular_density_per_100k_px"] == 20.0
    assert features["vascular"]["count_length"]["vascular_length_density_per_10k_px"] == 40.0
    assert features["contour_firmness"]["midface_support_decline"]["midface_surface_continuity"] == 0.3
    assert features["contour_firmness"]["lower_face_sagging"]["jawline_continuity"] == 0.8
