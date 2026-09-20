from __future__ import annotations

from src.aisia_medical_report.word_scoring import (
    apply_word_scores,
    conservative_word_score,
    has_population_evidence_consensus,
    load_word_acne_2d_reference,
    load_word_population_profile,
    load_word_wrinkle_2d_references,
)
from src.scoring_bridge.population_profile import (
    PopulationMetricSpec,
    build_population_profile,
)


def test_word_scoring_preserves_old_four_and_adds_population_score_without_internal_terms() -> None:
    observations = tuple({
        "oil_tendency": {"coverage": {"ratio": index / 1000.0}}
    } for index in range(1000))
    profile = build_population_profile(
        observations=observations,
        metric_specs=(PopulationMetricSpec(
            "oil_tendency", "coverage", "ratio", "higher_burden",
            "ratio", 1.0, 1.0,
        ),),
        module_ids=("oil_tendency",),
    )
    payload = {
        "报告信息": {},
        "检测模块": [
            {
                "模块编号": f"{index:02d}",
                "综合得分": 42.0 if index == 1 else None,
                "程度等级": "中度" if index == 1 else None,
                "正式评分说明": None,
            }
            for index in range(1, 12)
        ],
    }
    complete = {
        "scoring_features": {
            "oil_tendency": {"coverage": {"ratio": 0.8}}
        }
    }

    apply_word_scores(payload, complete, profile)

    pores = payload["检测模块"][0]
    oil = payload["检测模块"][1]
    assert pores["综合得分"] == 42.0
    assert oil["综合得分"] is not None
    assert oil["Word评分展示"] is True
    visible = str(payload)
    assert "production_proxy" not in visible
    assert "三锚点" not in visible
    assert "代理" not in visible
    assert "V1" not in visible


def test_default_word_population_asset_is_sha_pinned_and_scores_all_new_modules() -> None:
    profile = load_word_population_profile()
    institution = load_word_population_profile(capture_profile="institution")

    assert profile.development_count == 1000
    assert len(profile.modules) == 7
    assert all(module.status == "candidate" for module in profile.modules.values())
    assert institution.development_count == 1000
    assert len(institution.modules) == 7
    assert all(module.status == "candidate" for module in institution.modules.values())
    wrinkle = load_word_wrinkle_2d_references()
    assert set(wrinkle) == {"stable_wrinkles", "structural_grooves"}
    assert all(len(reference.segment_count) >= 950 for reference in wrinkle.values())
    acne = load_word_acne_2d_reference()
    assert len(acne.candidate_count) >= 100


def test_word_acne_score_is_zero_when_formal_detector_has_no_targets() -> None:
    observations = tuple({
        "acne_activity": {"count": {"density": index / 1000.0}}
    } for index in range(1000))
    profile = build_population_profile(
        observations=observations,
        metric_specs=(PopulationMetricSpec(
            "acne_activity", "count", "density", "higher_burden",
            "ratio", 1.0, 1.0,
        ),),
        module_ids=("acne_activity",),
    )
    payload = {
        "报告信息": {},
        "检测模块": [
            {"模块编号": f"{index:02d}", "综合得分": None, "程度等级": None}
            for index in range(1, 12)
        ],
    }
    complete = {
        "scoring_features": {
            "acne_activity": {"count": {"density": 0.8}}
        },
        "detector_results": {
            "acne": {"metrics": {"detection": {"count": 0}}}
        },
    }

    apply_word_scores(payload, complete, profile)

    acne = payload["检测模块"][5]
    assert acne["综合得分"] == 0.0
    assert acne["程度等级"] == "未见明显"


def test_direct_evidence_profiles_prevent_single_target_extreme_scores() -> None:
    payload = {
        "报告信息": {},
        "检测模块": [
            {"模块编号": f"{index:02d}", "综合得分": None, "程度等级": None}
            for index in range(1, 12)
        ],
    }
    complete = {
        "scoring_features": {},
        "detector_results": {
            "acne": {
                "metrics": {
                    "detection": {
                        "count": 1,
                        "detections": [{"box_area": 2_320.0, "confidence": 0.31}],
                    },
                },
                "public_metrics": {
                    "overall_metrics": {
                        "auxiliary_metrics": {
                            "analysis_scope": {"valid_skin_area_px": 285_000},
                        },
                    },
                },
            },
            "wrinkle": {
                "metrics": {
                    "region_metrics": [
                        {"region_name": "额头纹", "segment_count": 7, "wrinkle_pixels": 339, "max_segment_length": 112},
                        {"region_name": "左法令纹", "segment_count": 1, "wrinkle_pixels": 130, "max_segment_length": 130},
                    ],
                },
            },
        },
    }
    apply_word_scores(
        payload,
        complete,
        None,
        load_word_wrinkle_2d_references(),
        load_word_acne_2d_reference(),
    )

    modules = {row["模块编号"]: row for row in payload["检测模块"]}
    assert modules["06"]["综合得分"] < 60.0
    assert modules["09"]["综合得分"] < 30.0


def test_new_word_score_compression_never_emits_significant_band() -> None:
    assert conservative_word_score(0.0) == 0.0
    assert conservative_word_score(50.0) == 41.0
    assert conservative_word_score(100.0) == 77.0
    assert conservative_word_score(100.0, allow_high=False) == 64.9


def test_population_high_band_requires_multiple_high_groups() -> None:
    observations = tuple({
        "oil_tendency": {
            "coverage": {"ratio": index / 1000.0},
            "intensity": {"value": index / 1000.0},
            "continuity": {"ratio": index / 1000.0},
        }
    } for index in range(1000))
    specs = tuple(
        PopulationMetricSpec(
            "oil_tendency", group, metric, "higher_burden", "ratio", 1 / 3, 1.0,
        )
        for group, metric in (("coverage", "ratio"), ("intensity", "value"), ("continuity", "ratio"))
    )
    profile = build_population_profile(
        observations=observations,
        metric_specs=specs,
        module_ids=("oil_tendency",),
    )
    one_high = {
        "oil_tendency": {
            "coverage": {"ratio": 0.99},
            "intensity": {"value": 0.10},
            "continuity": {"ratio": 0.10},
        }
    }
    all_high = {
        "oil_tendency": {
            "coverage": {"ratio": 0.99},
            "intensity": {"value": 0.99},
            "continuity": {"ratio": 0.99},
        }
    }
    assert not has_population_evidence_consensus(one_high, profile, "oil_tendency")
    assert has_population_evidence_consensus(all_high, profile, "oil_tendency")


def test_visually_limited_modules_stay_mid_band_until_evidence_improves() -> None:
    for module_id, report_index in (("vascular", 4), ("contour_firmness", 10)):
        observations = tuple({
            module_id: {
                "first": {"burden": index / 1000.0},
                "second": {"burden": index / 1000.0},
                "third": {"burden": index / 1000.0},
            }
        } for index in range(1000))
        specs = tuple(
            PopulationMetricSpec(
                module_id, group, "burden", "higher_burden", "ratio", 1 / 3, 1.0,
            )
            for group in ("first", "second", "third")
        )
        profile = build_population_profile(
            observations=observations,
            metric_specs=specs,
            module_ids=(module_id,),
        )
        payload = {
            "报告信息": {},
            "检测模块": [
                {"模块编号": f"{index:02d}", "综合得分": None, "程度等级": None}
                for index in range(1, 12)
            ],
        }
        complete = {
            "scoring_features": {
                module_id: {
                    "first": {"burden": 0.99},
                    "second": {"burden": 0.99},
                    "third": {"burden": 0.99},
                },
            },
        }

        apply_word_scores(payload, complete, profile)

        module = payload["检测模块"][report_index]
        assert module["综合得分"] <= 64.9
        assert module["程度等级"] == "中度"
