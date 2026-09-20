from __future__ import annotations

import src.scoring_bridge as scoring_bridge


def _runtime_profile() -> dict:
    finalize = getattr(scoring_bridge, "finalize_hybrid_profile_document")
    model_document = getattr(scoring_bridge, "compatibility_model_document")
    method = getattr(scoring_bridge, "CompatibilityMethod")
    fit = getattr(scoring_bridge, "fit_compatibility_model")
    row_type = getattr(scoring_bridge, "CompatibilityRow")
    group_type = getattr(scoring_bridge, "GroupScore")
    rows = tuple(
        row_type(
            f"subject-{index}",
            float(index),
            float(index + 10),
            (group_type("burden", float(index)),),
            (group_type("burden", float(index + 10)),),
        )
        for index in range(100)
    )
    model = fit(method.SCALAR_PAVA, rows, group_weights={"burden": 1.0})
    return finalize({
        "schema_version": "aisia_hybrid_scoring_profile_v1",
        "profile_id": "consumer_rgb_hybrid_v2_promoted",
        "status": "promoted",
        "capture_profile": "consumer",
        "institution_temporary_alias_id": "institution_hybrid_v2_temporary_alias",
        "score_direction": "higher_is_more_burden",
        "numerical_core_sha256": "1" * 64,
        "provenance": {"confirmation_manifest_sha256": "2" * 64},
        "legacy_dimensions": {
            "visible_pores": {
                "status": "promoted",
                "route": "compatibility_v2",
                "model": model_document(model),
            },
            "diffuse_redness": {
                "status": "blocked",
                "route": "v011_fallback",
                "model": {"method": "identity"},
            },
        },
        "population_profile": {"profile_id": "population"},
        "medical_boundary": "engineering only",
    })


def test_hybrid_runtime_applies_promoted_dimension_and_falls_back_per_dimension() -> None:
    input_type = getattr(scoring_bridge, "HybridLegacyDimensionInput", None)
    group_type = getattr(scoring_bridge, "GroupScore")
    score = getattr(scoring_bridge, "score_hybrid_legacy_dimensions", None)
    assert callable(input_type)
    assert callable(score)
    inputs = {
        "visible_pores": input_type(
            "visible_pores", 40.0, (group_type("burden", 40.0),)
        ),
        "diffuse_redness": input_type(
            "diffuse_redness", 55.0, (group_type("burden", 55.0),)
        ),
    }

    results = score(profile=_runtime_profile(), dimensions=inputs)

    assert results["visible_pores"].score > 40.0
    assert results["visible_pores"].score_source == "compatibility_v2"
    assert results["diffuse_redness"].score == 55.0
    assert results["diffuse_redness"].score_source == "v011_fallback"
