from __future__ import annotations

import src.scoring_bridge as scoring_bridge


def _candidate() -> dict:
    finalize = getattr(scoring_bridge, "finalize_hybrid_profile_document")
    return finalize({
        "schema_version": "aisia_hybrid_scoring_profile_v1",
        "profile_id": "consumer_rgb_hybrid_v2_candidate",
        "status": "candidate",
        "capture_profile": "consumer",
        "institution_temporary_alias_id": "institution_hybrid_v2_temporary_alias",
        "score_direction": "higher_is_more_burden",
        "numerical_core_sha256": "1" * 64,
        "provenance": {
            "confirmation_manifest_sha256": "2" * 64,
            "code_sha": "3" * 40,
        },
        "legacy_dimensions": {
            "visible_pores": {
                "status": "candidate",
                "technical_status": "promoted",
                "route": "compatibility_v2",
                "fallback_profile": "v011",
                "model": {"method": "identity"},
            }
        },
        "population_modules": {
            "acne_activity": {
                "status": "internal_candidate",
                "technical_status": "promoted",
                "doctor_approval": "pending",
                "user_report_exposure": False,
            }
        },
        "population_profile": {"profile_id": "population"},
        "medical_boundary": "engineering only",
    })


def _approval(candidate: dict, scope: str, item_id: str):
    approval_type = getattr(scoring_bridge, "ScoringApprovalRecord")
    return approval_type(
        approval_scope=scope,
        item_id=item_id,
        reviewer="reviewer-01",
        reviewed_at="2026-08-27",
        profile_sha256=candidate["profile_sha256"],
        dataset_sha256="2" * 64,
        conclusion="approved",
        textual_conclusion="确认通过；保持工程边界。",
    )


def test_hybrid_promotion_requires_user_approval_for_old_dimensions() -> None:
    promote = getattr(scoring_bridge, "promote_hybrid_profile", None)
    assert callable(promote)
    candidate = _candidate()

    try:
        promote(candidate, approvals=())
    except ValueError as error:
        assert "visible_pores" in str(error)
    else:
        raise AssertionError("old dimension was promoted without user approval")


def test_population_stays_internal_until_medical_approval() -> None:
    promote = getattr(scoring_bridge, "promote_hybrid_profile")
    candidate = _candidate()

    promoted = promote(
        candidate,
        approvals=(_approval(candidate, "user_release", "visible_pores"),),
    )

    assert promoted["status"] == "promoted"
    assert promoted["legacy_dimensions"]["visible_pores"]["status"] == "promoted"
    assert promoted["population_modules"]["acne_activity"]["status"] == "internal_candidate"
    assert promoted["population_modules"]["acne_activity"]["doctor_report_exposure"] is False


def test_medical_approval_is_sha_and_dataset_bound() -> None:
    promote = getattr(scoring_bridge, "promote_hybrid_profile")
    candidate = _candidate()
    approvals = (
        _approval(candidate, "user_release", "visible_pores"),
        _approval(candidate, "medical_report", "acne_activity"),
    )

    promoted = promote(candidate, approvals=approvals)

    module = promoted["population_modules"]["acne_activity"]
    assert module["status"] == "doctor_approved"
    assert module["doctor_report_exposure"] is True
    assert module["user_report_exposure"] is False
    assert promoted["profile_sha256"] != candidate["profile_sha256"]
