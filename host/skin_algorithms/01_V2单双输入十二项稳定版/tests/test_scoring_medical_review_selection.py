from __future__ import annotations

import src.scoring_bridge as scoring_bridge


def test_medical_review_pack_has_six_subjects_per_quantile_anchor() -> None:
    row_type = getattr(scoring_bridge, "MedicalReviewCandidate", None)
    select = getattr(scoring_bridge, "select_medical_review_pack", None)
    assert callable(row_type)
    assert callable(select)
    candidates = tuple(
        row_type(
            subject_id=f"subject-{index}",
            score=float(index),
            qc_status="PASS" if index % 2 else "WARNING",
            pose_stratum=("front", "side", "unknown")[index % 3],
            relative_path=f"batch/sample-{index}.jpg",
        )
        for index in range(100)
    )

    selected = select(candidates, per_anchor=6)

    assert len(selected) == 30
    assert len({row.subject_id for row in selected}) == 30
    assert {
        anchor: sum(row.quantile_anchor == anchor for row in selected)
        for anchor in (5, 25, 50, 75, 95)
    } == {5: 6, 25: 6, 50: 6, 75: 6, 95: 6}
    assert all(not row.relative_path.startswith("/") for row in selected)


def test_medical_review_pack_rejects_insufficient_subjects() -> None:
    row_type = getattr(scoring_bridge, "MedicalReviewCandidate")
    select = getattr(scoring_bridge, "select_medical_review_pack")
    candidates = tuple(
        row_type(f"subject-{index}", float(index), "PASS", "front", f"{index}.jpg")
        for index in range(29)
    )

    try:
        select(candidates, per_anchor=6)
    except ValueError as error:
        assert "30" in str(error)
    else:
        raise AssertionError("insufficient medical review candidates were accepted")
