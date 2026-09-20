from __future__ import annotations

import src.scoring_bridge as scoring_bridge


def test_confirmation_selector_is_disjoint_exact_and_grade_covered() -> None:
    candidate_type = getattr(scoring_bridge, "ConfirmationCandidate", None)
    select = getattr(scoring_bridge, "select_confirmation_pool", None)
    assert callable(candidate_type)
    assert callable(select)
    candidates = tuple(
        candidate_type(
            subject_id=f"subject-{index}",
            relative_path=f"batch_{20 + index % 5:06d}/{index}.jpg/00_输入图片.jpg",
            batch_name=f"batch_{20 + index % 5:06d}",
            grades=(index % 5, (index // 5) % 5, (index // 25) % 5, (index // 125) % 5),
            quality_band=("high", "medium", "low")[index % 3],
            image_suffix=".jpg",
            rank_key=f"{index:064x}",
        )
        for index in range(1000)
    )

    selected = select(
        candidates,
        development_subjects={"subject-1", "subject-2"},
        count=250,
        minimum_per_grade=15,
        seed="20260827-confirmation-v1",
    )

    assert len(selected) == 250
    assert len({row.subject_id for row in selected}) == 250
    assert not {row.subject_id for row in selected}.intersection({"subject-1", "subject-2"})
    assert all(
        sum(row.grades[dimension] == grade for row in selected) >= 15
        for dimension in range(4)
        for grade in range(5)
    )
