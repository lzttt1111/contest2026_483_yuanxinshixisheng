from __future__ import annotations

from src.scoring_bridge.reserve_activation import build_activation_plan


def _pair(relative_path: str, split: str) -> dict:
    return {
        "source_relative_path": relative_path,
        "subject_id": relative_path,
        "split": split,
    }


def test_reserve_activation_replaces_missing_primary_split_only() -> None:
    pairs = [
        _pair("train-a.jpg", "train"),
        _pair("train-b.jpg", "train"),
        _pair("validation-a.jpg", "validation"),
        _pair("reserve-a.jpg", "reserve"),
        _pair("reserve-b.jpg", "reserve"),
    ]
    eligible = {
        "train-a.jpg",
        "validation-a.jpg",
        "reserve-a.jpg",
        "reserve-b.jpg",
    }

    plan = build_activation_plan(pairs, eligible_paths=eligible)

    assert [row["source_relative_path"] for row in plan.active_pairs] == [
        "train-a.jpg",
        "validation-a.jpg",
        "reserve-a.jpg",
    ]
    assert [row["split"] for row in plan.active_pairs] == [
        "train",
        "validation",
        "train",
    ]
    assert plan.target_split_counts == {"train": 2, "validation": 1}
    assert plan.activated_split_counts == {"train": 1, "validation": 0}


def test_reserve_activation_fails_when_eligible_replacements_are_insufficient() -> None:
    pairs = [
        _pair("train-a.jpg", "train"),
        _pair("validation-a.jpg", "validation"),
        _pair("reserve-a.jpg", "reserve"),
    ]

    try:
        build_activation_plan(pairs, eligible_paths={"reserve-a.jpg"})
    except ValueError as exc:
        assert "eligible reserve" in str(exc)
    else:
        raise AssertionError("insufficient reserve must fail closed")
