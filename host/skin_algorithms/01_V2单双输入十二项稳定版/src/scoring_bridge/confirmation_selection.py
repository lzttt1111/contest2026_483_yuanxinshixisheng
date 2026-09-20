from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from src.scoring_bridge.manifest import proportional_quotas


@dataclass(frozen=True, slots=True)
class ConfirmationCandidate:
    subject_id: str
    relative_path: str
    batch_name: str
    grades: tuple[int, int, int, int]
    quality_band: str
    image_suffix: str
    rank_key: str


@dataclass(frozen=True, slots=True)
class ConfirmationSelectionError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _stable_key(seed: str, candidate: ConfirmationCandidate) -> str:
    value = "\0".join((
        seed,
        candidate.batch_name,
        candidate.quality_band,
        candidate.image_suffix,
        candidate.rank_key,
    ))
    return hashlib.sha256(value.encode()).hexdigest()


def _deduplicated(
    candidates: Sequence[ConfirmationCandidate],
    development_subjects: set[str],
) -> list[ConfirmationCandidate]:
    by_subject: dict[str, ConfirmationCandidate] = {}
    for candidate in sorted(candidates, key=lambda row: row.rank_key):
        if candidate.subject_id in development_subjects:
            continue
        by_subject.setdefault(candidate.subject_id, candidate)
    return list(by_subject.values())


def _mandatory_grade_coverage(
    candidates: list[ConfirmationCandidate],
    *,
    minimum_per_grade: int,
    seed: str,
) -> list[ConfirmationCandidate]:
    unmet = {
        (dimension, grade): minimum_per_grade
        for dimension in range(4)
        for grade in range(5)
    }
    remaining = list(candidates)
    selected: list[ConfirmationCandidate] = []
    while any(value > 0 for value in unmet.values()):
        scored = [
            (
                sum(
                    unmet[(dimension, candidate.grades[dimension])] > 0
                    for dimension in range(4)
                ),
                _stable_key(seed, candidate),
                candidate,
            )
            for candidate in remaining
        ]
        coverage, _, candidate = min(
            scored,
            key=lambda item: (-item[0], item[1]),
        )
        if coverage == 0:
            raise ConfirmationSelectionError(
                "candidate pool cannot satisfy per-grade coverage"
            )
        selected.append(candidate)
        remaining.remove(candidate)
        for dimension in range(4):
            key = (dimension, candidate.grades[dimension])
            unmet[key] = max(0, unmet[key] - 1)
    return selected


def select_confirmation_pool(
    candidates: Sequence[ConfirmationCandidate],
    *,
    development_subjects: set[str],
    count: int,
    minimum_per_grade: int,
    seed: str,
) -> tuple[ConfirmationCandidate, ...]:
    pool = _deduplicated(candidates, development_subjects)
    if len(pool) < count:
        raise ConfirmationSelectionError("candidate pool is smaller than requested count")
    selected = _mandatory_grade_coverage(
        pool,
        minimum_per_grade=minimum_per_grade,
        seed=seed,
    )
    if len(selected) > count:
        raise ConfirmationSelectionError("grade coverage exceeds requested count")
    selected_ids = {row.subject_id for row in selected}
    remaining = [row for row in pool if row.subject_id not in selected_ids]
    batch_counts = Counter(row.batch_name for row in pool)
    quotas = proportional_quotas(dict(batch_counts), count)
    used = Counter(row.batch_name for row in selected)
    ordered = sorted(remaining, key=lambda row: _stable_key(seed, row))
    for candidate in ordered:
        if len(selected) >= count:
            break
        if used[candidate.batch_name] >= quotas[candidate.batch_name]:
            continue
        selected.append(candidate)
        used[candidate.batch_name] += 1
    for candidate in ordered:
        if len(selected) >= count:
            break
        if candidate.subject_id in {row.subject_id for row in selected}:
            continue
        selected.append(candidate)
    if len(selected) != count:
        raise ConfirmationSelectionError("confirmation selection is incomplete")
    return tuple(sorted(selected, key=lambda row: _stable_key(seed, row)))


__all__ = [
    "ConfirmationCandidate",
    "ConfirmationSelectionError",
    "select_confirmation_pool",
]
