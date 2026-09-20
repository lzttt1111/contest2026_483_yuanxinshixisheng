from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


OUTER_FOLDS = 5
INNER_FOLDS = 4
GRADE_COUNT = 5


@dataclass(frozen=True, slots=True)
class FoldSample:
    subject_id: str
    relative_path: str
    stratum: str


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    subject_id: str
    relative_path: str
    stratum: str
    outer_fold: int
    inner_fold_by_outer: Mapping[int, int | None]


@dataclass(frozen=True, slots=True)
class ConfirmationMember:
    subject_id: str
    grades: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ConfirmationContractReport:
    subject_count: int
    per_dimension_grade_counts: tuple[Mapping[int, int], ...]


@dataclass(frozen=True, slots=True)
class ConfirmationContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _stable_key(seed: str, *parts: str) -> str:
    return hashlib.sha256("\0".join((seed, *parts)).encode()).hexdigest()


def _round_robin(
    samples: Sequence[FoldSample],
    *,
    folds: int,
    seed: str,
    label: str,
) -> dict[str, int]:
    by_stratum: dict[str, list[FoldSample]] = defaultdict(list)
    for sample in samples:
        by_stratum[sample.stratum].append(sample)
    assigned: dict[str, int] = {}
    for stratum, rows in sorted(by_stratum.items()):
        ordered = sorted(
            rows,
            key=lambda row: _stable_key(seed, label, stratum, row.subject_id),
        )
        offset = int(_stable_key(seed, label, stratum)[:16], 16) % folds
        for index, row in enumerate(ordered):
            assigned[row.subject_id] = (offset + index) % folds
    return assigned


def build_nested_fold_assignments(
    samples: Sequence[FoldSample],
    *,
    seed: str,
) -> tuple[FoldAssignment, ...]:
    if len({sample.subject_id for sample in samples}) != len(samples):
        raise ValueError("fold samples contain duplicate subjects")
    outer = _round_robin(samples, folds=OUTER_FOLDS, seed=seed, label="outer")
    inner_maps: dict[int, dict[str, int]] = {}
    for outer_fold in range(OUTER_FOLDS):
        training = tuple(
            sample for sample in samples
            if outer[sample.subject_id] != outer_fold
        )
        inner_maps[outer_fold] = _round_robin(
            training,
            folds=INNER_FOLDS,
            seed=seed,
            label=f"inner-{outer_fold}",
        )
    return tuple(
        FoldAssignment(
            subject_id=sample.subject_id,
            relative_path=sample.relative_path,
            stratum=sample.stratum,
            outer_fold=outer[sample.subject_id],
            inner_fold_by_outer={
                outer_fold: inner_maps[outer_fold].get(sample.subject_id)
                for outer_fold in range(OUTER_FOLDS)
            },
        )
        for sample in sorted(samples, key=lambda item: item.relative_path)
    )


def fold_assignments_sha256(assignments: Sequence[FoldAssignment]) -> str:
    payload = "".join(
        json.dumps(
            asdict(assignment),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        for assignment in sorted(assignments, key=lambda row: row.relative_path)
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_confirmation_contract(
    *,
    development_subjects: set[str],
    confirmation_members: Sequence[ConfirmationMember],
    minimum_subjects: int = 200,
    minimum_per_grade: int = 15,
) -> ConfirmationContractReport:
    subject_ids = [member.subject_id for member in confirmation_members]
    if len(set(subject_ids)) != len(subject_ids):
        raise ConfirmationContractError("confirmation contains duplicate subjects")
    if development_subjects.intersection(subject_ids):
        raise ConfirmationContractError("confirmation overlaps development subjects")
    if len(subject_ids) < minimum_subjects:
        raise ConfirmationContractError("confirmation subject count is insufficient")
    grade_counts: list[Mapping[int, int]] = []
    for dimension_index in range(4):
        counts = Counter(
            member.grades[dimension_index]
            for member in confirmation_members
        )
        if any(counts[grade] < minimum_per_grade for grade in range(GRADE_COUNT)):
            raise ConfirmationContractError(
                f"confirmation grade coverage is insufficient for dimension {dimension_index}"
            )
        grade_counts.append(dict(counts))
    return ConfirmationContractReport(
        subject_count=len(subject_ids),
        per_dimension_grade_counts=tuple(grade_counts),
    )


__all__ = [
    "ConfirmationContractError",
    "ConfirmationContractReport",
    "ConfirmationMember",
    "FoldAssignment",
    "FoldSample",
    "build_nested_fold_assignments",
    "fold_assignments_sha256",
    "validate_confirmation_contract",
]
