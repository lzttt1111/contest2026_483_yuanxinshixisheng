from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


QUANTILE_ANCHORS = (5, 25, 50, 75, 95)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True, slots=True)
class MedicalReviewCandidate:
    subject_id: str
    score: float
    qc_status: str
    pose_stratum: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class MedicalReviewSelection:
    subject_id: str
    score: float
    quantile_anchor: int
    qc_status: str
    pose_stratum: str
    relative_path: str


def _validate(candidates: Sequence[MedicalReviewCandidate]) -> None:
    if len({candidate.subject_id for candidate in candidates}) != len(candidates):
        raise ValueError("medical review candidates must have unique subjects")
    if any(
        candidate.relative_path.startswith("/")
        or _WINDOWS_ABSOLUTE.match(candidate.relative_path) is not None
        for candidate in candidates
    ):
        raise ValueError("medical review candidates require relative paths")


def select_medical_review_pack(
    candidates: Sequence[MedicalReviewCandidate],
    *,
    per_anchor: int = 6,
) -> tuple[MedicalReviewSelection, ...]:
    if per_anchor < 1:
        raise ValueError("medical review per-anchor count must be positive")
    required = len(QUANTILE_ANCHORS) * per_anchor
    if len(candidates) < required:
        raise ValueError(f"medical review pack requires at least {required} subjects")
    _validate(candidates)
    values = np.asarray([candidate.score for candidate in candidates], dtype=np.float64)
    selected: list[MedicalReviewSelection] = []
    used: set[str] = set()
    for anchor in QUANTILE_ANCHORS:
        target = float(np.percentile(values, anchor))
        available = sorted(
            (
                candidate for candidate in candidates
                if candidate.subject_id not in used
            ),
            key=lambda candidate: (
                abs(candidate.score - target),
                candidate.qc_status != "PASS",
                candidate.pose_stratum == "unknown",
                candidate.subject_id,
            ),
        )
        for candidate in available[:per_anchor]:
            used.add(candidate.subject_id)
            selected.append(MedicalReviewSelection(
                subject_id=candidate.subject_id,
                score=candidate.score,
                quantile_anchor=anchor,
                qc_status=candidate.qc_status,
                pose_stratum=candidate.pose_stratum,
                relative_path=candidate.relative_path,
            ))
    return tuple(selected)


__all__ = [
    "MedicalReviewCandidate",
    "MedicalReviewSelection",
    "QUANTILE_ANCHORS",
    "select_medical_review_pack",
]
