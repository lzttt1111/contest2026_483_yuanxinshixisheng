from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from src.scoring_bridge.legacy_scoring import (
    features_from_metrics,
    legacy_scores_from_features,
    prepare_references,
)


DIMENSION_IDS: Final = (
    "visible_pores",
    "combined_pigmentation",
    "diffuse_redness",
    "surface_smoothness_decline",
)


@dataclass(frozen=True, slots=True)
class LegacyCandidate:
    batch_name: str
    sample_name: str
    subject_id: str
    sample_dir: Path
    rank_key: str
    scores: tuple[float, float, float, float] | None = None
    score_decile: int | None = None


@dataclass(frozen=True, slots=True)
class BridgeSelection:
    train: tuple[LegacyCandidate, ...]
    validation: tuple[LegacyCandidate, ...]
    reserve: tuple[LegacyCandidate, ...]

    @property
    def primary(self) -> tuple[LegacyCandidate, ...]:
        return (*self.train, *self.validation)


def stable_key(seed: str, *parts: str) -> str:
    payload = "\0".join((seed, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def subject_id_from_sample_name(sample_name: str) -> str:
    stem = Path(sample_name).stem
    matched = re.fullmatch(r"(.+)_\d+", stem)
    return matched.group(1) if matched is not None else stem


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total < 0 or total > available or (total > 0 and available == 0):
        raise ValueError("配额总数超出可用样本")
    if total == 0:
        return {key: 0 for key in counts}
    exact = {key: total * value / available for key, value in counts.items()}
    quotas = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (exact[key] - quotas[key], key), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def discover_candidates(
    legacy_root: Path,
    *,
    batch_start: int,
    batch_end: int,
    seed: str,
) -> tuple[LegacyCandidate, ...]:
    """Enumerate only the first level of the approved historical batches."""

    candidates: list[LegacyCandidate] = []
    for number in range(batch_start, batch_end + 1):
        batch = legacy_root / f"batch_{number:06d}"
        with os.scandir(batch) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                subject_id = subject_id_from_sample_name(entry.name)
                candidates.append(
                    LegacyCandidate(
                        batch_name=batch.name,
                        sample_name=entry.name,
                        subject_id=subject_id,
                        sample_dir=Path(entry.path),
                        rank_key=stable_key(seed, subject_id, entry.name, batch.name),
                    )
                )
    return tuple(candidates)


def deduplicate_subjects(candidates: tuple[LegacyCandidate, ...]) -> tuple[LegacyCandidate, ...]:
    kept: list[LegacyCandidate] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.rank_key):
        if candidate.subject_id in seen:
            continue
        seen.add(candidate.subject_id)
        kept.append(candidate)
    return tuple(kept)


def candidate_pool(
    candidates: tuple[LegacyCandidate, ...],
    *,
    pool_size: int,
) -> tuple[LegacyCandidate, ...]:
    counts = Counter(candidate.batch_name for candidate in candidates)
    quotas = proportional_quotas(dict(counts), min(pool_size, len(candidates)))
    selected: list[LegacyCandidate] = []
    for batch_name, quota in quotas.items():
        batch_rows = sorted(
            (candidate for candidate in candidates if candidate.batch_name == batch_name),
            key=lambda item: item.rank_key,
        )
        selected.extend(batch_rows[:quota])
    return tuple(selected)


def attach_legacy_scores(
    candidates: tuple[LegacyCandidate, ...],
    *,
    references: dict[str, list[float]],
    profile_version: str,
) -> tuple[LegacyCandidate, ...]:
    del profile_version
    prepared = prepare_references(references)
    scored: list[LegacyCandidate] = []
    for candidate in candidates:
        metrics_path = candidate.sample_dir / "九项核心量化指标.json"
        if not metrics_path.is_file():
            continue
        scores = legacy_scores_from_features(
            features_from_metrics(metrics_path),
            prepared,
        )
        if scores is None:
            continue
        scored.append(replace(candidate, scores=scores))
    ordered = sorted(scored, key=lambda item: (float(np.mean(item.scores)), item.rank_key))
    count = len(ordered)
    return tuple(
        replace(candidate, score_decile=min(9, index * 10 // count))
        for index, candidate in enumerate(ordered)
    ) if count else ()


def _balanced_selection(
    candidates: tuple[LegacyCandidate, ...],
    *,
    total: int,
) -> tuple[LegacyCandidate, ...]:
    counts = Counter(candidate.batch_name for candidate in candidates)
    batch_quotas = proportional_quotas(dict(counts), total)
    decile_quotas = {
        index: total // 10 + int(index < total % 10)
        for index in range(10)
    }
    selected: list[LegacyCandidate] = []
    selected_ids: set[str] = set()
    batch_used: Counter[str] = Counter()
    decile_used: Counter[int] = Counter()
    ordered = sorted(candidates, key=lambda item: item.rank_key)
    for candidate in ordered:
        assert candidate.score_decile is not None
        if batch_used[candidate.batch_name] >= batch_quotas[candidate.batch_name]:
            continue
        if decile_used[candidate.score_decile] >= decile_quotas[candidate.score_decile]:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.rank_key)
        batch_used[candidate.batch_name] += 1
        decile_used[candidate.score_decile] += 1
    for candidate in ordered:
        if len(selected) >= total:
            break
        if candidate.rank_key in selected_ids:
            continue
        if batch_used[candidate.batch_name] >= batch_quotas[candidate.batch_name]:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.rank_key)
        batch_used[candidate.batch_name] += 1
    if len(selected) != total:
        raise ValueError(f"有效候选不足: expected={total}, actual={len(selected)}")
    return tuple(selected)


def _take_by_batch(
    candidates: tuple[LegacyCandidate, ...],
    *,
    count: int,
    seed: str,
    label: str,
) -> tuple[LegacyCandidate, ...]:
    batch_counts = Counter(candidate.batch_name for candidate in candidates)
    quotas = proportional_quotas(dict(batch_counts), count)
    selected: list[LegacyCandidate] = []
    for batch_name, quota in quotas.items():
        rows = sorted(
            (candidate for candidate in candidates if candidate.batch_name == batch_name),
            key=lambda item: stable_key(seed, label, item.subject_id, item.sample_name),
        )
        selected.extend(rows[:quota])
    return tuple(selected)


def split_selection(
    candidates: tuple[LegacyCandidate, ...],
    *,
    primary_count: int,
    validation_count: int,
    reserve_count: int,
    seed: str,
) -> BridgeSelection:
    chosen = _balanced_selection(candidates, total=primary_count + reserve_count)
    reserve = _take_by_batch(chosen, count=reserve_count, seed=seed, label="reserve")
    reserve_ids = {candidate.rank_key for candidate in reserve}
    primary = tuple(candidate for candidate in chosen if candidate.rank_key not in reserve_ids)
    validation = _take_by_batch(
        primary,
        count=validation_count,
        seed=seed,
        label="validation",
    )
    validation_ids = {candidate.rank_key for candidate in validation}
    train = tuple(candidate for candidate in primary if candidate.rank_key not in validation_ids)
    return BridgeSelection(train=train, validation=validation, reserve=reserve)
