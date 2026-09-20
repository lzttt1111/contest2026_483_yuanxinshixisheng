# /// script
# requires-python = "==3.10.*"
# dependencies = ["numpy==1.26.4", "pydantic==2.12.5"]
# ///
# ─── How to run ───
# PYTHONPATH=. python scripts/scoring/build_legacy_bridge_manifest.py --help

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.scoring_bridge.manifest import (
    DIMENSION_IDS,
    BridgeSelection,
    LegacyCandidate,
    attach_legacy_scores,
    candidate_pool,
    deduplicate_subjects,
    discover_candidates,
    split_selection,
)


SOURCE_PROFILE = "legacy_nine_full_face_v011"
TARGET_PROFILE = "consumer_rgb_twelve_v2_20260826"


def _input_image(sample_dir: Path) -> Path:
    with os.scandir(sample_dir) as entries:
        candidates = sorted(
            Path(entry.path)
            for entry in entries
            if entry.is_file(follow_symlinks=False)
            and entry.name.startswith("00_输入图片.")
        )
    if len(candidates) != 1:
        raise ValueError(f"输入图片数量异常: {sample_dir} -> {len(candidates)}")
    return candidates[0]


def _record(
    candidate: LegacyCandidate,
    *,
    legacy_root: Path,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _input_image(candidate.sample_dir)
    stat = image.stat()
    metrics = candidate.sample_dir / "九项核心量化指标.json"
    metrics_payload = metrics.read_bytes()
    metrics_sha = hashlib.sha256(metrics_payload).hexdigest()
    assert candidate.scores is not None
    manifest_row = {
        "relative_path": image.relative_to(legacy_root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "subject_id": candidate.subject_id,
        "legacy_record_sha256": metrics_sha,
        "split": split,
    }
    pair_row = {
        "sample_id": candidate.sample_name,
        "subject_id": candidate.subject_id,
        "batch_name": candidate.batch_name,
        "source_relative_path": manifest_row["relative_path"],
        "legacy_metrics_relative_path": metrics.relative_to(legacy_root).as_posix(),
        "legacy_record_sha256": metrics_sha,
        "legacy_scores": dict(zip(DIMENSION_IDS, candidate.scores)),
        "legacy_score_decile": candidate.score_decile,
        "split": split,
    }
    return manifest_row, pair_row


def _selection_rows(
    selection: BridgeSelection,
    *,
    legacy_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = (
        *((candidate, "train") for candidate in selection.train),
        *((candidate, "validation") for candidate in selection.validation),
        *((candidate, "reserve") for candidate in selection.reserve),
    )
    manifest_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for candidate, split in sorted(
        labels,
        key=lambda item: (item[0].batch_name, item[0].rank_key),
    ):
        manifest_row, pair_row = _record(
            candidate,
            legacy_root=legacy_root,
            split=split,
        )
        manifest_rows.append(manifest_row)
        pair_rows.append(pair_row)
    primary = [row for row in manifest_rows if row["split"] != "reserve"]
    reserve = [row for row in manifest_rows if row["split"] == "reserve"]
    return primary, reserve, pair_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从旧九项正式结果冻结评分Bridge同图样本")
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-start", type=int, default=20)
    parser.add_argument("--batch-end", type=int, default=30)
    parser.add_argument("--primary-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--reserve-count", type=int, default=200)
    parser.add_argument("--pool-size", type=int, default=4800)
    parser.add_argument("--seed", default="20260827")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    legacy_root = args.legacy_root.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    discovered = discover_candidates(
        legacy_root,
        batch_start=args.batch_start,
        batch_end=args.batch_end,
        seed=args.seed,
    )
    deduplicated = deduplicate_subjects(discovered)
    pool = candidate_pool(deduplicated, pool_size=args.pool_size)
    scored = attach_legacy_scores(
        pool,
        references=profile["references"],
        profile_version=profile["scoring_profile_version"],
    )
    selection = split_selection(
        scored,
        primary_count=args.primary_count,
        validation_count=args.validation_count,
        reserve_count=args.reserve_count,
        seed=args.seed,
    )
    primary, reserve, pairs = _selection_rows(selection, legacy_root=legacy_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_sha = _write_jsonl(output_dir / "bridge_manifest_primary_1000.jsonl", primary)
    reserve_sha = _write_jsonl(output_dir / "bridge_manifest_reserve_200.jsonl", reserve)
    pairs_sha = _write_jsonl(output_dir / "legacy_pairs_1200.jsonl", pairs)
    summary = {
        "schema_version": "legacy_bridge_manifest_v1",
        "source_profile": SOURCE_PROFILE,
        "target_profile": TARGET_PROFILE,
        "seed": args.seed,
        "batch_range": [args.batch_start, args.batch_end],
        "discovered_count": len(discovered),
        "deduplicated_subject_count": len(deduplicated),
        "candidate_pool_count": len(pool),
        "scored_candidate_count": len(scored),
        "train_count": len(selection.train),
        "validation_count": len(selection.validation),
        "reserve_count": len(selection.reserve),
        "batch_counts": dict(Counter(row["batch_name"] for row in pairs)),
        "decile_counts": dict(Counter(str(row["legacy_score_decile"]) for row in pairs)),
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "primary_manifest_sha256": primary_sha,
        "reserve_manifest_sha256": reserve_sha,
        "legacy_pairs_sha256": pairs_sha,
    }
    summary_path = output_dir / "bridge_manifest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
