from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ActivationPlan:
    active_pairs: tuple[dict[str, Any], ...]
    activated_pairs: tuple[dict[str, Any], ...]
    target_split_counts: dict[str, int]
    activated_split_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReserveActivationError(ValueError):
    needed: int
    available: int

    def __str__(self) -> str:
        return (
            "eligible reserve samples are insufficient: "
            f"needed={self.needed}, available={self.available}"
        )


def build_activation_plan(
    pairs: Iterable[dict[str, Any]],
    *,
    eligible_paths: set[str],
) -> ActivationPlan:
    rows = tuple(dict(pair) for pair in pairs)
    target_splits = ("train", "validation")
    target_counts = Counter(
        str(row.get("split"))
        for row in rows
        if row.get("split") in target_splits
    )
    primary = tuple(
        row
        for row in rows
        if row.get("split") in target_splits
        and str(row.get("source_relative_path")) in eligible_paths
    )
    present_counts = Counter(str(row["split"]) for row in primary)
    deficits = {
        split: target_counts[split] - present_counts[split]
        for split in target_splits
    }
    needed = sum(deficits.values())
    eligible_reserve = tuple(
        row
        for row in rows
        if row.get("split") == "reserve"
        and str(row.get("source_relative_path")) in eligible_paths
    )
    if len(eligible_reserve) < needed:
        raise ReserveActivationError(needed, len(eligible_reserve))
    activated: list[dict[str, Any]] = []
    remaining = dict(deficits)
    for row in eligible_reserve[:needed]:
        split = "train" if remaining["train"] > 0 else "validation"
        replacement = dict(row)
        replacement["split"] = split
        activated.append(replacement)
        remaining[split] -= 1
    return ActivationPlan(
        active_pairs=(*primary, *activated),
        activated_pairs=tuple(activated),
        target_split_counts={
            split: target_counts[split]
            for split in target_splits
        },
        activated_split_counts={
            split: deficits[split]
            for split in target_splits
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    from src.scoring_bridge.builder import load_latest_observations

    parser = argparse.ArgumentParser(
        description="按主清单缺口激活已成功的冻结备用评分样本",
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--legacy-pairs", type=Path, required=True)
    parser.add_argument("--reserve-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    job_dir = args.job_dir.resolve(strict=True)
    pairs_path = args.legacy_pairs.resolve(strict=True)
    reserve_manifest_path = args.reserve_manifest.resolve(strict=True)
    observations, _ = load_latest_observations(
        sorted(job_dir.glob("scoring_features.*.jsonl")),
    )
    plan = build_activation_plan(
        _read_jsonl(pairs_path),
        eligible_paths=set(observations),
    )
    activated_splits = {
        str(row["source_relative_path"]): str(row["split"])
        for row in plan.activated_pairs
    }
    activated_manifest = []
    for row in _read_jsonl(reserve_manifest_path):
        relative_path = str(row["relative_path"])
        if relative_path not in activated_splits:
            continue
        activated = dict(row)
        activated["split"] = activated_splits[relative_path]
        activated_manifest.append(activated)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    active_pairs_path = output / "bridge_active_pairs_1000.jsonl"
    activated_manifest_path = output / "bridge_manifest_activated_reserve.jsonl"
    active_pairs_sha = _write_jsonl(active_pairs_path, plan.active_pairs)
    activated_manifest_sha = _write_jsonl(
        activated_manifest_path,
        activated_manifest,
    )
    audit = {
        "schema_version": "aisia_bridge_reserve_activation_v1",
        "active_pair_count": len(plan.active_pairs),
        "activated_reserve_count": len(plan.activated_pairs),
        "target_split_counts": plan.target_split_counts,
        "activated_split_counts": plan.activated_split_counts,
        "legacy_pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "reserve_manifest_sha256": hashlib.sha256(
            reserve_manifest_path.read_bytes()
        ).hexdigest(),
        "active_pairs_sha256": active_pairs_sha,
        "activated_manifest_sha256": activated_manifest_sha,
    }
    (output / "reserve_activation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


__all__ = [
    "ActivationPlan",
    "ReserveActivationError",
    "build_activation_plan",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
