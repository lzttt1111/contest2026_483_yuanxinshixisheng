from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_bridge.compatibility_artifacts import (  # noqa: E402
    load_compatibility_development_data,
)
from src.scoring_bridge.compatibility_profile import (  # noqa: E402
    CompatibilityProfileProvenance,
    build_compatibility_profile,
)
from src.scoring_bridge.split_contract import FoldAssignment  # noqa: E402
from src.scoring_calibration.v011.registry import registry_document  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignments(path: Path) -> dict[str, FoldAssignment]:
    assignments: dict[str, FoldAssignment] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assignment = FoldAssignment(
            subject_id=str(row["subject_id"]),
            relative_path=str(row["relative_path"]),
            stratum=str(row["stratum"]),
            outer_fold=int(row["outer_fold"]),
            inner_fold_by_outer={
                int(key): int(value) if value is not None else None
                for key, value in row["inner_fold_by_outer"].items()
            },
        )
        if assignment.subject_id in assignments:
            raise ValueError("fold manifest contains duplicate subjects")
        assignments[assignment.subject_id] = assignment
    return assignments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从冻结开发集与fold重建旧四项compatibility candidate",
    )
    parser.add_argument("--active-pairs", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--observation-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--formula-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    active = args.active_pairs.resolve(strict=True)
    official = args.official_profile.resolve(strict=True)
    folds = args.fold_manifest.resolve(strict=True)
    formula = args.formula_registry.resolve(strict=True)
    job_dir = args.job_dir.resolve(strict=True)
    observations = tuple(sorted(job_dir.glob("scoring_features.*.jsonl"))) + tuple(
        path.resolve(strict=True) for path in args.observation_jsonl
    )
    if not observations:
        raise ValueError("no development observation JSONL was found")
    data = load_compatibility_development_data(
        active_pairs_path=active,
        observation_paths=observations,
        legacy_root=args.legacy_root.resolve(strict=True),
        official_profile_path=official,
    )
    receipt = build_compatibility_profile(
        rows_by_dimension=data.rows_by_dimension,
        assignments=_assignments(folds),
        group_weights_by_dimension=data.group_weights_by_dimension,
        provenance=CompatibilityProfileProvenance(
            development_manifest_sha256=_sha(active),
            fold_manifest_sha256=_sha(folds),
            official_v011_profile_sha256=_sha(official),
            registry_sha256=str(registry_document()["registry_sha256"]),
            formula_registry_sha256=_sha(formula),
            code_sha=args.code_sha,
        ),
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps({
        "status": "candidate_built",
        "profile": receipt.profile_path.name,
        "profile_sha256": receipt.profile_sha256,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
