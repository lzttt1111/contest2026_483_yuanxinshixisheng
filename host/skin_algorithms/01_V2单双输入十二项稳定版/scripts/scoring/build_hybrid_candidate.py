from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_bridge.hybrid_candidate import build_hybrid_candidate  # noqa: E402
from src.scoring_bridge.confirmation_report import (  # noqa: E402
    verify_confirmation_report,
)
from src.scoring_bridge.lineage import (  # noqa: E402
    validate_confirmation_freeze_metadata,
    validate_confirmation_lineage,
    validate_development_fold_lineage,
)
from src.scoring_calibration.v011.registry import registry_document  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path.name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从冻结开发候选和一次性确认报告构建hybrid candidate",
    )
    parser.add_argument("--compatibility-profile", type=Path, required=True)
    parser.add_argument("--population-profile", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold-metadata", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-pairs", type=Path, required=True)
    parser.add_argument("--confirmation-metadata", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--formula-registry", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    compatibility = _json(args.compatibility_profile.resolve(strict=True))
    population = _json(args.population_profile.resolve(strict=True))
    confirmation = _json(args.confirmation_report.resolve(strict=True))
    verify_confirmation_report(confirmation)
    development_lineage = validate_development_fold_lineage(
        development_manifest_path=args.development_manifest.resolve(strict=True),
        fold_manifest_path=args.fold_manifest.resolve(strict=True),
        fold_metadata_path=args.fold_metadata.resolve(strict=True),
    )
    confirmation_lineage = validate_confirmation_lineage(
        manifest_path=args.confirmation_manifest.resolve(strict=True),
        pairs_path=args.confirmation_pairs.resolve(strict=True),
        legacy_root=args.legacy_root.resolve(strict=True),
    )
    validate_confirmation_freeze_metadata(
        metadata_path=args.confirmation_metadata.resolve(strict=True),
        lineage=confirmation_lineage,
    )
    receipt = build_hybrid_candidate(
        compatibility_profile=compatibility,
        population_profile=population,
        confirmation_report=confirmation,
        development_subject_ids=development_lineage.subject_ids,
        confirmation_subject_ids=confirmation_lineage.subject_ids,
        output_path=args.output,
        provenance={
            "development_manifest_sha256": (
                development_lineage.development_manifest_sha256
            ),
            "fold_manifest_sha256": development_lineage.fold_manifest_sha256,
            "confirmation_manifest_sha256": confirmation_lineage.manifest_sha256,
            "confirmation_pairs_sha256": confirmation_lineage.pairs_sha256,
            "official_v011_profile_sha256": _sha(
                args.official_profile.resolve(strict=True)
            ),
            "registry_sha256": registry_document()["registry_sha256"],
            "formula_registry_sha256": _sha(
                args.formula_registry.resolve(strict=True)
            ),
            "code_sha": args.code_sha,
        },
    )
    print(json.dumps({
        "status": "candidate_built_not_promoted",
        "profile": receipt.path.name,
        "profile_sha256": receipt.sha256,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
