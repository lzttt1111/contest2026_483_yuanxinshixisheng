from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_bridge.builder import load_latest_observations  # noqa: E402
from src.scoring_bridge.compatibility_artifacts import (  # noqa: E402
    load_compatibility_confirmation_data,
)
from src.scoring_bridge.compatibility_serialization import (  # noqa: E402
    compatibility_model_from_document,
)
from src.scoring_bridge.confirmation_evaluation import (  # noqa: E402
    evaluate_compatibility_confirmation,
    evaluate_population_confirmation,
)
from src.scoring_bridge.confirmation_report import (  # noqa: E402
    write_confirmation_report,
)
from src.scoring_bridge.population_artifacts import (  # noqa: E402
    NumericFeaturesDocument,
)
from src.scoring_bridge.population_serialization import (  # noqa: E402
    population_profile_from_document,
)
from src.scoring_bridge.confirmation_outputs import (  # noqa: E402
    write_confirmation_summary_md,
    write_medical_review_packs,
)
from src.scoring_bridge.lineage import (  # noqa: E402
    validate_confirmation_freeze_metadata,
    validate_confirmation_lineage,
)
from src.scoring_bridge.observation_lineage import (  # noqa: E402
    validate_confirmation_observation_lineage,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_profile(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = document.get("profile_sha256")
    body = {key: value for key, value in document.items() if key != "profile_sha256"}
    actual = hashlib.sha256(json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    if expected != actual:
        raise ValueError(f"profile SHA mismatch: {path.name}")
    return document


def _pairs(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _population_features(
    pairs: tuple[dict[str, Any], ...],
    observations: Mapping[str, dict[str, Any]],
) -> dict[str, Mapping[str, Mapping[str, Mapping[str, int | float | None]]]]:
    features: dict[str, Mapping[str, Mapping[str, Mapping[str, int | float | None]]]] = {}
    for pair in pairs:
        observation = observations.get(str(pair["source_relative_path"]))
        if observation is None:
            continue
        parsed = NumericFeaturesDocument.model_validate(
            observation.get("v2_scoring_features") or {}
        )
        features[str(pair["subject_id"])] = parsed.root
    return features


def _compatibility_document(result: Any) -> dict[str, Any]:
    return asdict(result)


def _population_document(result: Any) -> dict[str, Any]:
    document = asdict(result)
    document.pop("scores", None)
    return document


def _population_profile(document: Mapping[str, Any]):
    return population_profile_from_document({
        key: document[key]
        for key in ("profile_id", "status", "development_count", "modules")
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一次性apply冻结评分候选并生成确认报告；本命令不拟合模型",
    )
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-pairs", type=Path, required=True)
    parser.add_argument("--confirmation-metadata", type=Path, required=True)
    parser.add_argument("--observation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--compatibility-profile", type=Path, required=True)
    parser.add_argument("--population-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--review-pack-output-dir", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.confirmation_manifest.resolve(strict=True)
    pairs_path = args.confirmation_pairs.resolve(strict=True)
    legacy_root = args.legacy_root.resolve(strict=True)
    lineage = validate_confirmation_lineage(
        manifest_path=manifest,
        pairs_path=pairs_path,
        legacy_root=legacy_root,
    )
    validate_confirmation_freeze_metadata(
        metadata_path=args.confirmation_metadata.resolve(strict=True),
        lineage=lineage,
    )
    official = args.official_profile.resolve(strict=True)
    compatibility_path = args.compatibility_profile.resolve(strict=True)
    population_path = args.population_profile.resolve(strict=True)
    observations_paths = tuple(
        path.resolve(strict=True) for path in args.observation_jsonl
    )
    observation_lineage = validate_confirmation_observation_lineage(
        manifest_path=manifest,
        observation_paths=observations_paths,
    )
    pairs = _pairs(pairs_path)
    attempted_subjects = tuple(str(pair["subject_id"]) for pair in pairs)
    compatibility_profile = _verified_profile(compatibility_path)
    population_profile_document = _verified_profile(population_path)
    data = load_compatibility_confirmation_data(
        confirmation_pairs_path=pairs_path,
        observation_paths=observations_paths,
        legacy_root=legacy_root,
        official_profile_path=official,
    )
    models = {
        dimension_id: compatibility_model_from_document(dimension["model"])
        for dimension_id, dimension in compatibility_profile["dimensions"].items()
    }
    compatibility = evaluate_compatibility_confirmation(
        rows_by_dimension=data.rows_by_dimension,
        models_by_dimension=models,
        attempted_subject_ids=attempted_subjects,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    observations, _ = load_latest_observations(observations_paths)
    population = evaluate_population_confirmation(
        profile=_population_profile(population_profile_document),
        features_by_subject=_population_features(pairs, observations),
        attempted_subject_ids=attempted_subjects,
    )
    review_packs = write_medical_review_packs(
        output_dir=args.review_pack_output_dir,
        population=population,
        pairs=pairs,
        observations=observations,
    )
    receipt = write_confirmation_report(
        output_path=args.output,
        attempted_count=len(attempted_subjects),
        compatibility_results={
            key: _compatibility_document(value) for key, value in compatibility.items()
        },
        population_results={
            key: _population_document(value) for key, value in population.items()
        },
        provenance={
            "confirmation_manifest_sha256": lineage.manifest_sha256,
            "confirmation_pairs_sha256": lineage.pairs_sha256,
            "official_v011_profile_sha256": _sha(official),
            "compatibility_profile_sha256": compatibility_profile["profile_sha256"],
            "population_profile_sha256": population_profile_document["profile_sha256"],
            "observation_sha256": [
                {"name": name, "sha256": sha}
                for name, sha in observation_lineage.source_sha256
            ],
            "code_sha": args.code_sha,
        },
    )
    write_confirmation_summary_md(
        path=args.summary_md,
        attempted_count=len(attempted_subjects),
        compatibility=compatibility,
        population=population,
        report_sha=receipt.sha256,
    )
    print(json.dumps({
        "status": "evaluated",
        "report": receipt.path.name,
        "report_sha256": receipt.sha256,
        "compatibility": {key: value.status for key, value in compatibility.items()},
        "population": {key: value.status for key, value in population.items()},
        "medical_review_packs": review_packs,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
