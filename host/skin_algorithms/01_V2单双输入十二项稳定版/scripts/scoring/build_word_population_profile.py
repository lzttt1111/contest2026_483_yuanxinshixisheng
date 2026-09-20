from __future__ import annotations

# How to run:
# ../../.venv/bin/python scripts/scoring/build_word_population_profile.py --help

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, ConfigDict  # noqa: E402

from src.scoring_bridge.builder import load_latest_observations  # noqa: E402
from src.scoring_bridge.population_artifacts import NumericFeaturesDocument  # noqa: E402
from src.scoring_bridge.population_serialization import (  # noqa: E402
    population_profile_document,
    population_profile_from_document,
)
from src.scoring_bridge.word_population_profile import (  # noqa: E402
    build_word_population_profile,
)
from src.scoring_bridge.word_acne_2d import (  # noqa: E402
    acne_2d_reference_document,
    build_acne_2d_references,
    extract_acne_2d_metrics,
)
from src.scoring_bridge.word_wrinkle_2d import (  # noqa: E402
    build_wrinkle_2d_references,
    extract_wrinkle_2d_metrics,
    wrinkle_2d_reference_document,
)
from src.aisia_medical_report.institution_word_features import (  # noqa: E402
    INSTITUTION_METRIC_ALLOWLIST,
)


MODULE_IDS = (
    "oil_tendency",
    "vascular",
    "acne_activity",
    "dry_fine_lines",
    "stable_wrinkles",
    "structural_grooves",
    "contour_firmness",
)


class ActiveSubject(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subject_id: str
    source_relative_path: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_profile(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    return population_profile_from_document({
        key: document[key]
        for key in ("profile_id", "status", "development_count", "modules")
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从冻结1000张结果构建仅供Word展示的人群参考评分",
    )
    parser.add_argument("--active-pairs", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="保存1000套完整结果的batch_000001目录",
    )
    parser.add_argument("--strict-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    active_path = args.active_pairs.resolve(strict=True)
    job_dir = args.job_dir.resolve(strict=True)
    results_root = args.results_root.resolve(strict=True)
    strict_path = args.strict_profile.resolve(strict=True)
    subjects = tuple(
        ActiveSubject.model_validate_json(line)
        for line in active_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    observation_paths = tuple(sorted(job_dir.glob("scoring_features.*.jsonl")))
    observations, _ = load_latest_observations(observation_paths)
    features = []
    wrinkle_rows = {
        "stable_wrinkles": [],
        "structural_grooves": [],
    }
    acne_rows = []
    for subject in subjects:
        observation = observations.get(subject.source_relative_path)
        if observation is None:
            raise ValueError(f"missing development observation: {subject.subject_id}")
        features.append(NumericFeaturesDocument.model_validate(
            observation.get("v2_scoring_features") or {}
        ).root)
        sample_name = Path(subject.source_relative_path).parent.name
        wrinkle_path = (
            results_root
            / sample_name
            / "十二项检测/08_皱纹/皱纹量化指标.json"
        )
        if not wrinkle_path.is_file():
            wrinkle_metrics = None
        else:
            wrinkle_metrics = json.loads(wrinkle_path.read_text(encoding="utf-8"))
        if isinstance(wrinkle_metrics, dict):
            for module_id in wrinkle_rows:
                wrinkle_rows[module_id].append(
                    extract_wrinkle_2d_metrics(wrinkle_metrics, module_id)
                )
        complete_path = results_root / sample_name / "十二项完整量化指标.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            acne = (complete.get("detector_results") or {}).get("acne")
            if isinstance(acne, dict):
                acne_rows.append(extract_acne_2d_metrics(acne))
    consumer_profile = build_word_population_profile(
        strict_profile=_strict_profile(strict_path),
        observations=tuple(features),
        module_ids=MODULE_IDS,
        minimum_retained_group_weight=0.6,
    )
    institution_profile = build_word_population_profile(
        strict_profile=_strict_profile(strict_path),
        observations=tuple(features),
        module_ids=MODULE_IDS,
        minimum_retained_group_weight=0.2,
        metric_allowlist=INSTITUTION_METRIC_ALLOWLIST,
        profile_id="word_population_reference_1000_institution_alias",
    )
    blocked = [
        f"{profile_id}:{module_id}"
        for profile_id, profile in (
            ("consumer", consumer_profile),
            ("institution", institution_profile),
        )
        for module_id, module in profile.modules.items()
        if module.status != "candidate"
    ]
    if blocked:
        raise ValueError(f"Word population modules remain blocked: {blocked}")
    wrinkle_references = {
        module_id: build_wrinkle_2d_references(tuple(rows))
        for module_id, rows in wrinkle_rows.items()
    }
    acne_reference = build_acne_2d_references(tuple(acne_rows))
    body = {
        "schema_version": "aisia_word_population_profile_20260828",
        **population_profile_document(consumer_profile),
        "institution_profile": population_profile_document(institution_profile),
        "wrinkle_2d_profile": wrinkle_2d_reference_document(wrinkle_references),
        "acne_2d_profile": acne_2d_reference_document(acne_reference),
        "status": "provisional",
        "provenance": {
            "development_subject_count": len(subjects),
            "wrinkle_2d_subject_count": {
                module_id: len(rows)
                for module_id, rows in wrinkle_rows.items()
            },
            "acne_2d_subject_count": len(acne_rows),
            "active_pairs_sha256": _sha(active_path),
            "strict_profile_sha256": _sha(strict_path),
            "observation_sources": [
                {"name": path.name, "sha256": _sha(path)}
                for path in observation_paths
            ],
            "code_sha": args.code_sha,
        },
        "visible_report_boundary": (
            "reference-population relative scores; no overall score"
        ),
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    profile_sha = hashlib.sha256(canonical).hexdigest()
    document = {**body, "profile_sha256": profile_sha}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "provisional_profile_built",
        "profile_sha256": profile_sha,
        "modules": list(MODULE_IDS),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
