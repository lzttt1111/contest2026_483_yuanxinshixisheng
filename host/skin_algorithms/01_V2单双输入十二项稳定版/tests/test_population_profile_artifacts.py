from __future__ import annotations

import json
from pathlib import Path

import src.scoring_bridge as scoring_bridge


def test_population_artifact_builder_uses_formula_registry_and_active_subjects(
    tmp_path: Path,
) -> None:
    build = getattr(scoring_bridge, "build_population_profile_from_artifacts", None)
    assert callable(build)
    active = tmp_path / "active.jsonl"
    observations = tmp_path / "scoring_features.worker.jsonl"
    active.write_text(
        "".join(
            json.dumps({
                "subject_id": f"subject-{index}",
                "source_relative_path": f"sample-{index}.jpg",
            }) + "\n"
            for index in range(1000)
        ),
        encoding="utf-8",
    )
    observations.write_text(
        "".join(
            json.dumps({
                "signature": {"relative_path": f"sample-{index}.jpg"},
                "status": "success",
                "success_items": 12,
                "quality": {"status": "PASS", "flags": []},
                "scoring_features": {"fixture": index},
                "v2_scoring_features": {
                    "oil_tendency": {"coverage": {"variable": index / 1000.0}}
                },
            }) + "\n"
            for index in range(1000)
        ),
        encoding="utf-8",
    )
    formulas = tmp_path / "formulas.json"
    formulas.write_text(json.dumps({
        "formula_version": "fixture",
        "metric_formulas": {
            "oil_tendency.coverage.variable": {
                "module_id": "oil_tendency",
                "group_id": "coverage",
                "metric_id": "variable",
                "direction": "higher_burden",
                "unit": "ratio_0_1",
                "group_weight": 1.0,
                "metric_weight": 1.0,
            }
        },
    }), encoding="utf-8")

    receipt = build(
        active_pairs_path=active,
        observation_paths=(observations,),
        formula_registry_path=formulas,
        output_dir=tmp_path / "output",
        module_ids=("oil_tendency",),
        code_sha="a" * 40,
    )

    document = json.loads(receipt.profile_path.read_text(encoding="utf-8"))
    assert document["status"] == "candidate"
    assert document["development_count"] == 1000
    assert document["modules"]["oil_tendency"]["status"] == "candidate"
    assert document["profile_sha256"] == receipt.profile_sha256
    assert "/tmp/" not in json.dumps(document)
