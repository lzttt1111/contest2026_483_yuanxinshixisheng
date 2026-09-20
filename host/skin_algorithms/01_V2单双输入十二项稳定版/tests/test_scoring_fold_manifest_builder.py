from __future__ import annotations

import json
from pathlib import Path

import src.scoring_bridge as scoring_bridge


def test_development_fold_builder_freezes_subjects_and_sanitized_provenance(
    tmp_path: Path,
) -> None:
    build = getattr(scoring_bridge, "build_development_fold_manifest", None)
    assert callable(build)
    active_pairs = tmp_path / "active.jsonl"
    observations = tmp_path / "scoring_features.worker.jsonl"
    active_pairs.write_text(
        "".join(
            json.dumps({
                "subject_id": f"subject-{index}",
                "source_relative_path": f"batch_000020/{index}.jpg/00_输入图片.jpg",
                "batch_name": "batch_000020",
                "legacy_scores": {
                    "visible_pores": float(index * 10),
                    "combined_pigmentation": float(index * 10),
                    "diffuse_redness": float(index * 10),
                    "surface_smoothness_decline": float(index * 10),
                },
            }) + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    observations.write_text(
        "".join(
            json.dumps({
                "signature": {
                    "relative_path": f"batch_000020/{index}.jpg/00_输入图片.jpg"
                },
                "status": "success",
                "success_items": 12,
                "quality": {"status": "PASS", "flags": []},
                "scoring_features": {"fixture": index},
            }) + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )

    receipt = build(
        active_pairs_path=active_pairs,
        observation_paths=(observations,),
        output_dir=tmp_path / "output",
        seed="20260827-scoring-v2",
    )

    rows = [
        json.loads(line)
        for line in receipt.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(receipt.metadata_path.read_text(encoding="utf-8"))
    assert len(rows) == 10
    assert len({row["subject_id"] for row in rows}) == 10
    assert metadata["subject_count"] == 10
    assert metadata["manifest_sha256"] == receipt.manifest_sha256
    assert "/tmp/" not in json.dumps(metadata)
