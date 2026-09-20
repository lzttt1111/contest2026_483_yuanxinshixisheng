from __future__ import annotations

import hashlib
import json
from pathlib import Path

import src.scoring_bridge as scoring_bridge


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "legacy"
    sample = root / "batch_000020/sample.jpg"
    sample.mkdir(parents=True)
    image = sample / "00_输入图片.jpg"
    image.write_bytes(b"fixture-image")
    metrics = sample / "九项核心量化指标.json"
    metrics.write_text('{"fixture": true}\n', encoding="utf-8")
    stat = image.stat()
    manifest = tmp_path / "confirmation.jsonl"
    manifest.write_text(json.dumps({
        "relative_path": image.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "subject_id": "subject-1",
        "legacy_record_sha256": _sha(metrics),
        "split": "reserve",
    }) + "\n", encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({
        "subject_id": "subject-1",
        "source_relative_path": image.relative_to(root).as_posix(),
        "legacy_metrics_relative_path": metrics.relative_to(root).as_posix(),
        "legacy_record_sha256": _sha(metrics),
        "split": "confirmation",
    }) + "\n", encoding="utf-8")
    return root, manifest, pairs


def test_confirmation_lineage_binds_manifest_pairs_input_and_legacy_record(
    tmp_path: Path,
) -> None:
    validate = getattr(scoring_bridge, "validate_confirmation_lineage", None)
    assert callable(validate)
    root, manifest, pairs = _fixture(tmp_path)

    receipt = validate(
        manifest_path=manifest,
        pairs_path=pairs,
        legacy_root=root,
        expected_count=1,
    )

    assert receipt.subject_count == 1
    assert receipt.manifest_sha256 == _sha(manifest)
    assert receipt.pairs_sha256 == _sha(pairs)


def test_confirmation_lineage_rejects_cross_asset_mismatch(tmp_path: Path) -> None:
    validate = getattr(scoring_bridge, "validate_confirmation_lineage")
    root, manifest, pairs = _fixture(tmp_path)
    rows = [json.loads(line) for line in pairs.read_text().splitlines()]
    rows[0]["legacy_record_sha256"] = "0" * 64
    pairs.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    try:
        validate(
            manifest_path=manifest,
            pairs_path=pairs,
            legacy_root=root,
            expected_count=1,
        )
    except ValueError as error:
        assert "legacy record SHA" in str(error)
    else:
        raise AssertionError("cross-asset lineage mismatch was accepted")


def test_development_fold_lineage_binds_subjects_paths_and_metadata(
    tmp_path: Path,
) -> None:
    validate = getattr(scoring_bridge, "validate_development_fold_lineage", None)
    assert callable(validate)
    development = tmp_path / "development.jsonl"
    development.write_text(json.dumps({
        "subject_id": "subject-1",
        "source_relative_path": "batch/sample.jpg/00_输入图片.jpg",
    }) + "\n", encoding="utf-8")
    folds = tmp_path / "folds.jsonl"
    folds.write_text(json.dumps({
        "subject_id": "subject-1",
        "relative_path": "batch/sample.jpg/00_输入图片.jpg",
        "stratum": "fixture",
        "outer_fold": 0,
        "inner_fold_by_outer": {"0": None, "1": 0, "2": 1, "3": 2, "4": 3},
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    metadata = tmp_path / "folds.metadata.json"
    metadata.write_text(json.dumps({
        "subject_count": 1,
        "active_pairs_sha256": _sha(development),
        "manifest_sha256": _sha(folds),
    }), encoding="utf-8")

    receipt = validate(
        development_manifest_path=development,
        fold_manifest_path=folds,
        fold_metadata_path=metadata,
        expected_count=1,
    )

    assert receipt.subject_count == 1
    assert receipt.development_manifest_sha256 == _sha(development)
    assert receipt.fold_manifest_sha256 == _sha(folds)


def test_confirmation_observation_lineage_rejects_stale_signature(
    tmp_path: Path,
) -> None:
    validate = getattr(
        scoring_bridge,
        "validate_confirmation_observation_lineage",
        None,
    )
    assert callable(validate)
    root, manifest, _ = _fixture(tmp_path)
    manifest_row = json.loads(manifest.read_text(encoding="utf-8"))
    observation = tmp_path / "observations.jsonl"
    observation.write_text(json.dumps({
        "signature": {
            "relative_path": manifest_row["relative_path"],
            "size": manifest_row["size"] + 1,
            "mtime_ns": manifest_row["mtime_ns"],
        },
        "status": "success",
        "success_items": 12,
    }) + "\n", encoding="utf-8")

    try:
        validate(
            manifest_path=manifest,
            observation_paths=(observation,),
            expected_count=1,
        )
    except ValueError as error:
        assert "observation signature" in str(error)
    else:
        raise AssertionError("stale confirmation observation was accepted")


def test_confirmation_freeze_metadata_rejects_insufficient_grade_coverage(
    tmp_path: Path,
) -> None:
    receipt_type = getattr(scoring_bridge, "ConfirmationLineageReceipt")
    validate = getattr(scoring_bridge, "validate_confirmation_freeze_metadata")
    lineage = receipt_type(
        subject_count=250,
        subject_ids=frozenset(f"subject-{index}" for index in range(250)),
        manifest_sha256="a" * 64,
        pairs_sha256="b" * 64,
    )
    grade_counts = {
        dimension: {"0": 50, "1": 50, "2": 50, "3": 50, "4": 50}
        for dimension in (
            "visible_pores",
            "combined_pigmentation",
            "diffuse_redness",
            "surface_smoothness_decline",
        )
    }
    grade_counts["visible_pores"] = {
        "0": 14,
        "1": 59,
        "2": 59,
        "3": 59,
        "4": 59,
    }
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "candidate_count": 250,
        "manifest_sha256": "a" * 64,
        "pairs_sha256": "b" * 64,
        "grade_counts": grade_counts,
    }), encoding="utf-8")

    try:
        validate(metadata_path=metadata, lineage=lineage)
    except ValueError as error:
        assert "grade coverage" in str(error)
    else:
        raise AssertionError("insufficient confirmation grade coverage was accepted")
