from __future__ import annotations

import json
from pathlib import Path

import src.scoring_bridge as scoring_bridge
from src.scoring_calibration.v011.registry import REGISTRY


def test_artifact_loader_rebuilds_all_four_dimensions_from_explicit_paths(
    tmp_path: Path,
) -> None:
    load = getattr(scoring_bridge, "load_compatibility_development_data", None)
    assert callable(load)
    features: dict[str, dict] = {}
    references: dict[str, list[float]] = {}
    for dimension in REGISTRY:
        for group in dimension.groups:
            for metric in group.metrics:
                item, group_name, metric_name = metric.source.split(".")
                features.setdefault(item, {}).setdefault(group_name, {})[
                    metric_name
                ] = 1.0
                references[metric.id] = [0.0, 1.0, 2.0]
    legacy_metrics = tmp_path / "legacy/sample/九项核心量化指标.json"
    legacy_metrics.parent.mkdir(parents=True)
    legacy_metrics.write_text(
        json.dumps({
            "九项": {
                item: {"评分输入": payload}
                for item, payload in features.items()
            }
        }),
        encoding="utf-8",
    )
    active_pairs = tmp_path / "active.jsonl"
    active_pairs.write_text(json.dumps({
        "subject_id": "subject",
        "source_relative_path": "sample/input.jpg",
        "legacy_metrics_relative_path": "legacy/sample/九项核心量化指标.json",
    }) + "\n", encoding="utf-8")
    observations = tmp_path / "scoring_features.worker.jsonl"
    observations.write_text(json.dumps({
        "signature": {"relative_path": "sample/input.jpg"},
        "status": "success",
        "success_items": 12,
        "quality": {"status": "PASS", "flags": []},
        "scoring_features": features,
    }) + "\n", encoding="utf-8")
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"references": references}), encoding="utf-8")

    data = load(
        active_pairs_path=active_pairs,
        observation_paths=(observations,),
        legacy_root=tmp_path,
        official_profile_path=official,
    )

    assert set(data.rows_by_dimension) == {
        "visible_pores",
        "combined_pigmentation",
        "diffuse_redness",
        "surface_smoothness_decline",
    }
    assert all(len(rows) == 1 for rows in data.rows_by_dimension.values())
    assert all(
        abs(sum(weights.values()) - 1.0) < 1e-9
        for weights in data.group_weights_by_dimension.values()
    )


def test_confirmation_loader_records_missing_current_result_as_attrition(
    tmp_path: Path,
) -> None:
    load = getattr(scoring_bridge, "load_compatibility_confirmation_data", None)
    assert callable(load)
    features: dict[str, dict] = {}
    references: dict[str, list[float]] = {}
    for dimension in REGISTRY:
        for group in dimension.groups:
            for metric in group.metrics:
                item, group_name, metric_name = metric.source.split(".")
                features.setdefault(item, {}).setdefault(group_name, {})[
                    metric_name
                ] = 1.0
                references[metric.id] = [0.0, 1.0, 2.0]
    for subject in ("success", "rejected"):
        metrics = tmp_path / f"legacy/{subject}/九项核心量化指标.json"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(json.dumps({
            "九项": {
                item: {"评分输入": payload}
                for item, payload in features.items()
            }
        }), encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(
        json.dumps({
            "subject_id": subject,
            "source_relative_path": f"sample/{subject}.jpg",
            "legacy_metrics_relative_path": f"legacy/{subject}/九项核心量化指标.json",
            "legacy_record_sha256": __import__("hashlib").sha256(
                (tmp_path / f"legacy/{subject}/九项核心量化指标.json").read_bytes()
            ).hexdigest(),
        }) + "\n"
        for subject in ("success", "rejected")
    ), encoding="utf-8")
    observations = tmp_path / "observations.jsonl"
    observations.write_text(json.dumps({
        "signature": {"relative_path": "sample/success.jpg"},
        "status": "success",
        "success_items": 12,
        "quality": {"status": "PASS", "flags": []},
        "scoring_features": features,
    }) + "\n", encoding="utf-8")
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"references": references}), encoding="utf-8")

    data = load(
        confirmation_pairs_path=pairs,
        observation_paths=(observations,),
        legacy_root=tmp_path,
        official_profile_path=official,
    )

    assert data.unavailable_subjects == ("rejected",)
    assert all(len(rows) == 1 for rows in data.rows_by_dimension.values())
