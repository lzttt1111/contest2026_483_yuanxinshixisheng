from __future__ import annotations

import json
from pathlib import Path

from src.nine_analysis.v2_proxy_projection import build_v2_proxy_modules


ROOT = Path(__file__).resolve().parents[1]
SCORING_FIXTURES = ROOT / "tests/fixtures/scoring"


def test_v2_projection_fills_all_registered_core_inputs_with_provenance() -> None:
    complete = json.loads(
        (SCORING_FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8")
    )
    detectors = complete["detector_results"]
    registry = json.loads(
        (ROOT / "calibration" / "metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )

    modules, scoring = build_v2_proxy_modules(detectors, registry)

    assert list(modules) == list(registry["modules"])
    assert list(scoring) == list(registry["modules"])
    assert sum(
        len(group["metrics"])
        for module in modules.values()
        for group in module["groups"].values()
    ) == 137
    for module in modules.values():
        assert module["completeness"]["missing_required"] == []
        assert module["completeness"]["complete"] is True
        for group in module["groups"].values():
            for metric in group["metrics"].values():
                assert isinstance(metric["value"], float)
                assert metric["availability"] in {"explicit_direct", "explicit_formula"}
                assert metric["source_metrics"]
                assert metric["formula_version"] == "doctor_v2_explicit_proxy_v1"
                assert metric["formula"]["operator"] != "mean_abs"
                assert metric["source_detector"]
                assert metric["source_metric_ids"]
                assert metric["medical_rationale"]
                assert not any(
                    token in source.lower()
                    for source in metric["source_metrics"]
                    for token in (
                        "timing", "elapsed", "image_size",
                        "device", "weights", "output_dir",
                    )
                )
        for region in module["regions"]:
            region_projection = module["region_groups"][region["id"]]
            assert region_projection["weight"] == region["weight"]
            assert set(region_projection["groups"]) == set(module["groups"])


def test_v2_projection_marks_required_inputs_incomplete_when_evidence_is_absent() -> None:
    registry = json.loads(
        (ROOT / "calibration" / "metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )

    modules, _ = build_v2_proxy_modules({}, registry)

    assert all(module["completeness"]["complete"] is False for module in modules.values())
    assert all(module["completeness"]["missing_required"] for module in modules.values())
    assert {
        metric["availability"]
        for module in modules.values()
        for group in module["groups"].values()
        for metric in group["metrics"].values()
    } == {"unavailable"}
    assert {
        metric["value"]
        for module in modules.values()
        for group in module["groups"].values()
        for metric in group["metrics"].values()
    } == {None}


def test_empty_wrinkle_report_group_only_invalidates_its_module() -> None:
    complete = json.loads(
        (SCORING_FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8")
    )
    detectors = complete["detector_results"]
    detectors["wrinkle"]["metrics"]["report_group_overlays"] = {
        "07": {"selected_region_keys": ["left_under_eye"]},
        "08": {"selected_region_keys": ["forehead"]},
        "09": {"selected_region_keys": []},
    }
    registry = json.loads(
        (ROOT / "calibration/metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )

    modules, scoring = build_v2_proxy_modules(detectors, registry)

    assert modules["dry_fine_lines"]["completeness"]["complete"] is True
    assert modules["stable_wrinkles"]["completeness"]["complete"] is True
    assert modules["structural_grooves"]["completeness"]["complete"] is False
    assert modules["structural_grooves"]["completeness"]["scope_unavailable_reason"] == (
        "wrinkle_report_group_09_unavailable"
    )
    assert {
        value
        for group in scoring["structural_grooves"].values()
        for value in group.values()
    } == {None}


def test_explicit_formula_registry_covers_every_v2_group() -> None:
    registry = json.loads(
        (ROOT / "calibration" / "metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )
    formulas = json.loads(
        (ROOT / "calibration" / "v2_explicit_formula_registry_v1.json").read_text(encoding="utf-8")
    )
    expected = {
        f"{module_id}.{group['id']}.{metric['id']}": {
            "module_id": module_id,
            "group_id": group["id"],
            "metric_id": metric["id"],
            "unit": metric["unit"],
            "direction": metric["direction"],
            "metric_role": "core",
            "group_weight": group["weight"],
            "metric_weight": 1.0 / len(group["core_metrics"]),
        }
        for module_id, module in registry["modules"].items()
        for group in module["groups"]
        for metric in group["core_metrics"]
    }
    assert "group_formulas" not in formulas
    assert set(formulas["metric_formulas"]) == set(expected)
    assert formulas["formula_version"] == "doctor_v2_explicit_proxy_v1"
    assert len(expected) == 137
    required_fields = {
        "module_id", "group_id", "metric_id", "source_detector",
        "source_metric_ids", "formula", "direction", "unit", "metric_role",
        "region_scope", "group_weight", "metric_weight", "proxy_boundary",
        "algorithm_version", "roi_version", "medical_rationale",
    }
    for key, definition in formulas["metric_formulas"].items():
        assert required_fields <= set(definition)
        for field, expected_value in expected[key].items():
            assert definition[field] == expected_value
        assert definition["source_metric_ids"]
        assert definition["formula"]["operator"] != "mean_abs"
    grouped_signatures: dict[tuple[str, str], set[tuple[tuple[str, ...], str]]] = {}
    for definition in formulas["metric_formulas"].values():
        group_key = (definition["module_id"], definition["group_id"])
        signature = (
            tuple(definition["source_metric_ids"]),
            json.dumps(definition["formula"], sort_keys=True),
        )
        grouped_signatures.setdefault(group_key, set()).add(signature)
    assert all(
        len(signatures) == sum(
            definition["module_id"] == module_id
            and definition["group_id"] == group_id
            for definition in formulas["metric_formulas"].values()
        )
        for (module_id, group_id), signatures in grouped_signatures.items()
    )
    all_signatures = [
        (
            tuple(definition["source_metric_ids"]),
            json.dumps(definition["formula"], sort_keys=True),
        )
        for definition in formulas["metric_formulas"].values()
    ]
    assert len(set(all_signatures)) == 137
    assert not any(
        token in source.lower()
        for definition in formulas["metric_formulas"].values()
        for source in definition["source_metric_ids"]
        for token in formulas["forbidden_source_tokens"]
    )
    assert not any(
        "max_recall" in source
        for definition in formulas["metric_formulas"].values()
        for source in definition["source_metric_ids"]
    )
    acne_definitions = [
        definition
        for definition in formulas["metric_formulas"].values()
        if definition["module_id"] == "acne_activity"
    ]
    assert not any(
        source in {"acne.detection.raw_count"}
        or source.startswith("porphyrin.")
        for definition in acne_definitions
        for source in definition["source_metric_ids"]
    )


def test_acne_papule_metrics_use_distinct_explicit_semantic_formulas() -> None:
    formulas = json.loads(
        (ROOT / "calibration" / "v2_explicit_formula_registry_v1.json").read_text(encoding="utf-8")
    )["metric_formulas"]
    definitions = [
        formulas[f"acne_activity.follicular_papule.{metric_id}"]
        for metric_id in (
            "papule_density", "papule_area_ratio", "p90_papule_redness",
            "p90_papule_relief",
        )
    ]

    assert len({tuple(item["source_metric_ids"]) for item in definitions}) == 4
    assert len({json.dumps(item["formula"], sort_keys=True) for item in definitions}) == 4
