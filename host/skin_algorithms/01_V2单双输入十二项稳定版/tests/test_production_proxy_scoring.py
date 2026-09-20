from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.nine_analysis import production_proxy_scoring
from src.nine_analysis.production_proxy_scoring import score_production_proxy
from src.nine_analysis.v2_proxy_projection import build_v2_proxy_modules


ROOT = Path(__file__).resolve().parents[1]
SCORING_FIXTURES = ROOT / "tests/fixtures/scoring"
ANCHOR_FIXTURES = {
    "clinic28-09_RGB_M": "clinic28-09_r1",
    "clinic28-23_RGB_M": "clinic28-23_r1",
    "clinic28-25_RGB_M": "clinic28-25_r1",
}


def _load_detector_fixture(fixture_id: str) -> dict[str, dict]:
    return json.loads(
        (SCORING_FIXTURES / f"{fixture_id}.json").read_text(encoding="utf-8")
    )


def _formal_fast_detector_results(complete: dict) -> dict[str, dict]:
    detectors = json.loads(json.dumps(complete["detector_results"], ensure_ascii=False))
    acne = detectors["acne"]["metrics"]
    count = int(acne["detection"]["count"])
    acne["artifact_policy"] = "formal-fast"
    acne["max_recall"] = {
        "status": "skipped",
        "reason": "formal_fast_policy",
        "yolo_count": count,
        "unsupervised_focal_count": 0,
        "diffuse_erythema_region_count": 0,
        "combined_count": count,
    }
    return detectors


def test_production_proxy_scores_all_eleven_modules_with_trace() -> None:
    complete = _load_detector_fixture("clinic28-25_r3")
    registry = json.loads(
        (ROOT / "calibration/metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )
    modules, _ = build_v2_proxy_modules(_formal_fast_detector_results(complete), registry)

    result = score_production_proxy(modules)

    assert result["scoring_profile_version"] == "production_proxy_v1"
    assert result["score_status"] == "production_proxy"
    assert len(result["module_scores"]) == 11
    assert sum(
        len(group["metrics"])
        for module in result["module_scores"].values()
        for group in module["groups"].values()
    ) == 137
    assert all(0.0 <= module["score"] <= 100.0 for module in result["module_scores"].values())
    assert all(module["grade"] for module in result["module_scores"].values())
    assert all(module["groups"] for module in result["module_scores"].values())
    assert all(module["score_valid"] is True for module in result["module_scores"].values())
    assert result["module_scores"]["acne_activity"]["score_valid"] is True
    assert all(
        abs(
            sum(group["base_score_contribution"] for group in module["groups"].values())
            - module["base_engineering_score"]
        ) < 1e-4
        for module in result["module_scores"].values()
    )
    assert result["score_trace"]["anchor_registry_sha256"]
    assert result["calibration_boundary"]["population_calibrated"] is False
    assert all(
        metric["formula"] != "legacy_projection"
        for module in result["module_scores"].values()
        for group in module["groups"].values()
        for metric in group["metrics"].values()
    )


def test_frozen_anchor_scores_match_the_explicit_projection() -> None:
    registry = json.loads(
        (ROOT / "calibration/metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )
    formulas = json.loads(
        (ROOT / "calibration/v2_explicit_formula_registry_v1.json").read_text(encoding="utf-8")
    )

    manifest = json.loads(
        (SCORING_FIXTURES / "manifest.json").read_text(encoding="utf-8")
    )

    for anchor_id, fixture_id in ANCHOR_FIXTURES.items():
        complete = _load_detector_fixture(fixture_id)
        modules, _ = build_v2_proxy_modules(_formal_fast_detector_results(complete), registry)
        scored = score_production_proxy(modules)

        fixture = manifest["fixtures"][fixture_id]
        fixture_path = SCORING_FIXTURES / fixture["fixture_file"]
        assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == fixture["fixture_sha256"]
        assert fixture["source_document_sha256"] == (
            formulas["anchor_sources"][anchor_id]["evidence_document_sha256"]
        )
        assert sum(
            len(group["metrics"])
            for module in modules.values()
            for group in module["groups"].values()
        ) == 137
        assert all(module["completeness"]["complete"] for module in modules.values())
        assert all(module["score_valid"] for module in scored["module_scores"].values())
        assert all(
            abs(
                module["base_engineering_score"]
                - formulas["module_score_anchors"][module_id]["base_scores"][anchor_id]
            ) < 1e-6
            for module_id, module in scored["module_scores"].items()
        )


def test_formal_fast_acne_scoring_uses_filtered_count_and_bounded_coburden() -> None:
    complete = _load_detector_fixture(ANCHOR_FIXTURES["clinic28-25_RGB_M"])
    registry = json.loads(
        (ROOT / "calibration/metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8")
    )
    detectors = _formal_fast_detector_results(complete)
    acne = detectors["acne"]["metrics"]
    acne["detection"]["count"] = 2
    acne["detection"]["raw_count"] = 200

    modules, _ = build_v2_proxy_modules(detectors, registry)
    papule = modules["acne_activity"]["groups"]["follicular_papule"]["metrics"]
    pustule = modules["acne_activity"]["groups"]["follicular_pustule"]["metrics"]
    skin_pixels = float(acne["preprocess"]["skin_pixels"])
    raised_ratio = float(
        detectors["texture"]["metrics"]["medical_metrics_v2"]
        ["overall_metrics"]["core_metrics"]["scope_and_burden"]
        ["raised_like_area_ratio"]
    )

    assert abs(papule["papule_density"]["value"] - 2 * 100000.0 / skin_pixels) < 1e-9
    assert abs(
        papule["papule_area_ratio"]["value"]
        - min(1.0, 2 * raised_ratio * 100000.0 / skin_pixels)
    ) < 1e-9
    assert 0.0 <= papule["papule_area_ratio"]["value"] <= 1.0
    assert 0.0 <= papule["p90_papule_relief"]["value"] <= 1.0
    assert 0.0 <= pustule["pustule_area_ratio"]["value"] <= 1.0
    assert 0.0 <= pustule["p90_pustule_redness"]["value"] <= 1.0
    assert 0.0 <= pustule["p90_pustule_relief"]["value"] <= 1.0

    acne["detection"]["raw_count"] = 9999
    changed_raw_modules, _ = build_v2_proxy_modules(detectors, registry)
    assert changed_raw_modules["acne_activity"]["groups"] == modules["acne_activity"]["groups"]


def test_scoring_fixtures_are_small_sanitized_and_provenance_locked() -> None:
    manifest_path = SCORING_FIXTURES / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formulas_path = ROOT / "calibration/v2_explicit_formula_registry_v1.json"
    formulas = json.loads(formulas_path.read_text(encoding="utf-8"))
    required_source_count = len({
        source
        for definition in formulas["metric_formulas"].values()
        for source in definition["source_metric_ids"]
    })

    assert manifest["fixture_schema"] == "dermavision_scoring_fixture_manifest_v1"
    assert manifest["formula_registry_sha256"] == hashlib.sha256(
        formulas_path.read_bytes()
    ).hexdigest()
    assert set(manifest["fixtures"]) == {
        "clinic28-09_r1", "clinic28-23_r1", "clinic28-25_r1", "clinic28-25_r3",
    }
    for fixture in manifest["fixtures"].values():
        fixture_path = SCORING_FIXTURES / fixture["fixture_file"]
        fixture_bytes = fixture_path.read_bytes()
        fixture_text = fixture_bytes.decode("utf-8").lower()
        assert len(fixture_bytes) == fixture["fixture_size_bytes"]
        assert len(fixture_bytes) < 10_000
        assert hashlib.sha256(fixture_bytes).hexdigest() == fixture["fixture_sha256"]
        assert fixture["required_source_metric_count"] == required_source_count == 102
        assert not any(
            token in fixture_text
            for token in (
                "/home/", "/tmp/", "\\\\", "model_path", "weights_path",
                "output_dir", "device_name", "debug_info", "cuda_device",
            )
        )


def test_legacy_unavailable_scoring_helper_remains_explicitly_null() -> None:
    scorer = getattr(production_proxy_scoring, "score_consumer_unavailable", None)
    assert callable(scorer), "legacy unavailable scoring helper is missing"
    complete = _load_detector_fixture("clinic28-25_r3")
    registry = json.loads(
        (ROOT / "calibration/metric_registry_v2_proxy_20260728.json").read_text(
            encoding="utf-8"
        )
    )
    modules, _ = build_v2_proxy_modules(
        _formal_fast_detector_results(complete),
        registry,
    )

    result = scorer(modules)

    assert result["status"] == "uncalibrated"
    assert result["scoring_profile_version"] == "consumer_unavailable_v1"
    assert result["score_status"] == "unavailable_consumer_profile"
    assert len(result["module_scores"]) == 11
    assert all(module["score"] is None for module in result["module_scores"].values())
    assert all(
        module["grade"] == "不可评估"
        for module in result["module_scores"].values()
    )
    assert all(
        module["score_valid"] is False
        for module in result["module_scores"].values()
    )
    assert result["calibration_boundary"]["anchor_sample_count"] == 0
