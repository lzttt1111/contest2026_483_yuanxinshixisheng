"""Completeness gate: missing evidence must never score as 0 by accident."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.summary_scoring import load_default_assets, score_report_from_evidence

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
LEGACY_MODULES = ("01", "03", "04", "10")
PORES_LEAF = (
    "pores.metrics.medical_metrics_v2.overall_metrics.core_metrics."
    "scope_and_burden.feature_density_per_100k_skin_px"
)


@pytest.fixture(scope="module")
def assets():
    return load_default_assets()


@pytest.fixture()
def evidence() -> dict:
    return copy.deepcopy(
        json.loads((FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8"))[
            "detector_results"
        ]
    )


def _score(evidence: dict, *, gate: dict | None, assets) -> dict:
    return score_report_from_evidence(
        evidence,
        capture_profile="consumer",
        input_route="consumer_rgb",
        quality={"quality_status": "PASS"},
        input_quality_gate=gate,
        scoring_assets=assets,
    )


def test_missing_required_leaf_only_degrades_owning_module(evidence, assets) -> None:
    baseline = _score(evidence, gate={"status": "PASS"}, assets=assets)
    node = evidence["pores"]["metrics"]["medical_metrics_v2"]["overall_metrics"][
        "core_metrics"
    ]["scope_and_burden"]
    del node["feature_density_per_100k_skin_px"]
    gated = _score(evidence, gate={"status": "PASS"}, assets=assets)

    module = gated["modules"]["01"]
    assert module["word_display"]["score_valid"] is False
    assert PORES_LEAF in module["word_display"]["missing_inputs"]
    assert module["production_proxy_v1"]["score_valid"] is False
    assert PORES_LEAF in module["production_proxy_v1"]["missing_inputs"]
    for module_id in ("02", "05", "06", "07", "08", "09", "11"):
        assert gated["modules"][module_id] == baseline["modules"][module_id]


def test_legitimate_zero_target_stays_scorable_zero(evidence, assets) -> None:
    evidence["acne"]["metrics"]["detection"] = {"count": 0}
    out = _score(evidence, gate={"status": "PASS"}, assets=assets)
    acne = out["modules"]["06"]["word_display"]
    assert acne["score_valid"] is True
    assert acne["score"] == 0.0
    assert acne["grade"] == "未见明显"


def test_missing_acne_count_is_not_faked_as_zero(evidence, assets) -> None:
    evidence["acne"]["metrics"].pop("detection")
    out = _score(evidence, gate={"status": "PASS"}, assets=assets)
    acne = out["modules"]["06"]["word_display"]
    assert acne["score_valid"] is False
    assert acne["score"] is None
    assert "acne.metrics.detection.count" in acne["missing_inputs"]


def test_missing_quality_gate_degrades_legacy_word_only(evidence, assets) -> None:
    baseline = _score(evidence, gate={"status": "PASS"}, assets=assets)
    out = _score(evidence, gate=None, assets=assets)
    for module_id in LEGACY_MODULES:
        word = out["modules"][module_id]["word_display"]
        assert word["score_valid"] is False
        assert word["score"] is None
        assert word["grade"] == "不可评估"
        assert "missing_input_quality_gate" in word["reason_codes"]
        assert out["modules"][module_id]["production_proxy_v1"] == (
            baseline["modules"][module_id]["production_proxy_v1"]
        )


def test_present_pass_gate_emits_formal_legacy_scores(evidence, assets) -> None:
    out = _score(evidence, gate={"status": "PASS"}, assets=assets)
    for module_id in ("01", "04", "10"):
        word = out["modules"][module_id]["word_display"]
        assert word["score_valid"] is True
        assert word["score"] is not None


def test_overall_algorithm_missing_degrades_cross_module_dependents(evidence, assets) -> None:
    baseline = _score(evidence, gate={"status": "PASS"}, assets=assets)
    evidence.pop("wrinkle")
    out = _score(evidence, gate={"status": "PASS"}, assets=assets)
    for module_id in ("07", "08", "09", "11"):
        assert out["modules"][module_id]["word_display"]["score_valid"] is False
        assert out["modules"][module_id]["production_proxy_v1"]["score_valid"] is False
    for module_id in ("01", "02", "04", "05", "10"):
        assert out["modules"][module_id] == baseline["modules"][module_id]
