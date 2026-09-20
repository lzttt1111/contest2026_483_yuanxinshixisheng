"""Summary aggregation task: validation, rejection codes, degradation, release.

CPU-only, no broker, no GPU/model: fixture ``detector_results`` are projected
through the B1 ``scoring_input`` builder exactly like the consumption-loop test,
then fed to the task body directly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from aisia_contracts.canonicalization import fingerprint_v1
from aisia_contracts.overall_summary.v1 import SUMMARY_MODULES, SummaryRequestV1
from src.scoring_input import build_scoring_input
from src.summary import release as summary_release
from src.summary import worker as summary_worker
from src.summary.request import recompute_fingerprint

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores",
    "purple", "acne", "wrinkle", "surface_gloss", "vascular", "contour_firmness",
)
FAST_FIVE = {"redness", "spots", "brown", "texture", "pores"}
MODULES = tuple(f"{index:02d}" for index in range(1, 12))


def _runtime(algorithm: str, source: Mapping[str, Any]) -> dict[str, Any]:
    if algorithm in FAST_FIVE:
        metrics = dict(source[algorithm]["metrics"])
        medical = metrics.pop("medical_metrics_v2", None)
        return {"metrics": metrics, "medical_metrics_v2": medical}
    if algorithm == "purple":
        return {
            "full_metrics": {
                "uv_spots": source["uv_spots"]["metrics"]["uv_spots"],
                "porphyrin": source["porphyrin"]["metrics"]["porphyrin"],
            }
        }
    if algorithm in {"surface_gloss", "vascular", "contour_firmness"}:
        return {"full_metrics": source[algorithm]["metrics"]}
    if algorithm == "wrinkle":
        return {"summary": source["wrinkle"]["metrics"]}
    if algorithm == "acne":
        metrics = source["acne"]["metrics"]
        return {
            "summary": {
                "detection": metrics.get("detection"),
                "preprocess": metrics.get("preprocess"),
            }
        }
    raise AssertionError(algorithm)


def _scoring_input(
    algorithm: str,
    source: Mapping[str, Any],
    *,
    profile: str,
    route: str,
) -> dict[str, Any]:
    return build_scoring_input(
        algorithm_name=algorithm,
        detection_schema_version="3",
        detection_impl_version="1",
        report_id="fixture-report",
        detection_attempt_id=f"attempt-{algorithm}",
        source_image_sha256="f" * 64,
        capture_profile=profile,
        input_route=route,
        runtime=_runtime(algorithm, source),
    )


def _base_payload(
    case: str = "clinic28-25_r1",
    *,
    profile: str = "consumer",
    route: str = "consumer_queue",
) -> dict[str, Any]:
    source = json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))[
        "detector_results"
    ]
    scoring_inputs: dict[str, Any] = {}
    outcomes: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        scoring_input = _scoring_input(algorithm, source, profile=profile, route=route)
        assert scoring_input["evidence_status"] == "present", algorithm
        scoring_inputs[algorithm] = scoring_input
        outcomes.append(
            {
                "algorithm_name": algorithm,
                "status": "success",
                "detection_attempt_id": f"attempt-{algorithm}",
                "detection_schema_version": "3",
                "evidence_status": scoring_input["evidence_status"],
            }
        )
    return {
        "schema_version": "overall_summary_request_v1",
        "report_id": "fixture-report",
        "attempt_id": "summary-attempt-1",
        "capture_profile": profile,
        "input_route": route,
        "expected_algorithms": list(ALGORITHMS),
        "algorithm_outcomes": outcomes,
        "scoring_inputs": scoring_inputs,
        "scoring_release": summary_worker.RELEASE_ID,
        "input_fingerprint": "0" * 64,
    }


def _finalize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, then set the canonical fingerprint (mirrors backend C3)."""
    dumped = SummaryRequestV1.model_validate(dict(payload)).model_dump(mode="json")
    dumped["input_fingerprint"] = fingerprint_v1(
        {key: value for key, value in dumped.items() if key != "input_fingerprint"}
    )
    return dumped


def _run(payload: Mapping[str, Any]) -> dict[str, Any]:
    return summary_worker.run_aggregate_summary(payload)


def test_positive_request_produces_valid_eleven_module_result() -> None:
    payload = _finalize(_base_payload())
    result = _run(payload)

    assert result["status"] == "success"
    assert result["error"] is None
    assert result["report_id"] == "fixture-report"
    assert result["attempt_id"] == "summary-attempt-1"
    assert result["input_fingerprint"] == payload["input_fingerprint"]
    assert result["scoring_release"] == summary_worker.RELEASE_ID

    raw = result["raw_result"]
    assert [module["module_no"] for module in raw["modules"]] == list(MODULES)
    assert [module["name"] for module in raw["modules"]] == [
        external for _, external, _ in SUMMARY_MODULES
    ]
    for module in raw["modules"]:
        for system in ("word_display", "production_proxy_v1"):
            entry = module[system]
            assert set(entry) == {
                "score", "grade", "score_valid", "score_status",
                "quality_gate", "missing_inputs", "reason_codes",
            }
            assert isinstance(entry["quality_gate"], str)
            if entry["score_valid"]:
                assert isinstance(entry["score"], float)
                assert entry["grade"] != "不可评估"
            else:
                assert entry["score"] is None
                assert entry["grade"] == "不可评估"

    completeness = raw["completeness"]
    assert completeness["expected_algorithms"] == list(ALGORITHMS)
    assert completeness["terminal_success"] == list(ALGORITHMS)
    assert completeness["terminal_failed"] == []
    assert completeness["not_requested"] == []
    assert completeness["word_display_valid_count"] == sum(
        1 for module in raw["modules"] if module["word_display"]["score_valid"]
    )
    assert completeness["production_proxy_valid_count"] == sum(
        1 for module in raw["modules"] if module["production_proxy_v1"]["score_valid"]
    )
    assert completeness["production_proxy_valid_count"] > 0

    asset_sha256 = raw["scoring_versions"]["asset_sha256"]
    for name, pinned in summary_release.PINNED_ASSET_SHA256.items():
        assert asset_sha256[name] == pinned
    assert asset_sha256["scoring_input_field_list_v3.json"] == (
        "e4f4cc694ba48df6fe63e404ed53a6ff416deb900660958076da4c151c0190ff"
    )
    assert "elapsed_ms" in result["debug_info"]
    assert "/" not in json.dumps(result["debug_info"])


def test_fingerprint_rule_matches_backend_c3() -> None:
    payload = _finalize(_base_payload())
    model = SummaryRequestV1.model_validate(payload)
    canonical = {key: value for key, value in model.model_dump(mode="json").items()
                 if key != "input_fingerprint"}
    assert recompute_fingerprint(model) == fingerprint_v1(canonical)
    assert recompute_fingerprint(model) == payload["input_fingerprint"]


def test_task_object_runs_directly_without_broker() -> None:
    assert summary_worker.aggregate_summary.name == "dermavision.aggregate_summary"
    result = summary_worker.aggregate_summary(_finalize(_base_payload()))
    assert result["status"] == "success"


@pytest.mark.parametrize("case", ("clinic28-09_r1", "clinic28-25_r3"))
def test_positive_other_fixtures(case: str) -> None:
    result = _run(_finalize(_base_payload(case)))
    assert result["status"] == "success"
    assert len(result["raw_result"]["modules"]) == 11


def test_request_invalid_is_reported_not_raised() -> None:
    result = _run({"schema_version": "overall_summary_request_v1"})
    assert result["status"] == "failed"
    assert result["raw_result"] is None
    assert result["error"]["code"] == "request_invalid"
    assert result["error"]["retryable"] is False


def test_fingerprint_tamper_is_rejected() -> None:
    payload = _finalize(_base_payload())
    payload["input_fingerprint"] = "a" * 64
    result = _run(payload)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "fingerprint_mismatch"


def test_source_image_mismatch_is_rejected() -> None:
    payload = _base_payload()
    payload["scoring_inputs"]["redness"]["source_image_sha256"] = "e" * 64
    result = _run(_finalize(payload))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "identity_mismatch"


def test_capture_profile_mismatch_is_rejected() -> None:
    payload = _base_payload()
    payload["capture_profile"] = "institution"
    result = _run(_finalize(payload))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "identity_mismatch"


def test_scoring_release_mismatch_is_rejected() -> None:
    payload = _base_payload()
    payload["scoring_release"] = "other-release"
    result = _run(_finalize(payload))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "scoring_release_mismatch"


def test_release_asset_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _finalize(_base_payload())
    drifted = dict(summary_release.PINNED_ASSET_SHA256)
    drifted["metric_registry_v2_proxy_20260728.json"] = "0" * 64
    monkeypatch.setattr(summary_release, "PINNED_ASSET_SHA256", drifted)

    result = _run(payload)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "scoring_asset_mismatch"
    assert result["raw_result"] is None


def test_missing_evidence_degrades_module_but_task_succeeds() -> None:
    payload = _base_payload()
    missing = build_scoring_input(
        algorithm_name="acne",
        detection_schema_version="2",
        detection_impl_version="2",
        report_id="fixture-report",
        detection_attempt_id="attempt-acne",
        source_image_sha256="f" * 64,
        capture_profile="consumer",
        input_route="consumer_queue",
        runtime={},
    )
    assert missing["evidence_status"] == "missing"
    payload["scoring_inputs"]["acne"] = missing
    payload["algorithm_outcomes"] = [
        {**outcome, "evidence_status": "missing"}
        if outcome["algorithm_name"] == "acne"
        else outcome
        for outcome in payload["algorithm_outcomes"]
    ]

    result = _run(_finalize(payload))
    assert result["status"] == "success"
    modules = {module["module_no"]: module for module in result["raw_result"]["modules"]}
    # Module 06 (follicular acne activity) depends on the missing acne evidence.
    for system in ("word_display", "production_proxy_v1"):
        entry = modules["06"][system]
        assert entry["score_valid"] is False
        assert entry["score"] is None
        assert entry["grade"] == "不可评估"
        assert entry["missing_inputs"]
    # Other modules remain scored; the task itself is still a success.
    assert any(
        modules[module_id]["production_proxy_v1"]["score_valid"]
        for module_id in MODULES
        if module_id != "06"
    )


def test_scoring_exception_maps_to_scoring_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.summary_scoring as summary_scoring

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(summary_scoring, "score_report_from_evidence", _boom)
    result = _run(_finalize(_base_payload()))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "scoring_error"
    assert "boom" in result["error"]["message"]


def test_load_release_pins_all_expected_assets() -> None:
    release = summary_release.load_release()
    assert release.release_id == summary_release.RELEASE_ID
    assert dict(release.asset_sha256) == dict(summary_release.PINNED_ASSET_SHA256)
    assert set(release.asset_sha256) == {
        "scoring_input_field_list_v3.json",
        "v2_explicit_formula_registry_v1.json",
        "metric_registry_v2_proxy_20260728.json",
        "word_population_reference_1000.json",
        "全量历史ECDF评分配置.json",
    }


def test_copy_of_positive_payload_is_independent() -> None:
    payload = _finalize(_base_payload())
    mutated = copy.deepcopy(payload)
    mutated["scoring_inputs"]["purple"]["evidence"] = {}
    assert payload["scoring_inputs"]["purple"]["evidence"]
