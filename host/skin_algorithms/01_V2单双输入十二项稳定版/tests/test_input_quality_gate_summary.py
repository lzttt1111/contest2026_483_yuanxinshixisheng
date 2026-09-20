"""summary 闭环：持久化信封 → request → task 消费真实门禁。

- gate=PASS（合成图经真实 ``compute_input_quality_gate`` 算出）→ 01/04/10 formal。
- gate=REJECT（合成中心拼接）→ V0.1.1 真实拒绝语义，不伪造 formal。
- gate 缺失/不一致 → None 降级（既有诚实行为）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from aisia_contracts.canonicalization import fingerprint_v1
from aisia_contracts.overall_summary.v1 import SummaryRequestV1
from src.scoring_input import build_scoring_input
from src.scoring_input.gate import compute_input_quality_gate
from src.summary import worker as summary_worker
from src.summary.request import (
    GATE_CONSISTENT,
    GATE_INCONSISTENT,
    GATE_MISSING,
    resolve_input_quality_gate,
)

from input_gate_test_utils import StubPreprocessor, center_collage, clean_synthetic, encode_png

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"
ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores",
    "purple", "acne", "wrinkle", "surface_gloss", "vascular", "contour_firmness",
)
FAST_FIVE = {"redness", "spots", "brown", "texture", "pores"}
LEGACY_MODULES = ("01", "03", "04", "10")


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


def _payload(gate: dict[str, Any] | None) -> dict[str, Any]:
    source = json.loads(
        (FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8")
    )["detector_results"]
    scoring_inputs: dict[str, Any] = {}
    outcomes: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        scoring_input = build_scoring_input(
            algorithm_name=algorithm,
            detection_schema_version="3",
            detection_impl_version="1",
            report_id="gate-report",
            detection_attempt_id=f"attempt-{algorithm}",
            source_image_sha256="f" * 64,
            capture_profile="consumer",
            input_route="consumer_queue",
            runtime=_runtime(algorithm, source),
            input_quality_gate=gate,
        )
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
        "report_id": "gate-report",
        "attempt_id": "gate-summary-attempt",
        "capture_profile": "consumer",
        "input_route": "consumer_queue",
        "expected_algorithms": list(ALGORITHMS),
        "algorithm_outcomes": outcomes,
        "scoring_inputs": scoring_inputs,
        "scoring_release": summary_worker.RELEASE_ID,
        "input_fingerprint": "0" * 64,
    }


def _finalize(payload: Mapping[str, Any]) -> dict[str, Any]:
    dumped = SummaryRequestV1.model_validate(dict(payload)).model_dump(mode="json")
    dumped["input_fingerprint"] = fingerprint_v1(
        {key: value for key, value in dumped.items() if key != "input_fingerprint"}
    )
    return dumped


def _pass_gate() -> dict[str, Any]:
    gate = compute_input_quality_gate(
        encode_png(clean_synthetic()),
        preprocessor=StubPreprocessor(flags=()),
    )
    assert gate == {"status": "PASS", "reason_codes": []}
    return gate


def _reject_gate() -> dict[str, Any]:
    gate = compute_input_quality_gate(
        encode_png(center_collage()),
        preprocessor=StubPreprocessor(),
    )
    assert gate == {"status": "REJECT", "reason_codes": ["collage"]}
    return gate


def test_pass_gate_enables_legacy_formal_scores() -> None:
    payload = _finalize(_payload(_pass_gate()))
    result = summary_worker.run_aggregate_summary(payload)

    assert result["status"] in {"success", "partial"}
    modules = {module["module_no"]: module for module in result["raw_result"]["modules"]}
    for module_id in ("01", "04", "10"):
        word = modules[module_id]["word_display"]
        assert word["score_valid"] is True, module_id
        assert word["score"] is not None
        assert word["quality_gate"] == "PASS"
    # 03 的 UV 密度在 fixture 中缺失，诚实不可评估而不是补零。
    assert modules["03"]["word_display"]["score_valid"] is False
    assert result["raw_result"]["completeness"]["word_display_valid_count"] == 10
    assert result["debug_info"]["input_quality_gate"] == {
        "status": "PASS",
        "reason_codes": [],
        "diagnosis": GATE_CONSISTENT,
    }


def test_reject_gate_degrades_legacy_modules_honestly() -> None:
    payload = _finalize(_payload(_reject_gate()))
    result = summary_worker.run_aggregate_summary(payload)

    modules = {module["module_no"]: module for module in result["raw_result"]["modules"]}
    for module_id in LEGACY_MODULES:
        word = modules[module_id]["word_display"]
        assert word["score_valid"] is False, module_id
        assert word["score"] is None
        if module_id != "03":
            assert "input_quality_gate_reject" in word["reason_codes"]
    assert result["debug_info"]["input_quality_gate"] == {
        "status": "REJECT",
        "reason_codes": ["collage"],
        "diagnosis": GATE_CONSISTENT,
    }


def test_missing_gate_still_degrades_legacy_modules() -> None:
    payload = _finalize(_payload(None))
    result = summary_worker.run_aggregate_summary(payload)

    modules = {module["module_no"]: module for module in result["raw_result"]["modules"]}
    for module_id in LEGACY_MODULES:
        word = modules[module_id]["word_display"]
        assert word["score_valid"] is False
        assert "missing_input_quality_gate" in word["reason_codes"]
    assert result["debug_info"]["input_quality_gate"]["diagnosis"] == GATE_MISSING


def test_resolve_consistent_gate() -> None:
    model = SummaryRequestV1.model_validate(_payload(_pass_gate()))
    gate, diagnosis = resolve_input_quality_gate(model)
    assert diagnosis == GATE_CONSISTENT
    assert gate == {"status": "PASS", "reason_codes": []}


def test_resolve_missing_gate() -> None:
    model = SummaryRequestV1.model_validate(_payload(None))
    gate, diagnosis = resolve_input_quality_gate(model)
    assert diagnosis == GATE_MISSING
    assert gate is None


def test_resolve_inconsistent_gate_is_unusable() -> None:
    payload = _payload(_pass_gate())
    # 单个 scoring_input 被篡改为 REJECT：不挑选、不伪造，整体降级 None。
    payload["scoring_inputs"]["acne"]["quality"]["input_quality_gate"] = {
        "status": "REJECT",
        "reason_codes": ["collage"],
    }
    model = SummaryRequestV1.model_validate(payload)
    gate, diagnosis = resolve_input_quality_gate(model)
    assert diagnosis == GATE_INCONSISTENT
    assert gate is None


def test_inconsistent_gate_degrades_in_task() -> None:
    payload = _payload(_pass_gate())
    payload["scoring_inputs"]["pores"]["quality"]["input_quality_gate"] = {
        "status": "REJECT",
        "reason_codes": ["collage"],
    }
    result = summary_worker.run_aggregate_summary(_finalize(payload))

    modules = {module["module_no"]: module for module in result["raw_result"]["modules"]}
    word = modules["01"]["word_display"]
    assert word["score_valid"] is False
    assert "missing_input_quality_gate" in word["reason_codes"]
    assert result["debug_info"]["input_quality_gate"]["diagnosis"] == GATE_INCONSISTENT
