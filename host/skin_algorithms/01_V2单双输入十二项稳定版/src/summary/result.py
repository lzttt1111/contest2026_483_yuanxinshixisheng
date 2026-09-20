"""Map the pure scoring entry output onto the ``SummaryResultV1`` contract.

The contract models are the single source of truth; this module only translates
the entry's plain dict into their shape and fixes the one representational gap
(``quality_gate`` is optional in the entry but a required ``str`` in the
contract).
"""

from __future__ import annotations

from typing import Any, Mapping

from aisia_contracts.overall_summary.v1 import (
    RESULT_SCHEMA_VERSION,
    SUMMARY_MODULES,
    SummaryRequestV1,
    SummaryResultV1,
)

from .release import Release

# The entry only knows "no gate was supplied"; the contract field is a str.
QUALITY_GATE_UNAVAILABLE = "unavailable"


def _score_entry(system: Mapping[str, Any]) -> dict[str, Any]:
    gate = system.get("quality_gate")
    return {
        "score": system.get("score"),
        "grade": system.get("grade"),
        "score_valid": bool(system.get("score_valid")),
        "score_status": str(system.get("score_status")),
        "quality_gate": QUALITY_GATE_UNAVAILABLE if gate is None else str(gate),
        "missing_inputs": [str(item) for item in (system.get("missing_inputs") or [])],
        "reason_codes": [str(item) for item in (system.get("reason_codes") or [])],
    }


def _modules(entry_modules: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "module_no": module_no,
            "name": external,
            "word_display": _score_entry(entry_modules[module_no]["word_display"]),
            "production_proxy_v1": _score_entry(
                entry_modules[module_no]["production_proxy_v1"]
            ),
        }
        for module_no, external, _ in SUMMARY_MODULES
    ]


def _completeness(
    request: SummaryRequestV1, entry_modules: Mapping[str, Any]
) -> dict[str, Any]:
    outcomes = request.algorithm_outcomes
    outcome_names = {outcome.algorithm_name for outcome in outcomes}
    return {
        "expected_algorithms": list(request.expected_algorithms),
        "terminal_success": [
            outcome.algorithm_name
            for outcome in outcomes
            if outcome.status == "success"
        ],
        "terminal_failed": [
            outcome.algorithm_name
            for outcome in outcomes
            if outcome.status == "failed"
        ],
        "not_requested": [
            name
            for name in request.expected_algorithms
            if name not in outcome_names
        ],
        "word_display_valid_count": sum(
            1 for module in entry_modules.values() if module["word_display"]["score_valid"]
        ),
        "production_proxy_valid_count": sum(
            1
            for module in entry_modules.values()
            if module["production_proxy_v1"]["score_valid"]
        ),
    }


def build_success_result(
    *,
    request: SummaryRequestV1,
    entry_result: Mapping[str, Any],
    release: Release,
    debug_info: dict[str, Any],
) -> SummaryResultV1:
    versions = entry_result["scoring_versions"]
    asset_sha256 = dict(versions["asset_sha256"])
    # The release table is the authoritative actual-byte proof; it must match.
    asset_sha256.update(release.asset_sha256)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "report_id": request.report_id,
        "attempt_id": request.attempt_id,
        "input_fingerprint": request.input_fingerprint,
        "capture_profile": request.capture_profile,
        "input_route": request.input_route,
        "scoring_release": request.scoring_release,
        "status": "success",
        "raw_result": {
            "scoring_versions": {
                "code_version": str(versions["code_version"]),
                "display_policy_version": str(versions["display_policy_version"]),
                "formula_registry_version": str(versions["formula_registry_version"]),
                "asset_sha256": asset_sha256,
            },
            "completeness": _completeness(request, entry_result["modules"]),
            "modules": _modules(entry_result["modules"]),
        },
        "error": None,
        "debug_info": debug_info,
    }
    return SummaryResultV1.model_validate(payload)


def build_failed_result(
    *,
    report_id: str,
    attempt_id: str,
    input_fingerprint: str,
    capture_profile: str,
    input_route: str,
    scoring_release: str,
    code: str,
    message: str,
    retryable: bool,
    debug_info: dict[str, Any],
) -> SummaryResultV1:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "report_id": report_id,
        "attempt_id": attempt_id,
        "input_fingerprint": input_fingerprint,
        "capture_profile": capture_profile,
        "input_route": input_route,
        "scoring_release": scoring_release,
        "status": "failed",
        "raw_result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": None,
        },
        "debug_info": debug_info,
    }
    return SummaryResultV1.model_validate(payload)


__all__ = ["QUALITY_GATE_UNAVAILABLE", "build_failed_result", "build_success_result"]
