"""Surface success assessment independent of process exit claims."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter

from scripts.acceptance.acceptance_types import AssessmentResult, SurfaceAssessment


Profile = Literal["baseline", "merged"]
JSON_ADAPTER = TypeAdapter(JsonValue)


def append_contract_error(
    assessment: SurfaceAssessment,
    error: str,
) -> SurfaceAssessment:
    return replace(
        assessment,
        contract_errors=(*assessment.contract_errors, error),
    )


def assess_surface(assessment: SurfaceAssessment) -> AssessmentResult:
    """Reject process success unless fresh, complete evidence proves success."""

    reasons: list[str] = []
    if assessment.exit_code != 0:
        reasons.append("nonzero_exit")
    if assessment.discovered_cases != assessment.expected_cases:
        reasons.append("missing_cases")
    if assessment.oldest_required_output_ns <= assessment.started_ns:
        reasons.append("stale_output")
    if assessment.require_cloud_reuse:
        if assessment.task_count != assessment.expected_task_count:
            reasons.append("task_count_mismatch")
        if assessment.projected_result_count != assessment.expected_projected_result_count:
            reasons.append("projected_result_count_mismatch")
        if assessment.service_process_starts != 3:
            reasons.append("service_process_start_mismatch")
        if assessment.pipeline_initializations != 3:
            reasons.append("pipeline_initialization_mismatch")
    if assessment.contract_errors:
        reasons.append("contract_violation")
    return AssessmentResult(
        status="success" if not reasons else "failed",
        reason_codes=tuple(reasons),
    )


def build_cloud_assessment(
    evidence: Path,
    results: Path,
    expected_cases: tuple[str, ...],
    started_ns: int,
    exit_code: int,
    profile: Profile,
) -> SurfaceAssessment:
    summary_path = evidence / "cloud_batch_summary.json"
    summary: dict[str, JsonValue] = {}
    if summary_path.is_file():
        value = JSON_ADAPTER.validate_json(summary_path.read_text(encoding="utf-8"))
        summary = value if isinstance(value, dict) else {}
    bundles = sorted(results.rglob("cloud_response_bundle.json"))
    discovered = tuple(
        case_id for case_id in expected_cases if (results / case_id / "cloud_response_bundle.json").is_file()
    )
    bundle_values = [
        JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8")) for path in bundles
    ]
    task_count = sum(
        len(value.get("tasks", {}))
        for value in bundle_values
        if isinstance(value, dict) and isinstance(value.get("tasks"), dict)
    )
    projected = task_count + sum(
        1
        for value in bundle_values
        if isinstance(value, dict)
        and isinstance(value.get("tasks"), dict)
        and "purple" in value["tasks"]
    )
    contract_error_list: list[str] = []
    if summary.get("status") != "success":
        contract_error_list.append("cloud_summary_failed")
    if summary.get("pid_reuse_consistent") is not True:
        contract_error_list.append("cloud_pid_reuse_failed")
    if summary.get("service_execution_order") != ["dermavision", "acne", "wrinkle"]:
        contract_error_list.append("cloud_service_order_failed")
    if summary.get("service_inference_overlap_allowed") is not False:
        contract_error_list.append("cloud_inference_overlap_failed")
    return SurfaceAssessment(
        exit_code=exit_code,
        started_ns=started_ns,
        expected_cases=expected_cases,
        discovered_cases=discovered,
        oldest_required_output_ns=min((path.stat().st_mtime_ns for path in bundles), default=0),
        task_count=task_count,
        projected_result_count=projected,
        service_process_starts=int(summary.get("service_process_starts", 0)),
        pipeline_initializations=int(summary.get("pipeline_initializations", 0)),
        expected_task_count=24 if profile == "baseline" else 33,
        expected_projected_result_count=27 if profile == "baseline" else 36,
        require_cloud_reuse=True,
        contract_errors=tuple(contract_error_list),
    )
