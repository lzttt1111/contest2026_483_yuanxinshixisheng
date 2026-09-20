from __future__ import annotations

import json
from pathlib import Path

from scripts.acceptance.acceptance_types import SurfaceAssessment
from scripts.acceptance.run_acceptance_surface import assess_surface
from scripts.acceptance.surface_assessment import (
    append_contract_error,
    build_cloud_assessment,
)


def test_assess_surface_rejects_exit_zero_with_missing_or_stale_outputs() -> None:
    # Given
    missing = SurfaceAssessment(
        exit_code=0,
        started_ns=100,
        expected_cases=("clinic28-25", "clinic28-09", "clinic28-23"),
        discovered_cases=("clinic28-25",),
        oldest_required_output_ns=200,
        task_count=0,
        projected_result_count=0,
        service_process_starts=0,
        pipeline_initializations=0,
    )
    stale = SurfaceAssessment(
        exit_code=0,
        started_ns=300,
        expected_cases=("clinic28-25", "clinic28-09", "clinic28-23"),
        discovered_cases=("clinic28-25", "clinic28-09", "clinic28-23"),
        oldest_required_output_ns=200,
        task_count=0,
        projected_result_count=0,
        service_process_starts=0,
        pipeline_initializations=0,
    )

    # When / Then
    assert assess_surface(missing).status == "failed"
    assert "missing_cases" in assess_surface(missing).reason_codes
    assert assess_surface(stale).status == "failed"
    assert "stale_output" in assess_surface(stale).reason_codes


def test_cloud_assessment_counts_tasks_purple_projection_and_reuse_evidence(
    tmp_path: Path,
) -> None:
    # Given
    evidence = tmp_path / "evidence"
    results = tmp_path / "results"
    evidence.mkdir()
    cases = ("clinic28-25", "clinic28-09", "clinic28-23")
    tasks = {
        name: {"response": {"status": "success"}}
        for name in (
            "redness",
            "spots",
            "brown",
            "texture",
            "pores",
            "purple",
            "acne",
            "wrinkle",
        )
    }
    for case_id in cases:
        sample = results / case_id
        sample.mkdir(parents=True)
        (sample / "cloud_response_bundle.json").write_text(
            json.dumps({"tasks": tasks}), encoding="utf-8"
        )
    (evidence / "cloud_batch_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "service_process_starts": 3,
                "pipeline_initializations": 3,
                "pid_reuse_consistent": True,
                "service_execution_order": ["dermavision", "acne", "wrinkle"],
                "service_inference_overlap_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    # When
    assessment = build_cloud_assessment(
        evidence=evidence,
        results=results,
        expected_cases=cases,
        started_ns=0,
        exit_code=0,
        profile="baseline",
    )

    # Then
    assert assessment.task_count == 24
    assert assessment.projected_result_count == 27
    assert assessment.service_process_starts == 3
    assert assessment.pipeline_initializations == 3
    assert assess_surface(assessment).status == "success"

    # When
    failed_freeze = append_contract_error(
        assessment,
        "inspection_error:AcceptanceInputError",
    )

    # Then
    assert failed_freeze.task_count == 24
    assert failed_freeze.projected_result_count == 27
    assert failed_freeze.service_process_starts == 3
    assert failed_freeze.pipeline_initializations == 3
    assert assess_surface(failed_freeze).status == "failed"
