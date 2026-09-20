"""Strict Worker task and added-algorithm contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import ValidationError

from cloud_contracts import validate_worker_envelope
from scripts.acceptance.acceptance_types import JsonDict, JsonValue


ADDED_THREE: Final = ("surface_gloss", "vascular", "contour_firmness")
DERMAVISION_TASK: Final = "dermavision.analyze_image"


def _mapping(value: JsonValue | Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def validate_task(
    algorithm: str,
    task: JsonDict,
    *,
    queue: str,
    task_name: str,
    argument_algorithm: str,
) -> JsonDict:
    errors: list[JsonValue] = []
    response = _mapping(task.get("response"))
    arguments = task.get("arguments")
    if task.get("queue") != queue:
        errors.append("queue")
    if task.get("task") != task_name:
        errors.append("task")
    if (
        not isinstance(arguments, list)
        or len(arguments) != 4
        or not isinstance(arguments[0], str)
        or not isinstance(arguments[1], str)
        or not isinstance(arguments[2], (str, type(None)))
        or arguments[3] != [argument_algorithm]
    ):
        errors.append("arguments")
    if not response or response.get("status") != "success":
        errors.append("success_response")
    else:
        try:
            validate_worker_envelope(algorithm, response)
        except (ValidationError, ValueError) as exc:
            errors.append(type(exc).__name__)
    return {"passed": not errors, "errors": errors}


def added_contract(task: JsonDict) -> JsonDict:
    response = _mapping(task.get("response"))
    raw_result = _mapping(response.get("raw_result"))
    name = _mapping(response.get("meta_data")).get("name")
    algorithm = str(name) if isinstance(name, str) else ""
    validation = validate_task(
        algorithm,
        task,
        queue=algorithm,
        task_name=DERMAVISION_TASK,
        argument_algorithm=algorithm,
    ) if algorithm in ADDED_THREE else {"passed": False, "errors": ["algorithm"]}
    metrics = raw_result.get("metrics")
    quality_status = raw_result.get("quality_status")
    quality_flags = raw_result.get("quality_flags")
    public_passed = (
        validation["passed"] is True
        and isinstance(raw_result.get("overlay"), str)
        and bool(raw_result.get("overlay"))
        and isinstance(raw_result.get("medical_report_csv_v2"), str)
        and bool(raw_result.get("medical_report_csv_v2"))
        and isinstance(metrics, dict)
        and bool(metrics)
        and isinstance(quality_status, str)
        and bool(quality_status)
        and isinstance(quality_flags, list)
        and all(isinstance(flag, str) for flag in quality_flags)
    )
    return {
        "public_contract_passed": public_passed,
        "validation": validation,
        "raw_result_keys": sorted(raw_result),
        "overlay": raw_result.get("overlay"),
        "metrics": raw_result.get("metrics"),
        "quality": {
            "score": raw_result.get("quality_score"),
            "status": quality_status,
            "flags": quality_flags,
        },
    }
