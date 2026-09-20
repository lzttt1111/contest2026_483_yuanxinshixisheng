"""Worker compatibility, migration, and manifest-staleness evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import assert_never

from scripts.acceptance.acceptance_types import JsonDict, JsonValue
from scripts.acceptance.deep_compare_worker_validation import (
    ADDED_THREE,
    DERMAVISION_TASK,
    added_contract,
    validate_task,
)


OLD_SEVEN: Final = (
    "redness",
    "spots",
    "brown",
    "texture",
    "pores",
    "purple",
    "wrinkle",
)
JSON_ADAPTER = TypeAdapter(JsonValue)


def _mapping(value: JsonValue | Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    return value


def _type_name(value: JsonValue) -> str:
    match value:
        case None:
            return "null"
        case bool():
            return "boolean"
        case int():
            return "integer"
        case float():
            return "number"
        case str():
            return "string"
        case list():
            return "array"
        case dict():
            return "object"
        case unreachable:
            assert_never(unreachable)


def structure_signature(value: JsonValue, path: str = "$") -> list[JsonValue]:
    rows: list[JsonValue] = [{"path": path, "type": _type_name(value)}]
    match value:
        case dict():
            for key in sorted(value):
                rows.extend(structure_signature(value[key], f"{path}.{key}"))
        case list():
            for index, item in enumerate(value):
                rows.extend(structure_signature(item, f"{path}[{index}]"))
        case None | bool() | int() | float() | str():
            pass
        case unreachable:
            assert_never(unreachable)
    return rows


def _task_contract(task: JsonDict) -> JsonDict:
    response = _mapping(task.get("response"))
    raw_result = _mapping(response.get("raw_result"))
    debug_info = _mapping(response.get("debug_info"))
    return {
        "queue": task.get("queue"),
        "task": task.get("task"),
        "arguments": task.get("arguments"),
        "envelope": structure_signature(response),
        "raw_result": structure_signature(raw_result),
        "debug_info": structure_signature(debug_info),
    }


def _signature_rows(value: JsonValue | None) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        return set()
    return {
        (str(row.get("path")), str(row.get("type")))
        for row in value
        if isinstance(row, dict)
    }


def _baseline_contract_preserved(baseline: JsonDict, merged: JsonDict) -> bool:
    if any(
        baseline.get(key) != merged.get(key)
        for key in ("queue", "task", "arguments")
    ):
        return False
    return all(
        _signature_rows(baseline.get(field)).issubset(
            _signature_rows(merged.get(field))
        )
        for field in ("envelope", "raw_result", "debug_info")
    )


def compare_worker_bundles(
    baseline_bundle: Mapping[str, JsonValue],
    merged_bundle: Mapping[str, JsonValue],
) -> JsonDict:
    baseline_tasks = _mapping(baseline_bundle.get("tasks"))
    merged_tasks = _mapping(merged_bundle.get("tasks"))
    old_rows: list[JsonValue] = []
    old_zero_drift = True
    old_compatible = True
    for name in OLD_SEVEN:
        baseline_task = _mapping(baseline_tasks.get(name))
        merged_task = _mapping(merged_tasks.get(name))
        baseline_contract = _task_contract(baseline_task)
        merged_contract = _task_contract(merged_task)
        task_name = "wrinkle.analyze_image" if name == "wrinkle" else DERMAVISION_TASK
        baseline_validation = validate_task(
            name,
            baseline_task,
            queue=name,
            task_name=task_name,
            argument_algorithm=name,
        )
        merged_validation = validate_task(
            name,
            merged_task,
            queue=name,
            task_name=task_name,
            argument_algorithm=name,
        )
        same = (
            baseline_validation["passed"] is True
            and merged_validation["passed"] is True
            and baseline_contract == merged_contract
        )
        compatible = (
            baseline_validation["passed"] is True
            and merged_validation["passed"] is True
            and _baseline_contract_preserved(baseline_contract, merged_contract)
        )
        old_zero_drift = old_zero_drift and same
        old_compatible = old_compatible and compatible
        old_rows.append(
            {
                "algorithm": name,
                "zero_drift": same,
                "baseline_compatible": compatible,
                "additive_paths": {
                    field: len(
                        _signature_rows(merged_contract.get(field))
                        - _signature_rows(baseline_contract.get(field))
                    )
                    for field in ("envelope", "raw_result", "debug_info")
                },
                "baseline": baseline_contract,
                "merged": merged_contract,
                "baseline_validation": baseline_validation,
                "merged_validation": merged_validation,
            }
        )
    baseline_acne = _mapping(baseline_tasks.get("acne"))
    merged_acne = _mapping(merged_tasks.get("acne_v2"))
    baseline_acne_validation = validate_task(
        "acne",
        baseline_acne,
        queue="acne",
        task_name="acne.analyze_image",
        argument_algorithm="acne",
    )
    merged_acne_validation = validate_task(
        "acne_v2",
        merged_acne,
        queue="acne_v2",
        task_name=DERMAVISION_TASK,
        argument_algorithm="acne",
    )
    added = {
        name: added_contract(_mapping(merged_tasks.get(name)))
        for name in ADDED_THREE
    }
    return {
        "old_seven_zero_drift": old_zero_drift,
        "old_seven_compatible": old_compatible,
        "old_seven": old_rows,
        "acne_v1_to_v2": {
            "baseline_target": "acne",
            "merged_target": "acne_v2",
            "baseline": _task_contract(baseline_acne),
            "merged": _task_contract(merged_acne),
            "baseline_contract_passed": baseline_acne_validation["passed"],
            "merged_contract_passed": merged_acne_validation["passed"],
            "classification": "intentional_migration",
        },
        "added_three": added,
    }


def audit_manifest_staleness(project_root: Path, bundle: Mapping[str, JsonValue]) -> JsonDict:
    manifest_path = project_root / "cloud" / "contracts" / "internal_dev_contracts.json"
    if not manifest_path.is_file():
        return {"status": "unavailable", "findings": []}
    manifest_value = JSON_ADAPTER.validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(manifest_value)
    repositories = _mapping(manifest.get("source_repositories"))
    tasks = _mapping(bundle.get("tasks"))
    representatives = {
        "dermavision": _mapping(tasks.get("redness")),
        "acne": _mapping(tasks.get("acne")) or _mapping(tasks.get("acne_v2")),
        "wrinkle": _mapping(tasks.get("wrinkle")),
    }
    findings: list[JsonValue] = []
    for repository, task in representatives.items():
        expected = _mapping(repositories.get(repository)).get("success_envelope")
        actual = sorted(_mapping(task.get("response")))
        expected_keys = sorted(str(item) for item in expected) if isinstance(expected, list) else []
        if expected_keys != actual:
            findings.append(
                {
                    "repository": repository,
                    "manifest_envelope": expected_keys,
                    "actual_envelope": actual,
                    "missing_from_manifest": sorted(set(actual) - set(expected_keys)),
                    "stale": True,
                }
            )
    return {
        "status": "stale" if findings else "current",
        "manifest": "cloud/contracts/internal_dev_contracts.json",
        "findings": findings,
    }
