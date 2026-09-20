from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True, slots=True)
class ConfirmationReportReceipt:
    path: Path
    sha256: str


def _canonical(document: Mapping[str, JsonValue]) -> bytes:
    body = {
        key: value for key, value in document.items()
        if key != "report_sha256"
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _contains_absolute_path(value: JsonValue) -> bool:
    if isinstance(value, str):
        return value.startswith("/") or _WINDOWS_ABSOLUTE.match(value) is not None
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    return False


def _all_promoted(results: Mapping[str, Mapping[str, JsonValue]]) -> bool:
    return bool(results) and all(
        result.get("status") == "promoted" for result in results.values()
    )


def write_confirmation_report(
    *,
    output_path: Path,
    attempted_count: int,
    compatibility_results: Mapping[str, Mapping[str, JsonValue]],
    population_results: Mapping[str, Mapping[str, JsonValue]],
    provenance: Mapping[str, JsonValue],
) -> ConfirmationReportReceipt:
    if attempted_count < 1:
        raise ValueError("confirmation attempted count must be positive")
    status = (
        "promoted"
        if _all_promoted(compatibility_results) and _all_promoted(population_results)
        else "blocked"
    )
    body: dict[str, JsonValue] = {
        "schema_version": "aisia_scoring_confirmation_report_v2",
        "status": status,
        "attempted_count": attempted_count,
        "compatibility_dimensions": dict(compatibility_results),
        "population_modules": dict(population_results),
        "provenance": dict(provenance),
        "confirmation_is_apply_only": True,
        "empty_evidence_policy": "null_not_zero",
    }
    if _contains_absolute_path(body):
        raise ValueError("confirmation report cannot contain absolute paths")
    report_sha = hashlib.sha256(_canonical(body)).hexdigest()
    document = {**body, "report_sha256": report_sha}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return ConfirmationReportReceipt(output_path, report_sha)


def verify_confirmation_report(document: Mapping[str, JsonValue]) -> str:
    expected = document.get("report_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("confirmation report SHA is missing")
    actual = hashlib.sha256(_canonical(document)).hexdigest()
    if actual != expected:
        raise ValueError("confirmation report SHA mismatch")
    if _contains_absolute_path(dict(document)):
        raise ValueError("confirmation report contains an absolute path")
    return actual


__all__ = [
    "ConfirmationReportReceipt",
    "verify_confirmation_report",
    "write_confirmation_report",
]
