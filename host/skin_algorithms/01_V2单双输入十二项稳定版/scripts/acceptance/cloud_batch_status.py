"""Normalized service status evidence for serial cloud batches."""

from __future__ import annotations

from collections.abc import Mapping

from scripts.acceptance.acceptance_types import JsonDict, JsonValue


def build_service_records(
    services: tuple[str, ...],
    return_codes: Mapping[str, int],
    summaries: Mapping[str, JsonValue],
) -> JsonDict:
    records: JsonDict = {}
    for service in services:
        summary = summaries.get(service)
        if isinstance(summary, dict):
            records[service] = summary
        elif service in return_codes:
            records[service] = {
                "service": service,
                "status": "failed",
                "return_code": return_codes[service],
            }
        else:
            records[service] = {
                "service": service,
                "status": "not_started",
                "return_code": None,
            }
    return records
