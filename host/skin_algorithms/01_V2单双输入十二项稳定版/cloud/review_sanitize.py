from __future__ import annotations

"""Public-safe projection of internal Worker evidence for review pages."""

import copy
from pathlib import PurePosixPath
from typing import Any


HTML_HIDDEN_RAW_FIELDS = frozenset(
    {
        "analysis_face",
        "preprocessed_face",
        "stage1_candidates",
        "vote_heatmap",
        "stage2_centerline",
        "face_filter_debug",
        "texture_reference",
        "comparison",
        "region_metrics_csv",
        "summary_json",
        "device",
        "successful_runs",
        "failed_runs",
    }
)


def _safe_oss_key(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return value


def sanitize_worker_response(response: dict[str, Any]) -> dict[str, Any]:
    """Keep six fields while removing internal paths and debug-only media."""

    result = copy.deepcopy(response)
    debug = result.get("debug_info")
    safe_debug: dict[str, Any] = {}
    if isinstance(debug, dict):
        for key in ("execution_mode", "timing_seconds"):
            if key in debug:
                safe_debug[key] = debug[key]
        report_csv = _safe_oss_key(debug.get("report_csv"))
        if report_csv is not None:
            safe_debug["report_csv"] = report_csv
    result["debug_info"] = safe_debug
    return result


def sanitize_review_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Sanitize every task response without changing internal evidence files."""

    result = copy.deepcopy(bundle)
    input_value = result.get("input")
    if isinstance(input_value, str):
        result["input"] = PurePosixPath(input_value).name
    tasks = result.get("tasks")
    if isinstance(tasks, dict):
        for item in tasks.values():
            if isinstance(item, dict) and isinstance(item.get("response"), dict):
                item["response"] = sanitize_html_response(item["response"])
    services = result.get("services")
    if isinstance(services, dict):
        for service in services.values():
            if not isinstance(service, dict):
                continue
            service_input = service.get("input")
            if isinstance(service_input, str):
                service["input"] = PurePosixPath(service_input).name
            service.pop("simulated_oss_root", None)
            service_tasks = service.get("tasks")
            if not isinstance(service_tasks, dict):
                continue
            for item in service_tasks.values():
                if isinstance(item, dict) and isinstance(item.get("response"), dict):
                    item["response"] = sanitize_html_response(item["response"])
    return result


def sanitize_html_response(response: dict[str, Any]) -> dict[str, Any]:
    """Hide internal raw evidence only from the human review page."""

    result = sanitize_worker_response(response)
    raw = result.get("raw_result")
    if isinstance(raw, dict):
        for key in HTML_HIDDEN_RAW_FIELDS:
            raw.pop(key, None)
    return result


__all__ = [
    "sanitize_html_response",
    "sanitize_review_bundle",
    "sanitize_worker_response",
]
