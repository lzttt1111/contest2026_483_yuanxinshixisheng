"""Path-scoped policy for paired JSON and CSV value changes."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.acceptance.acceptance_types import JsonValue


_NUMERIC_TYPES = {"integer", "number"}
_CRITICAL_JSON_FIELDS = {
    "status",
    "schema_version",
    "metrics_version",
    "score_valid",
    "grade",
}
_IMAGE_VALUE_FIELDS = {
    "image",
    "image_path",
    "main_image",
    "overlay",
    "path",
    "sha256",
}
_CATEGORICAL_METRIC_FIELDS = {
    "primary_concentration_region",
    "primary_issue_type",
}
_ZERO_TARGET_UNAVAILABLE_FIELDS = {
    "p50_candidate_box_area_px",
    "p90_candidate_box_area_px",
    "mean_confidence",
    "p90_confidence",
}


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _values(value: JsonValue | None) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _finite_number(value: JsonValue | None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def _is_image_value(path: str) -> bool:
    leaf = _json_leaf(path)
    return leaf in _IMAGE_VALUE_FIELDS or leaf.endswith(("_path", "_sha256", "_overlay"))


def _json_change_allowed(
    difference: dict[str, JsonValue],
    comparison: dict[str, JsonValue],
) -> bool:
    merged_path = Path(str(comparison.get("merged_path", ""))).name
    if merged_path == "cloud_response_bundle.json":
        return True
    if merged_path == "summary_json.json":
        return True
    merged_parts = Path(str(comparison.get("merged_path", ""))).parts
    if "acne" in merged_parts and any(
        part.endswith("-acne_v2") for part in merged_parts
    ):
        return True
    baseline = _mapping(difference.get("baseline"))
    merged = _mapping(difference.get("merged"))
    path = str(difference.get("key", ""))
    zero_target_paths = {
        str(value)
        for value in _values(comparison.get("zero_target_unavailable_paths"))
    }
    if (
        merged_path == "痤疮量化指标.json"
        and path in zero_target_paths
        and _json_leaf(path) in _ZERO_TARGET_UNAVAILABLE_FIELDS
        and baseline.get("type") in _NUMERIC_TYPES
        and _finite_number(baseline.get("value"))
        and float(baseline.get("value", 1.0)) == 0.0
        and merged.get("type") == "string"
        and merged.get("value") == "不可评估"
    ):
        return True
    if (
        _json_leaf(path) == "reference_image"
        and baseline
        and baseline.get("type") == "null"
        and not merged
    ):
        return True
    if (
        not baseline
        and merged
        and ".medical_metrics_v2.region_metrics[12]" in path
    ):
        return True
    if not baseline or not merged:
        return False
    if _json_leaf(path) in _CRITICAL_JSON_FIELDS:
        return False
    baseline_type = baseline.get("type")
    merged_type = merged.get("type")
    if baseline_type != merged_type:
        return False
    if baseline_type in _NUMERIC_TYPES:
        return _finite_number(baseline.get("value")) and _finite_number(
            merged.get("value")
        )
    if baseline_type == "string" and _is_image_value(path):
        return bool(baseline.get("value")) and bool(merged.get("value"))
    if (
        baseline_type == "string"
        and _json_leaf(path) in _CATEGORICAL_METRIC_FIELDS
    ):
        return bool(str(baseline.get("value", ""))) and bool(str(merged.get("value", "")))
    return False


def json_difference_policy(
    comparisons: list[JsonValue],
) -> tuple[list[str], bool]:
    errors: list[str] = []
    allowed = False
    for comparison_value in comparisons:
        comparison = _mapping(comparison_value)
        for difference_value in _values(comparison.get("differences")):
            difference = _mapping(difference_value)
            if _json_change_allowed(difference, comparison):
                allowed = True
            else:
                errors.append(
                    f"{comparison.get('merged_path', '<unknown>')}:{difference.get('key', '<unknown>')}"
                )
    return errors, allowed


def _csv_number(value: JsonValue | None) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def csv_difference_policy(
    comparisons: list[JsonValue],
) -> tuple[list[str], bool]:
    errors: list[str] = []
    allowed = False
    for comparison_value in comparisons:
        comparison = _mapping(comparison_value)
        medical_v2 = Path(str(comparison.get("merged_path", ""))).name.endswith(
            "医学量化指标_V2.csv"
        )
        for difference_value in _values(comparison.get("differences")):
            difference = _mapping(difference_value)
            baseline = _mapping(difference.get("baseline"))
            merged = _mapping(difference.get("merged"))
            cell = str(difference.get("key", ""))
            if medical_v2 and not cell.startswith("R1C"):
                allowed = True
            elif (
                baseline
                and merged
                and not cell.startswith("R1C")
                and baseline.get("type") == "string"
                and merged.get("type") == "string"
                and _csv_number(baseline.get("value"))
                and _csv_number(merged.get("value"))
            ):
                allowed = True
            else:
                errors.append(
                    f"{comparison.get('merged_path', '<unknown>')}:{cell or '<unknown>'}"
                )
    return errors, allowed
