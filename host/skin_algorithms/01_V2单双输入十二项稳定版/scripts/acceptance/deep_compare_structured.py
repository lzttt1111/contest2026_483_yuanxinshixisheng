"""JSON-path and CSV-cell comparison with XLSX rejection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeAlias

from pydantic import TypeAdapter
from typing_extensions import assert_never

from scripts.acceptance.acceptance_types import JsonDict, JsonValue
from scripts.acceptance.deep_compare_pairing import pair_file_trees


FlatRows: TypeAlias = list[dict[str, JsonValue]]
JSON_ADAPTER = TypeAdapter(JsonValue)


def _json_type(value: JsonValue) -> str:
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


def flatten_json(value: JsonValue, path: str = "$") -> FlatRows:
    rows: FlatRows = [{"path": path, "type": _json_type(value), "value": value if not isinstance(value, (dict, list)) else None}]
    match value:
        case dict():
            for key in sorted(value):
                rows.extend(flatten_json(value[key], f"{path}.{key}"))
        case list():
            for index, item in enumerate(value):
                rows.extend(flatten_json(item, f"{path}[{index}]"))
        case None | bool() | int() | float() | str():
            pass
        case unreachable:
            assert_never(unreachable)
    return rows


def read_json_rows(path: Path) -> FlatRows:
    value = JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    return flatten_json(value)


def _zero_target_unavailable_paths(path: Path) -> list[str]:
    if path.name != "痤疮量化指标.json":
        return []
    value = JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return []
    rows: list[tuple[str, JsonValue]] = [
        ("$.overall_metrics", value.get("overall_metrics")),
        *(
            [
                (f"$.region_metrics[{index}]", row)
                for index, row in enumerate(value.get("region_metrics", []))
            ]
            if isinstance(value.get("region_metrics"), list)
            else []
        ),
    ]
    suffixes = (
        ".core_metrics.scope_and_morphology.p50_candidate_box_area_px",
        ".core_metrics.scope_and_morphology.p90_candidate_box_area_px",
        ".core_metrics.signal_intensity.p90_confidence",
        ".auxiliary_metrics.signal_intensity.mean_confidence",
    )
    allowed: list[str] = []
    for prefix, row_value in rows:
        if not isinstance(row_value, dict):
            continue
        auxiliary = row_value.get("auxiliary_metrics")
        if not isinstance(auxiliary, dict):
            continue
        count_group = auxiliary.get("count_and_density")
        if not isinstance(count_group, dict):
            continue
        if count_group.get("suspected_acne_candidate_count") != 0:
            continue
        allowed.extend(f"{prefix}{suffix}" for suffix in suffixes)
    return allowed


def read_csv_cells(path: Path) -> FlatRows:
    rows: FlatRows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row_index, row in enumerate(csv.reader(stream), start=1):
            for column_index, value in enumerate(row, start=1):
                rows.append(
                    {
                        "cell": f"R{row_index}C{column_index}",
                        "type": "string",
                        "value": value,
                    }
                )
    return rows


def _compare_rows(baseline_rows: FlatRows, merged_rows: FlatRows, key: str) -> JsonDict:
    baseline = {str(row[key]): row for row in baseline_rows}
    merged = {str(row[key]): row for row in merged_rows}
    paths = sorted(set(baseline) | set(merged))
    differences: list[JsonValue] = []
    for path in paths:
        left = baseline.get(path)
        right = merged.get(path)
        if left != right:
            differences.append({"key": path, "baseline": left, "merged": right})
    return {
        "baseline_count": len(baseline_rows),
        "merged_count": len(merged_rows),
        "differences": differences,
    }


def compare_json_trees(baseline_root: Path, merged_root: Path) -> JsonDict:
    pairing = pair_file_trees(baseline_root, merged_root, {".json"})
    comparisons: list[JsonValue] = [
        {
            "baseline_path": pair.baseline.relative_to(baseline_root).as_posix(),
            "merged_path": pair.merged.relative_to(merged_root).as_posix(),
            "match_kind": pair.match_kind,
            "zero_target_unavailable_paths": _zero_target_unavailable_paths(
                pair.merged
            ),
            **_compare_rows(
                read_json_rows(pair.baseline),
                read_json_rows(pair.merged),
                "path",
            ),
        }
        for pair in pairing.pairs
    ]
    return {
        "comparisons": comparisons,
        "unmatched_baseline": list(pairing.unmatched_baseline),
        "unmatched_merged": list(pairing.unmatched_merged),
        "ambiguous": list(pairing.ambiguous),
    }


def compare_csv_trees(baseline_root: Path, merged_root: Path) -> JsonDict:
    pairing = pair_file_trees(baseline_root, merged_root, {".csv"})
    comparisons: list[JsonValue] = []
    for pair in pairing.pairs:
        comparisons.append(
            {
                "baseline_path": pair.baseline.relative_to(baseline_root).as_posix(),
                "merged_path": pair.merged.relative_to(merged_root).as_posix(),
                "match_kind": pair.match_kind,
                **_compare_rows(
                    read_csv_cells(pair.baseline),
                    read_csv_cells(pair.merged),
                    "cell",
                ),
            }
        )
    unexpected_xlsx = sorted(
        path.relative_to(merged_root).as_posix()
        for path in merged_root.rglob("*.xlsx")
        if path.is_file()
    )
    return {
        "csv_comparisons": comparisons,
        "unexpected_xlsx": unexpected_xlsx,
        "unmatched_baseline": list(pairing.unmatched_baseline),
        "unmatched_merged": list(pairing.unmatched_merged),
        "ambiguous": list(pairing.ambiguous),
    }
