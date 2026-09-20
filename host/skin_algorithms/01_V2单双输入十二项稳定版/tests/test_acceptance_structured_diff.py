from __future__ import annotations

import json
from pathlib import Path

from scripts.acceptance.acceptance_types import JsonDict
from scripts.acceptance.deep_compare_contracts import audit_manifest_staleness
from scripts.acceptance.deep_compare_structured import (
    compare_csv_trees,
    compare_json_trees,
    flatten_json,
)


def _task(name: str) -> JsonDict:
    return {
        "queue": name,
        "task": f"{name}.analyze_image",
        "arguments": [f"id-{name}", "input.jpg", "prefix/", [name]],
        "response": {
            "record_id": f"id-{name}",
            "status": "success",
            "schema_version": "1",
            "meta_data": {"name": name, "version": "1"},
            "raw_result": {},
            "debug_info": {},
        },
    }


def test_flatten_json_preserves_paths_types_and_scalar_values() -> None:
    # Given
    value = {"metrics": {"count": 3, "status": None}, "flags": ["ok"]}

    # When
    rows = flatten_json(value)
    by_path = {str(row["path"]): row for row in rows}

    # Then
    assert by_path["$.metrics.count"] == {
        "path": "$.metrics.count",
        "type": "integer",
        "value": 3,
    }
    assert by_path["$.metrics.status"]["type"] == "null"
    assert by_path["$.flags[0]"]["value"] == "ok"


def test_compare_csv_trees_reports_cells_and_rejects_xlsx(tmp_path: Path) -> None:
    # Given
    baseline = tmp_path / "baseline"
    merged = tmp_path / "merged"
    baseline.mkdir()
    merged.mkdir()
    (baseline / "metrics.csv").write_text("name,value\ncount,1\n", encoding="utf-8")
    (merged / "metrics.csv").write_text("name,value\ncount,2\n", encoding="utf-8")
    (merged / "forbidden.xlsx").write_bytes(b"xlsx")

    # When
    comparison = compare_csv_trees(baseline, merged)

    # Then
    csv_comparison = comparison["csv_comparisons"][0]
    assert csv_comparison["differences"] == [
        {
            "key": "R2C2",
            "baseline": {"cell": "R2C2", "type": "string", "value": "1"},
            "merged": {"cell": "R2C2", "type": "string", "value": "2"},
        }
    ]
    assert comparison["unexpected_xlsx"] == ["forbidden.xlsx"]


def test_audit_manifest_staleness_exposes_missing_schema_version(tmp_path: Path) -> None:
    # Given
    contract_root = tmp_path / "cloud" / "contracts"
    contract_root.mkdir(parents=True)
    repositories = {
        name: {
            "success_envelope": [
                "record_id",
                "status",
                "meta_data",
                "raw_result",
                "debug_info",
            ]
        }
        for name in ("dermavision", "acne", "wrinkle")
    }
    (contract_root / "internal_dev_contracts.json").write_text(
        json.dumps({"source_repositories": repositories}), encoding="utf-8"
    )
    bundle = {
        "tasks": {
            "redness": _task("redness"),
            "acne": _task("acne"),
            "wrinkle": _task("wrinkle"),
        }
    }

    # When
    result = audit_manifest_staleness(tmp_path, bundle)

    # Then
    assert result["status"] == "stale"
    assert len(result["findings"]) == 3
    assert all(
        finding["missing_from_manifest"] == ["schema_version"]
        for finding in result["findings"]
    )


def test_structured_comparison_reports_unmatched_and_ambiguous_files(
    tmp_path: Path,
) -> None:
    # Given
    baseline = tmp_path / "baseline"
    merged = tmp_path / "merged"
    for relative in ("a/result.json", "b/result.json", "only-baseline.json"):
        path = baseline / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    merged_result = merged / "c/result.json"
    merged_result.parent.mkdir(parents=True)
    merged_result.write_text("{}", encoding="utf-8")
    (merged / "only-merged.json").write_text("{}", encoding="utf-8")

    # When
    comparison = compare_json_trees(baseline, merged)

    # Then
    assert comparison["comparisons"] == []
    assert comparison["unmatched_baseline"] == [
        "a/result.json",
        "b/result.json",
        "only-baseline.json",
    ]
    assert comparison["unmatched_merged"] == [
        "c/result.json",
        "only-merged.json",
    ]
    assert comparison["ambiguous"] == [
        {
            "basename": "result.json",
            "baseline": ["a/result.json", "b/result.json"],
            "merged": ["c/result.json"],
        }
    ]
