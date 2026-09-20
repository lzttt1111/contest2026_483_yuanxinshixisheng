#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["numpy==1.26.4", "pillow==12.3.0", "pydantic==2.13.4"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly:
#      uv run scripts/acceptance/deep_compare.py --help
# 3. Or make executable and run:
#      chmod +x scripts/acceptance/deep_compare.py && ./scripts/acceptance/deep_compare.py --help
# ─────────────────

"""Produce machine JSON and human Markdown for baseline-versus-merged evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.acceptance.acceptance_types import JsonValue
from scripts.acceptance.deep_compare_contracts import (
    audit_manifest_staleness,
    compare_worker_bundles,
)
from scripts.acceptance.deep_compare_inventory import (
    IMAGE_SUFFIXES,
    compare_image_trees,
    compare_tracked_files,
    duplicate_file_groups,
    inventory_tree,
)
from scripts.acceptance.deep_compare_structured import compare_csv_trees, compare_json_trees
from scripts.acceptance.deep_compare_types import ComparisonRequest, DeepReport
from scripts.acceptance.deep_compare_verdict import evaluate_sections
from scripts.acceptance.deep_compare_word import inspect_word_tree, scan_output_path_leaks
from scripts.acceptance.deep_compare_word_formal import inspect_formal_word_tree


JSON_ADAPTER = TypeAdapter(JsonValue)


def _load_bundles(root: Path) -> list[tuple[str, dict[str, JsonValue]]]:
    bundles: list[tuple[str, dict[str, JsonValue]]] = []
    for path in sorted(root.rglob("cloud_response_bundle.json")):
        value = JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            bundles.append((path.parent.name, value))
    return bundles


def _contract_sections(request: ComparisonRequest) -> tuple[JsonValue, JsonValue, JsonValue, JsonValue]:
    baseline = dict(_load_bundles(request.baseline_cloud))
    merged = dict(_load_bundles(request.merged_cloud))
    comparisons: list[JsonValue] = []
    for case_id in sorted(set(baseline) & set(merged)):
        comparison = compare_worker_bundles(baseline[case_id], merged[case_id])
        added_three = comparison.get("added_three")
        if isinstance(added_three, dict):
            for item in added_three.values():
                if not isinstance(item, dict):
                    continue
                overlay = item.get("overlay")
                item["overlay_exists"] = isinstance(overlay, str) and (
                    request.merged_cloud / "simulated_oss" / overlay
                ).is_file()
        comparisons.append({"case_id": case_id, **comparison})
    first_baseline = next(iter(baseline.values()), {})
    staleness = audit_manifest_staleness(request.baseline_root, first_baseline)
    acne = [item["acne_v1_to_v2"] for item in comparisons if isinstance(item, dict)]
    added = [item["added_three"] for item in comparisons if isinstance(item, dict)]
    return comparisons, staleness, acne, added


def build_deep_report(request: ComparisonRequest) -> DeepReport:
    worker, staleness, acne, added = _contract_sections(request)
    sections: dict[str, JsonValue] = {
        "tracked_files": compare_tracked_files(request.baseline_root, request.merged_root),
        "file_tree": {
            "baseline_run": {
                "files": inventory_tree(request.baseline_run),
                "duplicate_image_groups": duplicate_file_groups(
                    request.baseline_run, IMAGE_SUFFIXES
                ),
            },
            "merged_run": {
                "files": inventory_tree(request.merged_run),
                "duplicate_image_groups": duplicate_file_groups(
                    request.merged_run, IMAGE_SUFFIXES, "十二项检测"
                ),
            },
            "baseline_cloud": {"files": inventory_tree(request.baseline_cloud)},
            "merged_cloud": {"files": inventory_tree(request.merged_cloud)},
        },
        "images": {
            "run": compare_image_trees(request.baseline_run, request.merged_run),
            "cloud": compare_image_trees(request.baseline_cloud, request.merged_cloud),
        },
        "json": {
            "run": compare_json_trees(request.baseline_run, request.merged_run),
            "cloud": compare_json_trees(request.baseline_cloud, request.merged_cloud),
        },
        "csv_xlsx": {
            "run": compare_csv_trees(request.baseline_run, request.merged_run),
            "cloud": compare_csv_trees(request.baseline_cloud, request.merged_cloud),
            "xlsx_policy": "forbidden_by_final_contract",
        },
        "worker_contracts": worker,
        "baseline_manifest_staleness": staleness,
        "acne_v1_to_v2": acne,
        "added_three": added,
        "word": {
            "baseline": inspect_word_tree(request.baseline_run),
            "merged": inspect_formal_word_tree(
                request.merged_run,
                request.formal_baseline_root,
            ),
        },
        "path_leaks": {
            "baseline_run": scan_output_path_leaks(request.baseline_run),
            "merged_run": scan_output_path_leaks(request.merged_run),
            "baseline_cloud": scan_output_path_leaks(request.baseline_cloud),
            "merged_cloud": scan_output_path_leaks(request.merged_cloud),
        },
    }
    return DeepReport(
        schema="aisia_deep_comparison_v1",
        sections=sections,
        verdict=evaluate_sections(sections),
    )


def write_deep_report(report: DeepReport, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.as_json(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    titles = {
        "tracked_files": "Tracked files",
        "file_tree": "File tree",
        "images": "Images",
        "json": "JSON paths, types, and values",
        "csv_xlsx": "CSV cells and XLSX rejection",
        "worker_contracts": "Worker contracts",
        "baseline_manifest_staleness": "Baseline manifest staleness",
        "acne_v1_to_v2": "Acne v1 to v2",
        "added_three": "Added three",
        "word": "Word",
        "path_leaks": "Path leaks",
    }
    lines = [
        "# AISIA baseline vs merged deep comparison",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(report.verdict.as_json(), ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for key, value in report.sections.items():
        lines.extend(
            [
                f"## {titles[key]}",
                "",
                "```json",
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                "```",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and merged acceptance evidence")
    for name in (
        "baseline-root",
        "merged-root",
        "baseline-run",
        "merged-run",
        "baseline-cloud",
        "merged-cloud",
        "formal-baseline-root",
        "json-output",
        "markdown-output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison_root = args.json_output.parent.resolve()
    if args.markdown_output.parent.resolve() != comparison_root:
        print("comparison outputs must share one directory", file=sys.stderr)
        return 2
    comparison_root.mkdir(parents=True, exist_ok=True)
    if any(comparison_root.iterdir()):
        print("comparison output directory must be empty", file=sys.stderr)
        return 2
    request = ComparisonRequest(
        args.baseline_root,
        args.merged_root,
        args.baseline_run,
        args.merged_run,
        args.baseline_cloud,
        args.merged_cloud,
        args.formal_baseline_root,
    )
    report = build_deep_report(request)
    write_deep_report(report, args.json_output, args.markdown_output)
    return 0 if report.verdict.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
