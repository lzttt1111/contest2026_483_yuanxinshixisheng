"""Fail-closed policy for deep comparison evidence."""

from __future__ import annotations

from scripts.acceptance.acceptance_types import JsonValue
from scripts.acceptance.deep_compare_diff_policy import (
    csv_difference_policy,
    json_difference_policy,
)
from scripts.acceptance.deep_compare_types import (
    ComparisonFinding,
    ComparisonVerdict,
)
from scripts.acceptance.deep_compare_unmatched_policy import allowed_unmatched


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _values(value: JsonValue | None) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def evaluate_sections(sections: dict[str, JsonValue]) -> ComparisonVerdict:
    """Classify expected changes separately from acceptance blockers."""

    blockers: list[ComparisonFinding] = []
    intentional: list[ComparisonFinding] = []

    def block(code: str, section: str, detail: str) -> None:
        blockers.append(ComparisonFinding(code=code, section=section, detail=detail))

    def allow(code: str, section: str, detail: str) -> None:
        intentional.append(ComparisonFinding(code=code, section=section, detail=detail))

    worker_contracts = sections.get("worker_contracts")
    worker_cases = _values(worker_contracts)
    if len(worker_cases) != 3:
        block("worker_cases_missing", "worker_contracts", "expected three paired cloud bundles")
    for index, value in enumerate(worker_cases):
        case = _mapping(value)
        compatible = case.get("old_seven_compatible")
        if compatible is None:
            compatible = case.get("old_seven_zero_drift")
        if compatible is not True:
            block("old_seven_drift", "worker_contracts", f"case {index + 1} old-seven contract drift")
        elif case.get("old_seven_zero_drift") is not True:
            allow(
                "old_seven_additive",
                "worker_contracts",
                f"case {index + 1} preserves all baseline paths and types with additive fields",
            )
        acne = _mapping(case.get("acne_v1_to_v2"))
        if (
            acne.get("baseline_contract_passed") is not True
            or acne.get("merged_contract_passed") is not True
            or acne.get("classification") != "intentional_migration"
        ):
            block("acne_contract", "worker_contracts", f"case {index + 1} acne migration invalid")
        added = _mapping(case.get("added_three"))
        if set(added) != {"surface_gloss", "vascular", "contour_firmness"}:
            block("added_three_contract", "worker_contracts", f"case {index + 1} added targets missing")
        for name, item_value in added.items():
            item = _mapping(item_value)
            if item.get("public_contract_passed") is not True or item.get("overlay_exists") is not True:
                block("added_three_contract", "worker_contracts", f"case {index + 1} {name} invalid")

    file_tree = _mapping(sections.get("file_tree"))
    for name in ("baseline_run", "merged_run", "baseline_cloud", "merged_cloud"):
        value = file_tree.get(name)
        tree = _mapping(value)
        files = _values(tree.get("files")) if tree else _values(value)
        if not files:
            block("artifacts_missing", "file_tree", f"{name} is empty")
    merged_tree = _mapping(file_tree.get("merged_run"))
    merged_duplicates = _values(merged_tree.get("duplicate_image_groups"))
    if any(
        _mapping(item).get("match_kind") in (None, "pixel_exact")
        for item in merged_duplicates
    ):
        block("duplicate_evidence", "file_tree", "merged run contains duplicate image groups")
    elif merged_duplicates:
        allow(
            "perceptual_similarity",
            "file_tree",
            "merged run perceptual-near groups recorded for visual review",
        )
    baseline_tree = _mapping(file_tree.get("baseline_run"))
    if _values(baseline_tree.get("duplicate_image_groups")):
        allow("baseline_duplicate_groups", "file_tree", "historical baseline duplicates recorded")

    word = _mapping(sections.get("word"))
    baseline_word = _mapping(word.get("baseline"))
    merged_word = _mapping(word.get("merged"))
    if merged_word.get("docx_count") != 6 or merged_word.get("merged_formal_contract_passed") is not True:
        block("word_contract", "word", "merged must contain six validated formal reports")
    baseline_reports = _values(baseline_word.get("reports"))
    merged_reports = _values(merged_word.get("reports"))
    if any(
        _mapping(report).get("chapter_count") != 11
        or _mapping(report).get("bookmark_count") != 13
        for report in baseline_reports
    ):
        allow("baseline_legacy_word", "word", "historical baseline uses the legacy combined report")
    for report_value in merged_reports:
        report = _mapping(report_value)
        if (
            report.get("valid_docx") is not True
            or report.get("chapter_count") != 11
            or report.get("bookmark_count") != 13
            or bool(_values(report.get("path_leaks")))
            or not _values(report.get("media"))
        ):
            block("word_contract", "word", "formal report structure or media is incomplete")

    leaks = _mapping(sections.get("path_leaks"))
    merged_public_leaks = [
        item for item in _values(leaks.get("merged_run"))
        if not str(_mapping(item).get("path", "")).startswith("_batch/")
    ]
    if merged_public_leaks:
        block("path_leak", "path_leaks", "public acceptance output contains an absolute path")
    if any(_values(leaks.get(name)) for name in ("baseline_run", "baseline_cloud", "merged_cloud")):
        allow("internal_path_leaks", "path_leaks", "historical or Worker debug artifacts retain internal paths and are report-only")

    csv_xlsx = _mapping(sections.get("csv_xlsx"))
    for channel in ("run", "cloud"):
        comparison = _mapping(csv_xlsx.get(channel))
        if _values(comparison.get("unexpected_xlsx")):
            block("unexpected_xlsx", "csv_xlsx", f"{channel} contains XLSX")
        ambiguous_csv = _values(comparison.get("ambiguous"))
        known_purple_split = {
            "紫区量化指标.csv",
            "紫区医学量化指标_V2.csv",
        }
        unexpected_ambiguous = [
            row for row in ambiguous_csv
            if _mapping(row).get("basename") not in known_purple_split
        ]
        if unexpected_ambiguous:
            allow(
                "topology_ambiguous",
                "csv_xlsx",
                f"{channel} topology changed; ambiguous CSV candidates retained in report",
            )
        elif ambiguous_csv:
            allow("purple_csv_split", "csv_xlsx", "historical purple CSV split into UV and porphyrin")
        unmatched_baseline = _values(comparison.get("unmatched_baseline"))
        unmatched_merged = _values(comparison.get("unmatched_merged"))
        unexpected = [
            str(path) for path in unmatched_merged
            if not allowed_unmatched(str(path), "csv_xlsx", channel)
        ]
        if unexpected:
            block("unexpected_unmatched", "csv_xlsx", f"{channel}: {unexpected}")
        elif unmatched_baseline or unmatched_merged:
            allow("unmatched_artifacts", "csv_xlsx", f"{channel} documented structural additions")

    images = _mapping(sections.get("images"))
    for channel in ("run", "cloud"):
        comparison_value = images.get(channel)
        comparison = _mapping(comparison_value)
        rows = _values(comparison.get("comparisons")) if comparison else _values(comparison_value)
        if _values(comparison.get("ambiguous")):
            block("ambiguous_pairs", "images", f"{channel} contains ambiguous image pairs")
        unmatched_baseline = _values(comparison.get("unmatched_baseline"))
        unmatched_merged = _values(comparison.get("unmatched_merged"))
        unexpected = [
            str(path) for path in unmatched_merged
            if not allowed_unmatched(str(path), "images", channel)
        ]
        if unexpected:
            block("unexpected_unmatched", "images", f"{channel}: {unexpected}")
        elif unmatched_baseline or unmatched_merged:
            allow("unmatched_artifacts", "images", f"{channel} documented structural additions")
        for row_value in rows:
            pixel_diff = _mapping(_mapping(row_value).get("pixel_diff"))
            if pixel_diff.get("same_dimensions") is False:
                block("image_dimensions", "images", f"{channel} image dimensions drift")
        if any(
            _mapping(row).get("baseline_sha256") != _mapping(row).get("merged_sha256")
            for row in rows
        ):
            allow("image_pixel_changes", "images", f"{channel} expected algorithm output changes")

    json_section = _mapping(sections.get("json"))
    for channel in ("run", "cloud"):
        value = json_section.get(channel)
        comparison = _mapping(value)
        rows = _values(comparison.get("comparisons")) if comparison else _values(value)
        if _values(comparison.get("ambiguous")):
            block("ambiguous_pairs", "json", f"{channel} contains ambiguous JSON pairs")
        unmatched_baseline = _values(comparison.get("unmatched_baseline"))
        unmatched_merged = _values(comparison.get("unmatched_merged"))
        unexpected = [
            str(path) for path in unmatched_merged
            if not allowed_unmatched(str(path), "json", channel)
        ]
        if unexpected:
            block("unexpected_unmatched", "json", f"{channel}: {unexpected}")
        elif unmatched_baseline or unmatched_merged:
            allow("unmatched_artifacts", "json", f"{channel} documented structural additions")
        json_errors, json_allowed = json_difference_policy(rows)
        if json_errors:
            block("json_contract_drift", "json", f"{channel}: {json_errors}")
        if json_allowed:
            allow("json_value_changes", "json", f"{channel} expected value changes")

    for channel in ("run", "cloud"):
        csv_rows = _values(_mapping(csv_xlsx.get(channel)).get("csv_comparisons"))
        csv_errors, csv_allowed = csv_difference_policy(csv_rows)
        if csv_errors:
            block("csv_contract_drift", "csv_xlsx", f"{channel}: {csv_errors}")
        if csv_allowed:
            allow("csv_cell_changes", "csv_xlsx", f"{channel} expected quantitative changes")

    tracked = _mapping(sections.get("tracked_files"))
    if any(_mapping(row).get("status") != "same" for row in _values(tracked.get("comparison"))):
        allow("tracked_file_changes", "tracked_files", "merged branch intentionally adds features")
    manifest = _mapping(sections.get("baseline_manifest_staleness"))
    if manifest.get("status") == "stale":
        allow("baseline_manifest_stale", "baseline_manifest_staleness", "known five-field prose is historical")
    elif manifest.get("status") != "current":
        block("manifest_unavailable", "baseline_manifest_staleness", "baseline manifest audit unavailable")
    if worker_cases:
        allow("acne_v1_to_v2", "worker_contracts", "explicit v1 to v2 migration")
        allow("added_algorithms", "worker_contracts", "three additive algorithms")
    return ComparisonVerdict(
        status="failed" if blockers else "passed",
        blocking_findings=tuple(blockers),
        intentional_differences=tuple(intentional),
    )
