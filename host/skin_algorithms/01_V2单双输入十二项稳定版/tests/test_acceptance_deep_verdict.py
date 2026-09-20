from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.acceptance.acceptance_types import JsonDict
from scripts.acceptance.deep_compare_verdict import evaluate_sections


def _passing_sections() -> JsonDict:
    worker_case = {
        "old_seven_zero_drift": True,
        "acne_v1_to_v2": {
            "baseline_contract_passed": True,
            "merged_contract_passed": True,
            "classification": "intentional_migration",
        },
        "added_three": {
            name: {"public_contract_passed": True, "overlay_exists": True}
            for name in ("surface_gloss", "vascular", "contour_firmness")
        },
    }
    report = {
        "valid_docx": True,
        "chapter_count": 11,
        "bookmark_count": 13,
        "path_leaks": [],
        "media": [{"sha256": "a" * 64}],
    }
    return {
        "tracked_files": {
            "comparison": [{"path": "feature.py", "status": "added"}]
        },
        "file_tree": {
            name: {
                "files": [{"path": "artifact", "sha256": "a" * 64}],
                "duplicate_image_groups": [],
            }
            for name in (
                "baseline_run",
                "merged_run",
                "baseline_cloud",
                "merged_cloud",
            )
        },
        "images": {
            "run": {
                "comparisons": [
                    {
                        "baseline_sha256": "a" * 64,
                        "merged_sha256": "b" * 64,
                        "pixel_diff": {"same_dimensions": True},
                    }
                ],
                "unmatched_baseline": [],
                "unmatched_merged": [],
                "ambiguous": [],
            },
            "cloud": {"comparisons": [], "unmatched_baseline": [], "unmatched_merged": [], "ambiguous": []},
        },
        "json": {
            "run": {
                "comparisons": [{
                    "baseline_path": "sample/红区量化指标.json",
                    "merged_path": "sample/红区量化指标.json",
                    "differences": [{
                        "key": "$.metrics.value",
                        "baseline": {"path": "$.metrics.value", "type": "number", "value": 1.0},
                        "merged": {"path": "$.metrics.value", "type": "number", "value": 2.0},
                    }],
                }],
                "ambiguous": [],
            },
            "cloud": {"comparisons": [], "ambiguous": []},
        },
        "csv_xlsx": {
            "run": {
                "csv_comparisons": [{
                    "baseline_path": "sample/红区量化指标.csv",
                    "merged_path": "sample/红区量化指标.csv",
                    "differences": [{
                        "key": "R2C2",
                        "baseline": {"cell": "R2C2", "type": "string", "value": "1.0"},
                        "merged": {"cell": "R2C2", "type": "string", "value": "2.0"},
                    }],
                }],
                "unexpected_xlsx": [],
                "ambiguous": [],
            },
            "cloud": {"csv_comparisons": [], "unexpected_xlsx": [], "ambiguous": []},
        },
        "worker_contracts": [deepcopy(worker_case) for _ in range(3)],
        "baseline_manifest_staleness": {"status": "stale", "findings": [{"stale": True}]},
        "acne_v1_to_v2": [],
        "added_three": [],
        "word": {
            "baseline": {"docx_count": 3, "reports": [deepcopy(report) for _ in range(3)]},
            "merged": {
                "docx_count": 6,
                "reports": [deepcopy(report) for _ in range(6)],
                "merged_formal_contract_passed": True,
            },
        },
        "path_leaks": {
            "baseline_run": [],
            "merged_run": [],
            "baseline_cloud": [],
            "merged_cloud": [],
        },
    }


def test_verdict_blocks_old_seven_field_drift() -> None:
    # Given
    sections = _passing_sections()
    sections["worker_contracts"][0]["old_seven_zero_drift"] = False

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "old_seven_drift" in {item.code for item in verdict.blocking_findings}


def test_verdict_blocks_missing_word_media() -> None:
    # Given
    sections = _passing_sections()
    sections["word"]["merged"]["reports"][0]["media"] = []

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "word_contract" in {item.code for item in verdict.blocking_findings}


def test_verdict_blocks_duplicate_merged_evidence_group() -> None:
    # Given
    sections = _passing_sections()
    sections["file_tree"]["merged_run"]["duplicate_image_groups"] = [
        {"sha256": "a" * 64, "paths": ["a.jpg", "b.jpg"]}
    ]

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "duplicate_evidence" in {item.code for item in verdict.blocking_findings}


def test_verdict_blocks_unknown_unmatched_item_artifact() -> None:
    # Given
    sections = _passing_sections()
    sections["json"]["run"]["unmatched_merged"] = [
        "batch_000001/clinic28-25/十二项检测/01_红区/internal_debug.json"
    ]

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "unexpected_unmatched" in {item.code for item in verdict.blocking_findings}


def test_verdict_blocks_known_public_name_in_wrong_item_directory() -> None:
    # Given
    sections = _passing_sections()
    sections["json"]["run"]["unmatched_merged"] = [
        "batch_000001/clinic28-25/十二项检测/01_红区/血管样结构量化指标.json"
    ]

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "unexpected_unmatched" in {item.code for item in verdict.blocking_findings}


@pytest.mark.parametrize("mutation", ("missing", "type", "status"))
def test_verdict_blocks_json_structure_type_and_status_drift(mutation: str) -> None:
    # Given
    sections = _passing_sections()
    difference = sections["json"]["run"]["comparisons"][0]["differences"][0]
    if mutation == "missing":
        difference["merged"] = None
    elif mutation == "type":
        difference["merged"] = {
            "path": "$.metrics.value", "type": "string", "value": "2.0"
        }
    else:
        difference["key"] = "$.status"
        difference["baseline"] = {
            "path": "$.status", "type": "string", "value": "success"
        }
        difference["merged"] = {
            "path": "$.status", "type": "string", "value": "failed"
        }

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "json_contract_drift" in {item.code for item in verdict.blocking_findings}


@pytest.mark.parametrize("mutation", ("missing", "status"))
def test_verdict_blocks_csv_missing_cell_and_status_drift(mutation: str) -> None:
    # Given
    sections = _passing_sections()
    difference = sections["csv_xlsx"]["run"]["csv_comparisons"][0]["differences"][0]
    if mutation == "missing":
        difference["merged"] = None
    else:
        difference["baseline"]["value"] = "success"
        difference["merged"]["value"] = "failed"

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"
    assert "csv_contract_drift" in {item.code for item in verdict.blocking_findings}


def test_verdict_allows_explicit_legacy_baseline_word_structure() -> None:
    # Given
    sections = _passing_sections()
    for report in sections["word"]["baseline"]["reports"]:
        report["chapter_count"] = 0
        report["bookmark_count"] = 0

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "passed"
    assert "baseline_legacy_word" in {
        item.code for item in verdict.intentional_differences
    }


@pytest.mark.parametrize("mutation", ("path_leak", "xlsx", "ambiguous"))
def test_verdict_blocks_leaks_xlsx_and_ambiguous_pairs(mutation: str) -> None:
    # Given
    sections = _passing_sections()
    if mutation == "path_leak":
        sections["path_leaks"]["merged_run"] = [{"path": "result.json"}]
    elif mutation == "xlsx":
        sections["csv_xlsx"]["run"]["unexpected_xlsx"] = ["stale.xlsx"]
    else:
        sections["images"]["run"]["ambiguous"] = [{"basename": "result.jpg"}]

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"


@pytest.mark.parametrize("mutation", ("acne", "added_three"))
def test_verdict_blocks_acne_and_added_three_failures(mutation: str) -> None:
    # Given
    sections = _passing_sections()
    worker = sections["worker_contracts"][1]
    if mutation == "acne":
        worker["acne_v1_to_v2"]["merged_contract_passed"] = False
    else:
        worker["added_three"]["vascular"]["public_contract_passed"] = False

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "failed"


def test_verdict_explicitly_classifies_expected_differences() -> None:
    # Given
    sections = _passing_sections()

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert verdict.status == "passed"
    assert {
        "tracked_file_changes",
        "image_pixel_changes",
        "json_value_changes",
        "csv_cell_changes",
        "baseline_manifest_stale",
        "acne_v1_to_v2",
        "added_algorithms",
    } <= {item.code for item in verdict.intentional_differences}


def test_verdict_allows_cloud_bundle_migration_only_with_strict_worker_pass() -> None:
    # Given
    sections = _passing_sections()
    difference = {
        "key": "$.tasks.acne.response.raw_result.new_field",
        "baseline": None,
        "merged": {"type": "number", "value": 1},
    }
    sections["json"]["cloud"]["comparisons"] = [{
        "merged_path": "clinic28-25/cloud_response_bundle.json",
        "differences": [difference],
    }]

    # When
    allowed = evaluate_sections(sections)
    sections["worker_contracts"][0]["old_seven_zero_drift"] = False
    rejected = evaluate_sections(sections)

    # Then
    assert allowed.status == "passed"
    assert rejected.status == "failed"


def test_verdict_allows_only_documented_added_cloud_artifacts() -> None:
    # Given
    sections = _passing_sections()
    sections["images"]["cloud"]["unmatched_merged"] = [
        "simulated_oss/report/visia2-clinic28-25-surface_gloss/"
        "surface_gloss/overlay.jpg"
    ]

    # When
    allowed = evaluate_sections(sections)
    sections["images"]["cloud"]["unmatched_merged"] = [
        "simulated_oss/report/visia2-clinic28-25-surface_gloss/"
        "surface_gloss/internal_debug.jpg"
    ]
    rejected = evaluate_sections(sections)

    # Then
    assert allowed.status == "passed"
    assert rejected.status == "failed"
