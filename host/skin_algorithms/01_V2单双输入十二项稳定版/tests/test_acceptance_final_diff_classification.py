from __future__ import annotations

from pathlib import Path

from scripts.acceptance.deep_compare_pairing import pair_file_trees
from scripts.acceptance.deep_compare_verdict import evaluate_sections
from scripts.acceptance.deep_compare_word import docx_content_signature
from scripts.acceptance.deep_compare_word_formal import _subject_identity_passed
from scripts.acceptance.deep_compare_diff_policy import (
    csv_difference_policy,
    json_difference_policy,
)
from scripts.acceptance.deep_compare_unmatched_policy import allowed_unmatched
import scripts.acceptance.deep_compare_contracts as contracts
from test_acceptance_deep_verdict import _passing_sections
from test_acceptance_word_signature import _write_docx
from test_acceptance_worker_compare import _success_task


OLD_SEVEN = ("redness", "spots", "brown", "texture", "pores", "purple", "wrinkle")


def _old_tasks() -> dict[str, dict]:
    tasks = {
        name: _success_task(name, {"overlay": f"{name}.jpg"}, {"mode": "same"})
        for name in OLD_SEVEN
    }
    tasks["wrinkle"]["task"] = "wrinkle.analyze_image"
    return tasks


def test_old_worker_additions_are_compatible_but_deletions_block(monkeypatch) -> None:
    monkeypatch.setattr(contracts, "validate_task", lambda *_args, **_kwargs: {"passed": True, "errors": []})
    baseline = _old_tasks()
    merged = _old_tasks()
    merged["redness"]["response"]["raw_result"]["chin"] = {"count": 1}
    comparison = contracts.compare_worker_bundles({"tasks": baseline}, {"tasks": merged})
    assert comparison["old_seven_zero_drift"] is False
    assert comparison["old_seven_compatible"] is True

    del merged["spots"]["response"]["raw_result"]["overlay"]
    rejected = contracts.compare_worker_bundles({"tasks": baseline}, {"tasks": merged})
    assert rejected["old_seven_compatible"] is False


def test_verdict_allows_perceptual_review_but_blocks_pixel_duplicates() -> None:
    sections = _passing_sections()
    sections["file_tree"]["merged_run"]["duplicate_image_groups"] = [
        {"match_kind": "perceptual_near", "paths": ["a.jpg", "b.jpg"], "phash_distance": 1}
    ]
    assert evaluate_sections(sections).status == "passed"

    sections["file_tree"]["merged_run"]["duplicate_image_groups"] = [
        {"match_kind": "pixel_exact", "paths": ["a.jpg", "b.jpg"], "phash_distance": 0}
    ]
    assert evaluate_sections(sections).status == "failed"


def test_normalized_word_content_ignores_dynamic_image_count_not_text(tmp_path: Path) -> None:
    approved = tmp_path / "approved.docx"
    generated = tmp_path / "generated.docx"
    changed = tmp_path / "changed.docx"
    _write_docx(approved, "SCORE 81", (b"a", b"b"), ("media/image1.png", "media/image2.png"))
    _write_docx(generated, "SCORE 81", (b"c",), ("media/image1.png",))
    _write_docx(changed, "SCORE 0", (b"c",), ("media/image1.png",))
    assert docx_content_signature(approved, normalize_image_targets=True) == (
        docx_content_signature(generated, normalize_image_targets=True)
    )
    assert docx_content_signature(approved, normalize_image_targets=True) != (
        docx_content_signature(changed, normalize_image_targets=True)
    )


def test_pairing_uses_sample_and_basename_before_global_basename(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    merged = tmp_path / "merged"
    for root, middle in ((baseline, "七项检测/红区"), (merged, "十二项检测/01_红区")):
        for case in ("01_clinic28-25_RGB_M", "02_clinic28-09_RGB_M"):
            path = root / "batch_000001" / case / middle / "量化.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    result = pair_file_trees(baseline, merged, {".json"})
    assert len(result.pairs) == 2
    assert all(pair.match_kind == "sample_basename" for pair in result.pairs)
    assert result.ambiguous == ()


def test_known_root_batch_and_acne_v2_json_additions_are_classified() -> None:
    assert allowed_unmatched(
        "batch_000001/01_clinic28-25_RGB_M/十二项完整量化指标.json",
        "json",
        "run",
    )
    assert allowed_unmatched(
        "_batch/final-acceptance/final-acceptance.summary.csv",
        "csv_xlsx",
        "run",
    )
    assert allowed_unmatched(
        "simulated_oss/report/visia2-clinic28-25-acne_v2/acne/acne_summary.json",
        "json",
        "cloud",
    )


def test_medical_chin_and_internal_wrinkle_changes_remain_reported_not_blocked() -> None:
    json_rows = [
        {
            "merged_path": "sample/metrics.json",
            "differences": [{
                "key": "$.medical_metrics_v2.region_metrics[12].analysis_region",
                "baseline": None,
                "merged": {"path": "x", "type": "string", "value": "下巴"},
            }],
        },
        {
            "merged_path": "worker/summary_json.json",
            "differences": [{"key": "$.report_group_overlays", "baseline": None, "merged": {"type": "object"}}],
        },
    ]
    assert json_difference_policy(json_rows) == ([], True)
    csv_rows = [{
        "merged_path": "sample/医学量化指标_V2.csv",
        "differences": [{"key": "R15C1", "baseline": None, "merged": {"type": "string", "value": "下巴"}}],
    }]
    assert csv_difference_policy(csv_rows) == ([], True)


def test_categorical_metrics_privacy_removal_and_acne_v2_internal_changes_are_allowed() -> None:
    rows = [
        {
            "merged_path": "sample/metrics.json",
            "differences": [
                {
                    "key": "$.reference_image",
                    "baseline": {"type": "null", "value": None},
                    "merged": None,
                },
                {
                    "key": "$.medical_metrics_v2.primary_issue_type",
                    "baseline": {"type": "string", "value": "凸起样"},
                    "merged": {"type": "string", "value": "凹陷样"},
                },
            ],
        },
        {
            "merged_path": "simulated_oss/report/id-acne_v2/acne/acne_summary.json",
            "differences": [{"key": "$.artifact_policy", "baseline": None, "merged": {"type": "string", "value": "formal-fast"}}],
        },
    ]
    assert json_difference_policy(rows) == ([], True)


def test_doctor_signature_allows_only_frozen_legacy_group_title_omissions(tmp_path: Path) -> None:
    approved = tmp_path / "approved.docx"
    generated = tmp_path / "generated.docx"
    _write_docx(approved, "红褐混合印记", ())
    _write_docx(generated, "", ())
    ignored = frozenset({"", "红褐混合印记"})
    assert docx_content_signature(approved, ignored_visible=ignored) == (
        docx_content_signature(generated, ignored_visible=ignored)
    )


def test_formal_subject_must_equal_the_sample_alias() -> None:
    assert _subject_identity_passed(
        {"受检者信息": {"姓名或编号": "clinic28-25"}}, "clinic28-25"
    )
    assert not _subject_identity_passed(
        {"受检者信息": {"姓名或编号": "AISIA-FINAL-ACCEPTANCE"}},
        "clinic28-25",
    )
