from __future__ import annotations

import json

from scripts.acceptance.acceptance_types import JsonDict
from scripts.acceptance.deep_compare_contracts import compare_worker_bundles


def _success_task(name: str, raw_result: JsonDict, debug_info: JsonDict) -> JsonDict:
    return {
        "queue": name,
        "task": "dermavision.analyze_image",
        "arguments": [f"id-{name}", "cloud/input.jpg", "prefix/", [name]],
        "response": {
            "record_id": f"id-{name}",
            "status": "success",
            "schema_version": "1",
            "meta_data": {"name": name, "version": "1"},
            "raw_result": raw_result,
            "debug_info": debug_info,
        },
    }


def test_compare_worker_bundles_classifies_migration_and_added_three() -> None:
    # Given
    baseline_tasks = {
        name: _success_task(name, {"overlay": f"{name}.jpg"}, {"mode": "same"})
        for name in ("redness", "spots", "brown", "texture", "pores", "purple", "wrinkle")
    }
    baseline_tasks["acne"] = _success_task("acne", {"legacy": 1}, {"mode": "formal"})
    merged_tasks = json.loads(json.dumps(baseline_tasks))
    merged_tasks.pop("acne")
    acne_v2 = _success_task(
        "acne_v2",
        {
            "overlay": "acne.jpg",
            "metrics": {"疑似痤疮数量": {"总计": 1}},
            "quality_score": None,
            "quality_status": None,
            "quality_flags": [],
        },
        {"execution_mode": "formal_fast"},
    )
    acne_v2["queue"] = "acne_v2"
    merged_tasks["acne_v2"] = acne_v2
    regions = ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
    added_metrics = {
        "surface_gloss": {
            "油光面积占比": {region: 0.1 for region in regions},
            "油光区域数量": {region: 1 for region in regions},
        },
        "vascular": {
            "血管样结构数量": {region: 1 for region in regions},
            "血管样结构总长度": {region: 10.0 for region in regions},
        },
        "contour_firmness": {
            "中面部曲面连续性": 0.8,
            "下颌缘连续性": 0.7,
            "左右轮廓差异": 0.1,
        },
    }
    for name in ("surface_gloss", "vascular", "contour_firmness"):
        merged_tasks[name] = _success_task(
            name,
            {
                "overlay": f"{name}.jpg",
                "medical_report_csv_v2": f"{name}.csv",
                "metrics": added_metrics[name],
                "quality_score": 90.0,
                "quality_status": "PASS",
                "quality_flags": [],
            },
            {
                "report_csv": f"{name}/report.csv",
                "timing_seconds": {"pipeline": 1.0},
                "execution_mode": "single_algorithm",
            },
        )
    comparison = compare_worker_bundles(
        {"tasks": baseline_tasks}, {"tasks": merged_tasks}
    )

    # When / Then
    assert comparison["old_seven_zero_drift"] is False
    assert comparison["acne_v1_to_v2"]["baseline_target"] == "acne"
    assert comparison["acne_v1_to_v2"]["merged_target"] == "acne_v2"
    assert all(
        item["public_contract_passed"]
        for item in comparison["added_three"].values()
    )


def test_compare_worker_bundles_rejects_empty_old_seven_responses() -> None:
    # Given
    tasks = {
        name: {
            "queue": name,
            "task": "wrinkle.analyze_image" if name == "wrinkle" else "dermavision.analyze_image",
            "arguments": [f"id-{name}", "input.jpg", "prefix/", [name]],
            "response": {},
        }
        for name in ("redness", "spots", "brown", "texture", "pores", "purple", "wrinkle")
    }

    # When
    comparison = compare_worker_bundles({"tasks": tasks}, {"tasks": tasks})

    # Then
    assert comparison["old_seven_zero_drift"] is False


def test_compare_worker_bundles_rejects_malformed_added_three_contracts() -> None:
    # Given
    malformed = {
        "overlay": "result.jpg",
        "medical_report_csv_v2": "metrics.csv",
        "metrics": {},
        "quality_score": "not-a-number",
        "quality_status": [],
        "quality_flags": [],
        "internal_debug": True,
    }
    merged_tasks = {
        name: _success_task(name, malformed, {"execution_mode": "single_algorithm"})
        for name in ("surface_gloss", "vascular", "contour_firmness")
    }

    # When
    comparison = compare_worker_bundles({"tasks": {}}, {"tasks": merged_tasks})

    # Then
    assert all(
        item["public_contract_passed"] is False
        for item in comparison["added_three"].values()
    )
