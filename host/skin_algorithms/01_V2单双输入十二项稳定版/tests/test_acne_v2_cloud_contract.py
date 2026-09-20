from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

import run_cloud_worker
import src.acne.worker as acne_worker
from cloud_contracts import public_metric_documentation_rows, validate_worker_envelope


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _acne_v2_payload() -> dict:
    return {
        "record_id": "contract-acne-v2",
        "status": "success",
        "schema_version": "2",
        "meta_data": {"name": "acne", "version": "2"},
        "raw_result": {
            "overlay": "report/contract-acne-v2/acne/02_痤疮圈选结果_原图.jpg",
            "metrics": {
                "疑似痤疮数量": {
                    "总计": 13,
                    "额头": 1,
                    "左脸颊": 2,
                    "右脸颊": 3,
                    "鼻部": 4,
                    "下巴": 3,
                },
                "痤疮严重程度等级": {
                    "等级": 2,
                    "注释": "本次为2级（偏轻）。",
                },
            },
            "quality_score": None,
            "quality_status": None,
            "quality_flags": [],
        },
        "debug_info": {
            "execution_mode": "formal_fast",
            "elapsed_seconds": 2.5,
        },
    }


def test_acne_v2_validates_exact_six_field_envelope_without_mutation() -> None:
    # Given
    payload = _acne_v2_payload()
    before = deepcopy(payload)

    # When
    returned = validate_worker_envelope("acne_v2", payload)

    # Then
    assert returned is payload
    assert payload == before
    assert set(payload) == {
        "record_id",
        "status",
        "schema_version",
        "meta_data",
        "raw_result",
        "debug_info",
    }
    assert set(payload["raw_result"]) == {
        "overlay",
        "metrics",
        "quality_score",
        "quality_status",
        "quality_flags",
    }


@pytest.mark.parametrize(
    ("mutation",),
    [
        (lambda payload: payload["raw_result"].update({"acne_summary": "legacy.json"}),),
        (lambda payload: payload["meta_data"].update({"version": "1"}),),
        (lambda payload: payload.update({"schema_version": "1"}),),
        (
            lambda payload: payload["raw_result"]["metrics"][
                "疑似痤疮数量"
            ].pop("下巴"),
        ),
    ],
)
def test_acne_v2_rejects_contract_drift(mutation) -> None:
    # Given
    payload = _acne_v2_payload()
    mutation(payload)

    # When / Then
    with pytest.raises(ValidationError):
        validate_worker_envelope("acne_v2", payload)


def test_acne_v2_target_is_additive_and_keeps_v1_launch_unchanged() -> None:
    # Given
    expected_v1 = (
        "src.acne.worker:celery_app",
        ("acne",),
        "acne.analyze_image",
    )

    # When
    v1 = run_cloud_worker.resolve_launch("acne")
    v2 = run_cloud_worker.resolve_launch("acne_v2")
    run_cloud_worker.validate_layout(v2)

    # Then
    assert (v1.app, v1.queues, v1.task) == expected_v1
    assert (v2.app, v2.queues, v2.task, v2.concurrency) == (
        "src.acne.worker:celery_app",
        ("acne_v2",),
        "dermavision.analyze_image",
        1,
    )


def test_acne_v2_manifest_locks_queue_envelope_and_raw_keys() -> None:
    # Given
    manifest = json.loads(
        (PROJECT_ROOT / "cloud/contracts/internal_dev_contracts.json").read_text(
            encoding="utf-8"
        )
    )

    # When
    extension = manifest["source_repositories"]["acne"]["v2_extension"]

    # Then
    assert extension == {
        "target": "acne_v2",
        "task": "dermavision.analyze_image",
        "queue": "acne_v2",
        "arguments": ["task_id", "oss_key", "oss_result_prefix", "algorithms"],
        "success_envelope": [
            "record_id",
            "status",
            "schema_version",
            "meta_data",
            "raw_result",
            "debug_info",
        ],
        "meta_data": {"name": "acne", "version": "2"},
        "raw_result_keys": [
            "overlay",
            "metrics",
            "quality_score",
            "quality_status",
            "quality_flags",
            "scoring_input",
        ],
        "metrics": {
            "疑似痤疮数量": ["总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"],
            "痤疮严重程度等级": ["等级", "注释"],
        },
        "v1_contract_unchanged": True,
    }


@pytest.mark.parametrize("capture_profile", ("institution", "consumer"))
def test_acne_v2_worker_uses_formal_fast_and_projects_only_public_fields(
    capture_profile: str,
    monkeypatch,
) -> None:
    # Given
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE", capture_profile)
    received: list[tuple] = []
    legacy_response = {
        "record_id": "worker-acne-v2",
        "status": "success",
        "schema_version": "1",
        "meta_data": {"name": "acne", "version": "1"},
        "raw_result": {
            "acne_count": 18,
            "region_counts": [
                {"region": "forehead", "count": 1},
                {"region": "subject_left_cheek", "count": 2},
                {"region": "subject_right_cheek", "count": 3},
                {"region": "nose", "count": 4},
                {"region": "chin", "count": 3},
                {"region": "subject_left_jaw", "count": 5},
            ],
            "量化结果": {
                "疑似痤疮圈选数量": 18,
                "痤疮严重程度等级": {
                    "等级": 2,
                    "注释": "本次为2级（偏轻）。",
                },
            },
            "legacy_only": "must-not-leak",
        },
        "_scoring_input": {
            "schema_version": "scoring_input_v1",
            "algorithm_name": "acne",
            "detection_schema_version": "2",
            "detection_impl_version": "2",
            "report_id": "worker-acne-v2",
            "detection_attempt_id": "worker-acne-v2",
            "source_image_sha256": "c" * 64,
            "capture_profile": "institution",
            "input_route": "institution_queue",
            "evidence_status": "present",
            "evidence_field_list_sha256": "d" * 64,
            "quality": {"algorithm_quality_status": None, "algorithm_quality_flags": []},
            "evidence": {"acne": {"metrics": {"detection": {"count": 18}}}},
            "missing_fields": [],
            "missing_reason": None,
        },
        "debug_info": {
            "display_result": {
                "overlay_original": "report/worker-acne-v2/acne/02_痤疮圈选结果_原图.jpg",
                "summary_json": "report/worker-acne-v2/acne/痤疮量化指标.json",
            },
            "algorithm_version": "1",
            "elapsed_seconds": 2.5,
        },
    }

    def fake_v1(*args):
        received.append(args)
        return legacy_response

    monkeypatch.setattr(acne_worker.analyze_image, "run", fake_v1)

    # When
    result = acne_worker.analyze_image_v2.run(
        "worker-acne-v2",
        "inputs/face.jpg",
        "ignored-prefix/",
        ["acne", "debug-artifacts"],
    )

    # Then
    assert received == [
        (
            "worker-acne-v2",
            "inputs/face.jpg",
            "ignored-prefix/",
            ["acne", "formal-fast"],
        )
    ]
    expected = _acne_v2_payload()
    expected["record_id"] = "worker-acne-v2"
    expected["raw_result"]["overlay"] = (
        "report/worker-acne-v2/acne/02_痤疮圈选结果_原图.jpg"
    )
    expected["raw_result"]["metrics"]["疑似痤疮数量"]["总计"] = 18
    expected["raw_result"]["scoring_input"] = legacy_response["_scoring_input"]
    assert result == expected


def test_acne_v2_worker_fails_when_required_region_is_unavailable(monkeypatch) -> None:
    # Given
    legacy_response = {
        "record_id": "missing-region",
        "status": "success",
        "schema_version": "1",
        "meta_data": {"name": "acne", "version": "1"},
        "raw_result": {
            "acne_count": 1,
            "region_counts": [{"region": "forehead", "count": 1}],
            "量化结果": {
                "疑似痤疮圈选数量": 1,
                "痤疮严重程度等级": {"等级": None, "注释": "未评级"},
            },
        },
        "debug_info": {
            "display_result": {"overlay_original": "report/missing/acne/overlay.jpg"},
            "algorithm_version": "1",
            "elapsed_seconds": 1.0,
        },
    }
    monkeypatch.setattr(acne_worker.analyze_image, "run", lambda *args: legacy_response)

    # When
    result = acne_worker.analyze_image_v2.run(
        "missing-region", "inputs/face.jpg", None, ["acne"]
    )

    # Then
    assert result["status"] == "failed"
    assert result["schema_version"] == "2"
    assert result["error_code"] == "v2_contract_projection_failed"


def test_acne_v2_task_registration_keeps_v1_task_and_queue() -> None:
    # Given / When / Then
    assert acne_worker.analyze_image.name == "acne.analyze_image"
    assert acne_worker.analyze_image_v2.name == "dermavision.analyze_image"
    assert set(acne_worker.celery_app.conf.task_queues) == {
        "acne",
        "acne_v2",
        "consumer_acne_v2",
    }


def test_acne_v2_metrics_publish_chinese_leaf_documentation() -> None:
    # Given
    raw_result = _acne_v2_payload()["raw_result"]

    # When
    rows = public_metric_documentation_rows("acne_v2", raw_result)

    # Then
    assert {row["json_path"] for row in rows} == {
        "raw_result.metrics.疑似痤疮数量.总计",
        "raw_result.metrics.疑似痤疮数量.额头",
        "raw_result.metrics.疑似痤疮数量.左脸颊",
        "raw_result.metrics.疑似痤疮数量.右脸颊",
        "raw_result.metrics.疑似痤疮数量.鼻部",
        "raw_result.metrics.疑似痤疮数量.下巴",
        "raw_result.metrics.痤疮严重程度等级.等级",
        "raw_result.metrics.痤疮严重程度等级.注释",
    }
    assert all(row["description"] for row in rows)
