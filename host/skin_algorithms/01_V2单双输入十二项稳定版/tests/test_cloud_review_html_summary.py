import json
from pathlib import Path

import cloud.review_api as review_api
from cloud.review_sanitize import sanitize_review_bundle
from cloud.simulate_cloud_request import _bundle_counts, _write_html


def test_cloud_review_summary_uses_bundle_task_and_result_counts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "face.jpg"
    input_path.write_bytes(b"fixture")
    _write_html(
        input_path,
        tmp_path,
        {"tasks": {}, "task_count": 8, "projected_result_count": 9},
        1.25,
    )
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "单RGB9项云端返回模拟" in html
    assert "覆盖 8 个独立云端任务、9 项检测" in html


def test_cloud_review_health_never_exposes_absolute_root() -> None:
    result = review_api.health()

    assert "/" not in result["review_root"]


def test_safe_single_algorithm_api_does_not_revalidate_internal_worker_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_path = tmp_path / "cloud_response_bundle.json"
    bundle_path.write_text(
        json.dumps({
            "tasks": {
                "wrinkle": {
                    "response": {
                        "record_id": "fixture",
                        "status": "success",
                        "schema_version": "1",
                        "meta_data": {"name": "wrinkle", "version": "1"},
                        "raw_result": {
                            "overlay": "result.jpg",
                            "stage1_candidates": "debug.jpg",
                            "metrics": {},
                        },
                        "debug_info": {
                            "execution_mode": "balanced",
                            "output_dir": "/home/private/runtime",
                        },
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(review_api, "BUNDLE_PATH", bundle_path)

    route = next(
        value
        for value in review_api.app.routes
        if getattr(value, "path", None) == "/api/results/wrinkle"
    )
    assert route.response_model is None
    document = route.endpoint()
    assert set(document) == {
        "record_id",
        "status",
        "schema_version",
        "meta_data",
        "raw_result",
        "debug_info",
    }
    assert "stage1_candidates" not in document["raw_result"]
    assert "/home/" not in json.dumps(document, ensure_ascii=False)


def test_cloud_bundle_count_rule_projects_purple_as_two_results() -> None:
    tasks = {"redness": {}, "purple": {}, "wrinkle": {}}

    task_count, projected_result_count = _bundle_counts(tasks)

    assert task_count == 3
    assert projected_result_count == 4


def test_cloud_review_html_hides_debug_media_and_absolute_paths(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "face.jpg"
    input_path.write_bytes(b"fixture")
    public = tmp_path / "simulated_oss/result.jpg"
    public.parent.mkdir()
    public.write_bytes(b"public")
    debug = tmp_path / "simulated_oss/debug.jpg"
    debug.write_bytes(b"debug")
    bundle = {
        "input": "/home/private/face.jpg",
        "tasks": {
            "wrinkle": {
                "queue": "consumer_wrinkle",
                "task": "dermavision.analyze_image",
                "elapsed_seconds": 1.0,
                "pydantic_validation": "passed",
                "response": {
                    "record_id": "fixture",
                    "status": "success",
                    "schema_version": "1",
                    "meta_data": {"name": "wrinkle", "version": "1"},
                    "raw_result": {
                        "overlay": "result.jpg",
                        "stage1_candidates": "debug.jpg",
                        "metrics": {},
                    },
                    "debug_info": {
                        "display_result": {
                            "stage1_candidates": {"oss_key": "debug.jpg"}
                        },
                        "output_dir": "/home/private/runtime",
                        "execution_mode": "balanced",
                    },
                },
            }
        },
        "task_count": 1,
        "projected_result_count": 1,
    }
    bundle["services"] = {
        "wrinkle": {
            "input": "/home/private/face.jpg",
            "simulated_oss_root": "/tmp/private-oss",
            "tasks": bundle["tasks"],
        }
    }

    _write_html(input_path, tmp_path, bundle, 1.0)

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "result.jpg" in html
    assert "stage1_candidates" not in html
    assert "debug.jpg" not in html
    assert "/home/private" not in html
    assert "display_result" not in html

    projected = sanitize_review_bundle(bundle)
    response = projected["tasks"]["wrinkle"]["response"]
    assert set(response) == {
        "record_id",
        "status",
        "schema_version",
        "meta_data",
        "raw_result",
        "debug_info",
    }
    assert "stage1_candidates" not in response["raw_result"]
    assert response["debug_info"] == {"execution_mode": "balanced"}
    assert projected["input"] == "face.jpg"
    service = projected["services"]["wrinkle"]
    assert service["input"] == "face.jpg"
    assert "simulated_oss_root" not in service
    nested = service["tasks"]["wrinkle"]["response"]
    assert "stage1_candidates" not in nested["raw_result"]
    assert "/home/" not in str(projected)
    assert "/tmp/" not in str(projected)
