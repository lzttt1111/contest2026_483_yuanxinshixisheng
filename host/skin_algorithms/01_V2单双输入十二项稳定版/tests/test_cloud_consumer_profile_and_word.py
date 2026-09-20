from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from cloud import cloud_word_report, simulate_cloud_request, simulate_service_task


def test_cloud_word_remains_disabled_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "simulate_cloud_request.py",
        "--input", str(tmp_path / "input.jpg"),
        "--output", str(tmp_path / "output"),
    ])

    args = simulate_cloud_request.parse_args()

    assert args.generate_medical_report is False


def test_single_rgb_cloud_environment_selects_consumer_profile(
    tmp_path: Path,
) -> None:
    environment = simulate_cloud_request._service_environment(
        "dermavision",
        tmp_path,
    )

    assert environment["DERMAVISION_CAPTURE_PROFILE"] == "consumer"


def test_single_rgb_cloud_tasks_use_consumer_queue_names(monkeypatch) -> None:
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE", "consumer")
    queue_name = getattr(simulate_service_task, "_queue_name", None)

    assert callable(queue_name)
    assert queue_name("brown") == "consumer_brown"
    assert queue_name("acne_v2") == "consumer_acne_v2"
    assert queue_name("wrinkle") == "consumer_wrinkle"


def test_cloud_word_switch_reuses_consumer_twelve_report_entrypoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generator = getattr(simulate_cloud_request, "_generate_medical_reports", None)
    assert callable(generator)
    input_path = tmp_path / "clinic28-25_RGB_M.jpg"
    input_path.write_bytes(b"rgb")
    output_root = tmp_path / "cloud-output"
    output_root.mkdir()
    captured_command: list[str] = []

    def fake_run(command: list[str], **_kwargs) -> SimpleNamespace:
        captured_command.extend(command)
        run_root = Path(command[command.index("--output-dir") + 1])
        sample_root = run_root / "batch_000001" / "clinic28-25_RGB_M.jpg"
        sample_root.mkdir(parents=True)
        (sample_root / "AISIA_clinic28-25_用户精简版.docx").write_bytes(b"user")
        (sample_root / "AISIA_clinic28-25_医生详细版.docx").write_bytes(b"doctor")
        return SimpleNamespace(returncode=0, stdout="success", stderr="")

    monkeypatch.setattr(simulate_cloud_request.subprocess, "run", fake_run)

    reports = generator(input_path, output_root, "clinic28-25")

    assert "--capture-profile" in captured_command
    assert captured_command[captured_command.index("--capture-profile") + 1] == "consumer"
    assert "--generate-medical-report" in captured_command
    assert captured_command[captured_command.index("--report-subject-id") + 1] == (
        "clinic28-25"
    )
    assert {path.name for path in reports} == {
        "AISIA_clinic28-25_用户精简版.docx",
        "AISIA_clinic28-25_医生详细版.docx",
    }
    assert all(path.parent == output_root and path.is_file() for path in reports)
    assert not (output_root / ".medical-report-input").exists()
    assert not (output_root / ".medical-report-run").exists()


def test_cloud_word_default_subject_uses_original_source_not_staged_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "02_clinic28-09_RGB_M.jpg"
    input_path.write_bytes(b"rgb")
    output_root = tmp_path / "cloud-output"
    output_root.mkdir()
    captured_command: list[str] = []

    def fake_run(command: list[str], **_kwargs) -> SimpleNamespace:
        captured_command.extend(command)
        run_root = Path(command[command.index("--output-dir") + 1])
        sample = run_root / "batch_000001" / "report-hash.jpg"
        sample.mkdir(parents=True)
        (sample / "用户精简版.docx").write_bytes(b"user")
        (sample / "医生详细版.docx").write_bytes(b"doctor")
        return SimpleNamespace(returncode=0, stdout="success", stderr="")

    monkeypatch.setattr(cloud_word_report.subprocess, "run", fake_run)

    cloud_word_report.generate_cloud_review_reports(input_path, output_root)

    assert captured_command[captured_command.index("--report-subject-id") + 1] == (
        "clinic28-09"
    )


def test_cloud_word_input_name_binds_subject_and_image_identity(
    tmp_path: Path,
) -> None:
    image = tmp_path / "RGB_M.jpg"
    image.write_bytes(b"same-rgb")

    first = cloud_word_report._identity_bound_input_name(image, "subject-a")
    repeated = cloud_word_report._identity_bound_input_name(image, "subject-a")
    second = cloud_word_report._identity_bound_input_name(image, "subject-b")

    assert first == repeated
    assert first != second
    assert "subject-a" not in first
    assert first.endswith(".jpg")


def test_cloud_simulator_publishes_sanitized_bundle_and_keeps_internal_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"rgb")
    output = tmp_path / "output"
    response = {
        "record_id": "wrinkle-id",
        "status": "success",
        "schema_version": "1",
        "meta_data": {"name": "wrinkle", "version": "1"},
        "raw_result": {
            "stage1_candidates": "internal/stage1.jpg",
            "stage2_overlay": "public/result.jpg",
            "region_metrics": [],
        },
        "debug_info": {
            "execution_mode": "single_algorithm_consumer",
            "output_dir": "/tmp/private",
        },
    }
    bundle = {
        "services": {},
        "tasks": {
            "wrinkle": {
                "queue": "consumer_wrinkle",
                "task": "wrinkle.analyze_image",
                "arguments": ["id", "key", "prefix", ["wrinkle"]],
                "elapsed_seconds": 1.0,
                "pydantic_validation": "passed",
                "response": response,
            },
        },
    }
    monkeypatch.setattr(simulate_cloud_request, "_execute_services", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(simulate_cloud_request, "_collect_payload", lambda *_args: bundle)
    monkeypatch.setattr(simulate_cloud_request, "_write_html", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", [
        "simulate_cloud_request.py", "--input", str(input_path), "--output", str(output),
    ])

    assert simulate_cloud_request.main() == 0

    public = json.loads((output / "cloud_response_bundle.json").read_text(encoding="utf-8"))
    internal = json.loads((output / "logs/cloud_response_bundle.internal.json").read_text(encoding="utf-8"))
    assert "stage1_candidates" not in public["tasks"]["wrinkle"]["response"]["raw_result"]
    assert public["tasks"]["wrinkle"]["response"]["debug_info"] == {
        "execution_mode": "single_algorithm_consumer",
    }
    assert internal["tasks"]["wrinkle"]["response"]["raw_result"]["stage1_candidates"] == (
        "internal/stage1.jpg"
    )
