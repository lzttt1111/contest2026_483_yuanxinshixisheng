from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.acceptance import cloud_batch_coordinator
from scripts.acceptance.cloud_batch_status import build_service_records
from scripts.acceptance.cloud_target_output import CloudWorkError
from scripts.acceptance.acceptance_cloud_service import (
    CloudServiceError,
    _definitions,
    _prepare_local_output,
)


SHARED_PYTHON = Path(__file__).resolve().parents[3] / ".venv/bin/python"


class _FakeProcess:
    next_pid = 100

    def __init__(
        self,
        service: str,
        events: list[tuple[str, str]],
        desired_returncode: int = 0,
    ) -> None:
        self.service = service
        self.events = events
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.returncode: int | None = None
        self.desired_returncode = desired_returncode
        events.append(("start", service))

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.events.append(("wait", self.service))
        self.returncode = self.desired_returncode
        return self.desired_returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15


def test_cloud_services_start_and_finish_in_strict_serial_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    events: list[tuple[str, str]] = []
    working_directories: list[Path] = []
    args = argparse.Namespace(
        tool_root=tmp_path,
        evidence=tmp_path / "evidence",
        python=Path("python"),
        project_root=tmp_path,
        inputs_manifest=tmp_path / "manifest.json",
        profile="merged",
    )

    def fake_popen(command, **_kwargs):
        service = command[command.index("--service") + 1]
        working_directories.append(Path(_kwargs["cwd"]))
        target_output = args.project_root / "output"
        assert target_output.is_symlink()
        (target_output / f"{service}.txt").write_text(service, encoding="utf-8")
        return _FakeProcess(service, events)

    monkeypatch.setattr(cloud_batch_coordinator.subprocess, "Popen", fake_popen)

    # When
    children = cloud_batch_coordinator._start_children(args, tmp_path / "work")
    for child in children:
        child.log_handle.close()

    # Then
    assert events == [
        ("start", "dermavision"),
        ("wait", "dermavision"),
        ("start", "acne"),
        ("wait", "acne"),
        ("start", "wrinkle"),
        ("wait", "wrinkle"),
    ]
    assert working_directories == [
        tmp_path / "evidence/runtime_work/dermavision",
        tmp_path / "evidence/runtime_work/acne",
        tmp_path / "evidence/runtime_work/wrinkle",
    ]
    assert not (args.project_root / "output").exists()
    assert all(
        (directory / "worker_output" / f"{service}.txt").is_file()
        for directory, service in zip(
            working_directories,
            ("dermavision", "acne", "wrinkle"),
        )
    )


@pytest.mark.parametrize("profile", ("baseline", "merged"))
def test_cloud_helper_comes_from_harness_and_worker_from_target(profile: str) -> None:
    # Given
    tool_root = Path(__file__).resolve().parents[1]
    project_root = tool_root if profile == "merged" else tool_root.parent / "01_dev_baseline"
    code = (
        "import json; from pathlib import Path; "
        "from scripts.acceptance.acceptance_cloud_service import _load_runtime_modules; "
        f"modules=_load_runtime_modules(Path({str(project_root)!r}), 'dermavision'); "
        "print(json.dumps({'helper':modules.helper.__file__,"
        "'versions':modules.simulation_versions.__file__,"
        "'contracts':modules.contracts.__file__,'worker':modules.worker.__file__}))"
    )

    # When
    result = subprocess.run(
        (str(SHARED_PYTHON), "-B", "-c", code),
        check=True,
        capture_output=True,
        text=True,
        cwd=project_root,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": os.pathsep.join((str(project_root), str(tool_root))),
        },
    )
    provenance = json.loads(result.stdout)

    # Then
    assert Path(provenance["helper"]).is_relative_to(tool_root)
    assert Path(provenance["versions"]).is_relative_to(tool_root)
    assert Path(provenance["contracts"]).is_relative_to(tool_root)
    assert Path(provenance["worker"]).is_relative_to(project_root)


def test_cloud_service_help_exits_without_loading_models() -> None:
    # Given
    tool_root = Path(__file__).resolve().parents[1]
    script = tool_root / "scripts/acceptance/acceptance_cloud_service.py"

    # When
    result = subprocess.run(
        (str(SHARED_PYTHON), "-B", str(script), "--help"),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )

    # Then
    assert "--service" in result.stdout


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("baseline", ("redness", "spots", "brown", "texture", "pores", "purple")),
        (
            "merged",
            (
                "redness", "spots", "brown", "texture", "pores", "purple",
                "surface_gloss", "vascular", "contour_firmness",
            ),
        ),
    ),
)
def test_dermavision_profile_projects_exact_requested_algorithms(
    profile: str,
    expected: tuple[str, ...],
) -> None:
    # Given
    helper = SimpleNamespace(DERMAVISION_ALGORITHMS=(
        "redness", "spots", "brown", "texture", "pores", "purple",
        "surface_gloss", "vascular", "contour_firmness",
    ))

    # When
    definitions = _definitions("dermavision", profile, helper, SimpleNamespace())

    # Then
    assert tuple(row[0] for row in definitions) == expected
    assert all(row[1] == row[0] for row in definitions)
    assert all(row[2] == "dermavision.analyze_image" for row in definitions)
    assert all(row[4] == row[0] for row in definitions)


@pytest.mark.parametrize(
    "algorithms",
    (
        ("spots", "brown", "texture", "pores", "purple"),
        (
            "redness", "spots", "brown", "texture", "pores", "purple",
            "surface_gloss", "vascular", "contour_firmness", "internal_extra",
        ),
    ),
)
def test_dermavision_profile_rejects_missing_or_extra_helper_contract(
    algorithms: tuple[str, ...],
) -> None:
    helper = SimpleNamespace(DERMAVISION_ALGORITHMS=algorithms)
    with pytest.raises(CloudServiceError, match="target"):
        _definitions("dermavision", "baseline", helper, SimpleNamespace())


def test_cloud_services_stop_after_first_nonzero_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    events: list[tuple[str, str]] = []
    args = argparse.Namespace(
        tool_root=tmp_path,
        evidence=tmp_path / "evidence",
        python=Path("python"),
        project_root=tmp_path,
        inputs_manifest=tmp_path / "manifest.json",
        profile="baseline",
    )

    def fake_popen(command, **_kwargs):
        service = command[command.index("--service") + 1]
        return _FakeProcess(service, events, desired_returncode=1)

    monkeypatch.setattr(cloud_batch_coordinator.subprocess, "Popen", fake_popen)

    # When
    children = cloud_batch_coordinator._start_children(args, tmp_path / "work")

    # Then
    assert [child.service for child in children] == ["dermavision"]
    assert events == [("start", "dermavision"), ("wait", "dermavision")]


def test_cloud_failure_evidence_marks_remaining_services_not_started(
    tmp_path: Path,
) -> None:
    # Given
    records = build_service_records(
        cloud_batch_coordinator.SERVICES,
        {"dermavision": 1},
        {},
    )

    # When
    cloud_batch_coordinator._aggregate_case(
        "clinic28-25",
        "01_clinic28-25_RGB_M.jpg",
        tmp_path / "work",
        tmp_path / "results",
        "baseline",
        records,
    )
    bundle = json.loads(
        (tmp_path / "results/clinic28-25/cloud_response_bundle.json").read_text()
    )

    # Then
    assert records["dermavision"]["status"] == "failed"
    assert records["acne"]["status"] == "not_started"
    assert records["wrinkle"]["status"] == "not_started"
    assert bundle["services"] == records
    assert bundle["status"] == "failed"


def test_cloud_service_rejects_stale_internal_work_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    stale = tmp_path / "evidence/runtime_work/dermavision/stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    args = argparse.Namespace(
        tool_root=tmp_path,
        evidence=tmp_path / "evidence",
        python=Path("python"),
        project_root=tmp_path / "target",
        inputs_manifest=tmp_path / "manifest.json",
        profile="baseline",
    )
    monkeypatch.setattr(
        cloud_batch_coordinator.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("stale work must fail before process"),
    )

    # When / Then
    with pytest.raises(CloudWorkError, match="not empty"):
        cloud_batch_coordinator._start_children(args, tmp_path / "work")


def test_cloud_worker_output_environment_is_internal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    monkeypatch.setenv("WORKER_OUTPUT_DIR", "original")
    monkeypatch.setenv("WRINKLE_OUTPUT_DIR", "original")

    # When
    output = _prepare_local_output(tmp_path)

    # Then
    assert output == tmp_path / "worker_output"
    assert os.environ["WORKER_OUTPUT_DIR"] == str(output)
    assert os.environ["WRINKLE_OUTPUT_DIR"] == str(output)
