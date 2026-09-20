"""Summary worker startup routing: CPU-only app, launch route, target units."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = PROJECT_ROOT / "deploy"

_LIGHT_IMPORT_SCRIPT = """
import sys
import src.summary.worker  # noqa: F401
heavy = sorted({
    "." in m and m.split(".")[0] or m
    for m in sys.modules
    if m.split(".")[0] in {"torch", "mediapipe", "ultralytics", "cv2", "docx"}
    or m.startswith("src.pipeline")
    or m == "src.worker"
})
print("HEAVY=" + repr(heavy))
"""

_PRINT_COMMAND_SCRIPT = """
import contextlib
import io
import json
import sys

sys.argv = ["run_cloud_worker.py", "--target", "summary", "--print-command"]
import run_cloud_worker

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    code = run_cloud_worker.main()
data = json.loads(buffer.getvalue())
heavy = sorted({
    m.split(".")[0]
    for m in sys.modules
    if m.split(".")[0] in {"torch", "mediapipe", "ultralytics", "cv2", "docx"}
    or m.startswith("src.pipeline")
    or m == "src.worker"
})
sys.stderr.write(json.dumps({"code": code, "data": data, "heavy": heavy}))
"""


def _run_embedded(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )


def test_import_summary_worker_is_lightweight() -> None:
    completed = _run_embedded(_LIGHT_IMPORT_SCRIPT)
    assert "HEAVY=[]" in completed.stdout, completed.stdout


def test_print_command_resolves_cpu_app_without_heavy_imports() -> None:
    completed = _run_embedded(_PRINT_COMMAND_SCRIPT)
    payload = json.loads(completed.stderr)
    assert payload["code"] == 0
    assert payload["heavy"] == []
    data = payload["data"]
    assert data["target"] == "summary"
    assert data["capture_profile"] == "institution"
    assert data["app"] == "src.summary.worker:celery_app"
    assert data["queues"] == ["summary"]
    assert data["task"] == "dermavision.aggregate_summary"
    assert data["concurrency"] == 2
    assert data["results"] == []


def test_run_cloud_worker_resolves_summary_in_process() -> None:
    from run_cloud_worker import resolve_launch, validate_layout

    launch = resolve_launch("summary")
    validate_layout(launch)
    assert launch.app == "src.summary.worker:celery_app"
    assert launch.queues == ("summary",)
    assert launch.task == "dermavision.aggregate_summary"
    assert launch.concurrency == 2


def test_registry_summary_entry_matches_ssot() -> None:
    import src.summary as summary_pkg
    from src.worker import NINE_ANALYSIS_ITEMS, WORKER_TARGETS

    entry = WORKER_TARGETS["summary"]
    assert entry["app"] == summary_pkg.SUMMARY_APP
    assert entry["queue"] == summary_pkg.SUMMARY_QUEUE
    assert entry["task"] == summary_pkg.SUMMARY_TASK
    assert entry["concurrency"] == summary_pkg.SUMMARY_CONCURRENCY
    assert entry["results"] == ()
    # The CPU aggregation target must not change the twelve-item result list.
    assert len(NINE_ANALYSIS_ITEMS) == 12


def _unit_lines(name: str) -> set[str]:
    return {
        line.strip()
        for line in (DEPLOY_ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.startswith(("Requires=", "After="))
    }


def test_default_target_includes_summary_others_do_not() -> None:
    default_lines = _unit_lines("dermavision-workers.target")
    assert "Requires=dermavision@summary.service" in default_lines
    assert "After=dermavision@summary.service" in default_lines
    for name in ("dermavision-workers-v1.target", "dermavision-consumer-workers.target"):
        lines = _unit_lines(name)
        assert "Requires=dermavision@summary.service" not in lines
        assert "After=dermavision@summary.service" not in lines


def test_summary_service_uses_shared_template() -> None:
    service = (DEPLOY_ROOT / "dermavision@.service").read_text(encoding="utf-8")
    assert "run_cloud_worker.py --target %i" in service
