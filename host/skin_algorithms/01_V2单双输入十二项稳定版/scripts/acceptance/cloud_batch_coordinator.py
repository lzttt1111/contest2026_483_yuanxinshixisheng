#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["pillow==12.3.0", "pydantic==2.13.4"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly:
#      uv run scripts/acceptance/cloud_batch_coordinator.py --help
# 3. Or make executable and run:
#      chmod +x scripts/acceptance/cloud_batch_coordinator.py && ./scripts/acceptance/cloud_batch_coordinator.py --help
# ─────────────────

"""Start three source services once and aggregate three ordered cloud bundles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from pydantic import JsonValue, TypeAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.acceptance.acceptance_io import load_fixed_inputs
from scripts.acceptance.cloud_batch_status import build_service_records
from scripts.acceptance.cloud_target_output import temporary_service_work


Profile = Literal["baseline", "merged"]
JSON_ADAPTER = TypeAdapter(JsonValue)
SERVICES = ("dermavision", "acne", "wrinkle")


@dataclass(frozen=True, slots=True)
class Child:
    service: str
    process: subprocess.Popen[str]
    log_handle: IO[str]


@dataclass(frozen=True, slots=True)
class CloudBatchError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "merged"), required=True)
    return parser.parse_args()


def _write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, JsonValue]:
    value = JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CloudBatchError(detail=f"expected JSON object: {path}")
    return value


def _start_children(args: argparse.Namespace, work_root: Path) -> list[Child]:
    """Run each source service to completion before starting the next."""

    helper = args.tool_root / "scripts" / "acceptance" / "acceptance_cloud_service.py"
    children: list[Child] = []
    for service in SERVICES:
        with temporary_service_work(args.project_root, args.evidence, service) as service_work:
            log_path = args.evidence / "logs" / f"{service}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            command = (
                str(args.python), "-B", str(helper), "--project-root",
                str(args.project_root), "--tool-root", str(args.tool_root),
                "--inputs-manifest", str(args.inputs_manifest), "--output",
                str(work_root), "--service", service, "--profile", args.profile,
            )
            process = subprocess.Popen(
                command, cwd=service_work, stdout=log_handle,
                stderr=subprocess.STDOUT, text=True,
            )
            process.wait()
            log_handle.close()
        children.append(Child(service=service, process=process, log_handle=log_handle))
        if process.returncode != 0:
            break
    return children


def _aggregate_case(
    case_id: str,
    input_name: str,
    work_root: Path,
    result_root: Path,
    profile: Profile,
    service_records: dict[str, JsonValue],
) -> tuple[int, int, bool, list[int]]:
    services = dict(service_records)
    tasks: dict[str, JsonValue] = {}
    complete = True
    for service in SERVICES:
        response_path = work_root / "responses" / service / f"{case_id}.json"
        if not response_path.is_file():
            complete = False
            continue
        response = _read_json(response_path)
        services[service] = response
        service_tasks = response.get("tasks")
        if isinstance(service_tasks, dict):
            tasks.update(service_tasks)
    task_count = len(tasks)
    projected_count = task_count + (1 if "purple" in tasks else 0)
    expected_tasks = 8 if profile == "baseline" else 11
    expected_results = 9 if profile == "baseline" else 12
    successful = complete and task_count == expected_tasks and projected_count == expected_results
    for task in tasks.values():
        task_map = task if isinstance(task, dict) else {}
        response = task_map.get("response")
        response_map = response if isinstance(response, dict) else {}
        successful = successful and response_map.get("status") == "success"
    sample_root = result_root / case_id
    sample_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        sample_root / "cloud_response_bundle.json",
        {
            "schema": "aisia_cloud_batch_bundle_v1",
            "profile": profile,
            "case_id": case_id,
            "input": input_name,
            "services": services,
            "tasks": tasks,
            "task_count": task_count,
            "projected_result_count": projected_count,
            "status": "success" if successful else "failed",
        },
    )
    service_pids = sorted(
        {
            int(task["service_pid"])
            for task in tasks.values()
            if isinstance(task, dict) and isinstance(task.get("service_pid"), int)
        }
    )
    return task_count, projected_count, successful, service_pids


def main() -> int:
    args = parse_args()
    inputs = load_fixed_inputs(args.inputs_manifest.resolve())
    results = args.results.resolve()
    evidence = args.evidence.resolve()
    work_root = evidence / "cloud_work"
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        evidence / "cloud_batch_manifest.json",
        {
            "schema": "aisia_cloud_batch_manifest_v1",
            "profile": args.profile,
            "ordered_cases": [asset.case_id for asset in inputs],
            "service_process_starts_expected": 3,
            "pipeline_initializations_expected": 3,
            "service_execution_order": list(SERVICES),
            "service_inference_overlap_allowed": False,
            "service_work_directories": {service: f"runtime_work/{service}" for service in SERVICES},
        },
    )
    started = time.perf_counter()
    children = _start_children(args, work_root)
    return_codes = {
        child.service: int(child.process.returncode)
        for child in children
        if child.process.returncode is not None
    }
    summaries = {
        service: _read_json(work_root / "service_summaries" / f"{service}.json")
        for service in SERVICES
        if (work_root / "service_summaries" / f"{service}.json").is_file()
    }
    service_records = build_service_records(SERVICES, return_codes, summaries)
    simulated_oss = work_root / "simulated_oss"
    if simulated_oss.is_dir():
        shutil.move(str(simulated_oss), str(results / "simulated_oss"))
    bundles: list[JsonValue] = []
    observed_task_pids: set[int] = set()
    success = len(summaries) == 3 and all(code == 0 for code in return_codes.values())
    for asset in inputs:
        task_count, projected_count, sample_success, service_pids = _aggregate_case(
            asset.case_id,
            asset.path.name,
            work_root,
            results,
            args.profile,
            service_records,
        )
        bundles.append(
            {
                "case_id": asset.case_id,
                "task_count": task_count,
                "projected_result_count": projected_count,
                "status": "success" if sample_success else "failed",
                "service_pids": service_pids,
            }
        )
        observed_task_pids.update(service_pids)
        success = success and sample_success
    process_starts = sum(int(item.get("process_starts", 0)) for item in summaries.values())
    pipeline_initializations = sum(
        int(item.get("pipeline_initializations", 0)) for item in summaries.values()
    )
    summary_pids = {
        int(item["pid"])
        for item in summaries.values()
        if isinstance(item.get("pid"), int)
    }
    pid_reuse_consistent = (
        len(summary_pids) == 3 and observed_task_pids == summary_pids
    )
    success = (
        success
        and process_starts == 3
        and pipeline_initializations == 3
        and pid_reuse_consistent
    )
    _write_json(
        evidence / "cloud_batch_summary.json",
        {
            "schema": "aisia_cloud_batch_summary_v1",
            "status": "success" if success else "failed",
            "wall_seconds": round(time.perf_counter() - started, 6),
            "service_process_starts": process_starts,
            "pipeline_initializations": pipeline_initializations,
            "service_pids": sorted(summary_pids),
            "pid_reuse_consistent": pid_reuse_consistent,
            "service_execution_order": list(SERVICES),
            "service_inference_overlap_allowed": False,
            "return_codes": return_codes,
            "services": service_records,
            "bundles": bundles,
        },
    )
    if success:
        shutil.rmtree(work_root / "responses", ignore_errors=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
