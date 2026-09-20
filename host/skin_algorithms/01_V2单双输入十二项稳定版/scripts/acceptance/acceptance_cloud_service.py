#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["pillow==12.3.0", "pydantic==2.13.4"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly:
#      uv run scripts/acceptance/acceptance_cloud_service.py --help
# 3. Or make executable and run:
#      chmod +x scripts/acceptance/acceptance_cloud_service.py && ./scripts/acceptance/acceptance_cloud_service.py --help
# ─────────────────

"""One persistent source-service process for all three cloud samples."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

from pydantic import JsonValue, TypeAdapter

TOOL_ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [entry for entry in sys.path if entry != str(TOOL_ROOT)]
sys.path.insert(0, str(TOOL_ROOT))

from scripts.acceptance.acceptance_io import load_fixed_inputs


JSON_ADAPTER = TypeAdapter(JsonValue)
Service = Literal["dermavision", "acne", "wrinkle"]
Profile = Literal["baseline", "merged"]
BASELINE_DERMAVISION_ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores", "purple",
)
MERGED_DERMAVISION_ALGORITHMS = (
    *BASELINE_DERMAVISION_ALGORITHMS,
    "surface_gloss", "vascular", "contour_firmness",
)


@dataclass(frozen=True, slots=True)
class CloudServiceError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class CloudRuntimeModules:
    helper: ModuleType
    simulation_versions: ModuleType
    contracts: ModuleType
    worker: ModuleType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", choices=("dermavision", "acne", "wrinkle"), required=True)
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


def _load_worker(service: Service) -> ModuleType:
    if service == "dermavision":
        return importlib.import_module("src.worker")
    if service == "acne":
        return importlib.import_module("src.acne.worker")
    return importlib.import_module("src.wrinkle.worker")


def _module_path(module: ModuleType) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        raise CloudServiceError(detail=f"module provenance unavailable: {module.__name__}")
    return Path(value).resolve()


def _load_runtime_modules(
    project_root: Path,
    service: Service,
) -> CloudRuntimeModules:
    helper = importlib.import_module("cloud.simulate_service_task")
    simulation_versions = importlib.import_module(
        helper.install_simulated_contract_versions.__module__
    )
    helper.install_simulated_contract_versions()
    contracts = importlib.import_module("cloud_contracts")
    if (
        not _module_path(helper).is_relative_to(TOOL_ROOT)
        or not _module_path(simulation_versions).is_relative_to(TOOL_ROOT)
        or not _module_path(contracts).is_relative_to(TOOL_ROOT)
    ):
        raise CloudServiceError(detail="cloud helper provenance mismatch")
    sys.path.insert(0, str(project_root))
    worker = _load_worker(service)
    if not _module_path(worker).is_relative_to(project_root):
        raise CloudServiceError(detail="cloud worker provenance mismatch")
    return CloudRuntimeModules(
        helper=helper,
        simulation_versions=simulation_versions,
        contracts=contracts,
        worker=worker,
    )


def _prepare_local_output(service_work: Path) -> Path:
    output = service_work / "worker_output"
    output.mkdir(exist_ok=True)
    os.environ["WORKER_OUTPUT_DIR"] = str(output)
    os.environ["WRINKLE_OUTPUT_DIR"] = str(output)
    return output


def _run(worker: ModuleType, entry: str, arguments: list[JsonValue]) -> dict[str, JsonValue]:
    task = getattr(worker, entry)
    value = task.run(*arguments)
    validated = JSON_ADAPTER.validate_python(value)
    if not isinstance(validated, dict):
        raise CloudServiceError(detail="worker response must be a JSON object")
    return validated


def _definitions(
    service: Service,
    profile: Profile,
    helper: ModuleType,
    worker: ModuleType,
) -> list[tuple[str, str, str, str, str]]:
    if service == "dermavision":
        available = tuple(str(item) for item in helper.DERMAVISION_ALGORITHMS)
        if (
            len(available) != len(set(available))
            or set(available) != set(MERGED_DERMAVISION_ALGORITHMS)
        ):
            raise CloudServiceError(
                detail="DermaVision target contract does not match harness"
            )
        algorithms = (
            BASELINE_DERMAVISION_ALGORITHMS
            if profile == "baseline"
            else MERGED_DERMAVISION_ALGORITHMS
        )
        return [
            (algorithm, algorithm, "dermavision.analyze_image", "analyze_image", algorithm)
            for algorithm in algorithms
        ]
    if service == "acne" and profile == "merged":
        if not hasattr(worker, "analyze_image_v2"):
            raise CloudServiceError(detail="merged acne_v2 entry is unavailable")
        return [("acne_v2", "acne_v2", "dermavision.analyze_image", "analyze_image_v2", "acne")]
    if service == "acne":
        return [("acne", "acne", "acne.analyze_image", "analyze_image", "acne")]
    return [("wrinkle", "wrinkle", "wrinkle.analyze_image", "analyze_image", "wrinkle")]


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    tool_root = args.tool_root.resolve()
    if tool_root != TOOL_ROOT:
        raise CloudServiceError(detail="tool root does not match acceptance script")
    inputs = load_fixed_inputs(args.inputs_manifest.resolve())
    local_output = _prepare_local_output(Path.cwd())
    modules = _load_runtime_modules(project_root, args.service)
    helper = modules.helper
    contracts = modules.contracts
    worker = modules.worker
    if args.service == "dermavision":
        worker.OUTPUT_ROOT = local_output
    definitions = _definitions(args.service, args.profile, helper, worker)
    output = args.output.resolve()
    oss_root = output / "simulated_oss"
    pipeline_initializations = 0
    completed_cases: list[str] = []
    status = "running"
    _write_json(
        output / "service_manifests" / f"{args.service}.json",
        {"service": args.service, "pid": os.getpid(), "process_starts": 1, "status": status},
    )
    try:
        for asset in inputs:
            input_bytes = asset.path.read_bytes()
            source_key = f"cloud-simulation/input/{asset.path.name}"
            worker.download_via_internal_api = lambda _key, data=input_bytes: data
            worker.upload_via_internal_api = (
                lambda local_path, report_id, algo_name, file_name: helper._copy_upload(
                    local_path, report_id, algo_name, file_name, oss_root
                )
            )
            tasks: dict[str, JsonValue] = {}
            sample_success = True
            for target, queue, task_name, entry, algorithm in definitions:
                record_id = f"visia2-{asset.case_id}-{target}"
                arguments: list[JsonValue] = [
                    record_id,
                    source_key,
                    "cloud-simulation/v1/",
                    [algorithm],
                ]
                pipeline_missing = getattr(worker, "_pipeline", None) is None
                started = time.perf_counter()
                response = _run(worker, entry, arguments)
                if pipeline_missing and getattr(worker, "_pipeline", None) is not None:
                    pipeline_initializations += 1
                validation = "baseline_aware_structural_passed"
                if response.get("status") == "success" and args.profile == "merged":
                    validator_name = target if target == "acne_v2" else algorithm
                    response = contracts.validate_worker_envelope(validator_name, response)
                    validation = "pydantic_passed"
                sample_success = sample_success and response.get("status") == "success"
                tasks[target] = {
                    "queue": queue,
                    "task": task_name,
                    "arguments": arguments,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "pydantic_validation": validation,
                    "response": response,
                    "service_pid": os.getpid(),
                }
            _write_json(
                output / "responses" / args.service / f"{asset.case_id}.json",
                {"service": args.service, "case_id": asset.case_id, "tasks": tasks},
            )
            completed_cases.append(asset.case_id)
            if not sample_success:
                status = "failed"
                break
        if status == "running":
            status = "success"
        return 0 if status == "success" else 1
    finally:
        pipeline = getattr(worker, "_pipeline", None)
        if pipeline is not None:
            pipeline.close()
            worker._pipeline = None
        _write_json(
            output / "service_summaries" / f"{args.service}.json",
            {
                "service": args.service,
                "pid": os.getpid(),
                "process_starts": 1,
                "pipeline_initializations": pipeline_initializations,
                "completed_cases": completed_cases,
                "status": status,
                "closed": True,
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
