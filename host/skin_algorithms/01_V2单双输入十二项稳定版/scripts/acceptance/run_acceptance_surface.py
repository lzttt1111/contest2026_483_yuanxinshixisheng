#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["pillow==12.3.0", "pydantic==2.13.4"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly:
#      uv run scripts/acceptance/run_acceptance_surface.py --help
# 3. Or make executable and run:
#      chmod +x scripts/acceptance/run_acceptance_surface.py && ./scripts/acceptance/run_acceptance_surface.py --help
# ─────────────────

"""Execute one branch/surface acceptance attempt."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from pydantic import JsonValue, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.acceptance.acceptance_io import (
    AcceptanceInputError,
    load_fixed_inputs,
    load_vendor_sidecar,
)
from scripts.acceptance.acceptance_process import run_managed_process
from scripts.acceptance.acceptance_types import (
    ProcessRequest,
    SurfaceAssessment,
)
from scripts.acceptance.formal_baseline import verify_formal_baseline
from scripts.acceptance.surface_assessment import (
    append_contract_error,
    assess_surface,
    build_cloud_assessment,
)
from scripts.acceptance.surface_baseline import prove_baseline
from scripts.acceptance.surface_commands import build_surface_command
from scripts.acceptance.surface_evidence import inspect_local_results
from scripts.acceptance.surface_local import (
    formal_runtime_environment,
    prepare_runtime_cache,
    temporary_venv_link,
)


def _write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sanitized_command(
    command: tuple[str, ...],
    replacements: dict[str, str],
) -> list[str]:
    sanitized: list[str] = []
    ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    for argument in command:
        value = argument
        for source, label in ordered:
            value = value.replace(source, label)
        sanitized.append(value)
    return sanitized


def _validated_python_path(candidate: Path, shared: Path) -> Path:
    """Normalize parent segments while preserving the venv Python symlink."""

    normalized = Path(os.path.abspath(candidate.expanduser()))
    expected = Path(os.path.abspath(shared.expanduser()))
    if normalized != expected or not normalized.is_file():
        raise AcceptanceInputError(detail="external Python is not the shared entry")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AISIA acceptance surface")
    parser.add_argument("--profile", choices=("baseline", "merged"), required=True)
    parser.add_argument("--surface", choices=("run", "cloud"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--vendor-manifest", type=Path, required=True)
    parser.add_argument("--formal-baseline-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool_root = Path(__file__).resolve().parents[2]
    project_root = args.project_root.resolve()
    python = _validated_python_path(
        args.python,
        tool_root.parents[1] / ".venv/bin/python",
    )
    inputs_manifest = args.inputs_manifest.resolve()
    output = args.output.resolve()
    if not project_root.is_dir():
        raise AcceptanceInputError(detail="project root or external Python unavailable")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AcceptanceInputError(detail=f"acceptance surface is not empty: {output}")
    evidence = output / "_evidence"
    results = output / "results"
    evidence.mkdir()
    results.mkdir()
    inputs = load_fixed_inputs(inputs_manifest)
    sidecar = load_vendor_sidecar(args.vendor_manifest.resolve())
    if args.profile == "baseline":
        prove_baseline(project_root)
    if args.profile == "merged" and args.surface == "run":
        if args.formal_baseline_root is None:
            raise AcceptanceInputError(detail="merged run requires --formal-baseline-root")
        verify_formal_baseline(project_root, args.formal_baseline_root.resolve())
    command = build_surface_command(
        args.profile,
        args.surface,
        project_root,
        tool_root,
        python,
        inputs_manifest,
        results,
        evidence,
    )
    runtime_cache = prepare_runtime_cache(evidence)
    environment = {
        **sidecar.environment,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(project_root), str(tool_root))),
        "MPLCONFIGDIR": str(runtime_cache.matplotlib),
        "XDG_CACHE_HOME": str(runtime_cache.xdg),
    }
    environment.update(formal_runtime_environment(
        args.profile == "merged" and args.surface == "run",
        args.formal_baseline_root.resolve()
        if args.formal_baseline_root is not None
        else None,
    ))
    _write_json(
        evidence / "surface_manifest.json",
        {
            "schema": "aisia_acceptance_surface_v1",
            "profile": args.profile,
            "surface": args.surface,
            "ordered_cases": [asset.case_id for asset in inputs],
            "input_sha256": {asset.case_id: asset.sha256 for asset in inputs},
            "vendor_module_sha256": sidecar.module_sha256,
            "runtime_cache": runtime_cache.relative_evidence(),
            "command": _sanitized_command(
                command,
                {
                    str(inputs_manifest): "<fixed-input-manifest>",
                    str(evidence): "<surface-evidence>",
                    str(results): "<surface-results>",
                    str(project_root): "<project-root>",
                    str(tool_root): "<tool-root>",
                    str(python): "<external-python>",
                },
            ),
        },
    )
    started_ns = time.time_ns()
    request = ProcessRequest(
        command=command,
        cwd=project_root,
        environment=environment,
        log_path=evidence / "surface.log",
        resource_path=evidence / "resources.jsonl",
        timeout_seconds=args.timeout_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
    )
    if args.surface == "run":
        with temporary_venv_link(project_root, python):
            outcome = run_managed_process(request)
    else:
        outcome = run_managed_process(request)
    expected_cases = tuple(asset.case_id for asset in inputs)
    inspection_error: str | None = None
    try:
        if args.surface == "run":
            inspection = inspect_local_results(
                profile=args.profile,
                result_root=results,
                expected_cases=expected_cases,
                started_ns=started_ns,
                require_word=args.profile == "merged",
            )
            assessment = SurfaceAssessment(
                exit_code=outcome.exit_code,
                started_ns=started_ns,
                expected_cases=expected_cases,
                discovered_cases=inspection.discovered_cases,
                oldest_required_output_ns=inspection.oldest_required_output_ns,
                task_count=0,
                projected_result_count=0,
                service_process_starts=0,
                pipeline_initializations=0,
                contract_errors=inspection.contract_errors,
            )
        else:
            assessment = build_cloud_assessment(
                evidence, results, expected_cases, started_ns, outcome.exit_code, args.profile
            )
    except (
        OSError,
        UnicodeError,
        ValidationError,
        csv.Error,
        ValueError,
        TypeError,
        AcceptanceInputError,
    ) as exc:
        inspection_error = f"inspection_error:{type(exc).__name__}"
        assessment = SurfaceAssessment(
            exit_code=outcome.exit_code,
            started_ns=started_ns,
            expected_cases=expected_cases,
            discovered_cases=(),
            oldest_required_output_ns=0,
            task_count=0,
            projected_result_count=0,
            service_process_starts=0,
            pipeline_initializations=0,
            contract_errors=(inspection_error,),
        )
    if args.profile == "baseline":
        try:
            prove_baseline(project_root)
        except AcceptanceInputError as exc:
            inspection_error = f"inspection_error:{type(exc).__name__}"
            assessment = append_contract_error(assessment, inspection_error)
    result = assess_surface(assessment)
    _write_json(
        evidence / "surface_result.json",
        {
            "schema": "aisia_acceptance_surface_result_v1",
            "status": result.status,
            "reason_codes": list(result.reason_codes),
            "contract_errors": list(assessment.contract_errors),
            "exit_code": outcome.exit_code,
            "timed_out": outcome.timed_out,
            "wall_seconds": outcome.wall_seconds,
            "peak_rss_kib": outcome.peak_rss_kib,
            "peak_swap_kib": outcome.peak_swap_kib,
            "peak_gpu_memory_mib": outcome.peak_gpu_memory_mib,
            "discovered_cases": list(assessment.discovered_cases),
            "task_count": assessment.task_count,
            "projected_result_count": assessment.projected_result_count,
            "service_process_starts": assessment.service_process_starts,
            "pipeline_initializations": assessment.pipeline_initializations,
            "inspection_error": inspection_error,
        },
    )
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
