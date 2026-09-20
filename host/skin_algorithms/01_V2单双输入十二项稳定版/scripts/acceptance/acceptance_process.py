"""Managed process groups with retained logs and resource evidence."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from scripts.acceptance.acceptance_types import ProcessOutcome, ProcessRequest


def _group_resources(process_group_id: int) -> tuple[list[int], int, int]:
    pids: list[int] = []
    rss_kib = 0
    swap_kib = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat[stat.rfind(")") + 2 :].split()
            if int(remainder[2]) != process_group_id:
                continue
            status = (entry / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        pid = int(entry.name)
        pids.append(pid)
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_kib += int(line.split()[1])
            elif line.startswith("VmSwap:"):
                swap_kib += int(line.split()[1])
    return sorted(pids), rss_kib, swap_kib


def _gpu_resources(group_pids: list[int]) -> tuple[list[int], int]:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [], 0
    if result.returncode != 0:
        return [], 0
    group_set = set(group_pids)
    gpu_pids: list[int] = []
    memory_mib = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            used = int(fields[1])
        except ValueError:
            continue
        if pid in group_set:
            gpu_pids.append(pid)
            memory_mib += used
    return sorted(gpu_pids), memory_mib


def _terminate_group(process: subprocess.Popen[str]) -> None:
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=3)
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=3)


def run_managed_process(request: ProcessRequest) -> ProcessOutcome:
    """Run one acceptance command and always reap its complete process group."""

    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    request.resource_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(request.environment)
    started = time.monotonic()
    stop_sampling = threading.Event()
    peak_rss_kib = 0
    peak_swap_kib = 0
    peak_gpu_memory_mib = 0
    with request.log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            request.command,
            cwd=request.cwd,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        process_group_id = os.getpgid(process.pid)

        def sample_resources() -> None:
            nonlocal peak_rss_kib, peak_swap_kib, peak_gpu_memory_mib
            with request.resource_path.open("w", encoding="utf-8") as resource_handle:
                while True:
                    pids, rss_kib, swap_kib = _group_resources(process_group_id)
                    gpu_pids, gpu_memory_mib = _gpu_resources(pids)
                    peak_rss_kib = max(peak_rss_kib, rss_kib)
                    peak_swap_kib = max(peak_swap_kib, swap_kib)
                    peak_gpu_memory_mib = max(peak_gpu_memory_mib, gpu_memory_mib)
                    resource_handle.write(
                        json.dumps(
                            {
                                "elapsed_seconds": round(time.monotonic() - started, 6),
                                "pids": pids,
                                "rss_kib": rss_kib,
                                "swap_kib": swap_kib,
                                "gpu_pids": gpu_pids,
                                "gpu_memory_mib": gpu_memory_mib,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    resource_handle.flush()
                    if stop_sampling.wait(request.sample_interval_seconds):
                        return

        sampler = threading.Thread(target=sample_resources, daemon=False)
        sampler.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(process)
            exit_code = process.wait(timeout=3)
        finally:
            _terminate_group(process)
            stop_sampling.set()
            sampler.join(timeout=max(1.0, request.sample_interval_seconds * 2))
    return ProcessOutcome(
        exit_code=exit_code,
        timed_out=timed_out,
        wall_seconds=round(time.monotonic() - started, 6),
        process_group_id=process_group_id,
        peak_rss_kib=peak_rss_kib,
        peak_swap_kib=peak_swap_kib,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
    )
