from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path


_VALUE_OPTIONS = frozenset(("--slots", "--worker-id", "--runtime-dir"))


def _without_parent_options(arguments: tuple[str, ...]) -> tuple[str, ...]:
    kept: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        option = token.split("=", 1)[0]
        if option in _VALUE_OPTIONS:
            index += 1 if "=" in token else 2
            continue
        if token == "--allow-shared-gpu":
            index += 1
            continue
        kept.append(token)
        index += 1
    return tuple(kept)


def child_slot_commands(
    arguments: tuple[str, ...],
    *,
    python: Path,
    script: Path,
    worker_id: str,
    runtime_root: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Build two isolated child invocations from one public run command."""

    common = _without_parent_options(arguments)
    commands: list[tuple[str, ...]] = []
    for label in ("a", "b"):
        slot_arguments = (
            *common,
            "--slots",
            "1",
            "--allow-shared-gpu",
            "--worker-id",
            f"{worker_id}-slot-{label}",
        )
        if runtime_root is not None:
            slot_arguments = (
                *slot_arguments,
                "--runtime-dir",
                str(runtime_root / f"slot-{label}"),
            )
        commands.append((str(python), str(script), *slot_arguments))
    return commands[0], commands[1]


def run_child_slots(commands: tuple[tuple[str, ...], tuple[str, ...]]) -> int:
    """Run both long-lived slots and propagate a coordinated interrupt."""

    processes = [subprocess.Popen(command) for command in commands]
    try:
        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        deadline = time.monotonic() + 5.0
        while (
            time.monotonic() < deadline
            and any(process.poll() is None for process in processes)
        ):
            time.sleep(0.05)
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        return_codes = [process.wait() for process in processes]
    return 130 if 130 in return_codes else max(return_codes)


__all__ = ["child_slot_commands", "run_child_slots"]
