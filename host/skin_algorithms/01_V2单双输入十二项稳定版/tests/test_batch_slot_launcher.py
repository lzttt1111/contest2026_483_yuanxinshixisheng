from __future__ import annotations

from pathlib import Path

from src.nine_analysis import batch_slot_launcher
from src.nine_analysis.batch_slot_launcher import child_slot_commands


def test_two_slot_commands_share_job_but_isolate_worker_and_runtime() -> None:
    commands = child_slot_commands(
        (
            "--input-dir",
            "/input",
            "--output-dir",
            "/output",
            "--job-name",
            "bridge",
            "--slots",
            "2",
            "--worker-id",
            "gpu-pc-03",
            "--runtime-dir",
            "/dev/shm/runtime",
        ),
        python=Path("/python"),
        script=Path("/repo/run.py"),
        worker_id="gpu-pc-03",
        runtime_root=Path("/dev/shm/runtime"),
    )

    assert len(commands) == 2
    assert all("--slots" in command for command in commands)
    assert all(command[command.index("--slots") + 1] == "1" for command in commands)
    assert {command[command.index("--worker-id") + 1] for command in commands} == {
        "gpu-pc-03-slot-a",
        "gpu-pc-03-slot-b",
    }
    assert {command[command.index("--runtime-dir") + 1] for command in commands} == {
        "/dev/shm/runtime/slot-a",
        "/dev/shm/runtime/slot-b",
    }
    assert all("--allow-shared-gpu" in command for command in commands)


def test_parent_does_not_double_interrupt_children_already_exiting(
    monkeypatch,
) -> None:
    sent: list[int] = []
    processes = []
    first_wait = True

    class FakeProcess:
        def __init__(self) -> None:
            self.poll_count = 0

        def wait(self) -> int:
            nonlocal first_wait
            if first_wait:
                first_wait = False
                raise KeyboardInterrupt
            return 130

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 130

        def send_signal(self, value: int) -> None:
            sent.append(value)

    def popen(_command):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(batch_slot_launcher.subprocess, "Popen", popen)

    code = batch_slot_launcher.run_child_slots((("a",), ("b",)))

    assert code == 130
    assert sent == []
