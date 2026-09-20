from __future__ import annotations

"""JSONL command-line boundary for the persistent local runtime."""

import argparse
from concurrent.futures import Future
import json
from pathlib import Path
import signal
import sys
import threading
from types import FrameType
from typing import TextIO

from pydantic import ValidationError

from src.detection_runtime.persistent_protocol import (
    DuplicateJobError,
    JobResult,
    JsonObject,
    OutputRootError,
    QueueCapacityError,
    RuntimeConfig,
    RuntimeJob,
    RuntimeStateError,
)
from src.detection_runtime.persistent_runtime import PersistentRuntime


class _JsonLineWriter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, payload: JsonObject) -> None:
        with self._lock:
            self._stream.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )
            self._stream.flush()

    def emit_result(self, result: JobResult) -> None:
        with self._lock:
            self._stream.write(result.model_dump_json() + "\n")
            self._stream.flush()


def _emit_result(writer: _JsonLineWriter, future: Future[JobResult]) -> None:
    writer.emit_result(future.result())


def _shutdown_receipt(
    runtime: PersistentRuntime,
    closed_slots: int,
) -> JsonObject:
    return {
        "event": "shutdown_complete",
        "closed_slots": closed_slots,
        "close_errors": [error.as_json() for error in runtime.close_errors],
    }


class JsonCommandError(Exception):
    """A JSONL record is not a command object."""


def serve_jsonl(
    runtime: PersistentRuntime,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int:
    """Serve commands until EOF, shutdown, SIGINT, or SIGTERM."""

    writer = _JsonLineWriter(output_stream)
    try:
        receipts = runtime.start()
        writer.emit(
            {
                "event": "ready",
                "max_slots": runtime.config.max_slots,
                "queue_capacity": runtime.config.queue_capacity,
                "cold_start": list(receipts),
            }
        )
        for line in input_stream:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise JsonCommandError("JSONL command must be an object")
                command = payload.get("command")
                if command == "shutdown":
                    writer.emit({"event": "shutdown_started"})
                    closed = runtime.shutdown()
                    writer.emit(_shutdown_receipt(runtime, closed))
                    return 0
                job = RuntimeJob.model_validate(payload)
                future = runtime.submit(job)
                writer.emit({"event": "accepted", "job_id": job.job_id})
                future.add_done_callback(
                    lambda completed, sink=writer: _emit_result(sink, completed)
                )
            except (
                json.JSONDecodeError,
                JsonCommandError,
                ValidationError,
                DuplicateJobError,
                OutputRootError,
                QueueCapacityError,
                RuntimeStateError,
            ) as exc:
                writer.emit(
                    {
                        "event": "rejected",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
    except KeyboardInterrupt:
        writer.emit({"event": "shutdown_started", "reason": "signal"})
    finally:
        closed = runtime.shutdown()
    writer.emit(_shutdown_receipt(runtime, closed))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DermaVision local persistent JSONL runtime"
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--max-slots", type=int, choices=(1, 2), default=1)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--cuda-device", default="0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime = PersistentRuntime(
        RuntimeConfig(
            runtime_root=args.runtime_root,
            max_slots=args.max_slots,
            queue_capacity=args.queue_capacity,
            cuda_device=args.cuda_device,
        )
    )

    def request_shutdown(_signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    return serve_jsonl(runtime, sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
