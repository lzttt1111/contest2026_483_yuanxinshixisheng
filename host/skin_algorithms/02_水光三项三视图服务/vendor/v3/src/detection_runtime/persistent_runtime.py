from __future__ import annotations

"""Bounded persistent slots for repeated local twelve-item jobs."""

from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import queue
import threading
import time
from typing import Callable

from typing_extensions import assert_never

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
from src.detection_runtime.persistent_slot import LocalRuntimeSlot, RuntimeSlot


SlotFactory = Callable[[int, Path], RuntimeSlot]


@dataclass(frozen=True, slots=True)
class CloseError:
    slot_id: int
    error_type: str
    error_message: str

    def as_json(self) -> JsonObject:
        return {
            "slot_id": self.slot_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class _QueuedJob:
    job: RuntimeJob
    future: Future[JobResult]
    submitted_at: str
    submitted_monotonic: float


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


_STOP = _Stop()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RuntimeState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class PersistentRuntime:
    """Mutable bounded scheduler that owns and closes all resident slots."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        slot_factory: SlotFactory | None = None,
    ) -> None:
        self.config = config
        self._factory = slot_factory or self._default_slot
        self._queue: queue.Queue[_QueuedJob | _Stop] = queue.Queue()
        self._pending_jobs = 0
        self._lock = threading.Lock()
        self._state = _RuntimeState.CREATED
        self._job_ids: set[str] = set()
        self._output_roots: set[Path] = set()
        self._slots: list[RuntimeSlot] = []
        self._threads: list[threading.Thread] = []
        self._closed_slots = 0
        self._close_errors: list[CloseError] = []

    def _default_slot(self, slot_id: int, slot_root: Path) -> RuntimeSlot:
        return LocalRuntimeSlot(slot_id, slot_root, self.config.cuda_device)

    @property
    def close_errors(self) -> tuple[CloseError, ...]:
        with self._lock:
            return tuple(
                sorted(self._close_errors, key=lambda failure: failure.slot_id)
            )

    @staticmethod
    def _close_slot(slot_id: int, slot: RuntimeSlot) -> CloseError | None:
        try:
            slot.close()
        except Exception as exc:  # noqa: BROAD_EXCEPT_OK - resource owner boundary
            return CloseError(
                slot_id=slot_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        return None

    def start(self) -> tuple[JsonObject, ...]:
        with self._lock:
            match self._state:
                case _RuntimeState.CREATED:
                    pass
                case _RuntimeState.STARTING | _RuntimeState.RUNNING | _RuntimeState.STOPPING | _RuntimeState.STOPPED:
                    raise RuntimeStateError(self._state.value)
                case unreachable:
                    assert_never(unreachable)
            self._state = _RuntimeState.STARTING
            self.config.runtime_root.mkdir(parents=True, exist_ok=True)
            ready = False
            try:
                for slot_id in range(self.config.max_slots):
                    slot = self._factory(
                        slot_id,
                        self.config.runtime_root / f"slot-{slot_id + 1}",
                    )
                    self._slots.append(slot)
                receipts = tuple(slot.start() for slot in self._slots)
                self._state = _RuntimeState.RUNNING
                for slot_id, slot in enumerate(self._slots):
                    thread = threading.Thread(
                        target=self._worker,
                        args=(slot_id, slot),
                        name=(
                            "dermavision-runtime-slot-"
                            f"{len(self._threads) + 1}"
                        ),
                    )
                    thread.start()
                    self._threads.append(thread)
                ready = True
                return receipts
            finally:
                if not ready:
                    for slot_id, slot in enumerate(self._slots):
                        error = self._close_slot(slot_id, slot)
                        if error is not None:
                            self._close_errors.append(error)
                    self._closed_slots = len(self._slots)
                    self._state = _RuntimeState.STOPPED

    def submit(self, job: RuntimeJob) -> Future[JobResult]:
        future: Future[JobResult] = Future()
        with self._lock:
            match self._state:
                case _RuntimeState.RUNNING:
                    pass
                case _RuntimeState.CREATED | _RuntimeState.STARTING | _RuntimeState.STOPPING | _RuntimeState.STOPPED:
                    raise RuntimeStateError(self._state.value)
                case unreachable:
                    assert_never(unreachable)
            if job.job_id in self._job_ids:
                raise DuplicateJobError(job.job_id)
            if self._pending_jobs >= self.config.queue_capacity:
                raise QueueCapacityError(self.config.queue_capacity)
            output_conflict = any(
                job.output_root.is_relative_to(root)
                or root.is_relative_to(job.output_root)
                for root in self._output_roots | {self.config.runtime_root}
            )
            if output_conflict or job.output_root.exists():
                raise OutputRootError(job.output_root)
            self._job_ids.add(job.job_id)
            self._output_roots.add(job.output_root)
            self._pending_jobs += 1
            submitted_monotonic = time.monotonic()
            submitted_at = _utc_now()
            self._queue.put_nowait(
                _QueuedJob(
                    job=job,
                    future=future,
                    submitted_at=submitted_at,
                    submitted_monotonic=submitted_monotonic,
                )
            )
        return future

    def _before_job_start(self, _job: RuntimeJob) -> None:
        """Test seam immediately after dequeue and before state transition."""

    @staticmethod
    def _cancelled_result(queued: _QueuedJob) -> JobResult:
        completed_monotonic = time.monotonic()
        completed_at = _utc_now()
        end_to_end_seconds = max(
            0.0,
            completed_monotonic - queued.submitted_monotonic,
        )
        return JobResult(
            job_id=queued.job.job_id,
            route=queued.job.route,
            status="cancelled",
            output_root=queued.job.output_root,
            error_type="RuntimeShutdown",
            error_message="job cancelled before execution",
            submitted_at=queued.submitted_at,
            started_at=None,
            completed_at=completed_at,
            queue_wait_seconds=end_to_end_seconds,
            execution_seconds=0.0,
            end_to_end_seconds=end_to_end_seconds,
        )

    def _worker(self, slot_id: int, slot: RuntimeSlot) -> None:
        try:
            while True:
                queued = self._queue.get()
                try:
                    match queued:
                        case _Stop():
                            return
                        case _QueuedJob(job=job, future=future):
                            with self._lock:
                                self._pending_jobs -= 1
                            self._before_job_start(job)
                            with self._lock:
                                may_run = (
                                    self._state is _RuntimeState.RUNNING
                                    and future.set_running_or_notify_cancel()
                                )
                                if may_run:
                                    started_monotonic = time.monotonic()
                                    started_at = _utc_now()
                            if not may_run:
                                if not future.done():
                                    future.set_result(self._cancelled_result(queued))
                                continue
                            try:
                                output = slot.execute(job)
                                status = "success"
                                error_type = None
                                error_message = None
                            except Exception as exc:  # noqa: BROAD_EXCEPT_OK - worker boundary
                                output = job.output_root
                                status = "failed"
                                error_type = type(exc).__name__
                                error_message = str(exc)
                            completed_monotonic = time.monotonic()
                            completed_at = _utc_now()
                            result = JobResult(
                                job_id=job.job_id,
                                route=job.route,
                                status=status,
                                output_root=output,
                                error_type=error_type,
                                error_message=error_message,
                                submitted_at=queued.submitted_at,
                                started_at=started_at,
                                completed_at=completed_at,
                                queue_wait_seconds=max(
                                    0.0,
                                    started_monotonic
                                    - queued.submitted_monotonic,
                                ),
                                execution_seconds=max(
                                    0.0,
                                    completed_monotonic - started_monotonic,
                                ),
                                end_to_end_seconds=max(
                                    0.0,
                                    completed_monotonic
                                    - queued.submitted_monotonic,
                                ),
                            )
                            future.set_result(result)
                        case unreachable:
                            assert_never(unreachable)
                finally:
                    self._queue.task_done()
        finally:
            error = self._close_slot(slot_id, slot)
            with self._lock:
                if error is not None:
                    self._close_errors.append(error)
                self._closed_slots += 1
                if self._closed_slots == len(self._slots):
                    self._state = _RuntimeState.STOPPED

    def shutdown(self, *, wait: bool = True) -> int:
        cancelled: list[_QueuedJob] = []
        with self._lock:
            match self._state:
                case _RuntimeState.CREATED:
                    self._state = _RuntimeState.STOPPED
                case _RuntimeState.RUNNING:
                    self._state = _RuntimeState.STOPPING
                    while True:
                        try:
                            queued = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(queued, _QueuedJob):
                            self._pending_jobs -= 1
                            cancelled.append(queued)
                        self._queue.task_done()
                    for _thread in self._threads:
                        self._queue.put_nowait(_STOP)
                case _RuntimeState.STARTING | _RuntimeState.STOPPING | _RuntimeState.STOPPED:
                    pass
                case unreachable:
                    assert_never(unreachable)
        for queued in cancelled:
            if not queued.future.done():
                queued.future.set_result(self._cancelled_result(queued))
        if wait:
            for thread in self._threads:
                thread.join()
        with self._lock:
            return self._closed_slots


__all__ = [
    "CloseError",
    "DuplicateJobError",
    "JobResult",
    "LocalRuntimeSlot",
    "OutputRootError",
    "PersistentRuntime",
    "QueueCapacityError",
    "RuntimeConfig",
    "RuntimeJob",
    "RuntimeStateError",
]
