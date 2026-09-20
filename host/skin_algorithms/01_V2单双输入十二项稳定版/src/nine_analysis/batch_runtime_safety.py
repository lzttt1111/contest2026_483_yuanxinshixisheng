from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

import psutil


GIB = 1024**3
MAX_SHARED_GPU_WORKERS = 2


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    available_gib: float
    swap_used_gib: float
    rss_gib: float


@dataclass(frozen=True, slots=True)
class GpuRunLeaseUnavailable(Exception):
    cuda_device: str
    allow_shared: bool

    def __str__(self) -> str:
        mode = "双进程" if self.allow_shared else "单进程"
        return f"CUDA {self.cuda_device} 的{mode}批跑名额已占满"


class GpuRunLease:
    """Lifetime file locks limiting one exclusive or two shared GPU runners."""

    __slots__ = ("_global_handle", "_slot_handle")

    def __init__(self, global_handle: TextIO, slot_handle: TextIO | None) -> None:
        self._global_handle = global_handle
        self._slot_handle = slot_handle

    def close(self) -> None:
        for handle in (self._slot_handle, self._global_handle):
            if handle is None or handle.closed:
                continue
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def __enter__(self) -> GpuRunLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ExportLease:
    """One-at-a-time final publication lease for a mechanical output disk."""

    __slots__ = ("_handle",)

    def __init__(self, handle: TextIO) -> None:
        self._handle = handle

    def close(self) -> None:
        if self._handle.closed:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()


def acquire_gpu_run_lease(
    lock_root: Path,
    cuda_device: str,
    *,
    allow_shared: bool,
) -> GpuRunLease:
    """Acquire the only exclusive runner or one of two explicit shared slots."""

    lock_root.mkdir(parents=True, exist_ok=True)
    safe_device = "".join(character if character.isalnum() else "_" for character in cuda_device)
    global_handle = (lock_root / f"cuda-{safe_device}.lock").open("a+")
    global_mode = fcntl.LOCK_SH if allow_shared else fcntl.LOCK_EX
    try:
        fcntl.flock(global_handle.fileno(), global_mode | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        global_handle.close()
        raise GpuRunLeaseUnavailable(cuda_device, allow_shared) from exc
    if not allow_shared:
        return GpuRunLease(global_handle, None)
    for slot in range(MAX_SHARED_GPU_WORKERS):
        slot_handle = (lock_root / f"cuda-{safe_device}.slot-{slot}.lock").open("a+")
        try:
            fcntl.flock(slot_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            slot_handle.close()
            continue
        return GpuRunLease(global_handle, slot_handle)
    fcntl.flock(global_handle.fileno(), fcntl.LOCK_UN)
    global_handle.close()
    raise GpuRunLeaseUnavailable(cuda_device, allow_shared)


def memory_snapshot() -> MemorySnapshot:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process(os.getpid())
    return MemorySnapshot(
        available_gib=memory.available / GIB,
        swap_used_gib=swap.used / GIB,
        rss_gib=process.memory_info().rss / GIB,
    )


def resource_recycle_reason(
    snapshot: MemorySnapshot,
    *,
    min_available_gib: float,
    max_swap_gib: float,
) -> str | None:
    if snapshot.available_gib < min_available_gib:
        return "available_memory_below_limit"
    if snapshot.swap_used_gib >= max_swap_gib:
        return "swap_above_limit"
    return None


def acquire_export_lease(output_root: Path) -> ExportLease:
    """Block until this worker owns the single final HDD publication slot."""

    lock_path = output_root / "_batch" / "_export.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return ExportLease(handle)
