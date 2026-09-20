"""Temporary target-output routing for immutable cloud surfaces."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CloudWorkError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@contextmanager
def temporary_service_work(
    project_root: Path,
    evidence_root: Path,
    service: str,
) -> Iterator[Path]:
    service_work = evidence_root / "runtime_work" / service
    service_work.mkdir(parents=True, exist_ok=True)
    if any(service_work.iterdir()):
        raise CloudWorkError(detail=f"cloud service work directory is not empty: {service}")
    worker_output = service_work / "worker_output"
    worker_output.mkdir()
    target_output = project_root / "output"
    if target_output.exists() or target_output.is_symlink():
        raise CloudWorkError(detail="target project output already exists")
    target_output.symlink_to(worker_output, target_is_directory=True)
    try:
        yield service_work
    finally:
        target_output.unlink()
