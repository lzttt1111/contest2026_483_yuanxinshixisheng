from __future__ import annotations

"""Typed JSONL contracts for the persistent local runtime."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator, model_validator

from src.detection_runtime.contracts import RuntimeRoute


_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
JsonObject = dict[str, JsonValue]


class ConsumerInputManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_path: Path


class ClinicInputManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_manifest_path: Path
    capture_alias: str

    @field_validator("capture_alias")
    @classmethod
    def validate_capture_alias(cls, value: str) -> str:
        if not value or Path(value).name != value:
            raise ValueError(  # noqa: GENERIC_ERR_OK - Pydantic validator contract
                "capture_alias must be one path-safe name"
            )
        return value


class RuntimeJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command: Literal["submit"]
    job_id: str
    route: RuntimeRoute
    input_manifest: ConsumerInputManifest | ClinicInputManifest
    output_root: Path
    generate_reports: bool

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not _JOB_ID_PATTERN.fullmatch(value):
            raise ValueError(  # noqa: GENERIC_ERR_OK - Pydantic validator contract
                "job_id must be a path-safe identifier"
            )
        return value

    @field_validator("output_root")
    @classmethod
    def resolve_output_root(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if resolved == resolved.parent:
            raise ValueError(  # noqa: GENERIC_ERR_OK - Pydantic validator contract
                "output_root must name one job directory"
            )
        return resolved

    @model_validator(mode="after")
    def validate_route_input(self) -> "RuntimeJob":
        match (self.route, self.input_manifest):  # noqa: MATCH_OK - correlated fields
            case (RuntimeRoute.CONSUMER_RGB, ConsumerInputManifest()):
                return self
            case (RuntimeRoute.CLINIC_FOUR_LIGHT, ClinicInputManifest()):
                return self
            case _:
                raise ValueError(  # noqa: GENERIC_ERR_OK - Pydantic validator contract
                    "input_manifest does not match route"
                )


class JobResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: Literal["result"] = "result"
    job_id: str
    route: RuntimeRoute
    status: Literal["success", "failed", "cancelled"]
    output_root: Path
    error_type: str | None = None
    error_message: str | None = None
    submitted_at: str
    started_at: str | None
    completed_at: str
    queue_wait_seconds: float
    execution_seconds: float
    end_to_end_seconds: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runtime_root: Path
    max_slots: int = 1
    queue_capacity: int = 4
    cuda_device: str = "0"

    def __post_init__(self) -> None:
        if self.max_slots not in {1, 2}:
            raise RuntimeConfigurationError("max_slots must be 1 or 2")
        if self.queue_capacity < 1:
            raise RuntimeConfigurationError("queue_capacity must be at least 1")
        object.__setattr__(self, "runtime_root", self.runtime_root.resolve())


@dataclass(frozen=True, slots=True)
class DuplicateJobError(Exception):
    job_id: str

    def __str__(self) -> str:
        return f"duplicate job_id: {self.job_id}"


@dataclass(frozen=True, slots=True)
class QueueCapacityError(Exception):
    capacity: int

    def __str__(self) -> str:
        return f"pending job queue is full: capacity={self.capacity}"


@dataclass(frozen=True, slots=True)
class RuntimeStateError(Exception):
    state: str

    def __str__(self) -> str:
        return f"runtime does not accept jobs while state={self.state}"


@dataclass(frozen=True, slots=True)
class OutputRootError(Exception):
    output_root: Path

    def __str__(self) -> str:
        return f"output_root is already reserved or exists: {self.output_root}"


@dataclass(frozen=True, slots=True)
class RuntimeConfigurationError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


__all__ = [
    "ClinicInputManifest",
    "ConsumerInputManifest",
    "DuplicateJobError",
    "JobResult",
    "JsonObject",
    "OutputRootError",
    "QueueCapacityError",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "RuntimeJob",
    "RuntimeStateError",
]
