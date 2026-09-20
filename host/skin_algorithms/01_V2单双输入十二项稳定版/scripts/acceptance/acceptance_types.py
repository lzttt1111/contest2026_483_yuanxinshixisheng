"""Typed values shared by acceptance entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

from pydantic import JsonValue

JsonDict: TypeAlias = dict[str, JsonValue]
Environment: TypeAlias = dict[str, str]


@dataclass(frozen=True, slots=True)
class InputAsset:
    case_id: str
    path: Path
    size_bytes: int
    width_pixels: int
    height_pixels: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VendorSidecar:
    root: Path
    module_path: Path
    module_sha256: str
    root_environment_variable: str
    sha256_environment_variable: str

    @property
    def environment(self) -> Environment:
        return {
            self.root_environment_variable: str(self.root),
            self.sha256_environment_variable: self.module_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProcessRequest:
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    log_path: Path
    resource_path: Path
    timeout_seconds: float
    sample_interval_seconds: float


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    exit_code: int
    timed_out: bool
    wall_seconds: float
    process_group_id: int
    peak_rss_kib: int
    peak_swap_kib: int
    peak_gpu_memory_mib: int


@dataclass(frozen=True, slots=True)
class SurfaceAssessment:
    exit_code: int
    started_ns: int
    expected_cases: tuple[str, ...]
    discovered_cases: tuple[str, ...]
    oldest_required_output_ns: int
    task_count: int
    projected_result_count: int
    service_process_starts: int
    pipeline_initializations: int
    expected_task_count: int = 0
    expected_projected_result_count: int = 0
    require_cloud_reuse: bool = False
    contract_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalInspection:
    discovered_cases: tuple[str, ...]
    oldest_required_output_ns: int
    contract_errors: tuple[str, ...]
