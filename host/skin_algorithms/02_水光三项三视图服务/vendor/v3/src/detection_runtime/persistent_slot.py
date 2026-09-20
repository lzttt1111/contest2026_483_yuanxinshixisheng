from __future__ import annotations

"""One serial persistent slot backed by one resident analysis bundle."""

from pathlib import Path
import os
import shutil
from typing import Protocol
import uuid

from typing_extensions import assert_never

from src.detection_runtime.atomic_publish import atomic_publish_directory
from src.detection_runtime.capture_manifest import (
    ClinicCaptureInput,
    load_clinic_capture_manifest,
)
from src.detection_runtime.contracts import (
    RuntimeRoute,
    TWELVE_DETECTION_ITEMS,
)
from src.detection_runtime.exporter import TwelveOutputExporter
from src.detection_runtime.orchestrator import ClinicTwelveAnalysisOrchestrator
from src.detection_runtime.orchestrator import TwelveAnalysisOrchestrator
from src.detection_runtime.route_plan import CONSUMER_ROUTE_PLAN
from src.detection_runtime.persistent_protocol import (
    ClinicInputManifest,
    ConsumerInputManifest,
    JsonObject,
    RuntimeJob,
)


class CaptureSelectionError(Exception):
    """The requested capture alias is not unique in its signed fixture."""


class OutputPublishError(Exception):
    """A final output cannot be published without clobbering another owner."""


class ConsumerRoute(Protocol):
    run_root: Path

    def analyze(
        self,
        image_path: Path,
        repetition: int = 1,
        sample_id: str | None = None,
    ) -> JsonObject: ...


class _ConsumerResidentView(TwelveAnalysisOrchestrator):
    """Consumer route profile sharing one slot's serial resident clients."""

    def __init__(self, base: TwelveAnalysisOrchestrator) -> None:
        self.output_root = base.output_root
        self.cuda_device = base.cuda_device
        self.stage_input = base.stage_input
        self.run_id = base.run_id
        self.run_root = base.run_root
        self.log_dir = base.log_dir
        self.clients = base.clients
        self.cold_start = base.cold_start
        self.route_plan = CONSUMER_ROUTE_PLAN
        self.algorithms = tuple(
            item.item_id for item in TWELVE_DETECTION_ITEMS
        )


def _finalize_report_names(staging: Path, final_alias: str) -> None:
    """Replace the hidden staging alias in report names and JSON references."""

    staging_alias = staging.name
    report_root = staging / "正式报告"
    if not report_root.is_dir():
        return
    for path in tuple(report_root.rglob("*")):
        if path.is_file() and staging_alias in path.name:
            path.rename(
                path.with_name(path.name.replace(staging_alias, final_alias))
            )
    json_paths = (
        staging / "十二项检测结果索引.json",
        *report_root.rglob("*.json"),
    )
    for path in json_paths:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        updated = content.replace(staging_alias, final_alias)
        if updated != content:
            path.write_text(updated, encoding="utf-8")


class RuntimeSlot(Protocol):
    def start(self) -> JsonObject: ...

    def execute(self, job: RuntimeJob) -> Path: ...

    def close(self) -> None: ...


class LocalRuntimeSlot:
    """One mutable resident bundle, owned by exactly one worker thread."""

    def __init__(self, slot_id: int, slot_root: Path, cuda_device: str) -> None:
        self.slot_id = slot_id
        self._orchestrator = ClinicTwelveAnalysisOrchestrator(
            slot_root,
            cuda_device=cuda_device,
            stage_input=False,
        )
        self._consumer: ConsumerRoute = _ConsumerResidentView(
            self._orchestrator.base
        )
        self._closed = False

    @classmethod
    def from_orchestrator(
        cls,
        slot_id: int,
        orchestrator: ClinicTwelveAnalysisOrchestrator,
        consumer: ConsumerRoute,
    ) -> "LocalRuntimeSlot":
        slot = cls.__new__(cls)
        slot.slot_id = slot_id
        slot._orchestrator = orchestrator
        slot._consumer = consumer
        slot._closed = False
        return slot

    def start(self) -> JsonObject:
        return self._orchestrator.start()

    @staticmethod
    def _capture(manifest: ClinicInputManifest) -> ClinicCaptureInput:
        captures = load_clinic_capture_manifest(manifest.fixture_manifest_path)
        matches = tuple(
            capture
            for capture in captures
            if capture.capture_alias == manifest.capture_alias
        )
        if len(matches) != 1:
            raise CaptureSelectionError(
                "capture_alias is missing or duplicated in manifest"
            )
        return matches[0]

    def execute(self, job: RuntimeJob) -> Path:
        match (job.route, job.input_manifest):
            case (RuntimeRoute.CONSUMER_RGB, ConsumerInputManifest(image_path=image)):
                sample_id = f"{job.job_id}--{image.stem}"
                manifest = self._consumer.analyze(
                    image,
                    sample_id=sample_id,
                )
                source_root = self._consumer.run_root / "repeat_1" / sample_id
            case (RuntimeRoute.CLINIC_FOUR_LIGHT, ClinicInputManifest() as input_manifest):
                capture = self._capture(input_manifest)
                manifest = self._orchestrator.analyze_capture(
                    capture,
                    job_namespace=job.job_id,
                )
                source_root = self._orchestrator.job_runtime_root(
                    job.job_id,
                    capture.capture_alias,
                )
            case unreachable:
                assert_never(unreachable)
        parent = job.output_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / (
            f".{job.output_root.name}.pending-{uuid.uuid4().hex[:8]}"
        )
        claim = parent / f".{job.output_root.name}.publish.lock"
        try:
            descriptor = os.open(
                claim,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise OutputPublishError(f"publish claim exists: {claim}") from exc
        os.close(descriptor)
        try:
            destination = TwelveOutputExporter(parent).export(
                source_root,
                manifest,
                destination_name=staging.name,
            )
            if job.generate_reports:
                from src.aisia_medical_report import (
                    generate_dual_reports_from_twelve_result,
                )

                generate_dual_reports_from_twelve_result(
                    destination,
                    subject_id=job.job_id,
                )
                _finalize_report_names(destination, job.output_root.name)
            atomic_publish_directory(destination, job.output_root)
            return job.output_root
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            claim.unlink(missing_ok=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._orchestrator.close()


__all__ = ["LocalRuntimeSlot", "OutputPublishError", "RuntimeSlot"]
