from __future__ import annotations

"""Optional four-light report aggregation; never changes per-algorithm envelopes."""

from dataclasses import replace
from pathlib import Path
import re
import tempfile

from src.detection_runtime import ClinicTwelveAnalysisOrchestrator, TwelveOutputExporter
from src.detection_runtime.cloud_input import parse_capture_images, CaptureInputError
from src.detection_runtime.cloud_task import CaptureTaskContext, _download_images
from src.aisia_medical_report.clinic_twelve_delivery import generate_clinic_dual_reports_from_twelve_result
from src.detection_runtime.final_delivery_layout import finalize_institution_delivery


def export_clinic_reports(orchestrator, capture, output_root: Path, subject_id: str, *, job_namespace="local_word"):
    """One full current-result analysis followed by the unchanged local report chain."""
    destination = output_root / capture.capture_alias
    if destination.exists():
        raise FileExistsError(f"report destination already exists: {destination.name}")
    result = orchestrator.analyze_capture(capture, job_namespace=job_namespace)
    source = orchestrator.job_runtime_root(job_namespace, capture.capture_alias)
    destination = TwelveOutputExporter(output_root).export(
        source, result, destination_name=capture.capture_alias)
    reports = generate_clinic_dual_reports_from_twelve_result(
        destination, subject_id=subject_id, runtime_items=result["十二项结果"])
    finalize_institution_delivery(destination, result["十二项结果"])
    return destination, reports


class CloudCaptureReportSession:
    """Explicit optional report batch; owns reusable local institution models.

    Worker public metrics alone are insufficient for the eleven-module report.
    This session computes full institution evidence once per requested report pair,
    just as the existing consumer Word helper aggregates its full evidence.
    It is separate from the single-algorithm Celery tasks.
    """

    def __init__(self, runtime_root: Path):
        self.orchestrator = ClinicTwelveAnalysisOrchestrator(runtime_root, stage_input=False)

    def __enter__(self):
        self.orchestrator.start()
        return self

    def __exit__(self, *args):
        self.orchestrator.close()

    def generate(self, capture_images, output_root: Path, *, subject_id: str,
                 download_by_key, maximum_bytes=32*1024*1024):
        sources = parse_capture_images(capture_images)
        if len(sources) != 4:
            raise CaptureInputError("invalid_capture_count", "institution reports require four images")
        if not isinstance(subject_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", subject_id):
            raise CaptureInputError("invalid_report_subject", "provide a safe report subject identifier")
        with tempfile.TemporaryDirectory(prefix="dermavision-report-capture-") as temporary:
            context = CaptureTaskContext(Path(temporary), sources)
            _download_images(context, download_by_key, None, maximum_bytes)
            capture = replace(context.capture, capture_alias=subject_id, source_group_alias=subject_id)
            return export_clinic_reports(self.orchestrator, capture, output_root, subject_id, job_namespace="cloud_word")
