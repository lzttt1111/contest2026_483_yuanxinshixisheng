from __future__ import annotations

"""Request-scoped route dispatch with process-local reusable models."""

import threading

from src.capture_profile import CaptureProfile
from src.detection_runtime.cloud_task import CURRENT_CAPTURE


_PIPELINES = {}
_PROVIDER = None
_LOCK = threading.RLock()


def _pipeline(profile, existing):
    if getattr(existing, "capture_profile", None) == profile:
        return existing
    if profile not in _PIPELINES:
        from src.pipeline import DermaVisionPipeline
        _PIPELINES[profile] = DermaVisionPipeline(capture_profile=profile)
    return _PIPELINES[profile]


def run_task_pipeline(service, existing, input_path, algorithms):
    context = CURRENT_CAPTURE.get()
    if context is None:
        return existing.process_single(input_path, algorithms)
    with _LOCK:
        if service != "dermavision":
            # Acne's shared profile reader uses the request ContextVar.
            return existing.process_single(input_path, algorithms)
        pipeline = _pipeline(context.profile, existing)
        if context.capture is None:
            return pipeline.process_single(input_path, algorithms)
        algorithm = algorithms[0]
        if algorithm in {"spots", "texture", "pores", "contour_firmness"}:
            return pipeline.process_single(input_path, [algorithm])
        global _PROVIDER
        if _PROVIDER is None:
            from src.detection_runtime.clinic_provider import ClinicFourLightProvider
            _PROVIDER = ClinicFourLightProvider()
        from src.detection_runtime.cloud_projection import project_clinic_result
        root = context.root / "clinic"
        prepared = _PROVIDER.prepare(context.capture, root)
        items, receipt = _PROVIDER.run_prepared(prepared, root, algorithms=(algorithm,))
        return project_clinic_result(algorithm, items, prepared, root / "public", receipt)


def close_cloud_pipelines():
    global _PROVIDER
    for pipeline in _PIPELINES.values():
        pipeline.close()
    _PIPELINES.clear()
    if _PROVIDER is not None:
        _PROVIDER.close()
        _PROVIDER = None
