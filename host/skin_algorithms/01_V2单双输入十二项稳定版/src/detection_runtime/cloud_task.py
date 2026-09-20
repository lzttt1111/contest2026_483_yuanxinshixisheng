from __future__ import annotations

"""One/four-image task adapter; existing workers remain the result publishers."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import json
import logging
from pathlib import Path
import tempfile
import threading

import cv2
import numpy as np

from src.capture_profile import CaptureProfile, use_capture_profile
from src.detection_runtime.capture_manifest import CaptureChannelInput, ClinicCaptureInput
from src.detection_runtime.cloud_input import CaptureInputError, parse_capture_images, ROLES
from src.detection_runtime.cloud_registration import (
    CaptureRegistration, RegistrationThresholds, assess_registration,
    canonical_sha, rgb_geometry_valid,
)


@dataclass
class CaptureTaskContext:
    root: Path
    images: tuple
    capture: ClinicCaptureInput | None = None

    @property
    def profile(self):
        return CaptureProfile.INSTITUTION if len(self.images) == 4 else CaptureProfile.CONSUMER


CURRENT_CAPTURE: ContextVar[CaptureTaskContext | None] = ContextVar("cloud_capture", default=None)
_REGISTRATION = CaptureRegistration()
_REGISTRATION_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


def _download_images(context, by_key, by_url, maximum_bytes):
    rows, decoded = [], {}
    for image in context.images:
        data = (by_url if image.by_url else by_key)(image.locator)
        if not isinstance(data, bytes) or not data or len(data) > maximum_bytes:
            raise CaptureInputError("invalid_capture_bytes", "empty or oversized capture image")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise CaptureInputError("capture_decode_failed", f"cannot decode {image.role}")
        # Original names never become filesystem paths.
        path = context.root / (image.role + Path(image.filename).suffix.lower())
        path.write_bytes(data)
        decoded[image.role] = frame
        rows.append(dict(role=image.role, path=path, sha256=hashlib.sha256(data).hexdigest(),
                         size_bytes=len(data), width=frame.shape[1], height=frame.shape[0]))
    if len(rows) == 1:
        return
    limits = RegistrationThresholds()
    with _REGISTRATION_LOCK:
        points = {role: _REGISTRATION.landmarks(decoded[role]) for role in ROLES}
    if not rgb_geometry_valid(points["RGB_M"], limits):
        raise CaptureInputError("capture_rgb_quality_failed", "RGB anchor failed local geometry gate")
    channels, evidence = [], {}
    for row in rows:
        role = row["role"]
        assessment = (
            {"status": "ANCHOR", "transform_sha256": None, "reason": "RGB_ENGINEERING_ANCHOR_PASSED"}
            if role == "RGB_M" else
            assess_registration(role, points[role], points["RGB_M"],
                                decoded[role].shape[:2], decoded["RGB_M"].shape[:2], limits)
        )
        if role != "RGB_M" and len(points[role]) < limits.minimum_landmark_count:
            raise CaptureInputError("capture_landmarks_failed", f"{role} has no independent face observation")
        evidence[role] = assessment
        authorized = assessment["status"] == "PASSED"
        channels.append(CaptureChannelInput(
            **row, registration_status=assessment["status"],
            registration_transform_sha256=assessment["transform_sha256"],
            cross_channel_authorized={key: authorized for key in ("co_location", "fusion", "overlap")},
            reason_codes=(assessment["reason"],),
        ))
    manifest = context.root / "capture_evidence.json"
    public_evidence = {
        "schema": "cloud_capture_preparation_v1",
        "input_sha256": {row["role"]: row["sha256"] for row in rows},
        "registration": evidence,
        "thresholds": vars(limits),
    }
    manifest.write_text(json.dumps(public_evidence, indent=2, allow_nan=False), encoding="utf-8")
    bad = [role for role in ("PP_M", "CP_M") if evidence[role]["status"] != "PASSED"]
    if bad:
        raise CaptureInputError("capture_registration_failed", "local registration gate rejected " + ",".join(bad))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    alias = "capture-" + canonical_sha(public_evidence["input_sha256"])[:16]
    context.capture = ClinicCaptureInput(alias, alias, tuple(channels),
                                        manifest, digest, manifest, digest)


def download_task_input(oss_key, by_key):
    context = CURRENT_CAPTURE.get()
    if context is None:
        return by_key(oss_key)
    image = context.images[0]
    return (context.root / (image.role + Path(image.filename).suffix.lower())).read_bytes()


def with_capture_images(service: str):
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            images = bound.arguments.get("capture_images")
            if images is None:
                return function(*args, **kwargs)
            module = function.__globals__
            task_id = bound.arguments.get("task_id")
            try:
                sources = parse_capture_images(images)
                algorithms = bound.arguments.get("algorithms")
                if service == "dermavision" and (not isinstance(algorithms, list) or len(algorithms) != 1
                                                or algorithms[0] not in module["_SUPPORTED_ALGORITHMS"]):
                    raise CaptureInputError("invalid_algorithm", "capture requests require one supported algorithm")
                with tempfile.TemporaryDirectory(prefix="dermavision-capture-") as temporary:
                    context = CaptureTaskContext(Path(temporary), sources)
                    try:
                        _download_images(context, module["download_via_internal_api"],
                                         None,
                                         getattr(module["settings"], "image_fetch_max_bytes", 32 * 1024 * 1024))
                        _retain_preparation(context, task_id, service)
                        token = CURRENT_CAPTURE.set(context)
                        try:
                            with use_capture_profile(context.profile):
                                bound.arguments["capture_images"] = None
                                bound.arguments["oss_key"] = sources[0].filename
                                result = function(*bound.args, **bound.kwargs)
                        finally:
                            CURRENT_CAPTURE.reset(token)
                        if result.get("status") == "failed":
                            _retain_failure(context, result.get("error_code", "task_failed"))
                        return result
                    except Exception:
                        _retain_failure(context, "capture_failed")
                        raise
            except Exception as exc:
                from src.core.storage import StorageError
                code = getattr(exc, "code", None) or ("capture_download_failed" if isinstance(exc, StorageError) else "capture_failed")
                message = str(exc) if isinstance(exc, CaptureInputError) else "capture processing failed"
                logger.exception("cloud capture failure service=%s code=%s", service, code)
                extra = {"schema_version": module["_ACNE_V2_SV"]} if service == "acne" and function.__name__ == "analyze_image_v2" else {}
                return module["_failed"](task_id, message, error_code=code, **extra)
        return wrapped
    return decorate


def _retain_preparation(context, report_id, service):
    """Keep four-light provenance internally, before temporary inputs disappear."""
    if context.capture is None:
        return
    root = Path(__file__).resolve().parents[2] / "output" / "cloud_capture_evidence"
    root.mkdir(parents=True, exist_ok=True)
    evidence = context.root / "capture_evidence.json"
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload.update(
        report_id=report_id, service=service, state="prepared",
        capture_digest=canonical_sha(payload["input_sha256"]),
        cross_channel_authorized={
            channel.role: dict(channel.cross_channel_authorized)
            for channel in context.capture.channels
        },
    )
    from uuid import uuid4
    # Never use caller identifiers as filesystem paths, or retain source locators.
    (root / (uuid4().hex + ".json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _retain_failure(context, code):
    # Keep only bounded diagnostics, not signed URLs or input photographs.
    root = Path(__file__).resolve().parents[2] / "output" / "cloud_capture_failures"
    root.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    evidence = context.root / "capture_evidence.json"
    payload = json.loads(evidence.read_text()) if evidence.is_file() else {}
    payload["error_code"] = code
    (root / (uuid4().hex + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def close_capture_runtime():
    _REGISTRATION.close()
    from src.detection_runtime.cloud_inference import close_cloud_pipelines
    close_cloud_pipelines()
