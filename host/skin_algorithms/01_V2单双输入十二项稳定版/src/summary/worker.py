"""Independent CPU-only Celery app for overall-summary aggregation.

Import-time chain is deliberately tiny: ``celery`` + settings + the contract
models + :mod:`src.summary.release`. The scoring chain (which loads numpy/cv2
and the Word scoring code) is imported lazily inside the task body, so starting
this app never touches the detection Pipeline, GPU models or Word rendering.

Task flow (pure functions in :mod:`src.summary.request` /
:mod:`src.summary.result`):
1. validate request -> ``SummaryRequestV1`` (``request_invalid`` on failure);
2. canonical fingerprint, cross-input identity and release checks;
3. verify pinned asset SHA (fail closed) and merge present/partial evidence;
4. call the shared B2a entry and map its 11 modules onto ``SummaryResultV1``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Mapping

from celery import Celery

from src.core.config import settings

from . import SUMMARY_CONCURRENCY, SUMMARY_CONSUMER_QUEUE, SUMMARY_QUEUE, SUMMARY_TASK
from .release import RELEASE_ID, ReleaseAssetError, load_release
from .request import (
    SCORING_RELEASE_MISMATCH,
    SummaryRejection,
    check_fingerprint,
    check_identity,
    merge_evidence,
    resolve_input_quality_gate,
    validate_request,
)
from .result import build_failed_result, build_success_result

logger = logging.getLogger(__name__)

SCORING_ASSET_MISMATCH = "scoring_asset_mismatch"
SCORING_ERROR = "scoring_error"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

celery_app = Celery(
    "dermavision_summary",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.summary.worker"],
)

# Institution queue plus the additive consumer queue (declared, not deployed).
celery_app.conf.task_queues = {
    SUMMARY_QUEUE: {
        "exchange": SUMMARY_QUEUE,
        "exchange_type": "direct",
        "routing_key": SUMMARY_QUEUE,
    },
    SUMMARY_CONSUMER_QUEUE: {
        "exchange": SUMMARY_CONSUMER_QUEUE,
        "exchange_type": "direct",
        "routing_key": SUMMARY_CONSUMER_QUEUE,
    },
}
celery_app.conf.task_routes = {SUMMARY_TASK: {"queue": SUMMARY_QUEUE}}
celery_app.conf.task_default_queue = SUMMARY_QUEUE
celery_app.conf.task_default_exchange = SUMMARY_QUEUE
celery_app.conf.task_default_routing_key = SUMMARY_QUEUE
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=SUMMARY_CONCURRENCY,
    task_soft_time_limit=settings.task_soft_time_limit,
    task_time_limit=settings.task_time_limit,
    result_expires=86400,
    result_serializer="json",
    task_serializer="json",
    accept_content=["json"],
)


def _debug_info(elapsed_ms: float) -> dict[str, Any]:
    """Minimal, path-free debug block."""
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "scoring_release": RELEASE_ID,
    }


def _failed_from_model(model, rejection: SummaryRejection, elapsed_ms: float):
    return build_failed_result(
        report_id=model.report_id,
        attempt_id=model.attempt_id,
        input_fingerprint=model.input_fingerprint,
        capture_profile=model.capture_profile,
        input_route=model.input_route,
        scoring_release=model.scoring_release,
        code=rejection.code,
        message=rejection.message,
        retryable=rejection.retryable,
        debug_info=_debug_info(elapsed_ms),
    )


def _failed_from_raw(raw: Any, rejection: SummaryRejection, elapsed_ms: float):
    def text(key: str) -> str:
        value = raw.get(key) if isinstance(raw, Mapping) else None
        return value if isinstance(value, str) else ""

    fingerprint = text("input_fingerprint")
    if not _HEX64.fullmatch(fingerprint):
        fingerprint = "0" * 64
    return build_failed_result(
        report_id=text("report_id"),
        attempt_id=text("attempt_id"),
        input_fingerprint=fingerprint,
        capture_profile=text("capture_profile"),
        input_route=text("input_route"),
        scoring_release=text("scoring_release"),
        code=rejection.code,
        message=rejection.message,
        retryable=rejection.retryable,
        debug_info=_debug_info(elapsed_ms),
    )


def run_aggregate_summary(request: Any) -> dict[str, Any]:
    """Pure task body; callable directly in tests without a broker."""
    started = time.perf_counter()
    try:
        model = validate_request(request)
    except SummaryRejection as rejection:
        return _failed_from_raw(request, rejection, (time.perf_counter() - started) * 1e3).model_dump(
            mode="json"
        )

    try:
        check_fingerprint(model)
        check_identity(model)
        if model.scoring_release != RELEASE_ID:
            raise SummaryRejection(
                SCORING_RELEASE_MISMATCH,
                f"scoring_release {model.scoring_release!r} != {RELEASE_ID!r}",
            )
        try:
            release = load_release()
        except ReleaseAssetError as error:
            raise SummaryRejection(SCORING_ASSET_MISMATCH, str(error)) from error

        expected_fields = release.asset_sha256["scoring_input_field_list_v3.json"]
        if any(item.evidence_field_list_sha256 != expected_fields
               for item in model.scoring_inputs.values()):
            raise SummaryRejection(SCORING_RELEASE_MISMATCH,
                                   "scoring_input field-list SHA does not match this release")
        merged = merge_evidence(model)
        # 同源门禁：从各 scoring_input 携带的 input_quality_gate 提取，全部
        # 携带者必须一致；缺失/不一致诚实降级 REVIEW，不挑选、不伪造。
        input_quality_gate, gate_diagnosis = resolve_input_quality_gate(model)
        # Lazy import: keeps app startup free of the heavy scoring/Word chain.
        from src.summary_scoring import score_report_from_evidence

        entry_result = score_report_from_evidence(
            merged,
            capture_profile=model.capture_profile,
            input_route=model.input_route,
            quality=None,
            input_quality_gate=input_quality_gate,
            scoring_assets=None,
        )
        debug_info = _debug_info((time.perf_counter() - started) * 1e3)
        debug_info["input_quality_gate"] = {
            "status": input_quality_gate["status"] if input_quality_gate else None,
            "reason_codes": list(input_quality_gate["reason_codes"])
            if input_quality_gate
            else [],
            "diagnosis": gate_diagnosis,
        }
        result = build_success_result(
            request=model,
            entry_result=entry_result,
            release=release,
            debug_info=debug_info,
        )
    except SummaryRejection as rejection:
        result = _failed_from_model(model, rejection, (time.perf_counter() - started) * 1e3)
    except Exception as error:  # noqa: BLE001 - reported, never raised to broker
        logger.exception("aggregate_summary scoring failed report_id=%s", model.report_id)
        result = build_failed_result(
            report_id=model.report_id,
            attempt_id=model.attempt_id,
            input_fingerprint=model.input_fingerprint,
            capture_profile=model.capture_profile,
            input_route=model.input_route,
            scoring_release=model.scoring_release,
            code=SCORING_ERROR,
            message=f"{type(error).__name__}: {error}",
            retryable=False,
            debug_info=_debug_info((time.perf_counter() - started) * 1e3),
        )
    return result.model_dump(mode="json")


@celery_app.task(name=SUMMARY_TASK, bind=True)
def aggregate_summary(self, request):
    """Celery entry point; ``run_aggregate_summary`` holds the logic."""
    return run_aggregate_summary(request)


__all__ = ["aggregate_summary", "celery_app", "run_aggregate_summary"]
