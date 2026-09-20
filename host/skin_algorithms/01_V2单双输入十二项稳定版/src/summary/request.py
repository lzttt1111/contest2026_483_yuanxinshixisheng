"""Request validation, canonical fingerprint and evidence merge.

The fingerprint recomputation mirrors the backend C3 rule byte for byte:
``SummaryRequestV1.model_dump(mode="json")`` with ``input_fingerprint`` removed,
then ``aisia_contracts.canonicalization.fingerprint_v1``. No extra normalisation
is applied, so a worker-side match proves the backend hashed the same bytes.

Contract-enforced invariants (``scoring_inputs`` keys ⊆ success outcomes,
``algorithm_name``/``report_id`` consistency) are not re-implemented here; only
the cross-input identity the contract does not cover is checked.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

from aisia_contracts.canonicalization import fingerprint_v1
from aisia_contracts.overall_summary.v1 import SummaryRequestV1

REQUEST_INVALID = "request_invalid"
FINGERPRINT_MISMATCH = "fingerprint_mismatch"
IDENTITY_MISMATCH = "identity_mismatch"
SCORING_RELEASE_MISMATCH = "scoring_release_mismatch"

# input_quality_gate 提取诊断：所有携带门禁的 scoring_input 必须来自同一原图
# 且 status/reason_codes 完全一致；否则视为不可用并诚实降级 REVIEW。
GATE_CONSISTENT = "consistent"
GATE_MISSING = "missing"
GATE_INCONSISTENT = "inconsistent"


class SummaryRejection(Exception):
    """A request was rejected before scoring; carries a stable error code."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"{code}: {message}")


def validate_request(raw: Any) -> SummaryRequestV1:
    """Validate an inbound mapping into ``SummaryRequestV1`` without raising raw."""
    if not isinstance(raw, Mapping):
        raise SummaryRejection(REQUEST_INVALID, "summary request must be a mapping")
    try:
        return SummaryRequestV1.model_validate(dict(raw))
    except ValidationError as error:
        raise SummaryRejection(REQUEST_INVALID, str(error)) from error


def recompute_fingerprint(request: SummaryRequestV1) -> str:
    """Canonical SHA256 of the request with ``input_fingerprint`` removed."""
    payload = request.model_dump(mode="json")
    payload.pop("input_fingerprint", None)
    return fingerprint_v1(payload)


def check_fingerprint(request: SummaryRequestV1) -> None:
    if recompute_fingerprint(request) != request.input_fingerprint:
        raise SummaryRejection(
            FINGERPRINT_MISMATCH,
            "input_fingerprint does not match canonical SummaryRequestV1 bytes",
        )


def check_identity(request: SummaryRequestV1) -> None:
    """Reject evidence spliced from different samples / profiles / routes."""
    image_hashes = {
        scoring_input.source_image_sha256
        for scoring_input in request.scoring_inputs.values()
    }
    if len(image_hashes) > 1:
        raise SummaryRejection(
            IDENTITY_MISMATCH, "scoring_inputs come from different source images"
        )
    for name, scoring_input in request.scoring_inputs.items():
        if scoring_input.capture_profile != request.capture_profile:
            raise SummaryRejection(
                IDENTITY_MISMATCH,
                f"{name} capture_profile {scoring_input.capture_profile!r} "
                f"!= request {request.capture_profile!r}",
            )
        if scoring_input.input_route != request.input_route:
            raise SummaryRejection(
                IDENTITY_MISMATCH,
                f"{name} input_route {scoring_input.input_route!r} "
                f"!= request {request.input_route!r}",
            )


def merge_evidence(request: SummaryRequestV1) -> dict[str, Any]:
    """Rebuild a ``detector_results`` subset from present/partial inputs.

    ``missing``/``failed`` algorithms contribute nothing, so the entry's honest
    completeness gate degrades their modules instead of inventing zeros.
    """
    merged: dict[str, Any] = {}
    for scoring_input in request.scoring_inputs.values():
        if scoring_input.evidence_status in ("missing", "failed"):
            continue
        for detector, node in scoring_input.evidence.items():
            merged.setdefault(detector, {}).update(node)
    return merged


def resolve_input_quality_gate(
    request: SummaryRequestV1,
) -> tuple[dict[str, Any] | None, str]:
    """提取跨 scoring_input 一致的原图派生门禁。

    一致性规则（同一 ``source_image_sha256`` 已由 :func:`check_identity` 保证）：

    - 所有**携带**门禁（非 ``None``）的 scoring_input，其
      ``{status, reason_codes}`` 必须完全一致；
    - 携带者一致 → 返回该门禁，诊断 ``consistent``；
    - 无人携带 → 返回 ``None``，诊断 ``missing``（交 B2a 诚实降级 REVIEW）；
    - 携带者互相冲突 → 返回 ``None``，诊断 ``inconsistent``（不挑选、不伪造）。

    返回 ``(gate | None, diagnosis)``。
    """

    seen: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for name, scoring_input in request.scoring_inputs.items():
        quality = scoring_input.quality
        gate = quality.input_quality_gate if quality is not None else None
        if gate is None:
            continue
        key = (str(gate.status), tuple(str(code) for code in gate.reason_codes))
        seen.setdefault(key, []).append(name)
    if not seen:
        return None, GATE_MISSING
    if len(seen) > 1:
        return None, GATE_INCONSISTENT
    (status, reasons), _ = next(iter(seen.items()))
    return {"status": status, "reason_codes": list(reasons)}, GATE_CONSISTENT


__all__ = [
    "FINGERPRINT_MISMATCH",
    "GATE_CONSISTENT",
    "GATE_INCONSISTENT",
    "GATE_MISSING",
    "IDENTITY_MISMATCH",
    "REQUEST_INVALID",
    "SCORING_RELEASE_MISMATCH",
    "SummaryRejection",
    "check_fingerprint",
    "check_identity",
    "merge_evidence",
    "recompute_fingerprint",
    "resolve_input_quality_gate",
    "validate_request",
]
