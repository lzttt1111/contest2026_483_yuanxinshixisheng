from __future__ import annotations

from datetime import date
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from src.scoring_bridge.hybrid_profile import (
    HybridDocument,
    finalize_hybrid_profile_document,
    verify_hybrid_profile_document,
)


class ScoringApprovalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_scope: Literal["user_release", "medical_report"]
    item_id: str
    reviewer: str
    reviewed_at: str
    profile_sha256: str
    dataset_sha256: str
    conclusion: Literal["approved", "blocked"]
    textual_conclusion: str

    @field_validator("reviewer", "item_id", "textual_conclusion")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("approval text fields cannot be empty")
        return value.strip()

    @field_validator("reviewed_at")
    @classmethod
    def _date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @field_validator("profile_sha256", "dataset_sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("approval SHA must be lowercase SHA256")
        return value


def _approval_index(
    approvals: Sequence[ScoringApprovalRecord],
) -> Mapping[tuple[str, str], ScoringApprovalRecord]:
    indexed: dict[tuple[str, str], ScoringApprovalRecord] = {}
    for approval in approvals:
        key = (approval.approval_scope, approval.item_id)
        if key in indexed:
            raise ValueError(f"duplicate scoring approval: {key}")
        indexed[key] = approval
    return indexed


def _validated_approval(
    indexed: Mapping[tuple[str, str], ScoringApprovalRecord],
    *,
    scope: str,
    item_id: str,
    profile_sha: str,
    dataset_sha: str,
    required: bool,
) -> ScoringApprovalRecord | None:
    approval = indexed.get((scope, item_id))
    if approval is None:
        if required:
            raise ValueError(f"required scoring approval is missing: {item_id}")
        return None
    if approval.profile_sha256 != profile_sha:
        raise ValueError(f"scoring approval profile SHA mismatch: {item_id}")
    if approval.dataset_sha256 != dataset_sha:
        raise ValueError(f"scoring approval dataset SHA mismatch: {item_id}")
    if approval.conclusion != "approved":
        if required:
            raise ValueError(f"scoring approval is blocked: {item_id}")
        return None
    return approval


def _approval_document(approval: ScoringApprovalRecord) -> dict[str, str]:
    return approval.model_dump(mode="json")


def promote_hybrid_profile(
    candidate: Mapping[str, Any],
    *,
    approvals: Sequence[ScoringApprovalRecord],
) -> HybridDocument:
    candidate_sha = verify_hybrid_profile_document(candidate)
    if candidate.get("status") != "candidate":
        raise ValueError("only a candidate hybrid profile can be promoted")
    provenance = candidate.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("hybrid candidate provenance is invalid")
    dataset_sha = provenance.get("confirmation_manifest_sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        raise ValueError("confirmation dataset SHA is missing")
    indexed = _approval_index(approvals)
    legacy_source = candidate.get("legacy_dimensions")
    population_source = candidate.get("population_modules")
    if not isinstance(legacy_source, dict) or not isinstance(population_source, dict):
        raise ValueError("hybrid candidate routes are invalid")
    legacy: dict[str, Any] = {}
    for dimension_id, raw_dimension in sorted(legacy_source.items()):
        if not isinstance(raw_dimension, dict):
            raise ValueError(f"hybrid dimension is invalid: {dimension_id}")
        dimension = dict(raw_dimension)
        if dimension.get("status") == "candidate":
            approval = _validated_approval(
                indexed,
                scope="user_release",
                item_id=dimension_id,
                profile_sha=candidate_sha,
                dataset_sha=dataset_sha,
                required=True,
            )
            assert approval is not None
            dimension["status"] = "promoted"
            dimension["release_approval"] = _approval_document(approval)
        legacy[dimension_id] = dimension
    population: dict[str, Any] = {}
    for module_id, raw_module in sorted(population_source.items()):
        if not isinstance(raw_module, dict):
            raise ValueError(f"hybrid population module is invalid: {module_id}")
        module = dict(raw_module)
        approval = _validated_approval(
            indexed,
            scope="medical_report",
            item_id=module_id,
            profile_sha=candidate_sha,
            dataset_sha=dataset_sha,
            required=False,
        )
        module["doctor_report_exposure"] = approval is not None
        module["user_report_exposure"] = False
        if approval is not None and module.get("status") == "internal_candidate":
            module["status"] = "doctor_approved"
            module["doctor_approval"] = _approval_document(approval)
        population[module_id] = module
    profile_id = str(candidate.get("profile_id") or "consumer_rgb_hybrid_v2_candidate")
    body = {
        key: value for key, value in candidate.items()
        if key != "profile_sha256"
    }
    body.update({
        "profile_id": profile_id.replace("_candidate", "_promoted"),
        "status": "promoted",
        "legacy_dimensions": legacy,
        "population_modules": population,
        "provenance": {**provenance, "source_candidate_sha256": candidate_sha},
    })
    return finalize_hybrid_profile_document(body)


__all__ = ["ScoringApprovalRecord", "promote_hybrid_profile"]
