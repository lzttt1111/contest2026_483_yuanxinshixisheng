from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
HybridDocument: TypeAlias = dict[str, JsonValue]


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _canonical_body(document: Mapping[str, JsonValue]) -> bytes:
    body = {
        key: value for key, value in document.items()
        if key != "profile_sha256"
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _contains_absolute_path(value: JsonValue) -> bool:
    if isinstance(value, str):
        return value.startswith("/") or _WINDOWS_ABSOLUTE.match(value) is not None
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    return False


def finalize_hybrid_profile_document(
    body: Mapping[str, JsonValue],
) -> HybridDocument:
    if "profile_sha256" in body:
        raise ValueError("hybrid profile body already contains profile SHA")
    if _contains_absolute_path(dict(body)):
        raise ValueError("hybrid profile cannot contain absolute paths")
    required = {
        "schema_version",
        "profile_id",
        "status",
        "capture_profile",
        "institution_temporary_alias_id",
        "score_direction",
        "numerical_core_sha256",
        "provenance",
        "legacy_dimensions",
        "population_profile",
        "medical_boundary",
    }
    missing = sorted(required.difference(body))
    if missing:
        raise ValueError(f"hybrid profile is missing fields: {missing}")
    document: HybridDocument = dict(body)
    document["profile_sha256"] = hashlib.sha256(
        _canonical_body(document)
    ).hexdigest()
    return document


def verify_hybrid_profile_document(
    document: Mapping[str, JsonValue],
) -> str:
    expected = document.get("profile_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("hybrid profile SHA is missing")
    actual = hashlib.sha256(_canonical_body(document)).hexdigest()
    if actual != expected:
        raise ValueError("hybrid profile SHA mismatch")
    if _contains_absolute_path(dict(document)):
        raise ValueError("hybrid profile contains an absolute path")
    return actual


def load_promoted_hybrid_profile(
    path: Path,
    *,
    allowlisted_sha256: set[str] | frozenset[str],
) -> HybridDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hybrid profile root must be an object")
    document: HybridDocument = payload
    profile_sha = verify_hybrid_profile_document(document)
    if profile_sha not in allowlisted_sha256:
        raise ValueError("hybrid profile SHA is not allowlisted")
    if document.get("status") != "promoted":
        raise ValueError("hybrid profile is not promoted")
    return document


def resolve_legacy_dimension_route(
    profile: Mapping[str, JsonValue],
    dimension_id: str,
) -> str:
    dimensions = profile.get("legacy_dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("hybrid profile legacy dimensions are invalid")
    dimension = dimensions.get(dimension_id)
    if not isinstance(dimension, dict):
        raise ValueError(f"unknown hybrid dimension: {dimension_id}")
    if dimension.get("status") != "promoted":
        return "v011_fallback"
    route = dimension.get("route")
    if not isinstance(route, str):
        raise ValueError(f"hybrid dimension route is invalid: {dimension_id}")
    return route


def capture_profile_identity(
    profile: Mapping[str, JsonValue],
    capture_profile: str,
) -> tuple[str, str]:
    if capture_profile == "consumer":
        profile_id = profile.get("profile_id")
    elif capture_profile == "institution":
        profile_id = profile.get("institution_temporary_alias_id")
    else:
        raise ValueError(f"unsupported capture profile: {capture_profile}")
    numerical_sha = profile.get("numerical_core_sha256")
    if not isinstance(profile_id, str) or not isinstance(numerical_sha, str):
        raise ValueError("hybrid profile identity is invalid")
    return profile_id, numerical_sha


__all__ = [
    "HybridDocument",
    "capture_profile_identity",
    "finalize_hybrid_profile_document",
    "load_promoted_hybrid_profile",
    "resolve_legacy_dimension_route",
    "verify_hybrid_profile_document",
]
