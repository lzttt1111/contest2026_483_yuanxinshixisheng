from __future__ import annotations

"""Stable, non-path report and subject identifiers for formal Word payloads."""

from dataclasses import dataclass
from pathlib import Path
import re


_FIXED_CAPTURE = re.compile(
    r"(?:\d+_)?(clinic28-\d+)(?:_(?:RGB|PP|CP|365)_M)?",
    re.IGNORECASE,
)
_IMAGE_SUFFIX = re.compile(r"\.(?:jpe?g|png|bmp|webp)$", re.IGNORECASE)


def _normalized_text(value: str) -> str:
    normalized = " ".join(str(value).split())
    if not normalized:
        raise ValueError("report identity must not be empty")
    return normalized


def normalize_explicit_subject_id(value: str) -> str:
    """Honor an explicit identifier, changing whitespace only."""

    return _normalized_text(value)


def subject_id_from_source(source: str | Path) -> str:
    """Derive a human identifier from the original source filename."""

    name = str(source).replace("\\", "/").rsplit("/", 1)[-1]
    stem = _IMAGE_SUFFIX.sub("", name)
    fixed = _FIXED_CAPTURE.fullmatch(stem)
    return _normalized_text(fixed.group(1) if fixed is not None else stem)


def resolve_subject_id(
    explicit: str | None,
    *,
    source: str | Path,
) -> str:
    if explicit is not None:
        return normalize_explicit_subject_id(explicit)
    return subject_id_from_source(source)


def clean_report_number(value: str) -> str:
    """Remove user paths and image suffixes while preserving the report token."""

    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return _normalized_text(_IMAGE_SUFFIX.sub("", name))


@dataclass(frozen=True, slots=True)
class ReportIdentity:
    report_id: str
    subject_id: str


def single_rgb_report_identity(
    *,
    source_image: str | Path,
    result_name: str,
    subject_id: str | None,
    report_id: str | None,
) -> ReportIdentity:
    resolved_subject = resolve_subject_id(subject_id, source=source_image)
    resolved_report = (
        clean_report_number(report_id)
        if report_id is not None
        else f"AISIA-SINGLE-RGB-{clean_report_number(result_name)}"
    )
    return ReportIdentity(resolved_report, resolved_subject)


__all__ = [
    "ReportIdentity",
    "clean_report_number",
    "normalize_explicit_subject_id",
    "resolve_subject_id",
    "single_rgb_report_identity",
    "subject_id_from_source",
]
