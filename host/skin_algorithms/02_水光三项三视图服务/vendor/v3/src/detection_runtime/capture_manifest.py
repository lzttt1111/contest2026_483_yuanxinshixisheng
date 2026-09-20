from __future__ import annotations

"""Bounded local capture-manifest loader used by the clinic runtime."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from src.detection_runtime.contracts import infer_runtime_route


ROLE_MAP = {"RGB": "RGB_M", "PP": "PP_M", "CP": "CP_M", "365": "365_M"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CaptureChannelInput:
    role: str
    path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    registration_status: str
    registration_transform_sha256: str | None
    cross_channel_authorized: dict[str, bool]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ClinicCaptureInput:
    capture_alias: str
    source_group_alias: str
    channels: tuple[CaptureChannelInput, ...]
    fixture_manifest_path: Path
    fixture_manifest_sha256: str
    signed_manifest_path: Path
    signed_manifest_sha256: str

    def channel(self, role: str) -> CaptureChannelInput:
        matches = [item for item in self.channels if item.role == role]
        if len(matches) != 1:
            raise ValueError(f"INVALID_CAPTURE_SET: missing or duplicate {role}")
        return matches[0]


def _bounded_file(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("INVALID_CAPTURE_SET: absolute input path is forbidden")
    candidate = (root / relative).resolve(strict=True)
    if candidate.is_symlink() or not candidate.is_file() or not candidate.is_relative_to(root):
        raise ValueError("INVALID_CAPTURE_SET: input escaped manifest root")
    return candidate


def load_clinic_capture_manifest(path: Path) -> tuple[ClinicCaptureInput, ...]:
    manifest_path = path.resolve(strict=True)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("INVALID_CAPTURE_SET: manifest path is invalid")
    root = manifest_path.parent.resolve(strict=True)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "clinic_fast3_local_fixture_v1":
        raise ValueError("INVALID_CAPTURE_SET: unsupported capture manifest schema")
    if payload.get("source_authority_hmac_status") != "VERIFIED":
        raise ValueError("INVALID_CAPTURE_SET: source authority is not verified")
    signed_row = payload.get("signed_authority_manifest") or {}
    signed_path = _bounded_file(root, str(signed_row.get("relative_path") or ""))
    signed_sha = _sha256(signed_path)
    if signed_sha != signed_row.get("sha256"):
        raise ValueError("INVALID_CAPTURE_SET: signed authority hash mismatch")

    signed = json.loads(signed_path.read_text(encoding="utf-8"))
    signed_captures = {
        item["capture_alias"]: item
        for item in (signed.get("payload") or {}).get("captures", [])
    }
    rows = payload.get("images") or []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("capture_alias")), []).append(row)
    if int(payload.get("capture_count", -1)) != len(grouped):
        raise ValueError("INVALID_CAPTURE_SET: capture count mismatch")
    if int(payload.get("image_count", -1)) != len(rows):
        raise ValueError("INVALID_CAPTURE_SET: image count mismatch")

    captures: list[ClinicCaptureInput] = []
    for alias, image_rows in grouped.items():
        authority = signed_captures.get(alias)
        if authority is None:
            raise ValueError(f"INVALID_CAPTURE_SET: unsigned capture {alias}")
        authority_channels = {item["role"]: item for item in authority["channels"]}
        channels: list[CaptureChannelInput] = []
        for row in image_rows:
            short_role = str(row.get("role"))
            role = ROLE_MAP.get(short_role)
            if role is None:
                raise ValueError(f"INVALID_CAPTURE_SET: unsupported role {short_role}")
            source = _bounded_file(root, str(row.get("relative_path") or ""))
            if source.stat().st_size != int(row.get("size_bytes", -1)):
                raise ValueError(f"INVALID_CAPTURE_SET: size mismatch {alias}/{role}")
            if _sha256(source) != row.get("sha256"):
                raise ValueError(f"INVALID_CAPTURE_SET: hash mismatch {alias}/{role}")
            signed_channel = authority_channels.get(short_role)
            comparable = (
                "sha256", "size_bytes", "width", "height",
                "registration_status", "registration_transform_sha256",
            )
            if signed_channel is None or any(
                signed_channel.get(key) != row.get(key) for key in comparable
            ):
                raise ValueError(f"INVALID_CAPTURE_SET: authority mismatch {alias}/{role}")
            channels.append(
                CaptureChannelInput(
                    role=role,
                    path=source,
                    sha256=str(row["sha256"]),
                    size_bytes=int(row["size_bytes"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    registration_status=str(row["registration_status"]),
                    registration_transform_sha256=row.get("registration_transform_sha256"),
                    cross_channel_authorized={
                        str(key): bool(value)
                        for key, value in (row.get("cross_channel_authorized") or {}).items()
                    },
                    reason_codes=tuple(str(value) for value in (row.get("reason_codes") or ())),
                )
            )
        infer_runtime_route(
            (channel.role for channel in channels),
            signed_manifest_sha256=signed_sha,
        )
        ordered = tuple(
            next(channel for channel in channels if channel.role == role)
            for role in ("RGB_M", "PP_M", "CP_M", "365_M")
        )
        captures.append(
            ClinicCaptureInput(
                capture_alias=alias,
                source_group_alias=str(authority["source_group_alias"]),
                channels=ordered,
                fixture_manifest_path=manifest_path,
                fixture_manifest_sha256=_sha256(manifest_path),
                signed_manifest_path=signed_path,
                signed_manifest_sha256=signed_sha,
            )
        )
    if tuple(item.capture_alias for item in captures) != (
        "clinic28-25", "clinic28-09", "clinic28-23",
    ):
        raise ValueError("INVALID_CAPTURE_SET: fixed Clinic-Fast3 order drifted")
    return tuple(captures)


__all__ = [
    "CaptureChannelInput",
    "ClinicCaptureInput",
    "load_clinic_capture_manifest",
]
