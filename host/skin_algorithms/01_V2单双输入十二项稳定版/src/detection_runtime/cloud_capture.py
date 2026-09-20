from __future__ import annotations

"""Strict signed OSS manifest contract for the clinic capture Worker."""

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.detection_runtime.capture_manifest import CaptureChannelInput, ClinicCaptureInput
from src.detection_runtime.contracts import infer_runtime_route


CLOUD_CAPTURE_SCHEMA = "clinic_capture_manifest_v1"
CLOUD_CAPTURE_ROLES = ("RGB_M", "PP_M", "CP_M", "365_M")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clean_oss_key(value: object) -> str:
    key = str(value or "").strip().replace("\\", "/")
    if not key or key.startswith("/") or ".." in Path(key).parts or "://" in key:
        raise ValueError("INVALID_CAPTURE_SET: invalid OSS key")
    return key


@dataclass(frozen=True)
class CloudCaptureChannel:
    role: str
    oss_key: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    registration_status: str
    registration_transform_sha256: str | None
    cross_channel_authorized: dict[str, bool]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedCloudCaptureManifest:
    capture_alias: str
    source_group_alias: str
    channels: tuple[CloudCaptureChannel, ...]
    key_id: str
    manifest_sha256: str


def verify_cloud_capture_manifest(
    raw: bytes,
    *,
    key_hex: str | None = None,
    expected_key_id: str | None = None,
) -> VerifiedCloudCaptureManifest:
    """Verify signature and exact four-role payload before any image download."""

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("INVALID_CAPTURE_SET: manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {"payload", "signature"}:
        raise ValueError("INVALID_CAPTURE_SET: manifest top-level fields drifted")
    payload, signature = document["payload"], document["signature"]
    if not isinstance(payload, dict) or not isinstance(signature, dict):
        raise ValueError("INVALID_CAPTURE_SET: malformed signed manifest")
    if payload.get("schema_version") != CLOUD_CAPTURE_SCHEMA:
        raise ValueError("INVALID_CAPTURE_SET: unsupported manifest schema")
    if set(signature) != {"algorithm", "key_id", "value"} or signature.get("algorithm") != "HMAC-SHA256":
        raise ValueError("INVALID_CAPTURE_SET: unsupported manifest signature")
    configured_key = (key_hex or os.getenv("DERMAVISION_CAPTURE_MANIFEST_HMAC_KEY_HEX", "")).strip()
    if len(configured_key) != 64:
        raise ValueError("INVALID_CAPTURE_SET: capture manifest key is unavailable")
    try:
        secret = bytes.fromhex(configured_key)
    except ValueError as exc:
        raise ValueError("INVALID_CAPTURE_SET: capture manifest key is invalid") from exc
    key_id = str(signature.get("key_id") or "")
    configured_key_id = expected_key_id or os.getenv("DERMAVISION_CAPTURE_MANIFEST_KEY_ID", "")
    if configured_key_id and key_id != configured_key_id:
        raise ValueError("INVALID_CAPTURE_SET: manifest key ID mismatch")
    expected = hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(signature.get("value") or "")):
        raise ValueError("INVALID_CAPTURE_SET: manifest signature mismatch")

    alias = str(payload.get("capture_alias") or "").strip()
    source_group = str(payload.get("source_group_alias") or "").strip()
    rows = payload.get("channels")
    if not alias or Path(alias).name != alias or not source_group or not isinstance(rows, list):
        raise ValueError("INVALID_CAPTURE_SET: invalid capture identity")
    if len(rows) != 4 or tuple(row.get("role") for row in rows if isinstance(row, dict)) != CLOUD_CAPTURE_ROLES:
        raise ValueError("INVALID_CAPTURE_SET: exact RGB_M/PP_M/CP_M/365_M roles required")
    channels: list[CloudCaptureChannel] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("INVALID_CAPTURE_SET: invalid channel row")
        sha = str(row.get("sha256") or "").lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ValueError("INVALID_CAPTURE_SET: invalid channel SHA256")
        width, height, size = int(row.get("width", 0)), int(row.get("height", 0)), int(row.get("size_bytes", 0))
        if min(width, height, size) <= 0:
            raise ValueError("INVALID_CAPTURE_SET: invalid channel dimensions or size")
        channels.append(CloudCaptureChannel(
            role=str(row["role"]),
            oss_key=_clean_oss_key(row.get("oss_key")),
            sha256=sha,
            size_bytes=size,
            width=width,
            height=height,
            registration_status=str(row.get("registration_status") or "NOT_EVALUATED"),
            registration_transform_sha256=row.get("registration_transform_sha256"),
            cross_channel_authorized={str(k): bool(v) for k, v in (row.get("cross_channel_authorized") or {}).items()},
            reason_codes=tuple(str(v) for v in (row.get("reason_codes") or ())),
        ))
    infer_runtime_route(CLOUD_CAPTURE_ROLES, signed_manifest_sha256=_sha256_bytes(raw))
    return VerifiedCloudCaptureManifest(alias, source_group, tuple(channels), key_id, _sha256_bytes(raw))


def download_verified_cloud_capture(
    manifest: VerifiedCloudCaptureManifest,
    root: Path,
    downloader: Callable[[str], bytes],
    manifest_path: Path,
) -> ClinicCaptureInput:
    """Download exactly four objects and turn them into the local provider contract."""

    root.mkdir(parents=True, exist_ok=True)
    local_channels: list[CaptureChannelInput] = []
    for channel in manifest.channels:
        value = downloader(channel.oss_key)
        if len(value) != channel.size_bytes or _sha256_bytes(value) != channel.sha256:
            raise ValueError(f"INVALID_CAPTURE_SET: downloaded hash/size mismatch {channel.role}")
        suffix = Path(channel.oss_key).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"INVALID_CAPTURE_SET: unsupported image suffix {channel.role}")
        path = root / f"{channel.role}{suffix}"
        path.write_bytes(value)
        decoded = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or (decoded.shape[1], decoded.shape[0]) != (channel.width, channel.height):
            raise ValueError(f"INVALID_CAPTURE_SET: decoded dimensions mismatch {channel.role}")
        local_channels.append(CaptureChannelInput(
            role=channel.role,
            path=path,
            sha256=channel.sha256,
            size_bytes=channel.size_bytes,
            width=channel.width,
            height=channel.height,
            registration_status=channel.registration_status,
            registration_transform_sha256=channel.registration_transform_sha256,
            cross_channel_authorized=channel.cross_channel_authorized,
            reason_codes=channel.reason_codes,
        ))
    return ClinicCaptureInput(
        capture_alias=manifest.capture_alias,
        source_group_alias=manifest.source_group_alias,
        channels=tuple(local_channels),
        fixture_manifest_path=manifest_path,
        fixture_manifest_sha256=manifest.manifest_sha256,
        signed_manifest_path=manifest_path,
        signed_manifest_sha256=manifest.manifest_sha256,
    )


__all__ = [
    "CLOUD_CAPTURE_ROLES", "CLOUD_CAPTURE_SCHEMA", "CloudCaptureChannel",
    "VerifiedCloudCaptureManifest", "download_verified_cloud_capture",
    "verify_cloud_capture_manifest",
]
