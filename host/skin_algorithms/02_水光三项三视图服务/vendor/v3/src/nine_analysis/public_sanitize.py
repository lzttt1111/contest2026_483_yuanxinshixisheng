from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Final, Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


BLOCKED_KEYS: Final = frozenset({
    "weights", "weights_path", "source", "output_dir", "output_root",
    "model_path", "face_landmarker_model", "selfie_multiclass_model",
    "device", "gpu", "cuda", "gpu_name", "torch", "cuda_runtime",
    "compute_capability", "memory_fraction", "cpu_threads",
    "checkpoint", "checkpoint_sha256", "python_executable",
})


def _is_internal_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or normalized.startswith("models/")
        or "/models/" in normalized
    )


def sanitize_public_value(value: JsonValue, *, key: str = "") -> JsonValue:
    if key.lower() in BLOCKED_KEYS:
        return None
    if isinstance(value, str):
        return None if _is_internal_path(value) else value
    if isinstance(value, Mapping):
        sanitized: dict[str, JsonValue] = {}
        for child_key, child_value in value.items():
            cleaned = sanitize_public_value(child_value, key=str(child_key))
            if cleaned is not None:
                sanitized[str(child_key)] = cleaned
        return sanitized
    if isinstance(value, list):
        cleaned_items = [sanitize_public_value(item) for item in value]
        return [item for item in cleaned_items if item is not None]
    return value


def sanitize_public_document(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    cleaned = sanitize_public_value(value)
    return cleaned if isinstance(cleaned, dict) else {}


__all__ = ["sanitize_public_document"]
