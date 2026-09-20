"""Contained regular-file resolution for delivered acceptance artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class UnsafeArtifactPathError(RuntimeError):
    relative: str

    def __str__(self) -> str:
        return f"unsafe acceptance artifact path: {self.relative}"


def resolve_regular_file(root: Path, relative: str) -> Path:
    """Resolve one normalized POSIX path inside root without symlink escape."""

    normalized = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or normalized.is_absolute()
        or normalized.as_posix() != relative
        or ".." in normalized.parts
    ):
        raise UnsafeArtifactPathError(relative=relative)
    unresolved = root / Path(*normalized.parts)
    if unresolved.is_symlink():
        raise UnsafeArtifactPathError(relative=relative)
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise UnsafeArtifactPathError(relative=relative) from exc
    resolved_root = root.resolve()
    if not resolved.is_file() or resolved_root not in resolved.parents:
        raise UnsafeArtifactPathError(relative=relative)
    return resolved
