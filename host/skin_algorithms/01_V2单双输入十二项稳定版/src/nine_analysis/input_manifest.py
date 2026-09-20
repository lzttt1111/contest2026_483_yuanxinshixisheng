from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class InputManifestRow(BaseModel):
    """One frozen, previously selected calibration input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    subject_id: str = Field(min_length=1)
    legacy_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "validation", "reserve"]


@dataclass(frozen=True, slots=True)
class InputManifestError(Exception):
    manifest: Path
    line_number: int
    reason: str

    def __str__(self) -> str:
        return f"输入manifest无效: {self.manifest}:{self.line_number}: {self.reason}"


def _resolve_manifest_path(
    input_root: Path,
    row: InputManifestRow,
    *,
    manifest: Path,
    line_number: int,
) -> Path:
    relative = PurePosixPath(row.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise InputManifestError(manifest, line_number, "relative_path越出输入根目录")
    candidate = input_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(input_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise InputManifestError(manifest, line_number, f"输入文件不可用: {exc}") from exc
    if not resolved.is_file():
        raise InputManifestError(manifest, line_number, "输入路径不是普通文件")
    stat = resolved.stat()
    if stat.st_size != row.size or stat.st_mtime_ns != row.mtime_ns:
        raise InputManifestError(manifest, line_number, "输入文件大小或mtime与冻结记录不一致")
    return resolved


def iter_manifest_images(input_root: Path, manifest: Path) -> Iterator[Path]:
    """Yield only explicit manifest paths without enumerating the image root."""

    root = input_root.resolve(strict=True)
    manifest_path = manifest.resolve(strict=True)
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = InputManifestRow.model_validate_json(payload)
            except ValidationError as exc:
                raise InputManifestError(
                    manifest_path,
                    line_number,
                    exc.errors(include_url=False)[0]["msg"],
                ) from exc
            if row.relative_path in seen:
                raise InputManifestError(manifest_path, line_number, "relative_path重复")
            seen.add(row.relative_path)
            yield _resolve_manifest_path(
                root,
                row,
                manifest=manifest_path,
                line_number=line_number,
            )
