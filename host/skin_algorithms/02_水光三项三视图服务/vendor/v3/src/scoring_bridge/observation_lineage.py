from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from src.scoring_bridge.lineage import ConfirmationManifestRow


class ObservationSignature(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    relative_path: str
    size: int
    mtime_ns: int


class TerminalObservation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    signature: ObservationSignature
    status: str


@dataclass(frozen=True, slots=True)
class ConfirmationObservationLineageReceipt:
    observation_count: int
    source_sha256: tuple[tuple[str, str], ...]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_confirmation_observation_lineage(
    *,
    manifest_path: Path,
    observation_paths: Sequence[Path],
    expected_count: int = 250,
) -> ConfirmationObservationLineageReceipt:
    manifest_rows = tuple(
        ConfirmationManifestRow.model_validate_json(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected = {row.relative_path: row for row in manifest_rows}
    if len(expected) != expected_count:
        raise ValueError("confirmation observation manifest count mismatch")
    ordered = sorted(
        (path.resolve(strict=True) for path in observation_paths),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    latest: dict[str, TerminalObservation] = {}
    for path in ordered:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if raw.get("status") not in {"success", "partial_success", "failed"}:
                continue
            observation = TerminalObservation.model_validate(raw)
            latest[observation.signature.relative_path] = observation
    if set(latest) != set(expected):
        raise ValueError("confirmation observations do not match frozen manifest")
    for relative_path, observation in latest.items():
        manifest = expected[relative_path]
        if (
            observation.signature.size != manifest.size
            or observation.signature.mtime_ns != manifest.mtime_ns
        ):
            raise ValueError(
                f"confirmation observation signature mismatch: {relative_path}"
            )
    return ConfirmationObservationLineageReceipt(
        observation_count=len(latest),
        source_sha256=tuple((path.name, _sha(path)) for path in ordered),
    )


__all__ = [
    "ConfirmationObservationLineageReceipt",
    "validate_confirmation_observation_lineage",
]
