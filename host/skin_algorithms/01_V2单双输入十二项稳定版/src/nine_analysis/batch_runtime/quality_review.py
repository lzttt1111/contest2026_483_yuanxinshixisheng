from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

from src.nine_analysis.batch_runtime.identifiers import claim_key, safe_name


def quality_flags_from_record(record: dict[str, Any]) -> list[str]:
    errors = record.get("service_errors") or {}
    joined = " ".join(str(value) for value in errors.values())
    flags: list[str] = []
    for flag in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", joined.upper()):
        if flag not in flags:
            flags.append(flag)
    return flags


def quality_review_category(flags: Iterable[str]) -> str:
    values = set(flags)
    if values.intersection({
        "NO_FACE", "MULTIPLE_FACES", "INVALID_IMAGE",
        "INVALID_FACE_GEOMETRY", "FACE_TOO_SMALL", "FACE_TOO_LARGE",
    }):
        return "01_人脸异常"
    if values.intersection({
        "POSE_YAW", "POSE_PITCH", "POSE_ROLL", "EXTREME_POSE", "PARTIAL_FACE",
    }):
        return "02_姿态与裁切"
    if values.intersection({
        "UNEVEN_LIGHTING", "SEVERE_LOCAL_LIGHTING", "UNDEREXPOSED",
        "OVEREXPOSED", "EXCESSIVE_HIGHLIGHT",
    }):
        return "03_光照与曝光"
    if "BLUR" in values:
        return "04_模糊"
    if values.intersection({
        "FOREHEAD_OCCLUDED", "INSUFFICIENT_SKIN_AREA", "SEGMENTATION_FALLBACK",
    }):
        return "05_遮挡与有效域"
    return "99_其他"


def archive_quality_rejection(
    review_root: Path,
    image: Path,
    input_root: Path,
    record: dict[str, Any],
) -> Path | None:
    if record.get("failure_category") != "input_quality_reject":
        return None
    flags = quality_flags_from_record(record)
    category = quality_review_category(flags)
    signature = record.get("signature") or {}
    relative = str(
        signature.get("relative_path")
        or image.relative_to(input_root).as_posix()
    )
    digest = claim_key(dict(signature))[:12]
    sample = safe_name(str(record.get("sample_id") or image.parent.name))
    target_dir = review_root / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sample}__{digest}{image.suffix.lower()}"
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    shutil.copy2(image, temporary)
    os.replace(temporary, target)
    metadata = {
        "review_status": "pending",
        "category": category,
        "quality_flags": flags,
        "source_relative_path": relative,
        "sample_id": record.get("sample_id"),
        "signature": signature,
        "finished_at": record.get("finished_at"),
    }
    metadata_path = target.with_name(target.name + ".json")
    metadata_temporary = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{uuid.uuid4().hex}"
    )
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)
    return target


__all__ = [
    "archive_quality_rejection",
    "quality_flags_from_record",
    "quality_review_category",
]
