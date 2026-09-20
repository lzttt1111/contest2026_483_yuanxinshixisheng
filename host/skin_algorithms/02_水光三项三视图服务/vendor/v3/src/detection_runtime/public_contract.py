from __future__ import annotations

"""Mechanical checks for the public twelve-item delivery tree."""

import csv
import json
from pathlib import Path
import re
from typing import Any


_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")
_FORBIDDEN_KEYS = {
    "audit",
    "reason_code",
    "reason_codes",
    "weights",
    "output_dir",
    "device",
    "gpu",
    "model_path",
    "input_path",
    "source_path",
}


def _walk(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                errors.append(f"{location}: forbidden public key {key}")
            _walk(child, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{location}[{index}]", errors)
    elif isinstance(value, str):
        if "/home/" in value or "/tmp/" in value or _WINDOWS_PATH.search(value):
            errors.append(f"{location}: leaked absolute path")
        if value.endswith("Analyzer") or ".Analyzer" in value:
            errors.append(f"{location}: leaked internal class name")


def validate_public_delivery(root: Path) -> None:
    errors: list[str] = []
    index_path = root / "十二项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    items = index.get("十二项结果") or {}
    if len(items) != 12:
        errors.append("twelve item index is incomplete")
    for item_id, item in items.items():
        image_fields = [item.get("主结果图"), *(item.get("附加结果图") or [])]
        for relative in image_fields:
            if not relative or not (root / relative).is_file():
                errors.append(f"{item_id}: missing indexed image {relative}")
        for key in ("量化JSON", "量化CSV", "医学V2CSV"):
            relative = item.get(key)
            if not relative or not (root / relative).is_file():
                errors.append(f"{item_id}: missing {key}")
    for item_id in ("redness", "brown"):
        item = items.get(item_id) or {}
        if 1 + len(item.get("附加结果图") or []) != 2:
            errors.append(f"{item_id}: expected exactly vendor base and final result images")
    for item_id in ("uv_spots", "porphyrin"):
        item = items.get(item_id) or {}
        if 1 + len(item.get("附加结果图") or []) < 2:
            errors.append(f"{item_id}: expected base and result images")

    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"{path.relative_to(root)}: invalid JSON: {exc}")
            continue
        _walk(payload, path.relative_to(root).as_posix(), errors)
    for path in root.rglob("*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or not rows[0]:
            errors.append(f"{path.relative_to(root)}: empty CSV")
        for row in rows:
            for value in row:
                if "/home/" in value or "/tmp/" in value or _WINDOWS_PATH.search(value):
                    errors.append(f"{path.relative_to(root)}: leaked absolute path")
    if errors:
        raise ValueError("public delivery contract failed:\n" + "\n".join(errors))


__all__ = ["validate_public_delivery"]
