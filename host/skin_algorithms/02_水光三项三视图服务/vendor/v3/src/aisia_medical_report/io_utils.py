from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须为对象: {file_path}")
    return value


def existing_files(paths: list[str | Path]) -> list[str]:
    result: list[str] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            result.append(str(path))
    return result


def value(data: dict[str, Any], key: str, default: Any = "不可评估") -> Any:
    current = data.get(key, default)
    if current in (None, ""):
        return default
    return current


def module_regions(raw_regions: Any, keys: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw_regions, list):
        return []
    output: list[dict[str, Any]] = []
    for row in raw_regions:
        if not isinstance(row, dict):
            continue
        output.append({k: row.get(k, "") for k in keys})
    return output
