from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path


def prepare_mapping_database(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS legacy_samples (
                sample_id TEXT PRIMARY KEY,
                result_path TEXT NOT NULL,
                contract_version TEXT NOT NULL,
                signature_json TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS legacy_metric_values (
                sample_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                metric_path TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (sample_id, item_id, metric_path),
                FOREIGN KEY (sample_id) REFERENCES legacy_samples(sample_id)
            );
            CREATE TABLE IF NOT EXISTS mapping_pairs (
                image_sha256 TEXT PRIMARY KEY,
                legacy_sample_id TEXT NOT NULL,
                new_sample_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'prepared', 'fitted', 'validated', 'non_convertible'
                )),
                FOREIGN KEY (legacy_sample_id) REFERENCES legacy_samples(sample_id)
            );
            CREATE TABLE IF NOT EXISTS mapping_models (
                module_id TEXT NOT NULL,
                source_contract TEXT NOT NULL,
                target_contract TEXT NOT NULL,
                model_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'prepared', 'fitted', 'validated', 'non_convertible'
                )),
                PRIMARY KEY (module_id, source_contract, target_contract)
            );
        """)


def _flatten(value: object, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, bool):
                continue
            if isinstance(child, (int, float)) and math.isfinite(float(child)):
                rows.append((path, float(child)))
            else:
                rows.extend(_flatten(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_flatten(child, f"{prefix}[{index}]"))
    return rows


def import_legacy_result(result_root: Path, database: Path) -> dict[str, object]:
    metrics_path = result_root / "九项核心量化指标.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    signature_path = result_root / "批处理元数据.json"
    signature = (
        json.loads(signature_path.read_text(encoding="utf-8"))
        .get("signature", {})
        if signature_path.is_file()
        else {"result_name": result_root.name}
    )
    contract_version = str(document.get("指标版本") or "legacy_nine_unknown")
    identity = json.dumps(
        {"signature": signature, "contract_version": contract_version},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sample_id = hashlib.sha256(identity).hexdigest()
    metric_rows = [
        (sample_id, str(item_id), metric_path, value)
        for item_id, item in (document.get("九项") or {}).items()
        for metric_path, value in _flatten(item)
    ]
    prepare_mapping_database(database)
    with sqlite3.connect(database) as connection:
        if connection.execute(
            "SELECT 1 FROM legacy_samples WHERE sample_id=?", (sample_id,)
        ).fetchone():
            return {
                "status": "duplicate",
                "sample_id": sample_id,
                "metric_count": len(metric_rows),
            }
        connection.execute(
            "INSERT INTO legacy_samples VALUES (?, ?, ?, ?, ?)",
            (
                sample_id,
                str(result_root.resolve()),
                contract_version,
                json.dumps(signature, ensure_ascii=False, sort_keys=True),
                datetime.now().astimezone().isoformat(),
            ),
        )
        connection.executemany(
            "INSERT INTO legacy_metric_values VALUES (?, ?, ?, ?)", metric_rows
        )
    return {
        "status": "imported",
        "sample_id": sample_id,
        "metric_count": len(metric_rows),
    }


def register_mapping_pair(
    database: Path,
    *,
    image_sha256: str,
    legacy_sample_id: str,
    new_sample_id: str,
) -> None:
    if len(image_sha256) != 64:
        raise ValueError("image_sha256 must contain 64 hexadecimal characters")
    prepare_mapping_database(database)
    with sqlite3.connect(database) as connection:
        if not connection.execute(
            "SELECT 1 FROM legacy_samples WHERE sample_id=?", (legacy_sample_id,)
        ).fetchone():
            raise KeyError(legacy_sample_id)
        connection.execute(
            "INSERT OR REPLACE INTO mapping_pairs VALUES (?, ?, ?, 'prepared')",
            (image_sha256, legacy_sample_id, new_sample_id),
        )


__all__ = [
    "import_legacy_result",
    "prepare_mapping_database",
    "register_mapping_pair",
]
