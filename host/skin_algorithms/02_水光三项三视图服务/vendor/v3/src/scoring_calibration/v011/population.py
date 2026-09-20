from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .quality import evaluate_quality
from .registry import REGISTRY, registry_document
from .scoring import transform_value


SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY,
  content_sha256 TEXT,
  perceptual_hash TEXT,
  quality_status TEXT NOT NULL,
  reason_codes_json TEXT NOT NULL,
  algorithm_version TEXT,
  metrics_schema_version TEXT,
  source_path TEXT,
  source_line INTEGER
);
CREATE TABLE IF NOT EXISTS evidence (
  sample_id TEXT NOT NULL,
  metric_id TEXT NOT NULL,
  raw_value REAL NOT NULL,
  transformed_value REAL NOT NULL,
  PRIMARY KEY(sample_id, metric_id)
);
CREATE INDEX IF NOT EXISTS evidence_metric_value
  ON evidence(metric_id, transformed_value);
CREATE TABLE IF NOT EXISTS imports (
  source_path TEXT PRIMARY KEY,
  size_bytes INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  last_line INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS perceptual_hash_bands (
  sample_id TEXT NOT NULL,
  band INTEGER NOT NULL,
  band_value TEXT NOT NULL,
  PRIMARY KEY(sample_id, band)
);
CREATE INDEX IF NOT EXISTS perceptual_band_lookup
  ON perceptual_hash_bands(band, band_value);
"""


def _path(document: dict[str, Any], dotted: str) -> Any:
    current: Any = document
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _sample_id(observation: dict[str, Any]) -> str:
    signature = observation.get("signature") or {}
    content_hash = observation.get("content_sha256") or signature.get("sha256")
    if content_hash:
        return str(content_hash)
    canonical = json.dumps(
        {"relative_path": signature.get("relative_path"), "size": signature.get("size"), "mtime_ns": signature.get("mtime_ns")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(SCHEMA)
    return connection


def _perceptual_duplicate(connection: sqlite3.Connection, perceptual_hash: str | None) -> str | None:
    if not perceptual_hash or len(perceptual_hash) != 16:
        return None
    candidates: set[str] = set()
    for band in range(4):
        value = perceptual_hash[band * 4:(band + 1) * 4]
        candidates.update(row[0] for row in connection.execute(
            "SELECT sample_id FROM perceptual_hash_bands WHERE band=? AND band_value=?", (band, value)
        ))
    target = int(perceptual_hash, 16)
    for sample_id in candidates:
        row = connection.execute("SELECT perceptual_hash FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        if row and row[0] and (target ^ int(row[0], 16)).bit_count() <= 4:
            return sample_id
    return None


def _insert_perceptual_bands(connection: sqlite3.Connection, sample_id: str, perceptual_hash: str | None) -> None:
    if not perceptual_hash or len(perceptual_hash) != 16:
        return
    connection.executemany(
        "INSERT OR IGNORE INTO perceptual_hash_bands VALUES (?,?,?)",
        [(sample_id, band, perceptual_hash[band * 4:(band + 1) * 4]) for band in range(4)],
    )


def ingest_jsonl(
    jsonl_paths: Iterable[Path],
    database: Path,
    *,
    workspace_limit_bytes: int = 10 * 1024 ** 3,
) -> dict[str, int]:
    """流式、可断点、只追加地收集评分证据，不修改任何原始文件。"""
    connection = initialize_database(database)
    counters = {"seen": 0, "inserted": 0, "duplicate": 0, "excluded": 0, "invalid": 0}
    try:
        for source in sorted(Path(item).resolve() for item in jsonl_paths):
            stat = source.stat()
            last_line = 0
            previous = connection.execute(
                "SELECT size_bytes,mtime_ns,last_line FROM imports WHERE source_path=?", (str(source),)
            ).fetchone()
            start_line = int(previous[2]) if previous and previous[:2] == (stat.st_size, stat.st_mtime_ns) else 0
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    last_line = line_number
                    if line_number <= start_line or not line.strip():
                        continue
                    counters["seen"] += 1
                    try:
                        observation = json.loads(line)
                        features = observation.get("features") or observation.get("scoring_features")
                        if not isinstance(features, dict):
                            raise ValueError("missing features")
                    except (json.JSONDecodeError, ValueError):
                        counters["invalid"] += 1
                        continue
                    gate = evaluate_quality(observation)
                    sample_id = _sample_id(observation)
                    if connection.execute("SELECT 1 FROM samples WHERE sample_id=?", (sample_id,)).fetchone():
                        counters["duplicate"] += 1
                        continue
                    perceptual_hash = observation.get("perceptual_hash")
                    if _perceptual_duplicate(connection, perceptual_hash):
                        counters["duplicate"] += 1
                        continue
                    connection.execute(
                        "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            sample_id, observation.get("content_sha256"), perceptual_hash,
                            gate["status"], json.dumps(gate["reason_codes"], ensure_ascii=False),
                            observation.get("algorithm_version"), observation.get("metrics_schema_version"),
                            str(source), line_number,
                        ),
                    )
                    _insert_perceptual_bands(connection, sample_id, perceptual_hash)
                    if gate["status"] != "PASS":
                        counters["excluded"] += 1
                    else:
                        rows = []
                        for dimension in REGISTRY:
                            for group in dimension.groups:
                                for metric in group.metrics:
                                    value = _path(features, metric.source)
                                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                                        continue
                                    value = float(value)
                                    if not math.isfinite(value):
                                        continue
                                    rows.append((sample_id, metric.id, value, transform_value(value, metric.transform)))
                        connection.executemany("INSERT INTO evidence VALUES (?,?,?,?)", rows)
                    counters["inserted"] += 1
                    if counters["seen"] % 500 == 0:
                        connection.commit()
                        if database.stat().st_size > workspace_limit_bytes:
                            raise RuntimeError("评分工作库超过10GiB限制，已安全停止且保留断点")
            connection.execute(
                "INSERT OR REPLACE INTO imports VALUES (?,?,?,?)",
                (str(source), stat.st_size, stat.st_mtime_ns, last_line),
            )
            connection.commit()
    finally:
        connection.close()
    return counters


def build_exact_profile(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    try:
        references: dict[str, list[float]] = {}
        effective_counts: dict[str, int] = {}
        for dimension in REGISTRY:
            for group in dimension.groups:
                for metric in group.metrics:
                    values = [float(row[0]) for row in connection.execute(
                        "SELECT transformed_value FROM evidence WHERE metric_id=? ORDER BY transformed_value", (metric.id,)
                    )]
                    references[metric.id] = values
                    effective_counts[metric.id] = len(values)
        sample_counts = dict(connection.execute(
            "SELECT quality_status,COUNT(*) FROM samples GROUP BY quality_status"
        ).fetchall())
    finally:
        connection.close()
    return {
        **registry_document(),
        "reference_storage": "full_empirical_cdf_sorted_values",
        "reference_sample_counts": sample_counts,
        "effective_metric_counts": effective_counts,
        "references": references,
    }
