from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


QUANTILES = {
    "q005": 0.005, "q10": 0.10, "q25": 0.25, "q50": 0.50,
    "q75": 0.75, "q90": 0.90, "q95": 0.95, "q995": 0.995,
}


def transform_name(metric: str) -> str:
    if "占比" in metric or "比例" in metric or "强度" in metric or "置信度" in metric or "响应" in metric:
        return "identity"
    if any(token in metric for token in ("数量", "密度", "面积", "长度", "像素", "聚集")):
        return "log1p"
    return "identity"


def transform_value(value: float, transform: str) -> float:
    return math.log1p(max(0.0, value)) if transform == "log1p" else value


def iter_feature_rows(paths: Iterable[Path]) -> Iterable[tuple[str, str, str, float, str]]:
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from iter_document_rows(data)


def iter_document_rows(data: dict[str, Any]) -> Iterable[tuple[str, str, str, float, str]]:
    for item, groups in (data.get("九项评分输入") or {}).items():
        for group, metrics in (groups or {}).items():
            for metric, raw in (metrics or {}).items():
                if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                    continue
                transform = transform_name(metric)
                yield item, group, metric, transform_value(float(raw), transform), transform


def _init_incremental_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS source_files (
            content_sha256 TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            subject_id TEXT,
            device_id TEXT,
            age_group TEXT,
            sex TEXT,
            skin_type TEXT,
            quality_status TEXT,
            imported_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS metric_values (
            content_sha256 TEXT NOT NULL,
            item TEXT NOT NULL,
            group_name TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            transform TEXT NOT NULL,
            PRIMARY KEY (content_sha256, item, group_name, metric),
            FOREIGN KEY (content_sha256) REFERENCES source_files(content_sha256)
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_metric_key ON metric_values(item, group_name, metric)")


def ingest_feature_paths(
    feature_paths: Iterable[Path],
    database: Path,
    metadata_by_path: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Idempotently collect scoring inputs for long-running population jobs."""
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    counters = {"seen": 0, "imported": 0, "duplicates": 0, "invalid": 0, "metrics": 0}
    metadata_by_path = metadata_by_path or {}
    try:
        _init_incremental_database(connection)
        for raw_path in feature_paths:
            counters["seen"] += 1
            path = raw_path.resolve()
            try:
                payload = path.read_bytes()
                data = json.loads(payload.decode("utf-8"))
                rows = list(iter_document_rows(data))
                if not rows:
                    raise ValueError("没有有效评分指标")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                counters["invalid"] += 1
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if connection.execute(
                "SELECT 1 FROM source_files WHERE content_sha256=?", (digest,)
            ).fetchone():
                counters["duplicates"] += 1
                continue
            stat = path.stat()
            meta = metadata_by_path.get(str(path), metadata_by_path.get(str(raw_path), {}))
            connection.execute(
                "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    digest, str(path), stat.st_size, stat.st_mtime_ns,
                    meta.get("subject_id"), meta.get("device_id"), meta.get("age_group"),
                    meta.get("sex"), meta.get("skin_type"), meta.get("quality_status"),
                    datetime.now().astimezone().isoformat(),
                ),
            )
            connection.executemany(
                "INSERT INTO metric_values VALUES (?, ?, ?, ?, ?, ?)",
                [(digest, *row) for row in rows],
            )
            counters["imported"] += 1
            counters["metrics"] += len(rows)
            if counters["imported"] % 1000 == 0:
                connection.commit()
        connection.commit()
        return counters
    finally:
        connection.close()


def ingest_feature_jsonl(
    jsonl_paths: Iterable[Path],
    database: Path,
) -> dict[str, int]:
    """流式导入 ``run.py --output-profile scoring`` 的逐图观测。

    一行对应一张图片，不把整份 JSONL 或全部指标加载到内存。内容摘要
    用作幂等键；重复执行相同文件不会重复计入常模。
    """

    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    counters = {
        "seen": 0,
        "imported": 0,
        "duplicates": 0,
        "invalid": 0,
        "metrics": 0,
    }
    try:
        _init_incremental_database(connection)
        for raw_path in jsonl_paths:
            path = raw_path.resolve()
            try:
                handle = path.open("r", encoding="utf-8")
            except OSError:
                counters["invalid"] += 1
                continue
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    counters["seen"] += 1
                    try:
                        observation = json.loads(line)
                        scoring_features = observation.get("scoring_features") or {}
                        rows = list(iter_document_rows({
                            "九项评分输入": scoring_features,
                        }))
                        if not rows:
                            raise ValueError("没有有效评分指标")
                        canonical = json.dumps(
                            {
                                "signature": observation.get("signature"),
                                "scoring_features": scoring_features,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ):
                        counters["invalid"] += 1
                        continue
                    digest = hashlib.sha256(canonical).hexdigest()
                    if connection.execute(
                        "SELECT 1 FROM source_files WHERE content_sha256=?",
                        (digest,),
                    ).fetchone():
                        counters["duplicates"] += 1
                        continue
                    signature = observation.get("signature") or {}
                    quality = observation.get("quality") or {}
                    source_name = (
                        f"{path}#{line_number}:"
                        f"{signature.get('relative_path', 'unknown')}"
                    )
                    connection.execute(
                        "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            digest,
                            source_name,
                            len(line.encode("utf-8")),
                            0,
                            None,
                            None,
                            None,
                            None,
                            None,
                            quality.get("status"),
                            datetime.now().astimezone().isoformat(),
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO metric_values VALUES (?, ?, ?, ?, ?, ?)",
                        [(digest, *row) for row in rows],
                    )
                    counters["imported"] += 1
                    counters["metrics"] += len(rows)
                    if counters["imported"] % 1000 == 0:
                        connection.commit()
        connection.commit()
        return counters
    finally:
        connection.close()


def database_progress(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    try:
        samples = int(connection.execute("SELECT COUNT(*) FROM source_files").fetchone()[0])
        metrics = int(connection.execute("SELECT COUNT(*) FROM metric_values").fetchone()[0])
        by_quality = dict(connection.execute(
            "SELECT COALESCE(quality_status, 'unknown'), COUNT(*) FROM source_files GROUP BY quality_status"
        ).fetchall())
        by_device = dict(connection.execute(
            "SELECT COALESCE(device_id, 'unknown'), COUNT(*) FROM source_files GROUP BY device_id"
        ).fetchall())
        return {"samples": samples, "metric_values": metrics, "by_quality": by_quality, "by_device": by_device}
    finally:
        connection.close()


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    pos = (len(sorted_values) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def build_profile(feature_paths: list[Path], output: Path, minimum_samples: int = 1000) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    database = output.with_suffix(".sqlite3")
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE values_table (item TEXT, group_name TEXT, metric TEXT, value REAL, transform TEXT)")
        batch = []
        for row in iter_feature_rows(feature_paths):
            batch.append(row)
            if len(batch) >= 10000:
                connection.executemany("INSERT INTO values_table VALUES (?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO values_table VALUES (?, ?, ?, ?, ?)", batch)
        connection.commit()
        keys = connection.execute(
            "SELECT item, group_name, metric, transform, COUNT(*) FROM values_table GROUP BY item, group_name, metric, transform"
        ).fetchall()
        metrics: dict[str, Any] = {}
        minimum_count = min((int(row[4]) for row in keys), default=0)
        for item, group, metric, transform, count in keys:
            values = [float(row[0]) for row in connection.execute(
                "SELECT value FROM values_table WHERE item=? AND group_name=? AND metric=? ORDER BY value",
                (item, group, metric),
            )]
            entry = {
                "count": int(count),
                "transform": transform,
                "min": values[0],
                "max": values[-1],
                "usable": values[-1] - values[0] > 1e-12,
            }
            entry.update({name: _quantile(values, q) for name, q in QUANTILES.items()})
            metrics.setdefault(item, {}).setdefault(group, {})[metric] = entry
        calibrated = bool(keys) and minimum_count >= minimum_samples
        profile = {
            "profile_version": "aisia_reference_distribution_v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "calibrated" if calibrated else "insufficient_samples",
            "reason": "" if calibrated else f"至少一个指标有效样本数低于 {minimum_samples}",
            "input_files": len(feature_paths),
            "minimum_metric_samples": minimum_count,
            "required_minimum_samples": minimum_samples,
            "winsor_limits": [0.005, 0.995],
            "metrics": metrics,
            "medical_boundary": "AISIA内部参考人群相对分布，不代表医学正常值。",
        }
        output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile
    finally:
        connection.close()


def build_profile_from_database(database: Path, output: Path, minimum_samples: int = 1000) -> dict[str, Any]:
    """Finalize exact reference quantiles from the resumable collection DB."""
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        _init_incremental_database(connection)
        keys = connection.execute(
            "SELECT item, group_name, metric, transform, COUNT(*) FROM metric_values "
            "GROUP BY item, group_name, metric, transform"
        ).fetchall()
        metrics: dict[str, Any] = {}
        minimum_count = min((int(row[4]) for row in keys), default=0)
        for item, group, metric, transform, count in keys:
            values = [float(row[0]) for row in connection.execute(
                "SELECT value FROM metric_values WHERE item=? AND group_name=? AND metric=? ORDER BY value",
                (item, group, metric),
            )]
            entry = {
                "count": int(count),
                "transform": transform,
                "min": values[0],
                "max": values[-1],
                "usable": values[-1] - values[0] > 1e-12,
            }
            entry.update({name: _quantile(values, q) for name, q in QUANTILES.items()})
            metrics.setdefault(item, {}).setdefault(group, {})[metric] = entry
        sample_count = int(connection.execute("SELECT COUNT(*) FROM source_files").fetchone()[0])
        calibrated = bool(keys) and minimum_count >= minimum_samples
        profile = {
            "profile_version": "aisia_reference_distribution_v2_20260728",
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "candidate" if calibrated else "insufficient_samples",
            "reason": "等待医生盲评和分层验证" if calibrated else f"至少一个指标有效样本数低于 {minimum_samples}",
            "input_files": sample_count,
            "minimum_metric_samples": minimum_count,
            "required_minimum_samples": minimum_samples,
            "winsor_limits": [0.005, 0.995],
            "metrics": metrics,
            "medical_boundary": "AISIA内部参考人群相对分布，不代表医学正常值；医生审核前不得用于正式等级。",
        }
        output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile
    finally:
        connection.close()


def read_feature_manifest(path: Path) -> list[Path]:
    rows = csv.DictReader(path.open(encoding="utf-8-sig"))
    result = []
    for row in rows:
        raw = row.get("scoring_features_path") or row.get("评分输入路径")
        if raw and Path(raw).is_file():
            result.append(Path(raw).resolve())
    return result


def read_feature_manifest_with_metadata(path: Path) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    rows = csv.DictReader(path.open(encoding="utf-8-sig"))
    paths: list[Path] = []
    metadata: dict[str, dict[str, Any]] = {}
    aliases = {
        "subject_id": ("subject_id", "受检者ID"),
        "device_id": ("device_id", "设备ID"),
        "age_group": ("age_group", "年龄组"),
        "sex": ("sex", "性别"),
        "skin_type": ("skin_type", "肤质"),
        "quality_status": ("quality_status", "质量状态"),
    }
    for row in rows:
        raw = row.get("scoring_features_path") or row.get("评分输入路径")
        if not raw or not Path(raw).is_file():
            continue
        resolved = Path(raw).resolve()
        paths.append(resolved)
        metadata[str(resolved)] = {
            target: next((row.get(name) for name in names if row.get(name)), None)
            for target, names in aliases.items()
        }
    return paths, metadata
