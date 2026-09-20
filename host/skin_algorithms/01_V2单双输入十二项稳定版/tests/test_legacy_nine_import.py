from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.nine_analysis.legacy_import import (
    import_legacy_result,
    prepare_mapping_database,
    register_mapping_pair,
)


def test_legacy_import_is_versioned_idempotent_and_pair_ready(tmp_path: Path) -> None:
    result = tmp_path / "legacy-sample"
    result.mkdir()
    (result / "九项核心量化指标.json").write_text(
        json.dumps({
            "指标版本": "nine_metrics_compact_v2",
            "九项": {
                "pores": {
                    "核心总体指标": {
                        "单位面积密度（个/10万有效皮肤像素）": 123.4,
                        "特征面积占比": 0.04,
                    }
                }
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (result / "批处理元数据.json").write_text(
        json.dumps({
            "signature": {
                "relative_path": "clinic28-25_RGB_M.jpg",
                "size": 10,
                "mtime_ns": 20,
            }
        }),
        encoding="utf-8",
    )
    database = tmp_path / "mapping.sqlite3"
    prepare_mapping_database(database)

    first = import_legacy_result(result, database)
    second = import_legacy_result(result, database)
    register_mapping_pair(
        database,
        image_sha256="a" * 64,
        legacy_sample_id=first["sample_id"],
        new_sample_id="new-sample",
    )

    assert first["status"] == "imported"
    assert first["metric_count"] == 2
    assert second["status"] == "duplicate"
    with sqlite3.connect(database) as connection:
        assert connection.execute("select count(*) from legacy_samples").fetchone()[0] == 1
        assert connection.execute("select count(*) from legacy_metric_values").fetchone()[0] == 2
        pair = connection.execute(
            "select image_sha256, status from mapping_pairs"
        ).fetchone()
    assert pair == ("a" * 64, "prepared")
