from __future__ import annotations

import csv
import json

from src.detection_runtime.final_delivery_layout import (
    materialize_consumer_scoring_artifacts,
)


def test_materialize_consumer_scoring_artifacts_keeps_core_json_and_csv(tmp_path) -> None:
    complete = {
        "schema_version": "complete",
        "detector_results": {
            "redness": {
                "status": "success",
                "public_metrics": {"area_ratio": 0.25, "nested": {"p90": 0.8}},
            },
            "pores": {
                "status": "success",
                "public_metrics": {"count": 12},
            },
        },
    }
    (tmp_path / "十二项完整量化指标.json").write_text(
        json.dumps(complete),
        encoding="utf-8",
    )

    materialize_consumer_scoring_artifacts(tmp_path)

    compact = json.loads((tmp_path / "十二项核心量化指标.json").read_text(encoding="utf-8"))
    assert compact["十二项"]["redness"]["nested"]["p90"] == 0.8
    with (tmp_path / "十二项核心量化指标.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["指标路径"] for row in rows} == {
        "area_ratio",
        "nested.p90",
        "count",
    }
