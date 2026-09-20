from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.acceptance.verify_redness_brown_forehead import verify


def _write(path: Path, area: int, count: int, count_column: str) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["检测范围", "有效皮肤面积（像素）", count_column],
        )
        writer.writeheader()
        writer.writerow({"检测范围": "额头", "有效皮肤面积（像素）": area, count_column: count})


def test_forehead_anchor_requires_noncollapsed_redness_and_brown(tmp_path: Path) -> None:
    anchor = Path(__file__).parent / "fixtures" / "redness_brown_forehead_anchor_clinic28_20260831.json"
    contract = json.loads(anchor.read_text(encoding="utf-8"))
    for case, minimum in contract["release_minimums"].items():
        sample = tmp_path / case / "十二项检测"
        _write(
            sample / "01_红区" / "红区医学量化指标_V2.csv",
            minimum["redness_area"],
            minimum["redness_count"],
            "核心-局灶红色实例数量（个）",
        )
        _write(
            sample / "03_棕区" / "棕区医学量化指标_V2.csv",
            minimum["brown_area"],
            minimum["brown_count"],
            "辅助-特征数量（个）",
        )

    result = verify(tmp_path, anchor)

    assert result["status"] == "passed"
    assert set(result["cases"]) == {"clinic28-25", "clinic28-09", "clinic28-23"}
