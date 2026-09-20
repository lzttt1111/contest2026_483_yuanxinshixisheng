from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.engines.rbx_engine import ErythemaAnalyzer
from src.engines.spots_engine import SpotsEngine
from src.engines.visia_regions import compact_region_counts


class MetricsContractTest(unittest.TestCase):
    """锁定云端 CSV/JSON 共用的精简量化口径。"""

    @staticmethod
    def _read_csv(path: Path) -> dict[str, int]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) != 2:
            raise AssertionError(f"CSV 必须严格为两行: {rows}")
        return {
            key: int(value)
            for key, value in zip(rows[0], rows[1])
        }

    def test_redness_full_face_compact_metrics_match_csv(self) -> None:
        metrics = {
            "red_feature_count": 21,
            "quality_flags": [],
            "red_feature_region_distribution": {
                "forehead": {"count": 1},
                "left_cheek": {"count": 2},
                "right_cheek": {"count": 3},
                "nose": {"count": 4},
                "chin": {"count": 11},
            },
        }
        compact = ErythemaAnalyzer._compact_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "红区量化指标.csv"
            ErythemaAnalyzer._write_metrics_csv(compact, path)
            self.assertEqual(self._read_csv(path), compact)

    def test_redness_partial_face_only_returns_total(self) -> None:
        compact = ErythemaAnalyzer._compact_metrics(
            {
                "red_feature_count": 9,
                "quality_flags": ["PARTIAL_FACE"],
                "red_feature_region_distribution": {},
            }
        )
        self.assertEqual(compact, {"总计": 9})

    def test_spots_full_face_compact_metrics_match_csv(self) -> None:
        result = SimpleNamespace(
            spot_count=15,
            region_distribution={
                "forehead": {"count": 1},
                "left_cheek": {"count": 2},
                "right_cheek": {"count": 3},
                "nose": {"count": 4},
                "chin": {"count": 5},
            },
        )
        compact = SpotsEngine._compact_metrics(result, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "02_Spots量化指标.csv"
            SpotsEngine._write_metrics_csv(compact, str(path))
            self.assertEqual(self._read_csv(path), compact)

    def test_spots_partial_face_only_returns_total(self) -> None:
        result = SimpleNamespace(
            spot_count=8,
            region_distribution={},
        )
        compact = SpotsEngine._compact_metrics(
            result,
            ["PARTIAL_FACE"],
        )
        self.assertEqual(compact, {"总计": 8})

    def test_shared_brown_texture_pores_region_labels_are_chinese(self) -> None:
        locations = [
            {"region": "forehead"},
            {"region": "left_cheek"},
            {"region": "right_cheek"},
            {"region": "nose"},
            {"region": "chin"},
        ]
        headers, values = compact_region_counts(locations, partial_face=False)
        self.assertEqual(
            headers,
            ["总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"],
        )
        self.assertEqual(values, [5, 1, 1, 1, 1, 1])

    def test_shared_partial_face_metrics_only_return_chinese_total(self) -> None:
        headers, values = compact_region_counts(
            [{"region": "left_cheek"}, {"region": "nose"}],
            partial_face=True,
        )
        self.assertEqual(headers, ["总计"])
        self.assertEqual(values, [2])


if __name__ == "__main__":
    unittest.main()
