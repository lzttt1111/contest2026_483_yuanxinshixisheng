from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _forehead(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    matches = [row for row in rows if row.get("检测范围") == "额头"]
    if len(matches) != 1:
        raise AssertionError(f"forehead row count: {path}: {len(matches)}")
    return matches[0]


def verify(root: Path, anchor: Path) -> dict[str, object]:
    contract = json.loads(anchor.read_text(encoding="utf-8"))
    results = {}
    for case, minimum in contract["release_minimums"].items():
        sample = root / case / "十二项检测"
        red = _forehead(sample / "01_红区" / "红区医学量化指标_V2.csv")
        brown = _forehead(sample / "03_棕区" / "棕区医学量化指标_V2.csv")
        observed = {
            "redness_area": int(float(red["有效皮肤面积（像素）"])),
            "redness_count": int(float(red["核心-局灶红色实例数量（个）"])),
            "brown_area": int(float(brown["有效皮肤面积（像素）"])),
            "brown_count": int(float(brown["辅助-特征数量（个）"])),
        }
        for key, expected in minimum.items():
            if observed[key] < int(expected):
                raise AssertionError(
                    f"{case}/{key}: observed={observed[key]} minimum={expected}"
                )
        results[case] = observed
    return {"status": "passed", "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.anchor), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
