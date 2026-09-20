from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nine_analysis.legacy_import import import_legacy_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将历史九项结果导入新旧映射数据库；不运行图片算法"
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    roots = sorted(
        path.parent
        for path in args.results_root.resolve().glob("**/九项核心量化指标.json")
    )
    counters = {"seen": 0, "imported": 0, "duplicates": 0, "invalid": 0}
    for root in roots:
        counters["seen"] += 1
        try:
            result = import_legacy_result(root, args.database)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            counters["invalid"] += 1
            continue
        counters[result["status"]] += 1
    print(json.dumps({"counters": counters, "database": str(args.database.resolve())}, ensure_ascii=False, indent=2))
    return 0 if counters["invalid"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
