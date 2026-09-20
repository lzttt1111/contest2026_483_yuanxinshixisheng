from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nine_analysis.calibration import (
    database_progress,
    ingest_feature_jsonl,
    ingest_feature_paths,
    read_feature_manifest_with_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="断点收集九项评分输入，供38万张参考分布标定")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--results-root", type=Path)
    source.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        help="run.py scoring模式产生的JSONL；可重复指定多台机器文件",
    )
    parser.add_argument("--database", type=Path, default=Path("calibration/population_metrics.sqlite3"))
    args = parser.parse_args()
    if args.jsonl:
        counters = ingest_feature_jsonl(args.jsonl, args.database)
        progress = database_progress(args.database)
        print(json.dumps({"本次": counters, "累计": progress, "数据库": str(args.database.resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.manifest:
        paths, metadata = read_feature_manifest_with_metadata(args.manifest)
    else:
        paths = sorted(args.results_root.resolve().glob("**/scoring_features.json"))
        metadata = {}
    counters = ingest_feature_paths(paths, args.database, metadata)
    progress = database_progress(args.database)
    print(json.dumps({"本次": counters, "累计": progress, "数据库": str(args.database.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
