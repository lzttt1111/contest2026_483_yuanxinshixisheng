from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nine_analysis.calibration import build_profile, build_profile_from_database, read_feature_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="从九项评分输入构建AISIA参考人群候选常模")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="含 scoring_features_path 列的CSV")
    source.add_argument("--results-root", type=Path, help="递归查找 scoring_features.json 的结果目录")
    source.add_argument("--database", type=Path, help="scripts/scoring/update_score_statistics.py 生成的断点统计数据库")
    parser.add_argument("--output", type=Path, default=Path("calibration/profiles/reference_v1.json"))
    parser.add_argument("--minimum-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.database:
        profile = build_profile_from_database(args.database, args.output, args.minimum_samples)
        print(json.dumps({
            "状态": profile["status"], "输入样本": profile["input_files"],
            "最小指标样本数": profile["minimum_metric_samples"], "输出": str(args.output.resolve()),
        }, ensure_ascii=False, indent=2))
        return 0
    paths = read_feature_manifest(args.manifest) if args.manifest else sorted(args.results_root.resolve().glob("**/scoring_features.json"))
    if not paths:
        raise SystemExit("清单中没有可读取的 scoring_features.json")
    profile = build_profile(paths, args.output, args.minimum_samples)
    print(json.dumps({
        "状态": profile["status"], "输入样本": profile["input_files"],
        "最小指标样本数": profile["minimum_metric_samples"], "输出": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
