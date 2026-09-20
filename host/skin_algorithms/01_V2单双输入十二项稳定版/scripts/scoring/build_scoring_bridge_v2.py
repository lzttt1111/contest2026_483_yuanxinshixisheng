from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_bridge.builder import (
    build_paired_score_rows,
    fit_dimension_bridges,
    load_legacy_pairs,
    load_latest_observations,
)
from src.scoring_bridge.fitting import build_ecdf_profile


SOURCE_PROFILE = "legacy_nine_full_face_v011"
TARGET_PROFILE = "consumer_rgb_twelve_v2_20260826"
DIMENSION_NAMES = {
    "visible_pores": "可见毛孔",
    "combined_pigmentation": "综合色素",
    "diffuse_redness": "弥漫性泛红",
    "surface_smoothness_decline": "表面平整度下降",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "relative_path", "subject_id", "split", "dimension_id",
        "legacy_score", "current_score", "mapped_score", "raw_error",
        "mapped_error", "legacy_grade", "current_grade", "mapped_grade",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(profile: dict, rejected: list[dict]) -> str:
    diagnostics = profile["validation"]
    lines = [
        "# DermaVision 旧九项到十二项评分Bridge验收",
        "",
        f"- 状态：`{diagnostics['status']}`",
        f"- 自动发布门：`{diagnostics['promotion_gate']}`",
        f"- 可入组样本：{profile['successful_observation_count']}",
        "- 分组：800张映射拟合组 + 200张独立验证组。",
        "- 说明：这里不训练检测模型，只校准新旧评分刻度。",
        f"- 当前无法成对样本：{profile['unavailable_pair_count']}",
        f"- 全运行历史QC拒绝/失败记录：{len(rejected)}",
        "- 分数方向：越高表示问题负担越明显。",
        "- 医学边界：本配置为1000张工程初版映射，不代表临床校准。",
        "",
        "|维度|独立验证数|映射前MAE|映射后MAE|Spearman|严重反转|结论|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dimension, splits in diagnostics["dimensions"].items():
        validation = splits.get("validation") or {}
        before = validation.get("before_mapping") or {}
        after = validation.get("after_mapping") or {}
        before_mae = before.get("mean_absolute_error")
        after_mae = after.get("mean_absolute_error")
        reversals = int(after.get("severe_reversal_count", 0))
        improved = (
            isinstance(before_mae, (int, float))
            and isinstance(after_mae, (int, float))
            and after_mae <= before_mae
        )
        conclusion = "MAE改善" if improved else "MAE变差，保留原刻度"
        if reversals:
            conclusion += f"；{reversals}个严重排序反转"
        lines.append(
            f"|{DIMENSION_NAMES.get(dimension, dimension)}|{after.get('count', 0)}|"
            f"{before_mae if before_mae is not None else '—'}|"
            f"{after_mae if after_mae is not None else '—'}|"
            f"{after.get('spearman', '—')}|"
            f"{reversals}|{conclusion}|"
        )
    lines.extend([
        "",
        "## 发布结论",
        "",
        "- `promotion_gate=blocked`：不得自动替换现有正式评分配置。",
        "- 映射改善只证明尺度漂移可部分修正；排序反转需后续算法和分层标定。",
        "- 现阶段保留为候选证据，继续使用已批准的旧评分代理。",
    ])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成旧九项到consumer十二项的成对单调评分Bridge")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--legacy-pairs", type=Path, required=True)
    parser.add_argument(
        "--official-profile",
        type=Path,
        default=PROJECT_ROOT / "00_runtime_assets/scoring_v011/全量历史ECDF评分配置.json",
    )
    parser.add_argument("--extra-jsonl", type=Path, action="append", default=[])
    parser.add_argument(
        "--active-pairs",
        type=Path,
        help="只使用该激活成对清单中的样本，原始1200清单仍用于追溯SHA",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-validation", type=int, default=150)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    job_dir = args.job_dir.resolve(strict=True)
    pairs_path = args.legacy_pairs.resolve(strict=True)
    official_path = args.official_profile.resolve(strict=True)
    jsonl_paths = sorted(job_dir.glob("scoring_features.*.jsonl"))
    jsonl_paths.extend(path.resolve(strict=True) for path in args.extra_jsonl)
    observations, rejected = load_latest_observations(jsonl_paths)
    official = json.loads(official_path.read_text(encoding="utf-8"))
    pairs_path_for_build = (
        args.active_pairs.resolve(strict=True)
        if args.active_pairs is not None
        else pairs_path
    )
    pair_rows = load_legacy_pairs(pairs_path_for_build)
    if args.active_pairs is not None:
        active_paths = {
            str(row["source_relative_path"])
            for row in pair_rows
        }
        observations = {
            relative_path: observation
            for relative_path, observation in observations.items()
            if relative_path in active_paths
        }
    paired, unavailable = build_paired_score_rows(
        observations,
        pair_rows,
        official["references"],
    )
    mappings, evaluated, diagnostics = fit_dimension_bridges(
        paired,
        minimum_validation=args.minimum_validation,
    )
    profile = {
        "schema_version": "aisia_scoring_bridge_v2_candidate_1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_profile": SOURCE_PROFILE,
        "target_profile": TARGET_PROFILE,
        "status": diagnostics["status"],
        "score_direction": "higher_is_more_burden",
        "official_v011_profile_sha256": _sha(official_path),
        "legacy_pairs_sha256": _sha(pairs_path),
        "active_pairs_sha256": (
            _sha(pairs_path_for_build)
            if args.active_pairs is not None
            else None
        ),
        "observation_jsonl": [str(path) for path in jsonl_paths],
        "successful_observation_count": len(observations),
        "unavailable_pair_count": len(unavailable),
        "mappings": mappings,
        "validation": diagnostics,
        "bootstrap_1000_v1": build_ecdf_profile([
            observation.get("v2_scoring_features") or {}
            for observation in observations.values()
        ]),
        "institution_profile_alias": {
            "status": "temporary",
            "profile": TARGET_PROFILE,
        },
        "medical_boundary": "工程初版映射；医生审核和完整人群重标定前不得宣称临床校准。",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "consumer_rgb_twelve_v2_scoring_bridge_candidate.json"
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "old_vs_new_mapping.csv", evaluated)
    (output / "unavailable_pairs.json").write_text(
        json.dumps(unavailable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "qc_rejections.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "评分Bridge中文验收.md").write_text(
        _markdown(profile, rejected),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": profile["status"],
        "successful_observations": len(observations),
        "paired_rows": len(evaluated),
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if profile["status"] == "candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
