"""Export human-readable acne detection reports from existing testout outputs.

Usage:
  .venv/bin/python scripts/export_detection_txt_report.py \
    --testout-dir testout --output-dir testout/text_reports

  .venv/bin/python scripts/export_detection_txt_report.py \
    --testout-dir testout --output-dir testout/text_reports --image 11.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acne.user_detection_report import build_user_report_from_paths


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_items(testout_dir: Path, single_image: str | None) -> list[tuple[Path, str]]:
    batch_summary_path = testout_dir / "batch_summary.json"
    if batch_summary_path.exists():
        payload = _load_json(batch_summary_path)
        results = payload.get("results", [])
        folders: list[tuple[Path, str]] = []
        for item in results:
            image = str(item.get("input_image", "")).strip()
            output_dir = str(item.get("output_dir", "")).strip()
            if not image or not output_dir:
                continue
            name = Path(image).name
            if single_image and name != single_image:
                continue
            folders.append((Path(output_dir), name))
        if folders:
            return folders

    if single_image:
        return [(testout_dir / single_image, single_image)]

    return [
        (entry, entry.name)
        for entry in sorted(testout_dir.iterdir())
        if entry.is_dir() and entry.name not in {"text_reports"} and (entry / "05_检测结果摘要.json").exists()
    ]


def _build_batch_report(items: list[tuple[Path, str]]) -> str:
    if not items:
        return "未找到可处理的检测结果。\n"

    lines = ["【批量检测自然语言汇总】", f"共处理 {len(items)} 张图片。", ""]
    total_ok = 0
    total_skip = 0
    total_raw = 0
    total_filtered = 0
    for folder, image_name in items:
        compact = _load_json(folder / "05_检测结果摘要.json")
        detection = compact.get("detection", {}) or {}
        grading = compact.get("grading", {}) or {}
        raw_count = int(detection.get("raw_count", 0) or 0)
        filtered_count = int(detection.get("filtered_count", detection.get("count", 0) or 0) or 0)
        total_raw += raw_count
        total_filtered += filtered_count
        if compact.get("status") == "ok":
            total_ok += 1
        elif compact.get("status") == "skipped":
            total_skip += 1
        lines.append(
            f"- {image_name}：模式 {compact.get('input_mode', 'unknown')}，"
            f"圈选保留 {filtered_count} 个候选，严重程度状态 {grading.get('status', 'unknown')}。"
        )

    lines.extend(
        [
            "",
            f"成功解析摘要：{total_ok} 张，跳过摘要：{total_skip} 张。",
            f"原始候选总数：{total_raw} 个。",
            f"过滤后候选总数：{total_filtered} 个。",
            "",
            "说明：",
            "- 每张图片的详细用户版报告位于各自图片输出目录中的 `08_用户检测报告.txt`。",
            "- 本批次结果仅供皮肤状态观察与研发辅助参考，不作为医学诊断。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="导出可读的痤疮检测 txt 报告")
    parser.add_argument("--testout-dir", default="testout")
    parser.add_argument("--output-dir", default="testout/text_reports")
    parser.add_argument("--image", default=None, help="只导出单张，例如 11.jpg；不传则导出批量目录")
    args = parser.parse_args()

    testout_dir = Path(args.testout_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not testout_dir.exists():
        print(f"测试输出目录不存在：{testout_dir}")
        return 1

    items = _collect_items(testout_dir, args.image)
    if not items:
        print("未找到可处理的图片输出目录。")
        return 1

    written: list[tuple[str, str]] = []
    for folder, name in items:
        compact_path = folder / "05_检测结果摘要.json"
        grading_path = folder / "06_严重程度评估.json"
        csv_path = folder / "04_检测结果表.csv"

        if compact_path.exists():
            report = build_user_report_from_paths(compact_path, grading_path, csv_path)
        else:
            report = (
                "【用户版痤疮检测报告】\n"
                f"图片名称：{name}\n"
                "本次未找到 `05_检测结果摘要.json`，因此无法生成完整用户报告。\n"
            )

        report_path = folder / "08_用户检测报告.txt"
        report_path.write_text(report, encoding="utf-8")

        legacy_path = folder / f"{name}.txt"
        if legacy_path.exists() and legacy_path != report_path:
            legacy_path.unlink()

        written.append((name, str(report_path)))

    if args.image is None:
        batch_report_path = output_dir / "批量检测自然语言汇总.txt"
        batch_report_path.write_text(_build_batch_report(items), encoding="utf-8")
        print(f"已生成批量汇总：{batch_report_path}")
    else:
        print("已生成单图可读报告。")

    print(f"共输出 {len(written)} 个图片文本：")
    for name, path in written:
        print(f"- {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
