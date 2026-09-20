from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregator import build_report_payload
from .io_utils import load_json
from .report import render_docx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AISIA九项皮肤检测医学汇总报告生成器")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="根据结果清单生成汇总JSON和DOCX")
    generate.add_argument("--bundle", required=True, help="报告输入清单JSON")
    generate.add_argument("--output-dir", required=True, help="报告输出目录")
    generate.add_argument("--template", default=None, help="DOCX模板路径；默认使用项目模板")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate":
        return _generate(args)
    return 2


def _generate(args: argparse.Namespace) -> int:
    bundle_path = Path(args.bundle).expanduser().resolve()
    bundle = load_json(bundle_path)
    payload = build_report_payload(bundle)
    report_id = str(payload["报告信息"]["报告编号"])
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in report_id)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"AISIA_面部多指标检测报告_{safe_id}.json"
    docx_path = output_dir / f"AISIA_面部多指标检测报告_{safe_id}.docx"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if args.template:
        template = Path(args.template).expanduser().resolve()
    else:
        template = Path(__file__).resolve().parents[2] / "templates" / "AISIA_面部多指标检测汇总模板_V2.docx"
    render_docx(payload, template, docx_path)
    print(json.dumps({"status": "success", "json": str(json_path), "docx": str(docx_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
