from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aisia_medical_report import generate_report_from_review_result


def main() -> int:
    parser = argparse.ArgumentParser(description="AISIA 九项皮肤检测医学汇总报告生成器（可选模块）")
    parser.add_argument("--result-dir", type=Path, required=True, help="单张图片的九项 review 结果目录")
    parser.add_argument("--output-dir", type=Path, help="报告输出目录；默认写入结果目录/医学报告")
    parser.add_argument("--report-id", help="报告编号；不传时自动生成")
    parser.add_argument("--subject-id", default="匿名受检者", help="受检者姓名或编号")
    parser.add_argument("--template", type=Path, help="可选DOCX模板路径")
    args = parser.parse_args()
    output = generate_report_from_review_result(
        args.result_dir,
        args.output_dir,
        report_id=args.report_id,
        subject_id=args.subject_id,
        template_path=args.template,
    )
    print(json.dumps({"状态": "success", "输出": output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
