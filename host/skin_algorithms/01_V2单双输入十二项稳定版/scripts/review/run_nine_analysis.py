from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nine_analysis import NineAnalysisOrchestrator
from src.nine_analysis.review_output import ReviewOutputExporter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 批量运行的安全默认值：只生成九项结果，不生成 DOCX 医学报告。
GENERATE_MEDICAL_REPORT_DEFAULT = False


def collect_images(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    if input_path is not None:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        return [input_path]
    directory = input_dir or Path("data/test_images")
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"未找到图片: {directory}")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="AISIA 九项 GPU 皮肤检测统一入口")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--layout", choices=("archive", "review"), default="archive")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument(
        "--generate-medical-report",
        action="store_true",
        default=GENERATE_MEDICAL_REPORT_DEFAULT,
        help="九项成功后为每张图片生成可选医学汇总JSON和DOCX；默认关闭",
    )
    parser.add_argument("--report-subject-id", default="匿名受检者", help="可选医学报告中的受检者姓名或编号")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须 >= 1")
    if args.layout == "review" and args.repeat != 1:
        parser.error("review 输出模式固定使用 --repeat 1")
    if args.generate_medical_report and args.layout != "review":
        parser.error("--generate-medical-report 仅支持 --layout review，批量归档模式默认不生成报告")

    images = collect_images(args.input, args.input_dir)
    output_root = args.output.resolve()
    runtime_root = output_root / ".runtime" if args.layout == "review" else output_root
    exporter = ReviewOutputExporter(output_root) if args.layout == "review" else None
    orchestrator = NineAnalysisOrchestrator(runtime_root, cuda_device=args.cuda_device)
    all_success = False
    try:
        cold = orchestrator.start()
        print(json.dumps({"冷启动": cold, "运行目录": str(orchestrator.run_root)}, ensure_ascii=False, indent=2))
        summaries = []
        for repetition in range(1, args.repeat + 1):
            for image in images:
                result = orchestrator.analyze(image, repetition=repetition)
                if exporter is not None and result["状态"] == "success":
                    source_root = orchestrator.run_root / f"repeat_{repetition}" / image.stem
                    exported = exporter.export(source_root, result)
                    print(f"  已整理人工验收结果: {exported}")
                    if args.generate_medical_report:
                        from src.aisia_medical_report import generate_report_from_review_result

                        report = generate_report_from_review_result(exported, subject_id=args.report_subject_id)
                        print(f"  已生成可选医学报告: {report['docx']}")
                summaries.append({"图片": image.name, "轮次": repetition, "状态": result["状态"], "成功项目数": result["成功项目数"]})
                print(f"[{repetition}/{args.repeat}] {image.name}: {result['状态']} ({result['成功项目数']}/9)")
        (orchestrator.run_root / "run_summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_success = all(item["状态"] == "success" for item in summaries)
        return 0 if all_success else 2
    finally:
        orchestrator.close()
        if args.layout == "review" and all_success:
            shutil.rmtree(orchestrator.run_root, ignore_errors=True)
            if runtime_root.is_dir() and not any(runtime_root.iterdir()):
                runtime_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
