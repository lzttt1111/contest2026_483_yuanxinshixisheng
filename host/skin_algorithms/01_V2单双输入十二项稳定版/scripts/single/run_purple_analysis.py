# -*- coding: utf-8 -*-
"""DermaVision Purple Analysis 独立运行入口。

示例：
    python run_purple_analysis.py --input data/raw_images/test.jpg
    python run_purple_analysis.py --input data/raw_images --output output/purple_analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.purple_analysis_engine import PurpleAnalysisEngine
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.utils.io_utils import cv_imread


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行标准化 UV / 荧光 UV Purple Analysis，不修改 Pipeline。"
    )
    parser.add_argument("--input", required=True, help="输入图片或图片目录")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "purple_analysis"),
        help="输出根目录，默认 output/purple_analysis",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    files = _input_files(input_path)
    if not files:
        print(f"❌ 未找到可处理图片: {input_path}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = ImagePreprocessor(None, None, None)
    engine = PurpleAnalysisEngine()
    failures: list[tuple[str, str]] = []
    try:
        for image_path in files:
            image = cv_imread(str(image_path))
            if image is None:
                failures.append((image_path.name, "图片读取失败"))
                continue
            preprocess_result = preprocessor.preprocess_image(image)
            flags = ",".join(preprocess_result.quality_flags) or "NONE"
            print(
                f"[Quality] {image_path.name}: "
                f"{preprocess_result.quality_score:.2f} "
                f"{preprocess_result.quality_status} [{flags}]"
            )
            overlay = engine.process_preprocess_result(
                preprocess_result,
                str(output_dir),
                image_path.name,
            )
            if overlay is None:
                failures.append((image_path.name, "质量门禁拒绝或未生成结果"))
                continue
            print(f"✅ {image_path.name} -> {os.path.dirname(overlay)}")
    finally:
        engine.close()
        preprocessor.close()

    if failures:
        print("\n⚠️ 以下图片未完成：")
        for name, reason in failures:
            print(f"- {name}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
