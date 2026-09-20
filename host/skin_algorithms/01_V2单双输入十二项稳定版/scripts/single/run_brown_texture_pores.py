#!/usr/bin/env python3
"""Batch runner for Brown Spots, Texture, and Pores local validation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.pores_engine import PoresAnalyzer
from src.engines.texture_engine import TextureAnalyzer
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.utils.io_utils import cv_imread, cv_imwrite


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行棕色斑、纹理和毛孔三项检测"
    )
    parser.add_argument(
        "--input-dir",
        default="test-output/visia报告测试原图",
        help="输入目录；默认使用从四份VISIA报告提取的标准照片",
    )
    parser.add_argument(
        "--output-dir",
        default="test-output/brown_texture_pores",
        help="输出目录",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = _path(args.input_dir)
    output_dir = _path(args.output_dir)
    sources = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not sources:
        raise FileNotFoundError(f"没有可测试图片: {input_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    preprocessor = ImagePreprocessor(
        str(input_dir),
        str(output_dir / "preprocessed"),
        str(PROJECT_ROOT / "data" / "reference.jpg"),
    )
    brown = BrownAreaAnalyzer()
    texture = TextureAnalyzer()
    pores = PoresAnalyzer()
    records = []
    try:
        for source in sources:
            print(f"\n{'=' * 72}\n处理 {source.name}\n{'=' * 72}")
            image = cv_imread(str(source))
            if image is None:
                records.append({"source": source.name, "status": "failed", "error": "读取失败"})
                continue
            try:
                pre = preprocessor.preprocess_image(image)
                if pre.quality_status == "REJECT":
                    raise ValueError(f"质量门禁拒绝: {pre.quality_flags}")
                preview_dir = output_dir / "预处理" / source.stem
                preview_dir.mkdir(parents=True, exist_ok=True)
                cv_imwrite(str(preview_dir / "01_标准化分析图.jpg"), pre.analysis_image)
                cv_imwrite(str(preview_dir / "02_有效皮肤Mask.png"), pre.skin_mask)

                brown_result = brown.detect_brown(pre)
                BrownAreaAnalyzer.save_result(
                    brown_result, str(output_dir / "棕色斑"), source.stem
                )
                texture_result = texture.detect_texture(pre)
                TextureAnalyzer.save_result(
                    texture_result, str(output_dir / "纹理"), source.stem
                )
                pores_result = pores.detect_pores(pre)
                PoresAnalyzer.save_result(
                    pores_result, str(output_dir / "毛孔"), source.stem
                )
                records.append(
                    {
                        "source": source.name,
                        "status": "success",
                        "quality_status": pre.quality_status,
                        "quality_flags": pre.quality_flags,
                        "brown_spot_count": brown_result.brown_spot_count,
                        "texture_feature_count": texture_result.texture_feature_count,
                        "texture_raised_like_count": texture_result.raised_like_count,
                        "texture_depressed_like_count": texture_result.depressed_like_count,
                        "pore_count": pores_result.pore_count,
                    }
                )
                print(
                    f"棕色斑={brown_result.brown_spot_count} | "
                    f"纹理={texture_result.texture_feature_count}"
                    f"（黄={texture_result.raised_like_count}，"
                    f"蓝={texture_result.depressed_like_count}） | "
                    f"毛孔={pores_result.pore_count}"
                )
            except Exception as exc:
                records.append(
                    {
                        "source": source.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"失败: {type(exc).__name__}: {exc}")
    finally:
        brown.close()
        preprocessor.close()

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "visia_reference_note": (
            "visia3.pdf 的原始特征计数为棕色斑232、纹理1826、毛孔733；"
            "其他报告只提供常模百分位，不能直接与实例数比较。"
        ),
        "records": records,
    }
    (output_dir / "批量检测汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    success = sum(item["status"] == "success" for item in records)
    print(f"\n完成：成功 {success}/{len(records)}，输出目录：{output_dir}")
    return 0 if success == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
