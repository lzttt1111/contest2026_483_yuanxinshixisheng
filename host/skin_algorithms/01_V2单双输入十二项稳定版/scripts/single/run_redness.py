#!/usr/bin/env python3
"""DermaVision RBX-like Visible Redness 独立批量测试入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.engines.rbx_engine import Config as RBXConfig
from src.engines.rbx_engine import ErythemaAnalyzer
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.utils.io_utils import cv_imread, cv_imwrite


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
RBX_IMAGE_FILES = (
    "01_RBX输入图.jpg",
    "02A_RBX人脸抠图.png",
    "02B_RBX颜色映射图.png",
    "03_RBX红区结果图.jpg",
    "03_有效皮肤区域.png",
    "04_RBX红区强度图.png",
    "04_毛发遮挡区域.png",
    "05_RBX完整人脸Mask.png",
    "05_已过滤毛发误检.png",
    "06_VISIA红色区实例图.jpg",
    "07_VISIA红色区实例Mask.png",
)
RBX_RESULT_FILES = set(RBX_IMAGE_FILES) | {
    "红区量化指标.json",
    "红区量化指标.csv",
}


def _resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行 DermaVision RBX 红区检测")
    parser.add_argument(
        "--input-dir",
        default="data/test_images",
        help="输入图片目录，默认 data/test_images",
    )
    parser.add_argument(
        "--output-dir",
        default="test-output/redness",
        help="输出目录，默认 test-output/redness",
    )
    parser.add_argument(
        "--visia-pairs-dir",
        default=None,
        help="可选 VISIA 输入/输出配对目录；设置后额外生成对齐质量评估",
    )
    parser.add_argument(
        "--review-output-dir",
        default="test-output/redness_quality_review",
        help="VISIA 对比输出目录，默认 test-output/redness_quality_review",
    )
    parser.add_argument(
        "--rebuild-visia-profile",
        action="store_true",
        help="使用 --visia-pairs-dir 中的示例1输入/输出重建正式配对配置",
    )
    parser.add_argument(
        "--color-transfer",
        choices=("off", "on"),
        default="off",
        help=(
            "RBX 脸部 LAB 颜色映射开关；默认 off，on 时只映射有效脸部皮肤"
        ),
    )
    parser.add_argument(
        "--color-reference",
        default="data/reference.jpg",
        help=(
            "开启颜色映射时使用的目标参考图，默认 data/reference.jpg；"
            "显式参考图会绕过 VISIA pair profile"
        ),
    )
    return parser.parse_args(argv)


def image_sources(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"测试图片目录不存在: {input_dir}")
    sources = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not sources:
        raise FileNotFoundError(f"目录中没有可测试图片: {input_dir}")
    stems = [path.stem for path in sources]
    if len(stems) != len(set(stems)):
        raise ValueError("测试图片存在同名但不同扩展名的文件，无法创建唯一子目录")
    return sources


def validate_result(sample_dir: Path) -> dict[str, Any]:
    actual = {path.name for path in sample_dir.iterdir() if path.is_file()}
    if actual != RBX_RESULT_FILES:
        raise AssertionError(
            f"{sample_dir.name} 输出文件不完整，缺失={RBX_RESULT_FILES - actual}，"
            f"多余={actual - RBX_RESULT_FILES}"
        )
    for filename in RBX_IMAGE_FILES:
        path = sample_dir / filename
        if path.stat().st_size <= 0 or cv_imread(str(path)) is None:
            raise AssertionError(f"图片为空或无法解码: {path}")
    metrics = json.loads(
        (sample_dir / "红区量化指标.json").read_text(encoding="utf-8")
    )
    csv_path = sample_dir / "红区量化指标.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or "检测范围" not in rows[0] or "评估状态" not in rows[0]:
        raise AssertionError(f"中文红区医学CSV格式不正确: {sample_dir}")
    if metrics.get("指标版本") != "medical_metrics_v1":
        raise AssertionError(f"红区JSON不是医学量化v1: {sample_dir}")
    if "总体指标" not in metrics or "分区指标" not in metrics:
        raise AssertionError(f"红区医学JSON缺少总体或分区指标: {sample_dir}")
    for key in metrics:
        if not any("\u4e00" <= char <= "\u9fff" for char in key):
            raise AssertionError(f"红区量化指标名称必须使用中文: {sample_dir}")
    return metrics


def batch_report(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary["images"]:
        if item["status"] == "success":
            rows.append(
                "| {name} | 成功 | {count} |".format(
                    name=item["source_name"],
                    count=item["红色区特征点总数"],
                )
            )
        else:
            rows.append(f"| {item['source_name']} | 失败：{item['error']} | - |")
    return f"""# DermaVision 批量红区检测报告

- 生成时间：`{summary['generated_at']}`
- 输入目录：`{summary['input_directory']}`
- 输出目录：`{summary['output_directory']}`
- 图片总数：`{summary['image_count']}`
- 成功：`{summary['success_count']}`
- 失败：`{summary['failure_count']}`

| 照片 | 状态 | 红色区特征点总数 |
|---|---|---:|
{chr(10).join(rows)}

当前结果是 Visible Redness 工程量化，不代表医学炎症诊断或严重程度分级。
"""


def process_one(
    source: Path,
    output_dir: Path,
    preprocessor: ImagePreprocessor,
    analyzer: ErythemaAnalyzer,
) -> dict[str, Any]:
    image = cv_imread(str(source))
    if image is None:
        raise ValueError("图片无法读取")
    preprocess_result = preprocessor.preprocess_image(image)
    if preprocess_result.quality_status == "REJECT":
        flags = ",".join(preprocess_result.quality_flags) or "UNKNOWN"
        raise ValueError(f"图像质量门禁拒绝: {flags}")

    result_path = analyzer.process_preprocess_result(
        preprocess_result,
        str(output_dir),
        source.name,
    )
    if not result_path:
        raise RuntimeError("RBX 引擎未生成结果")

    sample_dir = output_dir / source.stem
    metrics = validate_result(sample_dir)
    return {
        "source_name": source.name,
        "status": "success",
        "quality_status": preprocess_result.quality_status,
        "quality_score": preprocess_result.quality_score,
        "红色区特征点总数": metrics["总体指标"]["局灶红色实例数量（个）"],
        "result_path": str(Path(result_path).resolve()),
    }


def _blockiness_index(image: np.ndarray, block_size: int) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    all_energy = float(np.mean(vertical) + np.mean(horizontal) + 1e-6)
    vertical_boundaries = vertical[:, block_size - 1 :: block_size]
    horizontal_boundaries = horizontal[block_size - 1 :: block_size, :]
    boundary_energy = float(
        np.mean(vertical_boundaries) + np.mean(horizontal_boundaries)
    )
    return boundary_energy / all_energy


def _edge_correlation(
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    generated_edge = cv2.magnitude(
        cv2.Sobel(generated_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(generated_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    target_edge = cv2.magnitude(
        cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    valid = mask > 0
    if np.count_nonzero(valid) < 64:
        return 0.0
    left = generated_edge[valid]
    right = target_edge[valid]
    if float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _comparison_sheet(images: list[np.ndarray]) -> np.ndarray:
    height = 320
    cells = []
    for image in images:
        h, w = image.shape[:2]
        scale = min(320.0 / max(h, 1), 320.0 / max(w, 1))
        resized = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        cell = np.full((height, height, 3), 255, dtype=np.uint8)
        y0 = (height - resized.shape[0]) // 2
        x0 = (height - resized.shape[1]) // 2
        cell[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cells.append(cell)
    return np.hstack(cells)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_visia_pair(pairs_dir: Path, pair_number: int) -> tuple[Path, Path]:
    prefix = f"visia示例图{pair_number}"
    inputs = sorted(
        path
        for path in pairs_dir.iterdir()
        if path.is_file()
        and path.stem == f"{prefix}-输入"
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    outputs = sorted(
        path
        for path in pairs_dir.iterdir()
        if path.is_file()
        and path.stem == f"{prefix}-输出"
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{pairs_dir} 中无法确定唯一的 {prefix} 输入/输出")
    return inputs[0], outputs[0]


def _align_target_to_preprocess(
    target: np.ndarray,
    source_shape: tuple[int, int],
    preprocess_result,
) -> np.ndarray:
    source_height, source_width = source_shape
    if target.shape[:2] != (source_height, source_width):
        target = cv2.resize(
            target,
            (source_width, source_height),
            interpolation=cv2.INTER_AREA,
        )
    analysis = preprocess_result.analysis_image
    return cv2.warpAffine(
        target,
        preprocess_result.face_transform_matrix,
        (analysis.shape[1], analysis.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def rebuild_visia_pair1_profile(
    pairs_dir: Path,
    preprocessor: ImagePreprocessor,
    analyzer: ErythemaAnalyzer,
) -> dict[str, Any]:
    input_path, output_path = _find_visia_pair(pairs_dir, 1)
    source = cv_imread(str(input_path))
    target = cv_imread(str(output_path))
    if source is None or target is None:
        raise ValueError("VISIA 示例 1 输入或输出无法读取")
    preprocess_result = preprocessor.preprocess_image(source)
    if preprocess_result.quality_status == "REJECT":
        flags = ",".join(preprocess_result.quality_flags) or "UNKNOWN"
        raise ValueError(f"VISIA 示例 1 预处理拒绝: {flags}")
    aligned_target = _align_target_to_preprocess(
        target,
        source.shape[:2],
        preprocess_result,
    )
    profile = analyzer.build_visia_pair_profile(
        preprocess_result=preprocess_result,
        aligned_target=aligned_target,
        source_name=input_path.name,
        target_name=output_path.name,
        source_sha256=_sha256(input_path),
        target_sha256=_sha256(output_path),
    )
    profile_path = RBXConfig.COLOR_PROFILE_PATH
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile


def _histogram_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    bins: int = 128,
) -> float:
    left_hist, _ = np.histogram(left, bins=bins, range=(0.0, 1.0), density=False)
    right_hist, _ = np.histogram(right, bins=bins, range=(0.0, 1.0), density=False)
    left_cdf = np.cumsum(left_hist) / max(int(left_hist.sum()), 1)
    right_cdf = np.cumsum(right_hist) / max(int(right_hist.sum()), 1)
    return float(np.mean(np.abs(left_cdf - right_cdf)))


def run_visia_review(
    pairs_dir: Path,
    review_dir: Path,
    preprocessor: ImagePreprocessor,
    analyzer: ErythemaAnalyzer,
) -> dict[str, Any]:
    if not pairs_dir.is_dir():
        raise FileNotFoundError(f"VISIA 配对目录不存在: {pairs_dir}")
    input_files = sorted(
        path
        for path in pairs_dir.iterdir()
        if path.is_file()
        and "-输入" in path.stem
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not input_files:
        raise FileNotFoundError(f"VISIA 配对目录没有 *-输入 图片: {pairs_dir}")

    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)
    generated_root = review_dir / "_generated"
    results: list[dict[str, Any]] = []

    try:
        for input_path in input_files:
            output_stem = input_path.stem.replace("-输入", "-输出")
            matches = [
                path
                for path in pairs_dir.iterdir()
                if path.is_file()
                and path.stem == output_stem
                and path.suffix.lower() in SUPPORTED_SUFFIXES
            ]
            if len(matches) != 1:
                raise ValueError(f"{input_path.name} 无法确定唯一 VISIA 输出配对")
            target_path = matches[0]
            source = cv_imread(str(input_path))
            target = cv_imread(str(target_path))
            if source is None or target is None:
                raise ValueError(f"VISIA 配对图无法读取: {input_path.name}")

            preprocess_result = preprocessor.preprocess_image(source)
            if preprocess_result.quality_status == "REJECT":
                flags = ",".join(preprocess_result.quality_flags) or "UNKNOWN"
                raise ValueError(f"{input_path.name} 预处理拒绝: {flags}")

            aligned_target = _align_target_to_preprocess(
                target,
                source.shape[:2],
                preprocess_result,
            )

            result_path = analyzer.process_preprocess_result(
                preprocess_result,
                str(generated_root),
                input_path.name,
            )
            if not result_path:
                raise RuntimeError(f"{input_path.name} 未生成 RBX 结果")
            generated_dir = Path(result_path).parent
            generated = cv_imread(str(generated_dir / "03_RBX红区结果图.jpg"))
            score_u8 = cv_imread(str(generated_dir / "04_RBX红区强度图.png"))
            face_mask = cv_imread(str(generated_dir / "05_RBX完整人脸Mask.png"))
            if (
                generated is None
                or score_u8 is None
                or face_mask is None
            ):
                raise RuntimeError(f"{input_path.name} 正式结果或 Mask 读取失败")
            if score_u8.ndim == 3:
                score_u8 = cv2.cvtColor(score_u8, cv2.COLOR_BGR2GRAY)
            if face_mask.ndim == 3:
                face_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)

            pair_dir = review_dir / input_path.stem.replace("-输入", "")
            pair_dir.mkdir(parents=True)
            projected = cv2.warpAffine(
                generated,
                preprocess_result.inverse_transform_matrix,
                (source.shape[1], source.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            comparison = _comparison_sheet([source, projected, target])
            shutil.copy2(
                input_path,
                pair_dir / f"01_VISIA原始输入图{input_path.suffix.lower()}",
            )
            cv_imwrite(
                str(pair_dir / "02_正式版_原始构图.jpg"),
                projected,
            )
            shutil.copy2(
                target_path,
                pair_dir / f"03_VISIA原始输出图{target_path.suffix.lower()}",
            )
            cv_imwrite(str(pair_dir / "04_三图原始构图对比.jpg"), comparison)
            cv_imwrite(str(pair_dir / "05_内部对齐目标图.png"), aligned_target)

            valid_mask = cv2.bitwise_and(
                (preprocess_result.skin_mask > 0).astype(np.uint8) * 255,
                (face_mask > 0).astype(np.uint8) * 255,
            )
            score_map = score_u8.astype(np.float32) / 255.0
            target_density = analyzer._visia_target_density(
                aligned_target,
                face_mask,
            )
            valid = valid_mask > 0
            generated_quantiles = np.percentile(
                score_map[valid],
                [50.0, 90.0, 95.0],
            )
            target_quantiles = np.percentile(
                target_density[valid],
                [50.0, 90.0, 95.0],
            )
            metrics = {
                "source": input_path.name,
                "target": target_path.name,
                "comparison_order": [
                    "VISIA原始输入图",
                    "正式自然版（反投影到原始构图）",
                    "VISIA原始输出图",
                ],
                "quality_status": preprocess_result.quality_status,
                "quality_score": preprocess_result.quality_score,
                "generated_density_quantiles": {
                    "p50": float(generated_quantiles[0]),
                    "p90": float(generated_quantiles[1]),
                    "p95": float(generated_quantiles[2]),
                },
                "target_density_quantiles": {
                    "p50": float(target_quantiles[0]),
                    "p90": float(target_quantiles[1]),
                    "p95": float(target_quantiles[2]),
                },
                "quantile_absolute_error": {
                    "p50": float(abs(generated_quantiles[0] - target_quantiles[0])),
                    "p90": float(abs(generated_quantiles[1] - target_quantiles[1])),
                    "p95": float(abs(generated_quantiles[2] - target_quantiles[2])),
                },
                "histogram_wasserstein": _histogram_wasserstein(
                    score_map[valid],
                    target_density[valid],
                ),
                "blockiness_8": _blockiness_index(generated, 8),
                "blockiness_16": _blockiness_index(generated, 16),
                "edge_correlation": _edge_correlation(
                    generated,
                    aligned_target,
                    valid_mask,
                ),
            }
            (pair_dir / "对比指标.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results.append(metrics)
    finally:
        if generated_root.exists():
            shutil.rmtree(generated_root)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pairs_directory": str(pairs_dir),
        "pair_count": len(results),
        "images": results,
    }
    (review_dir / "VISIA对比汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = _resolve_cli_path(args.input_dir).resolve()
    output_dir = _resolve_cli_path(args.output_dir).resolve()
    sources = image_sources(input_dir)
    color_transfer_enabled = args.color_transfer == "on"
    color_reference = _resolve_cli_path(args.color_reference).resolve()
    if color_transfer_enabled and not color_reference.is_file():
        raise FileNotFoundError(f"颜色映射参考图不存在: {color_reference}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    preprocessor = ImagePreprocessor(None, None, None)
    analyzer = ErythemaAnalyzer(
        color_transfer_enabled=color_transfer_enabled,
        color_transfer_reference=(
            color_reference if color_transfer_enabled else None
        ),
    )
    results: list[dict[str, Any]] = []
    visia_summary: dict[str, Any] | None = None
    try:
        if args.rebuild_visia_profile:
            if not args.visia_pairs_dir:
                raise ValueError(
                    "--rebuild-visia-profile 必须同时提供 --visia-pairs-dir"
                )
            profile = rebuild_visia_pair1_profile(
                _resolve_cli_path(args.visia_pairs_dir).resolve(),
                preprocessor,
                analyzer,
            )
            print(
                "✅ 已重建 VISIA 示例 1 配对配置 -> "
                f"{RBXConfig.COLOR_PROFILE_PATH}"
            )
            print(
                "   输入/输出 SHA-256: "
                f"{profile['source_sha256'][:12]} / "
                f"{profile['target_sha256'][:12]}"
            )
            # 当前 analyzer 可能缓存旧配置；重建后重新加载，确保本批次
            # 使用刚写入的正式 V3 配置。
            analyzer.close()
            analyzer = ErythemaAnalyzer(
                color_transfer_enabled=color_transfer_enabled,
                color_transfer_reference=(
                    color_reference if color_transfer_enabled else None
                ),
            )
        for index, source in enumerate(sources, start=1):
            print(f"[{index}/{len(sources)}] {source.name}", flush=True)
            try:
                results.append(process_one(source, output_dir, preprocessor, analyzer))
            except Exception as exc:
                print(f"  失败: {type(exc).__name__}: {exc}", flush=True)
                failure_dir = output_dir / source.stem
                if failure_dir.exists():
                    shutil.rmtree(failure_dir)
                failure_dir.mkdir(parents=True)
                (failure_dir / "检测失败.txt").write_text(
                    f"图片：{source.name}\n"
                    f"状态：失败\n"
                    f"原因：{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
                results.append(
                    {
                        "source_name": source.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if args.visia_pairs_dir:
            visia_summary = run_visia_review(
                _resolve_cli_path(args.visia_pairs_dir).resolve(),
                _resolve_cli_path(args.review_output_dir).resolve(),
                preprocessor,
                analyzer,
            )
    finally:
        analyzer.close()
        preprocessor.close()

    success_count = sum(item["status"] == "success" for item in results)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "image_count": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "color_transfer_enabled": color_transfer_enabled,
        "color_transfer_reference": (
            str(color_reference) if color_transfer_enabled else None
        ),
        "images": results,
    }
    if visia_summary is not None:
        summary["visia_review"] = {
            "pair_count": visia_summary["pair_count"],
            "pairs_directory": visia_summary["pairs_directory"],
            "review_output_directory": str(
                _resolve_cli_path(args.review_output_dir).resolve()
            ),
        }
    (output_dir / "批量红区检测汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "批量红区检测报告.md").write_text(
        batch_report(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "image_count": len(results),
                "success_count": success_count,
                "failure_count": len(results) - success_count,
                "output_directory": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
