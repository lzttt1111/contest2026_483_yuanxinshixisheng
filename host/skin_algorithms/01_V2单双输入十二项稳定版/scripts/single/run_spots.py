#!/usr/bin/env python3
"""DermaVision Visible Spots V2 独立批量测试入口。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mediapipe as mp
import numpy as np

from src.engines.spots_engine import SpotsEngine, SpotsResultV2, build_region_masks
from src.preprocess.image_preprocessor import ImagePreprocessor, PreprocessResultV2
from src.utils.detailed_metrics import (
    build_medical_payload,
    write_medical_metrics,
)
from src.utils.io_utils import cv_imread, cv_imwrite


INPUT_DIR = PROJECT_ROOT / "data" / "test_images"
TEST_OUTPUT = PROJECT_ROOT / "test-output" / "spots"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
FOCUS_IMAGE = "1000085_1.jpg"

IMAGE_OUTPUT_NAMES = (
    "01_原始照片.jpg",
    "02_人脸标准化结果.jpg",
    "03_有效皮肤区域.png",
    "04_毛发遮挡区域.png",
    "05_已过滤毛发误检.png",
    "06_最终斑点检测结果.jpg",
)
DIRECTORY_OUTPUT_NAMES = set(IMAGE_OUTPUT_NAMES) | {
    "检测指标.json", "检测指标.csv", "结果报告.md"
}
ROOT_OUTPUT_NAMES = {"批量检测报告.md", "批量检测汇总.json"}


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "mediapipe": getattr(mp, "__version__", "unknown"),
        "scikit-image": importlib.metadata.version("scikit-image"),
    }


def image_sources() -> list[Path]:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"测试图片目录不存在: {INPUT_DIR}")
    sources = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not sources:
        raise FileNotFoundError(f"目录中没有可测试图片: {INPUT_DIR}")
    stems = [path.stem for path in sources]
    if len(stems) != len(set(stems)):
        raise ValueError("测试图片存在同名但不同扩展名的文件，无法创建唯一子目录")
    return sources


def spots_fingerprint(result: SpotsResultV2) -> tuple[object, ...]:
    return (
        result.spot_count,
        result.spot_area_ratio,
        result.spot_confidence,
        result.mean_deltaE,
        json.dumps(result.spot_locations, ensure_ascii=False, sort_keys=True),
        json.dumps(result.region_distribution, ensure_ascii=False, sort_keys=True),
        json.dumps(result.occlusion_filter_statistics, sort_keys=True),
        result.scale_vote_map.tobytes(),
        result.candidate_heatmap.tobytes(),
        result.candidate_mask.tobytes(),
        result.filtered_mask.tobytes(),
        result.hair_occlusion_mask.tobytes(),
        result.hair_rejected_mask.tobytes(),
    )


def preprocess_fingerprint(result: PreprocessResultV2) -> tuple[object, ...]:
    return (
        result.analysis_image.tobytes(),
        result.display_image.tobytes(),
        result.skin_mask.tobytes(),
        result.landmarks.tobytes(),
        result.face_transform_matrix.tobytes(),
        result.inverse_transform_matrix.tobytes(),
        result.quality_score,
        result.quality_status,
        tuple(result.quality_flags),
    )


def mask_statistics(
    result: PreprocessResultV2, debug: dict[str, np.ndarray]
) -> dict[str, Any]:
    semantic = debug["semantic_skin_mask"]
    geometry = debug["face_geometry_mask"]
    final = result.skin_mask
    geometry_area = int(np.count_nonzero(geometry))
    region_masks = (
        build_region_masks(result.landmarks, final)
        if result.landmarks.shape[0] > 334
        else {}
    )
    return {
        "semantic_skin_area": int(np.count_nonzero(semantic)),
        "geometry_mask_area": geometry_area,
        "final_skin_area": int(np.count_nonzero(final)),
        "semantic_geometry_overlap_ratio": round(
            float(np.count_nonzero((semantic > 0) & (geometry > 0)) / max(geometry_area, 1)),
            8,
        ),
        "forehead_completion_area": int(np.count_nonzero(debug["forehead_completion"])),
        "feature_exclusion_area": int(np.count_nonzero(debug["feature_exclusions"])),
        "nostril_exclusion_area": int(np.count_nonzero(debug["nostril_mask"])),
        "bulk_hair_area": int(np.count_nonzero(debug["bulk_hair_mask"])),
        "color_hair_area": int(np.count_nonzero(debug["color_hair_mask"])),
        "strand_hair_area": int(np.count_nonzero(debug["strand_hair_mask"])),
        "facial_hair_area": int(np.count_nonzero(debug["facial_hair_mask"])),
        "hair_occlusion_area": int(np.count_nonzero(debug["hair_mask"])),
        "region_skin_pixels": {
            name: int(np.count_nonzero(mask))
            for name, mask in region_masks.items()
            if name != "other"
        },
    }


def write_image(path: Path, image: np.ndarray) -> None:
    cv_imwrite(str(path), image)
    if not path.is_file() or path.stat().st_size <= 0:
        raise OSError(f"图片写入失败: {path}")


def save_outputs(
    output_dir: Path,
    source_image: np.ndarray,
    preprocess_result: PreprocessResultV2,
    spots: SpotsResultV2,
    combined_occlusion_mask: np.ndarray,
) -> None:
    write_image(output_dir / "01_原始照片.jpg", source_image)
    write_image(output_dir / "02_人脸标准化结果.jpg", preprocess_result.display_image)
    write_image(output_dir / "03_有效皮肤区域.png", preprocess_result.skin_mask)
    write_image(output_dir / "04_毛发遮挡区域.png", combined_occlusion_mask)
    write_image(output_dir / "05_已过滤毛发误检.png", spots.hair_rejected_mask)
    write_image(output_dir / "06_最终斑点检测结果.jpg", spots.spots_overlay)


def result_report(metrics: dict[str, Any]) -> str:
    return f"""# DermaVision 斑点检测结果

## 检测信息

- 输入照片：`{metrics['source_name']}`
- 质量状态：`{metrics['quality_status']}`
- 质量分数：`{metrics['quality_score']:.1f}`
- 毛发过滤前候选数：`{metrics['pre_occlusion_spot_count']}`
- 最终斑点数：`{metrics['spot_count']}`
- 大型可见斑数：`{metrics['large_spot_count']}`
- 显著颜色异常斑数：`{metrics['salient_spot_count']}`
- 小型斑点数：`{metrics['small_spot_count']}`
- 被毛发后处理过滤的实例数：`{metrics['hair_filtered_count']}`
- 被鼻孔排除过滤的实例数：`{metrics['nostril_filtered_count']}`
- 是否为半脸/大侧脸分析：`{metrics['partial_face_analysis']}`
- 斑点面积比例：`{metrics['spot_area_ratio']:.6%}`
- 大型可见斑面积比例：`{metrics['large_spot_area_ratio']:.6%}`
- 毛发与五官遮挡面积：`{metrics['spots_hair_occlusion_area']}` 像素

## 图片说明

- `01_原始照片.jpg`：本次检测使用的原始照片。
- `02_人脸标准化结果.jpg`：用于人工核查的自然正方形裁剪图；算法内部另用 analysis_image 与 skin_mask 做计算。
- `03_有效皮肤区域.png`：白色表示参与算法分析的皮肤区域。
- `04_毛发遮挡区域.png`：白色表示毛发、眉毛、眼睛、嘴唇等遮挡区域。
- `05_已过滤毛发误检.png`：白色表示被后处理剔除的候选斑点。
- `06_最终斑点检测结果.jpg`：最终提供给用户查看的斑点结果。

## 注意

当前结果属于普通 RGB 照片中的可见斑点分析，不是 Brown Spots、UV Spots
或医学诊断。毛发检测属于无训练模型的启发式后处理，浅色发丝及紧邻毛发的
真实斑点仍可能需要人工复核。
"""


def validate_image_outputs(output_dir: Path, metrics: dict[str, Any]) -> None:
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != DIRECTORY_OUTPUT_NAMES:
        raise AssertionError(
            f"{output_dir.name} 文件集合不正确，缺失={DIRECTORY_OUTPUT_NAMES - actual}，"
            f"多余={actual - DIRECTORY_OUTPUT_NAMES}"
        )
    for name in IMAGE_OUTPUT_NAMES:
        path = output_dir / name
        if path.stat().st_size <= 0 or cv_imread(str(path)) is None:
            raise AssertionError(f"图片为空或无法解码: {path}")
    json.loads((output_dir / "检测指标.json").read_text(encoding="utf-8"))
    hard_failure_flags = {
        "NO_FACE",
        "MULTIPLE_FACES",
        "INVALID_IMAGE",
        "INVALID_FACE_GEOMETRY",
    }
    flags = set(metrics["quality_flags"])
    if hard_failure_flags.intersection(flags) and metrics["spot_count"] != 0:
        raise AssertionError("硬失败图片必须输出 0 个斑点")
    if metrics["partial_face_analysis"]:
        if metrics["quality_status"] != "WARNING":
            raise AssertionError("半脸/大侧脸必须以 WARNING 状态输出")
        if metrics["final_skin_area"] <= 0:
            raise AssertionError("半脸/大侧脸不得清空有效皮肤 Mask")
    if metrics["max_final_occlusion_overlap"] >= 0.12:
        raise AssertionError("最终候选与毛发遮挡区域重叠超出阈值")
    if metrics["max_final_nostril_overlap"] >= 0.05:
        raise AssertionError("最终候选仍与鼻孔区域重叠")
    if metrics["max_final_nostril_enclosure"] >= 0.08:
        raise AssertionError("最终候选轮廓仍包围鼻孔区域")
    if not metrics["deterministic_repeat"]:
        raise AssertionError("同一图片重复运行结果不一致")


def process_one(
    source: Path,
    preprocessor: ImagePreprocessor,
    spots_engine: SpotsEngine,
    branch: str,
    commit: str,
    versions: dict[str, str],
) -> dict[str, Any]:
    output_dir = TEST_OUTPUT / source.stem
    output_dir.mkdir(parents=True, exist_ok=False)
    image = cv_imread(str(source))
    if image is None:
        raise ValueError(f"无法读取图片: {source}")

    first_preprocess = preprocessor.preprocess_image(image)
    first_debug = {key: value.copy() for key, value in preprocessor.last_debug_masks.items()}
    first_spots = spots_engine.detect_spots(first_preprocess)
    second_preprocess = preprocessor.preprocess_image(image)
    second_spots = spots_engine.detect_spots(second_preprocess)
    deterministic = (
        preprocess_fingerprint(first_preprocess) == preprocess_fingerprint(second_preprocess)
        and spots_fingerprint(first_spots) == spots_fingerprint(second_spots)
    )

    combined_occlusion = cv2.bitwise_or(
        first_debug["hair_mask"], first_spots.hair_occlusion_mask
    )
    save_outputs(
        output_dir,
        image,
        first_preprocess,
        first_spots,
        combined_occlusion,
    )
    max_overlap = max(
        (float(item.get("occlusion_overlap_ratio", 0.0)) for item in first_spots.spot_locations),
        default=0.0,
    )
    max_nostril_overlap = max(
        (float(item.get("nostril_overlap_ratio", 0.0)) for item in first_spots.spot_locations),
        default=0.0,
    )
    max_nostril_enclosure = max(
        (float(item.get("nostril_enclosure_ratio", 0.0)) for item in first_spots.spot_locations),
        default=0.0,
    )
    metrics: dict[str, Any] = first_spots.metrics()
    metrics.update(mask_statistics(first_preprocess, first_debug))
    metrics.update(
        {
            "source_file": str(source.relative_to(PROJECT_ROOT)),
            "source_name": source.name,
            "source_sha256": sha256(source),
            "source_size": [int(image.shape[1]), int(image.shape[0])],
            "analysis_size": [
                int(first_preprocess.analysis_image.shape[1]),
                int(first_preprocess.analysis_image.shape[0]),
            ],
            "branch": branch,
            "commit": commit,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dependencies": versions,
            "quality_score": first_preprocess.quality_score,
            "quality_status": first_preprocess.quality_status,
            "quality_flags": first_preprocess.quality_flags,
            "partial_face_analysis": (
                "PARTIAL_FACE" in first_preprocess.quality_flags
                and int(np.count_nonzero(first_preprocess.skin_mask)) > 0
            ),
            "spots_hair_occlusion_area": int(np.count_nonzero(combined_occlusion)),
            "hair_rejected_area": int(np.count_nonzero(first_spots.hair_rejected_mask)),
            "max_final_occlusion_overlap": round(max_overlap, 6),
            "max_final_nostril_overlap": round(max_nostril_overlap, 6),
            "max_final_nostril_enclosure": round(max_nostril_enclosure, 6),
            "deterministic_repeat": deterministic,
        }
    )
    instances = []
    for item in first_spots.spot_locations:
        instance = dict(item)
        instance["intensity"] = min(float(item.get("mean_deltaE", 0.0)) / 12.0, 1.0)
        instance["morphology"] = "点状" if float(item.get("area", 0.0)) < 90 else "片状"
        instances.append(instance)
    detailed = build_medical_payload(
        project="spots",
        project_label="普通RGB可见斑点",
        analysis_mask=first_spots._medical_analysis_mask,
        landmarks=first_spots._medical_landmarks,
        instances=instances,
        score_map=np.maximum(
            first_spots.scale_vote_map.astype(np.float32) / max(len(first_spots.scale_vote_map.shape) + 2, 1),
            0.0,
        ),
        instance_mask=first_spots.filtered_mask,
        quality_control={
            "图像质量状态": first_preprocess.quality_status,
            "图像质量分数": first_preprocess.quality_score,
            "图像质量提示": first_preprocess.quality_flags,
            "遮挡过滤原因统计": first_spots.occlusion_filter_statistics,
            "重复运行结果一致": deterministic,
            "输入文件SHA256": metrics["source_sha256"],
            "依赖版本": versions,
        },
        limitations=[
            "普通白光RGB可见色差代理，不替代UV色斑或疾病诊断。",
            "工程置信度不是医学诊断概率。",
        ],
    )
    write_medical_metrics(
        output_dir / "检测指标.json", output_dir / "检测指标.csv", detailed, "spots"
    )
    (output_dir / "结果报告.md").write_text(result_report(metrics), encoding="utf-8")
    validate_image_outputs(output_dir, metrics)
    return metrics


def independence_checks(
    source: Path, preprocessor: ImagePreprocessor, spots_engine: SpotsEngine
) -> dict[str, bool]:
    image = cv_imread(str(source))
    if image is None:
        raise ValueError(f"无法读取重点检查图片: {source}")
    preprocess_result = preprocessor.preprocess_image(image)
    baseline = spots_engine.detect_spots(preprocess_result)
    rng = np.random.default_rng(20260710)

    display_noise = rng.integers(
        0, 256, size=preprocess_result.display_image.shape, dtype=np.uint8
    )
    display_result = spots_engine.detect_spots(
        replace(preprocess_result, display_image=display_noise)
    )

    analysis_noise = preprocess_result.analysis_image.copy()
    random_pixels = rng.integers(0, 256, size=analysis_noise.shape, dtype=np.uint8)
    outside = preprocess_result.skin_mask == 0
    analysis_noise[outside] = random_pixels[outside]
    background_result = spots_engine.detect_spots(
        replace(preprocess_result, analysis_image=analysis_noise)
    )
    return {
        "display_image_independent": spots_fingerprint(baseline)
        == spots_fingerprint(display_result),
        "mask_exterior_background_independent": spots_fingerprint(baseline)
        == spots_fingerprint(background_result),
    }


def batch_report(summary: dict[str, Any]) -> str:
    rows = [
        "| {name} | {quality} | {pre} | {large} | {salient} | {small} | {final} | {filtered} | {ratio:.4%} |".format(
            name=item["source_name"],
            quality=item["quality_status"],
            pre=item["pre_occlusion_spot_count"],
            large=item.get("large_spot_count", 0),
            salient=item.get("salient_spot_count", 0),
            small=item.get("small_spot_count", 0),
            final=item["spot_count"],
            filtered=item["hair_filtered_count"],
            ratio=item["spot_area_ratio"],
        )
        for item in summary["images"]
    ]
    return f"""# DermaVision 批量斑点检测报告

- 当前分支：`{summary['branch']}`
- 当前版本：`{summary['commit']}`
- 生成时间：`{summary['generated_at']}`
- 测试图片数量：`{summary['image_count']}`
- 质量状态统计：`{json.dumps(summary['quality_counts'], ensure_ascii=False)}`
- 输入隔离检查：`{json.dumps(summary['independence_checks'], ensure_ascii=False)}`

| 照片 | 质量状态 | 后处理前 | 大型斑 | 显著斑 | 小型斑 | 最终斑点 | 后处理过滤 | 斑点面积比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

每张照片的详细指标和六张关键阶段图片保存在同名子目录中。本报告不提供医学诊断，
也不能仅凭斑点数量变化判断检测召回率。
"""


def validate_root_outputs(sources: list[Path]) -> None:
    root_files = {path.name for path in TEST_OUTPUT.iterdir() if path.is_file()}
    root_dirs = {path.name for path in TEST_OUTPUT.iterdir() if path.is_dir()}
    expected_dirs = {source.stem for source in sources}
    if root_files != ROOT_OUTPUT_NAMES:
        raise AssertionError(f"test-output 根目录文件不正确: {root_files}")
    if root_dirs != expected_dirs:
        raise AssertionError(f"test-output 子目录不正确: {root_dirs}")
    json.loads((TEST_OUTPUT / "批量检测汇总.json").read_text(encoding="utf-8"))
    for directory in root_dirs:
        actual = {path.name for path in (TEST_OUTPUT / directory).iterdir() if path.is_file()}
        if actual != DIRECTORY_OUTPUT_NAMES:
            raise AssertionError(f"{directory} 没有严格包含 8 个规定文件")


def _resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行 DermaVision Visible Spots V2")
    parser.add_argument(
        "--input-dir",
        default="data/test_images",
        help="输入图片目录，默认 data/test_images",
    )
    parser.add_argument(
        "--output-dir",
        default="test-output/spots",
        help="输出目录，默认 test-output/spots",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global INPUT_DIR, TEST_OUTPUT
    args = parse_args(argv)
    INPUT_DIR = _resolve_cli_path(args.input_dir).resolve()
    TEST_OUTPUT = _resolve_cli_path(args.output_dir).resolve()
    sources = image_sources()
    source_hashes = {path: sha256(path) for path in sources}
    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)
    TEST_OUTPUT.mkdir(parents=True)

    branch = git_value("branch", "--show-current")
    commit = git_value("rev-parse", "HEAD")
    versions = dependency_versions()
    preprocessor = ImagePreprocessor(None, None, None)
    spots_engine = SpotsEngine()
    try:
        results = []
        for index, source in enumerate(sources, start=1):
            print(f"[{index}/{len(sources)}] {source.name}", flush=True)
            results.append(
                process_one(source, preprocessor, spots_engine, branch, commit, versions)
            )

        result_by_name = {item["source_name"]: item for item in results}
        for partial_name in ("1000065_1.jpg", "测试.png"):
            item = result_by_name.get(partial_name)
            if item is None:
                continue
            if (
                item["quality_status"] != "WARNING"
                or not item["partial_face_analysis"]
                or item["final_skin_area"] <= 0
                or item["spot_count"] <= 0
            ):
                raise AssertionError(f"半脸回归验收失败: {partial_name}: {item}")

        focus = next((path for path in sources if path.name == FOCUS_IMAGE), None)
        if focus is None:
            focus = next(
                path for path, result in zip(sources, results)
                if result["quality_status"] != "REJECT"
            )
        independence = independence_checks(focus, preprocessor, spots_engine)
        if not all(independence.values()):
            raise AssertionError(f"算法输入隔离检查失败: {independence}")

        quality_counts = {name: 0 for name in ("PASS", "WARNING", "REJECT")}
        for item in results:
            quality_counts[item["quality_status"]] += 1
        summary = {
            "branch": branch,
            "commit": commit,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_directory": str(INPUT_DIR),
            "output_directory": str(TEST_OUTPUT),
            "image_count": len(results),
            "quality_counts": quality_counts,
            "dependencies": versions,
            "independence_checks": independence,
            "images": results,
        }
        (TEST_OUTPUT / "批量检测汇总.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (TEST_OUTPUT / "批量检测报告.md").write_text(
            batch_report(summary), encoding="utf-8"
        )

        for source, original_hash in source_hashes.items():
            if sha256(source) != original_hash:
                raise AssertionError(f"原始测试图片发生变化: {source}")
        validate_root_outputs(sources)

        focus_result = next(
            item for item in results if item["source_name"] == focus.name
        )
        print(
            json.dumps(
                {
                    "branch": branch,
                    "commit": commit,
                    "image_count": len(results),
                    "quality_counts": quality_counts,
                    "focus_image": focus.name,
                    "focus_spot_count": focus_result["spot_count"],
                    "focus_large_spot_count": focus_result.get("large_spot_count", 0),
                    "files_per_image": len(DIRECTORY_OUTPUT_NAMES),
                    "test_output": str(TEST_OUTPUT),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        spots_engine.close()
        preprocessor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
