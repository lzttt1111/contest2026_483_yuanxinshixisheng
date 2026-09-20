#!/usr/bin/env python3
"""AISIA 九项皮肤检测大批量点火入口。

面向几十万张图片的长任务，提供递归扫描、分片、断点续跑、逐图落盘、
失败隔离和 ETA。三个 GPU 服务在一次进程生命周期内只冷启动一次；每张
图片内部并行执行 DermaVision 七项、皱纹和痤疮，图片之间顺序执行以避免
争抢同一块 GPU。
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from src.capture_profile import CaptureProfile
from src.aisia_medical_report.identity import subject_id_from_source
from src.detection_runtime import (
    ClinicTwelveAnalysisOrchestrator,
    TwelveOutputExporter,
    load_clinic_capture_manifest,
)
from src.nine_analysis import NineAnalysisOrchestrator
from src.nine_analysis.batch_runtime_safety import (
    GpuRunLeaseUnavailable,
    MemorySnapshot,
    acquire_export_lease,
    acquire_gpu_run_lease,
    memory_snapshot,
    resource_recycle_reason,
)
from src.nine_analysis.batch_runtime.summary import (
    BatchStatistics,
    STATE_VERSION,
    SUMMARY_CSV_FIELDS,
    summary_csv_row,
    write_summary,
)
from src.nine_analysis.batch_runtime.identifiers import (
    claim_key as _claim_key,
    safe_name as _safe_name,
)
from src.nine_analysis.batch_runtime.claims import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    WorkClaim,
    acquire_work_claim,
    preserve_terminal_claim,
    release_work_claim,
)
from src.nine_analysis.batch_runtime.quality_review import (
    archive_quality_rejection,
    quality_flags_from_record,
    quality_review_category,
)
from src.nine_analysis.batch_runtime.state_store import (
    append_state,
    commit_terminal_record,
    file_signature,
    load_completed,
    load_completed_job_states,
    load_completed_states,
    load_terminal_job_states,
    load_terminal_states,
    result_matches_signature,
    shard_number,
    should_skip_terminal,
    signature_key,
    write_batch_metadata,
    write_health_snapshot,
)
from src.nine_analysis.batch_slot_launcher import (
    child_slot_commands,
    run_child_slots,
)
from src.nine_analysis.input_manifest import (
    InputManifestError,
    iter_manifest_images,
)
from src.nine_analysis.review_output import ReviewOutputExporter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GENERATE_MEDICAL_REPORT_DEFAULT = False
LEGACY_NINE_ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores", "uv_spots",
    "porphyrin", "wrinkle", "acne",
)
ALL_TWELVE_ALGORITHMS = (
    *LEGACY_NINE_ALGORITHMS,
    "surface_gloss", "vascular", "contour_firmness",
)


def resolve_algorithms(values: list[str]) -> tuple[str, ...]:
    if values == ["all"]:
        return ALL_TWELVE_ALGORITHMS
    if values == ["legacy-nine"]:
        return LEGACY_NINE_ALGORITHMS
    invalid = sorted(set(values) - set(ALL_TWELVE_ALGORITHMS))
    if invalid:
        raise ValueError(f"未知算法: {invalid}")
    if len(values) != len(set(values)):
        raise ValueError("--algorithms 不能包含重复算法")
    return tuple(values)


def is_full_twelve_selection(values: tuple[str, ...]) -> bool:
    return (
        len(values) == len(ALL_TWELVE_ALGORITHMS)
        and set(values) == set(ALL_TWELVE_ALGORITHMS)
    )
DEFAULT_IMAGES_PER_GROUP = 2000


def default_runtime_dir(job_name: str, worker_id: str) -> Path:
    """Return a local scratch path, never a user-selected result disk.

    Nine analyzers still use file paths internally, but those files are only
    transfer stations used to assemble the formal result directory.  Keeping
    them under the output root is especially harmful when the root is a Windows
    HDD mount: every image creates and removes dozens of small debug files.
    ``/tmp`` is local to this Linux environment and is removed after a normal
    task.  The caller may still explicitly override it with ``--runtime-dir``.
    """
    return (
        Path(tempfile.gettempdir())
        / "aisia-nine-runtime"
        / job_name
        / worker_id
    )


def remove_runtime_tree(run_root: Path, runtime_root: Path) -> None:
    """Remove one completed run and its empty, job-specific scratch parents.

    Never walk above the caller's ``runtime_root``.  This deliberately leaves
    another concurrently running worker's directory intact while removing the
    otherwise confusing empty ``job/worker`` folders after a normal batch.
    """
    shutil.rmtree(run_root, ignore_errors=True)
    for directory in (runtime_root, runtime_root.parent):
        try:
            directory.rmdir()
        except OSError:
            break


def report_subject_id_for_image(path: Path) -> str:
    """Use a stable human-facing sample id, never the filename extension."""

    return subject_id_from_source(path)


def iter_directory_entries(
    root: Path,
    *,
    recursive: bool,
) -> Iterable[tuple[Path, str]]:
    """用 ``os.scandir`` 流式列目录，不构造几十万文件名列表。"""

    pending = [root]
    while pending:
        current = pending.pop()
        child_directories: list[Path] = []
        with os.scandir(current) as entries:
            for entry in entries:
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    # 单个损坏目录项不应终止几十万张长任务。
                    continue
                if is_directory:
                    if recursive:
                        child_directories.append(Path(entry.path))
                    continue
                yield current, entry.name
        # 目录数量通常远小于文件数量；只暂存目录，不暂存图片名。
        pending.extend(reversed(child_directories))


def iter_images(
    input_dir: Path,
    *,
    recursive: bool = True,
    patterns: Iterable[str] = (),
    excluded_root: Path | None = None,
) -> Iterable[Path]:
    """惰性遍历图片；输出目录位于输入目录内时自动排除。

    使用 os.scandir 惰性读取，不在 NFS 上逐文件 stat，也不把同一目录
    的几十万文件名一次性装入列表。
    """

    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    excluded = excluded_root.resolve() if excluded_root else None
    normalized_patterns = tuple(patterns)
    for parent, name in iter_directory_entries(root, recursive=recursive):
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        full = parent / name
        if excluded is not None and (full == excluded or excluded in full.parents):
            continue
        if normalized_patterns:
            relative = full.relative_to(root).as_posix()
            if not any(
                fnmatch.fnmatch(name, pattern)
                or fnmatch.fnmatch(relative, pattern)
                for pattern in normalized_patterns
            ):
                continue
        yield full


def count_stems(
    input_dir: Path,
    *,
    recursive: bool = True,
    patterns: Iterable[str] = (),
    excluded_root: Path | None = None,
) -> tuple[Counter[str], int]:
    """轻量第一遍：只统计文件名 stem 出现次数和图片总数，不持有路径列表。

    使用 os.scandir，不在 NFS 上逐文件 stat；仅显式 --precount 时调用。
    """

    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    excluded = excluded_root.resolve() if excluded_root else None
    normalized_patterns = tuple(patterns)
    stems: Counter[str] = Counter()
    total = 0
    for parent, name in iter_directory_entries(root, recursive=recursive):
        path = Path(name)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        full = parent / name
        if excluded is not None and (full == excluded or excluded in full.parents):
            continue
        if normalized_patterns:
            relative = full.relative_to(root).as_posix()
            if not any(
                fnmatch.fnmatch(path.name, pattern)
                or fnmatch.fnmatch(relative, pattern)
                for pattern in normalized_patterns
            ):
                continue
        stems[path.stem] += 1
        total += 1
    return stems, total


def compute_sample_id(
    path: Path, input_root: Path, stem_counts: Counter[str] | None
) -> str:
    """生成稳定结果目录名。

    全量预统计时沿用旧规则；默认流式模式不预扫目录，使用带扩展名的
    文件名，嵌套目录再追加相对路径哈希，避免不同目录同名时覆盖。
    """

    if stem_counts is None:
        relative = path.relative_to(input_root)
        safe_name = _safe_name(path.name)
        if relative.parent != Path("."):
            digest = hashlib.sha256(
                relative.as_posix().encode("utf-8")
            ).hexdigest()[:10]
            safe_name = f"{safe_name}__{digest}"
        return safe_name
    safe_stem = _safe_name(path.stem)
    if stem_counts[path.stem] > 1:
        relative = path.relative_to(input_root).as_posix()
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        safe_stem = f"{safe_stem}__{digest}"
    return safe_stem


def compute_output_group(global_index: int, images_per_group: int) -> str:
    """按遍历顺序分组，每个批次目录最多保存指定数量的图片结果。"""

    return f"batch_{global_index // images_per_group + 1:06d}"


def failure_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    """把九项失败压缩成适合几十万张批处理统计的小记录。"""

    items = manifest.get("九项结果") or {}
    failed_items = [
        key for key, value in items.items() if value.get("状态") != "success"
    ]
    service_errors: dict[str, str] = {}
    for service, response in (manifest.get("服务原始响应") or {}).items():
        if response.get("status") == "success":
            continue
        result = response.get("result") or {}
        message = result.get("message") or response.get("message") or "unknown error"
        service_errors[service] = str(message)
    joined = " ".join(service_errors.values()).upper()
    if "NO_FACE" in joined or "质量门禁拒绝" in joined:
        category = "input_quality_reject"
        retry_recommended = False
    elif "INDEXERROR" in joined or "OUT OF BOUNDS" in joined:
        category = "recoverable_algorithm_error"
        retry_recommended = True
    elif failed_items:
        category = "algorithm_error"
        retry_recommended = True
    else:
        category = "export_or_runner_error"
        retry_recommended = True
    return {
        "failure_category": category,
        "failed_items": failed_items,
        "service_errors": service_errors,
        "retry_recommended": retry_recommended,
    }


def batch_observation(
    manifest: dict[str, Any],
    source_root: Path,
    signature: dict[str, Any],
) -> dict[str, Any]:
    """生成评分分布用的一行记录，不包含坐标、Mask或本地输入副本。"""

    scoring_path = source_root / "scoring_features.json"
    try:
        scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        scoring = {
            "指标版本": "scoring_features_v1",
            "评分状态": "unavailable",
            "九项评分输入": {},
        }
    responses = manifest.get("服务原始响应") or {}
    derma_metadata = (
        (responses.get("dermavision") or {}).get("result", {}).get("metadata", {})
    )
    observation = {
        "state_version": STATE_VERSION,
        "capture_profile": manifest.get("采集Profile", "institution"),
        "signature": signature,
        "status": manifest.get("状态"),
        "success_items": int(manifest.get("成功项目数", 0)),
        "quality": {
            "score": derma_metadata.get("quality_score"),
            "status": derma_metadata.get("quality_status"),
            "flags": derma_metadata.get("quality_flags") or [],
            "metrics": derma_metadata.get("quality_metrics") or {},
        },
        "service_seconds": {
            key: value.get("seconds") for key, value in responses.items()
        },
        "scoring_features": scoring.get("九项评分输入") or {},
    }
    twelve_items = manifest.get("十二项结果") or {}
    if manifest.get("状态") == "success" and len(twelve_items) == 12:
        from src.nine_analysis.final_output import build_complete_document
        from src.nine_analysis.public_sanitize import sanitize_public_document

        detector_results = {}
        for item_id, item in twelve_items.items():
            metric_path = Path(str(
                item.get("完整量化JSON") or item["量化JSON"]
            ))
            metrics = sanitize_public_document(json.loads(
                metric_path.read_text(encoding="utf-8-sig")
            ))
            detector_results[item_id] = {
                "status": "success",
                "images": [],
                "public_metrics": metrics,
                "metrics": metrics,
            }
        profile = CaptureProfile(str(
            manifest.get("采集Profile", CaptureProfile.INSTITUTION.value)
        ))
        complete = build_complete_document(
            detector_results=detector_results,
            capture_profile=profile,
            input_signature={"relative_path": signature["relative_path"]},
            provenance={"capture_profile": profile.value},
            quality_control=observation["quality"],
        )
        observation["v2_scoring_features"] = complete["scoring_features"]
        observation["v2_scoring_results"] = complete["scoring_results"]
        observation["v2_scoring_completeness"] = {
            module_id: module["completeness"]
            for module_id, module in complete["medical_v2_modules"].items()
        }
    if manifest.get("状态") != "success":
        observation.update(failure_diagnostics(manifest))
    return observation


def preserve_failure_diagnostics(
    source_root: Path,
    failure_root: Path,
    sample_id: str,
    *,
    include_artifacts: bool = False,
) -> Path | None:
    """只保留失败索引和耗时，避免失败现场累积海量图片。"""

    target = failure_root / sample_id
    target.mkdir(parents=True, exist_ok=True)
    copied = False
    for name in ("manifest.json", "timing.json", "nine_metrics.json"):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, target / name)
            copied = True
    if include_artifacts:
        artifact_root = target / "partial_artifacts"
        for service in ("dermavision", "acne", "wrinkle"):
            source = source_root / service
            if source.is_dir():
                shutil.copytree(source, artifact_root / service)
                copied = True
    return target if copied else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AISIA 九项皮肤检测大批量点火程序（GPU 常驻、断点续跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  uv run --frozen python run.py \\
    --input-dir /data/skin-images \\
    --output-dir /data/skin-analysis-results

38万张评分统计请加 --output-profile scoring；逐图验收使用默认 review。
再次执行相同命令会断点续跑。
""",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        metavar="PATH",
        help="consumer单RGB输入图片根目录；与四光源manifest入口二选一",
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        help="institution固定四光源fixture_manifest.json",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help=(
            "consumer冻结样本JSONL；启用后只读取manifest明确列出的图片，"
            "不会枚举input-dir"
        ),
    )
    parser.add_argument(
        "--capture-manifest-dir",
        type=Path,
        help="包含一个或多个fixture_manifest.json的institution目录",
    )
    parser.add_argument(
        "--capture-alias",
        action="append",
        choices=("clinic28-25", "clinic28-09", "clinic28-23"),
        help="仅运行指定固定四光源样本；可重复，默认运行manifest全部样本",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="输出结果根目录（必须明确指定；--output 为兼容别名）",
    )
    parser.add_argument("--job-name", default="nine-analysis-batch")
    parser.add_argument(
        "--output-profile",
        choices=("review", "scoring"),
        default="review",
        help=(
            "review保存逐图正式结果；scoring仅保存状态和评分分布JSONL，"
            "用于几十万张常模统计，避免产生上千万小文件"
        ),
    )
    parser.add_argument(
        "--images-per-group",
        type=int,
        default=DEFAULT_IMAGES_PER_GROUP,
        metavar="N",
        help="每个 batch_xxxxxx 目录最多保存的图片结果数；默认 2000",
    )
    parser.add_argument(
        "--worker-id",
        default=f"{socket.gethostname()}-{os.getpid()}",
        help="多机运行标识，建议显式设置为机器名；默认使用主机名和进程号",
    )
    parser.add_argument(
        "--claim-timeout-seconds",
        type=int,
        default=DEFAULT_CLAIM_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="共享抢占锁的失效时间；默认 21600 秒（6小时）",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="图片过滤模式，可重复，例如 --glob '*.jpg' --glob 'visia*.png'",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="递归扫描输入目录；默认开启",
    )
    parser.add_argument(
        "--precount",
        action="store_true",
        help="处理前先完整统计目录和重名；默认关闭，NFS 大数据直接流式处理",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="根据图片路径、大小和修改时间断点续跑；默认开启",
    )
    parser.add_argument("--shard-count", type=int, default=1, help="总分片数")
    parser.add_argument("--shard-index", type=int, default=0, help="当前分片编号，从0开始")
    parser.add_argument("--start-index", type=int, default=0, help="在当前分片中跳过前N张")
    parser.add_argument("--limit", type=int, help="本次最多处理图片数；测试时使用")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument(
        "--slots",
        type=int,
        choices=(1, 2),
        default=1,
        help="一个run命令内部管理的持久批跑槽数量；默认1，可选2",
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="允许同一GPU最多两个显式批跑进程；默认只允许一个",
    )
    parser.add_argument(
        "--skip-recorded-failures",
        action="store_true",
        help="断点续跑时将已记录的partial/failed视为终态，避免反复阻塞",
    )
    parser.add_argument(
        "--qc-review-dir",
        type=Path,
        help="将门禁拒绝原图按原因分类复制到人工审核目录；默认关闭",
    )
    parser.add_argument(
        "--image-timeout-seconds",
        type=float,
        default=1800,
        help="单张图片三服务墙钟超时；默认1800秒",
    )
    parser.add_argument(
        "--recycle-every",
        type=int,
        default=8,
        help="每完成N张受控重建常驻服务；默认8，0表示仅资源门触发",
    )
    parser.add_argument(
        "--min-available-memory-gib",
        type=float,
        default=16.0,
        help="启动和运行资源门要求的最小WSL可用内存GiB",
    )
    parser.add_argument(
        "--max-swap-gib",
        type=float,
        default=6.0,
        help="达到该Swap使用量时停止继续取新图片并回收服务",
    )
    parser.add_argument(
        "--min-runtime-available-memory-gib",
        type=float,
        default=4.0,
        help="运行中低于该可用内存时安全停止；默认4GiB",
    )
    parser.add_argument(
        "--capture-profile",
        choices=tuple(profile.value for profile in CaptureProfile),
        default=CaptureProfile.CONSUMER.value,
        help="显式采集场景；正式环境不根据图片像素自动推断",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help=(
            "算法临时目录；默认使用 Linux 系统临时盘，避免在 D 盘等输出盘"
            "反复创建中间图片"
        ),
    )
    parser.add_argument(
        "--generate-medical-report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "默认不生成医学DOCX；review可用--generate-medical-report显式开启；"
            "scoring禁止开启"
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help=(
            "默认all十二项；review正式输出只支持all或legacy-nine，"
            "任意算法子集仅用于scoring工程证据"
        ),
    )
    parser.add_argument(
        "--report-subject-id",
        default=None,
        help="正式报告受检者编号；固定验收样本默认使用各自clinic别名",
    )
    parser.add_argument(
        "--save-debug-artifacts",
        "--keep-runtime",
        dest="keep_runtime",
        action="store_true",
        help=(
            "保留算法中间产物和 partial 现场；默认关闭，"
            "--keep-runtime 为兼容别名"
        ),
    )
    parser.add_argument(
        "--retain-scoring-artifacts",
        action="store_true",
        help="review批跑额外保留根目录十二项核心JSON/CSV，供评分映射",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="单张失败后立即停止；默认记录失败并继续下一张",
    )
    return parser


def resolve_medical_report_default(
    output_profile: str,
    requested: bool | None,
) -> bool:
    """Generate reports only when the review caller explicitly requests them."""
    if requested is not None:
        return requested
    return GENERATE_MEDICAL_REPORT_DEFAULT and output_profile == "review"


def _capture_manifest_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.capture_manifest is not None:
        return (args.capture_manifest.resolve(strict=True),)
    if args.capture_manifest_dir is None:
        raise ValueError("institution四光源入口缺少manifest")
    root = args.capture_manifest_dir.resolve(strict=True)
    candidates: list[Path] = []
    direct = root / "fixture_manifest.json"
    if direct.is_file():
        candidates.append(direct.resolve(strict=True))
    candidates.extend(
        candidate.resolve(strict=True)
        for child in sorted(root.iterdir(), key=lambda value: value.name)
        if child.is_dir() and (candidate := child / "fixture_manifest.json").is_file()
    )
    if not candidates:
        raise ValueError(f"未找到四光源fixture_manifest.json: {root}")
    return tuple(candidates)


def _run_clinic_mode(args: argparse.Namespace) -> int:
    if args.capture_profile != CaptureProfile.INSTITUTION.value:
        raise ValueError("四光源manifest只允许institution Profile")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    captures = tuple(
        capture
        for manifest in _capture_manifest_paths(args)
        for capture in load_clinic_capture_manifest(manifest)
    )
    if args.capture_alias:
        selected = frozenset(args.capture_alias)
        captures = tuple(
            capture for capture in captures if capture.capture_alias in selected
        )
        if {capture.capture_alias for capture in captures} != selected:
            raise ValueError("INVALID_CAPTURE_SET: selected alias missing from manifest")
    aliases = tuple(capture.capture_alias for capture in captures)
    if len(aliases) != len(set(aliases)):
        raise ValueError("INVALID_CAPTURE_SET: duplicate capture alias")
    runtime_root = (
        args.runtime_dir.resolve()
        if args.runtime_dir is not None
        else default_runtime_dir(_safe_name(args.job_name), _safe_name(args.worker_id))
    )
    print(f"四光源固定采集: {', '.join(aliases)}")
    print(f"算法临时目录: {runtime_root}")
    orchestrator = ClinicTwelveAnalysisOrchestrator(
        runtime_root,
        cuda_device=args.cuda_device,
        stage_input=False,
    )
    exporter = TwelveOutputExporter(output_root)
    completed = 0
    started = time.perf_counter()
    try:
        cold = orchestrator.start()
        print(f"GPU服务冷启动: {cold.get('orchestrator_wall_seconds', 'unknown')} 秒")
        for capture in captures:
            item_started = time.perf_counter()
            result = orchestrator.analyze_capture(capture)
            source_root = orchestrator.run_root / "repeat_1" / capture.capture_alias
            destination = exporter.export(
                source_root,
                result,
                destination_name=capture.capture_alias,
            )
            if args.generate_medical_report:
                from src.aisia_medical_report.clinic_twelve_delivery import (
                    generate_clinic_dual_reports_from_twelve_result,
                )

                generate_clinic_dual_reports_from_twelve_result(
                    destination,
                    subject_id=args.report_subject_id or capture.capture_alias,
                    runtime_items=result["十二项结果"],
                )
            else:
                from src.aisia_medical_report.controlled_evidence import (
                    materialize_delivery_evidence,
                )

                materialize_delivery_evidence(
                    source_image=destination / "00_输入图像" / "RGB_M.jpg",
                    runtime_items=result["十二项结果"],
                    result_root=destination,
                    capture_profile=CaptureProfile.INSTITUTION,
                )
            from src.detection_runtime.final_delivery_layout import (
                finalize_institution_delivery,
            )
            finalize_institution_delivery(
                destination,
                result["十二项结果"],
            )
            completed += 1
            print(
                f"{capture.capture_alias}: 12/12 -> {destination} "
                f"({time.perf_counter() - item_started:.2f}s)"
            )
            if not args.keep_runtime:
                shutil.rmtree(source_root, ignore_errors=True)
    finally:
        orchestrator.close()
        if not args.keep_runtime:
            remove_runtime_tree(orchestrator.run_root, runtime_root)
    print(f"四光源串行完成: {completed}/{len(captures)}，总耗时 {time.perf_counter() - started:.2f}s")
    return 0 if completed == len(captures) else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.slots == 2:
        if args.capture_manifest is not None or args.capture_manifest_dir is not None:
            parser.error("--slots 2当前只支持consumer单RGB批跑")
        source_arguments = tuple(argv if argv is not None else sys.argv[1:])
        commands = child_slot_commands(
            source_arguments,
            python=Path(sys.executable).absolute(),
            script=Path(__file__).resolve(),
            worker_id=_safe_name(args.worker_id),
            runtime_root=(
                args.runtime_dir.resolve()
                if args.runtime_dir is not None
                else None
            ),
        )
        print("双槽批跑: 一个run主命令管理slot-a/slot-b，统一中断与续跑。")
        return run_child_slots(commands)
    args.generate_medical_report = resolve_medical_report_default(
        args.output_profile,
        args.generate_medical_report,
    )
    if args.capture_manifest is not None or args.capture_manifest_dir is not None:
        try:
            return _run_clinic_mode(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.input_dir is None:
        parser.error("consumer单RGB入口必须提供--input-dir")
    if args.shard_count < 1:
        parser.error("--shard-count 必须 >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index 必须在 [0, shard-count) 范围内")
    if args.start_index < 0:
        parser.error("--start-index 必须 >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须 >= 1")
    if args.input_manifest is not None and args.precount:
        parser.error("--input-manifest禁止与--precount同时使用")
    if args.image_timeout_seconds <= 0:
        parser.error("--image-timeout-seconds 必须 > 0")
    if args.recycle_every < 0:
        parser.error("--recycle-every 必须 >= 0")
    if args.min_available_memory_gib <= 0:
        parser.error("--min-available-memory-gib 必须 > 0")
    if args.max_swap_gib <= 0:
        parser.error("--max-swap-gib 必须 > 0")
    if args.min_runtime_available_memory_gib <= 0:
        parser.error("--min-runtime-available-memory-gib 必须 > 0")
    if args.images_per_group < 1:
        parser.error("--images-per-group 必须 >= 1")
    if args.claim_timeout_seconds < 1:
        parser.error("--claim-timeout-seconds 必须 >= 1")
    if args.generate_medical_report and args.output_profile != "review":
        parser.error("--generate-medical-report 只能与 --output-profile review 同时使用")
    try:
        selected_algorithms = resolve_algorithms(args.algorithms)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.output_profile == "review"
        and selected_algorithms != LEGACY_NINE_ALGORITHMS
        and not is_full_twelve_selection(selected_algorithms)
    ):
        parser.error(
            "review正式输出只支持all或legacy-nine；"
            "任意算法子集请显式使用--output-profile scoring"
        )
    job_name = _safe_name(args.job_name)
    worker_id = _safe_name(args.worker_id)
    input_root = args.input_dir.resolve(strict=True)
    output_root = args.output_dir.resolve()

    # NFS 大数据默认不做第一遍全量目录统计，发现一张就过滤、抢占和处理
    # 一张。只有人工明确要求 --precount 时才沿用旧的全量预统计规则。
    stem_counts: Counter[str] | None = None
    total_images: int | None = None
    total_groups: int | None = None
    if args.input_manifest is not None:
        print(f"冻结输入manifest: {args.input_manifest.resolve(strict=True)}")
        print("Manifest模式: 不枚举输入目录，只读取冻结相对路径。")
    elif args.precount:
        print("正在扫描图片目录（预统计模式）...")
        stem_counts, total_images = count_stems(
            input_root,
            recursive=args.recursive,
            patterns=args.glob,
            excluded_root=output_root,
        )
        if total_images == 0:
            parser.error(f"未找到符合条件的图片: {input_root}")
        total_groups = (
            total_images + args.images_per_group - 1
        ) // args.images_per_group
        print(
            f"扫描完成: 共 {total_images} 张图片，"
            f"{total_groups} 个 batch 目录。"
        )
    else:
        print("流式模式: 不预扫描全目录，逐条读取并立即处理。")
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    group_message = f"分组规则: 每组最多 {args.images_per_group} 张"
    if total_groups is not None:
        group_message += f"，当前扫描结果共 {total_groups} 个 batch 目录"
    print(group_message + "。")

    state_job_name = (
        job_name
        if args.shard_count == 1
        else f"{job_name}-shard-{args.shard_index:04d}-of-{args.shard_count:04d}"
    )
    batch_root = output_root / "_batch"
    state_dir = batch_root / state_job_name
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"state.{worker_id}.jsonl"
    print(f"状态文件: {state_path}")
    print(f"执行节点: {worker_id}")
    completed = (
        load_terminal_job_states(
            batch_root,
            job_name,
            include_failures=args.skip_recorded_failures,
        )
        if args.resume
        else set()
    )
    if args.resume:
        print(f"跨分片断点记录: {len(completed)} 张已成功图片。")

    if args.runtime_dir is not None:
        runtime_root = args.runtime_dir.resolve()
    else:
        # ``review`` 也必须与 ``scoring`` 一样使用本地临时盘。
        # D 盘只保留 export() 白名单中的正式结果和断点状态，不能再
        # 留下 .runtime、日志、Mask、热力图或调试通道。
        runtime_root = default_runtime_dir(job_name, worker_id)
    print(f"算法临时目录: {runtime_root}")
    claim_root = output_root / "_batch" / "_claims"
    exporters: dict[str, ReviewOutputExporter] = {}
    initial_memory = memory_snapshot()
    initial_reason = resource_recycle_reason(
        initial_memory,
        min_available_gib=args.min_available_memory_gib,
        max_swap_gib=args.max_swap_gib,
    )
    if initial_reason is not None:
        parser.error(
            "启动资源门拒绝: "
            f"{initial_reason}; available={initial_memory.available_gib:.2f}GiB, "
            f"swap={initial_memory.swap_used_gib:.2f}GiB"
        )
    try:
        gpu_lease = acquire_gpu_run_lease(
            Path(tempfile.gettempdir()) / "aisia-twelve-gpu-locks",
            args.cuda_device,
            allow_shared=args.allow_shared_gpu,
        )
    except GpuRunLeaseUnavailable as exc:
        parser.error(str(exc))

    def build_orchestrator() -> NineAnalysisOrchestrator:
        return NineAnalysisOrchestrator(
            runtime_root,
            cuda_device=args.cuda_device,
            stage_input=True,
            algorithms=selected_algorithms,
            capture_profile=CaptureProfile(args.capture_profile),
        )

    # 每张输入先复制到Linux临时盘，避免三个常驻服务并发读取NFS/HDD源图。
    orchestrator = build_orchestrator()
    statistics = BatchStatistics()
    cold_start: dict[str, Any] | None = None
    interrupted = False
    progress: tqdm | None = None
    selected_count = 0
    skipped_count = 0
    position = 0
    completed_since_recycle = 0
    restart_count = 0
    resource_stop_reason: str | None = None
    retry_failed_terminal_before = datetime.now()
    scoring_path = state_dir / f"scoring_features.{worker_id}.jsonl"
    summary_csv_path = state_dir / f"{state_job_name}.{worker_id}.summary.csv"
    health_path = state_dir / f"health.{worker_id}.json"
    with (
        state_path.open("a", encoding="utf-8") as state_handle,
        scoring_path.open("a", encoding="utf-8") as scoring_handle,
        summary_csv_path.open("w", encoding="utf-8-sig", newline="") as summary_handle,
    ):
        summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_CSV_FIELDS)
        summary_writer.writeheader()
        try:
            cold_start = orchestrator.start()
            print(
                "GPU 服务冷启动完成: "
                f"{cold_start.get('orchestrator_wall_seconds', 'unknown')} 秒"
            )
            write_health_snapshot(
                health_path,
                {
                    "status": "running",
                    "worker_id": worker_id,
                    "processed": 0,
                    "restart_count": 0,
                    "memory": asdict(initial_memory),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print("正在流式扫描并处理图片（第二遍）...")
            started_batch = time.perf_counter()
            progress = tqdm(
                desc="批量检测",
                unit="张",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            # ── 冻结manifest或惰性目录流，逐张过滤并处理 ──
            shard_seen = 0  # 当前分片中已遇到的图片数（用于 start_index）
            image_stream = (
                iter_manifest_images(input_root, args.input_manifest)
                if args.input_manifest is not None
                else iter_images(
                    input_root,
                    recursive=args.recursive,
                    patterns=args.glob,
                    excluded_root=output_root,
                )
            )
            for global_index, image in enumerate(
                image_stream
            ):
                # 分片过滤
                if shard_number(image, input_root, args.shard_count) != args.shard_index:
                    continue
                # start_index 跳过
                if shard_seen < args.start_index:
                    shard_seen += 1
                    continue
                shard_seen += 1
                # 计算 sample_id 和 output_group（无需全量字典）
                sample_id = (
                    _safe_name(image.parent.name)
                    if args.input_manifest is not None
                    else compute_sample_id(image, input_root, stem_counts)
                )
                output_group = compute_output_group(global_index, args.images_per_group)
                expected_output = output_root / output_group / sample_id

                # 断点续跑过滤
                signature = file_signature(image, input_root)
                signature["capture_profile"] = args.capture_profile
                if (
                    args.output_profile == "review"
                    and result_matches_signature(expected_output, signature)
                ):
                    skipped_count += 1
                    continue
                result_index = expected_output / (
                    "十二项检测结果索引.json"
                    if selected_algorithms != LEGACY_NINE_ALGORITHMS
                    else "九项检测结果索引.json"
                )
                if should_skip_terminal(
                    signature_key(signature),
                    completed,
                    output_profile=args.output_profile,
                    skip_recorded_failures=args.skip_recorded_failures,
                    result_index_exists=result_index.is_file(),
                ):
                    skipped_count += 1
                    continue

                # limit 只限制本次实际新增处理数量；断点跳过不占额度。
                if args.limit is not None and position >= args.limit:
                    break
                selected_count += 1

                # ── 正式处理 ──
                position += 1
                started = time.perf_counter()
                record: dict[str, Any] = {
                    "state_version": STATE_VERSION,
                    "job_name": state_job_name,
                    "index": position,
                    "signature": signature,
                    "sample_id": sample_id,
                    "output_group": output_group,
                    "worker_id": worker_id,
                    "capture_profile": args.capture_profile,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                }
                claim = acquire_work_claim(
                    claim_root,
                    signature,
                    worker_id=worker_id,
                    stale_after_seconds=args.claim_timeout_seconds,
                    retry_failed_terminal=not args.skip_recorded_failures,
                    retry_failed_terminal_before=retry_failed_terminal_before,
                )
                if claim is None:
                    record.update(
                        status="claimed_elsewhere",
                        seconds=round(time.perf_counter() - started, 4),
                        finished_at=datetime.now().isoformat(timespec="seconds"),
                    )
                    append_state(state_handle, record)
                    summary_writer.writerow(summary_csv_row(record))
                    summary_handle.flush()
                    statistics.add(record)
                    progress.update(1)
                    progress.set_postfix_str("claimed_elsewhere")
                    tqdm.write(
                        f"[{position}] {signature['relative_path']}: "
                        "claimed_elsewhere，已由其他节点处理。"
                    )
                    continue
                export_lease = None
                try:
                    try:
                        result = orchestrator.analyze(
                            image,
                            repetition=1,
                            sample_id=sample_id,
                            timeout_seconds=args.image_timeout_seconds,
                        )
                        record["success_items"] = int(result.get("成功项目数", 0))
                        source_root = orchestrator.run_root / "repeat_1" / sample_id
                        observation = batch_observation(result, source_root, signature)
                        scoring_handle.write(
                            json.dumps(
                                observation,
                                ensure_ascii=False,
                                allow_nan=False,
                            ) + "\n"
                        )
                        scoring_handle.flush()
                        os.fsync(scoring_handle.fileno())

                        if (
                            result.get("状态") != "success"
                            or record["success_items"] != len(selected_algorithms)
                        ):
                            record.update(failure_diagnostics(result))
                            record["status"] = (
                                "partial_success"
                                if record["success_items"] > 0
                                else "failed"
                            )
                            diagnostics = preserve_failure_diagnostics(
                                source_root,
                                state_dir / "failures",
                                sample_id,
                                include_artifacts=(
                                    args.keep_runtime
                                    and record["success_items"] > 0
                                ),
                            )
                            if diagnostics is not None:
                                record["diagnostics_dir"] = diagnostics.relative_to(
                                    output_root
                                ).as_posix()
                            record["error"] = (
                                f"请求项目未全部成功: {result.get('状态')} "
                                f"({record['success_items']}/{len(selected_algorithms)})"
                            )
                        elif args.output_profile == "review":
                            export_lease = acquire_export_lease(output_root)
                            exporter = exporters.get(output_group)
                            if exporter is None:
                                exporter = ReviewOutputExporter(
                                    output_root / output_group
                                )
                                exporters[output_group] = exporter
                            exported = exporter.export(
                                source_root,
                                result,
                                destination_name=sample_id,
                                include_input=False,
                            )
                            if exported != expected_output:
                                raise RuntimeError(
                                    f"输出目录异常: expected={expected_output}, "
                                    f"actual={exported}"
                                )
                            write_batch_metadata(
                                exported,
                                signature=signature,
                                output_group=output_group,
                            )
                            if args.generate_medical_report and is_full_twelve_selection(selected_algorithms):
                                if args.capture_profile == CaptureProfile.CONSUMER.value:
                                    from src.aisia_medical_report import (
                                        generate_single_rgb_v012_reports,
                                    )

                                    generate_single_rgb_v012_reports(
                                        exported,
                                        source_image=image,
                                        subject_id=(
                                            args.report_subject_id
                                            or report_subject_id_for_image(image)
                                        ),
                                        runtime_items=result["十二项结果"],
                                    )
                                else:
                                    from src.aisia_medical_report import (
                                        generate_dual_reports_from_twelve_result,
                                    )

                                    generate_dual_reports_from_twelve_result(
                                        exported,
                                        source_image=image,
                                        subject_id=(
                                            args.report_subject_id
                                            or report_subject_id_for_image(image)
                                        ),
                                    )
                            elif is_full_twelve_selection(selected_algorithms):
                                from src.aisia_medical_report.controlled_evidence import (
                                    materialize_delivery_evidence,
                                )

                                materialize_delivery_evidence(
                                    source_image=image,
                                    runtime_items=result["十二项结果"],
                                    result_root=exported,
                                    capture_profile=CaptureProfile.CONSUMER,
                                )
                            elif args.generate_medical_report:
                                tqdm.write("结果未齐套，跳过Word；仅完整十二项允许生成正式报告。")
                            if is_full_twelve_selection(selected_algorithms):
                                from src.detection_runtime.final_delivery_layout import (
                                    finalize_consumer_delivery,
                                    materialize_consumer_scoring_artifacts,
                                )
                                finalize_consumer_delivery(exported, image)
                                if args.retain_scoring_artifacts:
                                    materialize_consumer_scoring_artifacts(exported)
                            export_lease.close()
                            export_lease = None
                            record.update(
                                status="success",
                                output_dir=exported.relative_to(output_root).as_posix(),
                            )
                        else:
                            record.update(
                                status="success",
                                output_dir=scoring_path.relative_to(output_root).as_posix(),
                            )
                        if not args.keep_runtime:
                            shutil.rmtree(source_root, ignore_errors=True)
                    except Exception as exc:
                        if export_lease is not None:
                            export_lease.close()
                            export_lease = None
                        record.setdefault("status", "failed")
                        record.setdefault(
                            "failure_category", "export_or_runner_error"
                        )
                        record.setdefault("retry_recommended", True)
                        record["error"] = f"{type(exc).__name__}: {exc}"
                except (KeyboardInterrupt, SystemExit):
                    # 未提交终态时保留claim，等待超时后由下一次续跑接管。
                    if export_lease is not None:
                        export_lease.close()
                    raise
                record["seconds"] = round(time.perf_counter() - started, 4)
                record["finished_at"] = datetime.now().isoformat(timespec="seconds")
                if args.qc_review_dir is not None:
                    try:
                        archived = archive_quality_rejection(
                            args.qc_review_dir.resolve(),
                            image,
                            input_root,
                            record,
                        )
                        if archived is not None:
                            record["qc_review_path"] = str(archived)
                    except Exception as exc:
                        record["qc_review_archive_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                commit_terminal_record(
                    state_handle,
                    record,
                    claim,
                    preserve_claim=(
                        args.skip_recorded_failures
                        or record.get("status") in {"partial_success", "failed"}
                    ),
                )
                summary_writer.writerow(summary_csv_row(record))
                summary_handle.flush()
                statistics.add(record)
                completed_since_recycle += 1

                progress.update(1)
                progress.set_postfix_str(
                    f"{record['status']} {record.get('success_items', 0)}/{len(selected_algorithms)}"
                )
                tqdm.write(
                    f"[{position}] {signature['relative_path']}: "
                    f"{record['status']} ({record.get('success_items', 0)}/{len(selected_algorithms)}), "
                    f"{record['seconds']:.2f}s"
                )
                current_memory = memory_snapshot()
                resource_stop_reason = resource_recycle_reason(
                    current_memory,
                    min_available_gib=args.min_runtime_available_memory_gib,
                    max_swap_gib=args.max_swap_gib,
                )
                if position % 10 == 0 or resource_stop_reason is not None:
                    write_health_snapshot(
                        health_path,
                        {
                            "status": (
                                "resource_stop"
                                if resource_stop_reason is not None
                                else "running"
                            ),
                            "worker_id": worker_id,
                            "processed": position,
                            "last_status": record["status"],
                            "restart_count": restart_count,
                            "memory": asdict(current_memory),
                            "reason": resource_stop_reason,
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                if resource_stop_reason is not None:
                    tqdm.write(f"资源门安全停止: {resource_stop_reason}")
                    interrupted = True
                    break
                should_recycle = (
                    args.recycle_every > 0
                    and completed_since_recycle >= args.recycle_every
                )
                should_recycle = should_recycle or (
                    record["status"] in {"failed", "partial_success"}
                    and bool(record.get("retry_recommended"))
                )
                if should_recycle:
                    old_run_root = orchestrator.run_root
                    orchestrator.close()
                    if not args.keep_runtime:
                        shutil.rmtree(old_run_root, ignore_errors=True)
                    orchestrator = build_orchestrator()
                    recycle_cold = orchestrator.start()
                    restart_count += 1
                    completed_since_recycle = 0
                    tqdm.write(
                        "常驻服务受控回收完成: "
                        f"{recycle_cold.get('orchestrator_wall_seconds', 'unknown')}秒"
                    )
                if (
                    record["status"] in {"failed", "partial_success"}
                    and args.stop_on_error
                ):
                    break
        except KeyboardInterrupt:
            interrupted = True
            tqdm.write("收到中断信号；已完成图片的状态已落盘，可使用 --resume 继续。")
        finally:
            if progress is not None:
                progress.close()
            orchestrator.close()
            gpu_lease.close()
            # 无论 review / scoring，正式结果已经导出或状态已落盘后，
            # 整个本次运行现场（服务日志、预处理图、中间 Mask 等）都
            # 不应留在磁盘。--keep-runtime 只给算法排障时显式使用。
            if not args.keep_runtime:
                remove_runtime_tree(orchestrator.run_root, runtime_root)
            final_memory = memory_snapshot()
            write_health_snapshot(
                health_path,
                {
                    "status": (
                        "resource_stop"
                        if resource_stop_reason is not None
                        else ("interrupted" if interrupted else "completed")
                    ),
                    "worker_id": worker_id,
                    "processed": position,
                    "restart_count": restart_count,
                    "memory": asdict(final_memory),
                    "reason": resource_stop_reason,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

    if position == 0 and not interrupted:
        print("所有选中图片均已成功完成，无需处理。")
    print(
        f"本次统计: 分片选择 {selected_count} 张，"
        f"断点跳过 {skipped_count} 张，实际处理 {position} 张。"
    )
    write_summary(
        state_dir,
        job_name=state_job_name,
        input_root=input_root,
        output_root=output_root,
        images_per_group=args.images_per_group,
        worker_id=worker_id,
        selected=selected_count,
        skipped=skipped_count,
        statistics=statistics,
        cold_start=cold_start,
        capture_profile=CaptureProfile(args.capture_profile),
    )
    if interrupted:
        return 130
    return 0 if statistics.all_successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
