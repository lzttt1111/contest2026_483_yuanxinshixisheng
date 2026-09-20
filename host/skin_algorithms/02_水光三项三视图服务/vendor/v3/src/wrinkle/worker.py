# -*- coding: utf-8 -*-
"""Celery 异步任务 Worker。

启动前必须将 SERVICE_NAME 替换为实际服务名（见 .env.example），否则启动守卫会拒绝启动。

后端对接契约（ai-skin-backend → Celery）：
  - Queue:       {SERVICE_NAME}
  - Task Name:   {SERVICE_NAME}.analyze_image
  - 参数:        task_id (str)
                 oss_key (str)
                 oss_result_prefix (str | None)
                 algorithms (list[str] | None)
  - 返回:        raw_result 返回全部产物；display_result 额外标记用户展示图；
                 metadata 含 algorithm_version（算法版本，供后端落 skin_tasks 快照）
                 失败时: {"record_id": task_id, "status": "failed", "error_message": "..."}

流程：经 backend internal-api 下载图片 → Pipeline 预处理+检测 → 经 internal-api 上传结果 → 返回 object_key。

启动命令:
  celery -A src.worker worker --loglevel=info --queues=$SERVICE_NAME
"""

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from aisia_contracts.algorithms.wrinkle.v1 import SCHEMA_VERSION as _WRINKLE_SV
from src.wrinkle.core.config import settings
from src.core.storage import (
    StorageError,
    download_via_internal_api,
    upload_via_internal_api,
)
from src.wrinkle.medical_v2_delivery import load_and_validate

logger = logging.getLogger(__name__)
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 队列名/任务名写死为 "wrinkle"（项目规则：worker 自包含，不经启动脚本注入）。
# 对应 backend implements/wrinkle_v1.py 的 celery_task="wrinkle.analyze_image" queue="wrinkle"。
_SERVICE_NAME = "wrinkle"

# ───────────────────────────────────────────────────────
# Celery 实例化
# ───────────────────────────────────────────────────────
celery_app = Celery(
    _SERVICE_NAME,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.wrinkle.worker"],
)

# 任务队列配置（队列名 = 服务名）
celery_app.conf.task_queues = {
    _SERVICE_NAME: {
        "exchange": _SERVICE_NAME,
        "exchange_type": "direct",
        "routing_key": _SERVICE_NAME,
    },
    "consumer_wrinkle": {
        "exchange": "consumer_wrinkle",
        "exchange_type": "direct",
        "routing_key": "consumer_wrinkle",
    },
}
celery_app.conf.task_default_queue = _SERVICE_NAME
celery_app.conf.task_default_exchange = _SERVICE_NAME
celery_app.conf.task_default_routing_key = _SERVICE_NAME

# 可靠性 / 超时 / 序列化
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=settings.task_soft_time_limit,
    task_time_limit=settings.task_time_limit,
    result_expires=86400,
    result_serializer="json",
    task_serializer="json",
    accept_content=["json"],
)

# 全局 Pipeline 实例（延迟初始化，避免 import 时加载模型）
_pipeline = None


def _failed(record_id: object, message: object, *, schema_version: str = _WRINKLE_SV, **details: object) -> dict:
    payload = {
        "record_id": str(record_id),
        "status": "failed",
        "schema_version": schema_version,
        "error_message": str(message),
        **details,
    }
    return payload


def _get_pipeline():
    """延迟初始化 Pipeline 单例（避免 Worker 启动时就加载模型）。"""
    global _pipeline
    if _pipeline is None:
        from src.wrinkle.pipeline import Pipeline

        _pipeline = Pipeline()
        logger.info("Pipeline 实例已初始化。")
    return _pipeline


@worker_process_init.connect
def _warm_worker_process(**_kwargs) -> None:
    if os.getenv("WRINKLE_WARMUP_ON_START", "true").strip().lower() in {
        "0", "false", "no", "off",
    }:
        logger.info("Wrinkle 子进程冷启动预热已关闭。")
        return
    started = time.perf_counter()
    metadata = _get_pipeline().warmup()
    logger.info(
        "Wrinkle 子进程冷启动完成 elapsed=%.3fs metadata=%s",
        time.perf_counter() - started,
        metadata,
    )


@worker_process_shutdown.connect
def _close_worker_process(**_kwargs) -> None:
    global _pipeline
    if _pipeline is not None:
        _pipeline.close()
        _pipeline = None


def _suffix_from_oss_key(oss_key: str) -> str:
    suffix = Path(urlparse(oss_key).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return suffix
    return ".jpg"


def _build_error_details(pipeline_result: dict) -> dict:
    detail_keys = (
        "returncode",
        "elapsed_seconds",
        "output_dir",
        "algorithm_args",
        "summary",
        "traceback",
    )
    return {
        key: pipeline_result[key]
        for key in detail_keys
        if pipeline_result.get(key) not in (None, "")
    }


def _resolve_upload_file_name(
    name: str,
    local_path: str,
    upload_relative_paths: dict[str, str],
) -> str:
    """计算上传文件名（保留 pipeline 声明的相对子路径），完整路径由 backend 统一拼接。"""
    relative_path = upload_relative_paths.get(name)
    if relative_path:
        safe_relative = Path(relative_path)
        if safe_relative.is_absolute() or ".." in safe_relative.parts:
            raise ValueError(f"Invalid upload relative path: {relative_path!r}")
        return safe_relative.as_posix()
    suffix = Path(local_path).suffix or ".jpg"
    return f"{name}{suffix}"


def _build_medical_v2_additions(
    *, record_id: str, output_dir: Path
) -> dict:
    """上传可选医学 V2；异常只关闭旁路，不改变旧皱纹成功结果。"""
    if not settings.enable_medical_metrics_v2:
        return {}
    try:
        json_path = output_dir / "皱纹量化指标.json"
        csv_path = output_dir / "皱纹医学量化指标_V2.csv"
        document = load_and_validate(json_path, csv_path)
        object_key = upload_via_internal_api(str(csv_path), record_id, "wrinkle", csv_path.name)
        return {
            "medical_metrics_v2": document,
            "medical_report_csv_v2": object_key,
        }
    except Exception:
        logger.exception(
            "Record %s: 医学 V2 旁路失败，保留旧皱纹合同成功返回",
            record_id,
        )
        return {}


def _build_display_result(
    display_manifest: dict,
    display_results: dict[str, str],
    raw_result: dict[str, str],
) -> dict:
    """从完整 raw_result 中标记当前需要给用户展示的文件。"""

    def build_fixed(result_key: str | None):
        if not result_key or result_key not in raw_result:
            return None
        local_path = display_results.get(result_key, "")
        return {
            "file_name": Path(local_path).name or Path(raw_result[result_key]).name,
            "oss_key": raw_result[result_key],
        }

    regions = []
    for item in display_manifest.get("regions", []):
        result_key = item.get("result_key")
        if not result_key or result_key not in raw_result:
            continue
        region = {key: value for key, value in item.items() if key != "result_key"}
        region["image_oss_key"] = raw_result[result_key]
        regions.append(region)

    return {
        "overview": build_fixed(display_manifest.get("overview_result_key")),
        "texture": build_fixed(display_manifest.get("texture_result_key")),
        "regions": regions,
    }


def _cleanup_success_output(output_dir: str | None, local_results: dict[str, str]) -> None:
    """成功上传后清理本次运行目录；失败任务不调用本函数。"""
    configured_root = Path(settings.wrinkle_output_dir)
    if not configured_root.is_absolute():
        configured_root = PROJECT_ROOT / configured_root
    configured_root = configured_root.resolve()

    if output_dir:
        candidate = Path(output_dir).resolve()
        if candidate != configured_root and configured_root in candidate.parents:
            shutil.rmtree(candidate, ignore_errors=False)
            return
        logger.warning("跳过不安全的运行目录清理路径: %s", candidate)
        return

    for local_path in local_results.values():
        try:
            local_file = Path(local_path).resolve()
            if configured_root in local_file.parents and local_file.is_file():
                local_file.unlink()
            elif local_file.is_file():
                logger.warning("跳过 output 根目录之外的文件清理: %s", local_file)
        except OSError as e:
            logger.warning("清理本地文件失败 %s: %s", local_path, e)


@celery_app.task(name=f"{_SERVICE_NAME}.analyze_image", queue=_SERVICE_NAME, bind=True)
def analyze_image(self, task_id, oss_key, oss_result_prefix=None, algorithms=None):
    """异步分析任务。

    Args:
        task_id (str): 任务标识（同时用作结果路径组件）
        oss_key (str): 待分析图片的 OSS key
        oss_result_prefix (str | None): [已停用] 保留入参兼容 backend 下发，结果路径由 backend 统一管理
        algorithms (list[str] | None): 算法列表，None 则由 Pipeline 决定

    Returns:
        dict: 成功时 raw_result 包含全部产物，display_result 标记用户展示图。
    """
    record_id = task_id
    logger.info(
        "Record %s: 开始处理 oss_key=%s algorithms=%s", record_id, oss_key, algorithms,
    )

    # 1. 经 backend internal-api 下载图片
    try:
        img_bytes = download_via_internal_api(oss_key)
    except StorageError as e:
        logger.exception("Record %s: 下载图片失败", record_id)
        return _failed(record_id, e)

    # 2. 写入临时文件（用 task_id 命名，避免并发任务写文件碰撞）
    with tempfile.TemporaryDirectory(prefix=f"{_SERVICE_NAME}_{record_id}_") as tmpdir:
        input_path = os.path.join(tmpdir, f"{record_id}{_suffix_from_oss_key(oss_key)}")
        with open(input_path, "wb") as f:
            f.write(img_bytes)

        # 3. 运行 Pipeline（纯计算，结果图写到 pipeline 自身的 output/ 目录）
        try:
            pipeline = _get_pipeline()
            pipeline_result = pipeline.process_single(input_path, algorithms)
        except Exception as e:
            logger.exception("Record %s: Pipeline 执行异常", record_id)
            return _failed(record_id, f"{type(e).__name__}: {e}")

        if pipeline_result.get("status") != "success":
            msg = pipeline_result.get("message", "分析处理失败")
            error_details = _build_error_details(pipeline_result)
            logger.warning("Record %s: Pipeline 返回失败 - %s", record_id, msg)
            if error_details:
                logger.warning("Record %s: Pipeline 诊断信息 - %s", record_id, error_details)
            return _failed(record_id, msg, error_details=error_details)

        local_results = pipeline_result.get("results", {})
        display_results = pipeline_result.get("display_results", {})
        display_manifest = pipeline_result.get("display_manifest", {})
        upload_relative_paths = pipeline_result.get("upload_relative_paths", {})
        metadata = pipeline_result.get("metadata", {})
        medical_v2_additions = {}
        # algorithm_version 已写死 "1"，不再从环境变量读取

        # 4. 经 backend internal-api 上传结果文件，回传所有结果的 object_key
        raw_result = {}
        for name, local_path in local_results.items():
            if not local_path or not os.path.exists(local_path):
                logger.warning("Record %s: 结果文件不存在 - %s", record_id, local_path)
                continue
            try:
                file_name = _resolve_upload_file_name(
                    name,
                    local_path,
                    upload_relative_paths,
                )
            except ValueError as e:
                logger.exception("Record %s: 非法上传路径", record_id)
                return _failed(record_id, e)
            try:
                object_key = upload_via_internal_api(local_path, str(record_id), "wrinkle", file_name)
            except StorageError as e:
                logger.exception("Record %s: 上传 %s 失败", record_id, name)
                return _failed(record_id, e)
            raw_result[name] = object_key
            logger.info("Record %s: 已上传 %s -> %s", record_id, name, object_key)

        # 5. raw_result 保持完整；display_result 只是额外标记用户展示子集。
        display_result = _build_display_result(
            display_manifest,
            display_results,
            raw_result,
        )
        medical_v2_additions = _build_medical_v2_additions(
            record_id=str(record_id),
            output_dir=Path(pipeline_result.get("output_dir", "")),
        )

        # 全部上传成功后才清理本次运行目录；任一失败会提前返回并保留现场。
        try:
            _cleanup_success_output(pipeline_result.get("output_dir"), local_results)
        except OSError as e:
            logger.warning("Record %s: 清理本地运行目录失败: %s", record_id, e)

    logger.info("Record %s: 任务完成 raw_result=%s", record_id, raw_result)
    response = {
        "record_id": record_id,
        "status": "success",
        "schema_version": _WRINKLE_SV,
        "meta_data": {"name": "wrinkle", "version": "1"},
        "raw_result": {
            # 13 个 OSS key 图片/CSV（最终结果）
            "analysis_face": raw_result.get("analysis_face"),
            "preprocessed_face": raw_result.get("preprocessed_face"),
            "stage1_candidates": raw_result.get("stage1_candidates"),
            "vote_heatmap": raw_result.get("vote_heatmap"),
            "stage2_overlay": raw_result.get("stage2_overlay"),
            "stage2_centerline": raw_result.get("stage2_centerline"),
            "face_filter_debug": raw_result.get("face_filter_debug"),
            "texture_reference": raw_result.get("texture_reference"),
            "comparison": raw_result.get("comparison"),
            "region_overlay": raw_result.get("region_overlay"),
            "region_tiles": raw_result.get("region_tiles"),
            "region_metrics_csv": raw_result.get("region_metrics_csv"),
            "summary_json": raw_result.get("summary_json"),
            # 结构化数据（前端需要）
            "region_metrics": metadata.get("summary", {}).get("region_metrics", []),
            "run_preset": metadata.get("run_preset"),
            "device": metadata.get("device"),
            "successful_runs": metadata.get("successful_runs"),
            "failed_runs": metadata.get("failed_runs"),
            "region_analysis_status": metadata.get("region_analysis_status"),
            **medical_v2_additions,
        },
        "debug_info": {
            "display_result": display_result,
            "algorithm_version": "1",
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            "output_dir": metadata.get("output_dir"),
        },
    }
    return response
