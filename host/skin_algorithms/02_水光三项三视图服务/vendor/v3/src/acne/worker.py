# -*- coding: utf-8 -*-
"""痤疮图像异步分析的 Celery Worker 入口。"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from aisia_contracts.algorithms.acne.v1 import SCHEMA_VERSION as _ACNE_SV
from cloud_contracts import validate_worker_envelope
from src.acne.acne_detector import DetectorUnavailableError
from src.acne.artifact_policy import FORMAL_FAST_TOKEN
from src.capture_profile import (
    capture_profile_from_environment,
    execution_mode_for_profile,
)
from src.acne.core.config import settings
from src.core.storage import (
    StorageError,
    download_via_internal_api,
    upload_via_internal_api,
)
from src.acne.medical_v2_delivery import load_and_validate
from src.acne.display_outputs import build_display_outputs

logger = logging.getLogger(__name__)
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

# 队列名/任务名写死为 "acne"（项目规则：worker 自包含，不经启动脚本注入）。
# 对应 backend implements/acne_v1.py 的 celery_task="acne.analyze_image" queue="acne"。
_SERVICE_NAME = "acne"
_ACNE_V2_QUEUE = "acne_v2"
_ACNE_V2_SV = "2"

celery_app = Celery(
    _SERVICE_NAME,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.acne.worker"],
)

celery_app.conf.task_queues = {
    queue: {
        "exchange": queue,
        "exchange_type": "direct",
        "routing_key": queue,
    }
    for queue in (_SERVICE_NAME, _ACNE_V2_QUEUE, "consumer_acne_v2")
}
celery_app.conf.task_default_queue = _SERVICE_NAME
celery_app.conf.task_default_exchange = _SERVICE_NAME
celery_app.conf.task_default_routing_key = _SERVICE_NAME
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

_pipeline = None


@dataclass(frozen=True, slots=True)
class AcneV2ProjectionError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _failed(record_id: object, message: object, *, schema_version: str = _ACNE_SV, **details: object) -> dict:
    payload = {
        "record_id": str(record_id),
        "status": "failed",
        "schema_version": schema_version,
        "error_message": str(message),
        **details,
    }
    return payload


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.acne.pipeline import Pipeline

        _pipeline = Pipeline()
        logger.info("Pipeline initialized.")
    return _pipeline


@worker_process_init.connect
def _warm_worker_process(**_kwargs) -> None:
    if os.getenv("ACNE_WARMUP_ON_START", "true").strip().lower() in {
        "0", "false", "no", "off",
    }:
        logger.info("Acne 子进程冷启动预热已关闭。")
        return
    started = time.perf_counter()
    metadata = _get_pipeline().warmup()
    logger.info(
        "Acne 子进程冷启动完成 elapsed=%.3fs metadata=%s",
        time.perf_counter() - started,
        metadata,
    )


@worker_process_shutdown.connect
def _close_worker_process(**_kwargs) -> None:
    global _pipeline
    if _pipeline is not None:
        _pipeline.close()
        _pipeline = None


def _upload_result_group(
    *,
    record_id: str,
    files: dict[str, str],
    use_basename: bool,
) -> dict[str, str]:
    """经 backend internal-api 上传结果产物，返回 {结果名: object_key}。

    OSS 路径由 backend 统一管理：report/{record_id}/acne/{file_name}。
    """
    uploaded: dict[str, str] = {}
    for name, local_path in files.items():
        path = Path(local_path)
        if not path.exists():
            logger.warning("Record %s: result file missing: %s", record_id, local_path)
            continue
        if use_basename:
            object_name = path.name
        else:
            suffix = path.suffix or ".bin"
            object_name = f"{name}{suffix}"
        try:
            object_key = upload_via_internal_api(str(path), record_id, "acne", object_name)
        except StorageError as exc:
            logger.exception("Record %s: result upload failed: %s", record_id, name)
            raise
        uploaded[name] = object_key
    return uploaded


def _build_medical_v2_additions(
    *, record_id: str, output_dir: Path
) -> dict:
    """返回可选医学 V2 旁路；失败时保留既有痤疮成功合同。"""
    if not settings.enable_medical_metrics_v2:
        return {}
    try:
        json_path = output_dir / "痤疮量化指标.json"
        csv_path = output_dir / "痤疮医学量化指标_V2.csv"
        document = load_and_validate(json_path, csv_path)
        object_key = upload_via_internal_api(str(csv_path), record_id, "acne", csv_path.name)
        return {
            "medical_metrics_v2": document,
            "medical_report_csv_v2": object_key,
        }
    except Exception:
        logger.exception(
            "Record %s: 医学 V2 旁路失败，保留旧痤疮合同成功返回",
            record_id,
        )
        return {}


@celery_app.task(name=f"{_SERVICE_NAME}.analyze_image", queue=_SERVICE_NAME, bind=True)
def analyze_image(self, task_id, oss_key, oss_result_prefix=None, algorithms=None):
    """从 backend internal-api 下载单张图片，运行分析流程，并经 internal-api 上传结果产物。

    oss_result_prefix 已停用：保留入参兼容 backend 下发，结果路径由 backend 统一管理
    （report/{task_id}/acne/）。
    """

    record_id = str(task_id)
    logger.info("Record %s: start oss_key=%s algorithms=%s", record_id, oss_key, algorithms)

    try:
        img_bytes = download_via_internal_api(oss_key)
    except StorageError as exc:
        logger.exception("Record %s: image download failed", record_id)
        return _failed(record_id, exc)

    with tempfile.TemporaryDirectory(prefix=f"{_SERVICE_NAME}_{record_id}_") as tmpdir:
        input_path = os.path.join(tmpdir, f"{record_id}.jpg")
        with open(input_path, "wb") as f:
            f.write(img_bytes)

        medical_v2_additions = {}
        try:
            pipeline_result = _get_pipeline().process_single(input_path, algorithms)
        except DetectorUnavailableError as exc:
            logger.exception("Record %s: detector deployment validation failed", record_id)
            return _failed(record_id, exc, error_code="detector_deployment_error")
        except Exception as exc:
            logger.exception("Record %s: pipeline crashed", record_id)
            return _failed(record_id, f"{type(exc).__name__}: {exc}")

        if pipeline_result.get("status") != "success":
            message = pipeline_result.get("message", "analysis_failed")
            logger.warning("Record %s: pipeline failed - %s", record_id, message)
            return _failed(record_id, message)

        raw_dir = Path(pipeline_result.get("output_dir", ""))
        summary_path = Path(pipeline_result.get("summary_path", ""))
        if not raw_dir.exists() or not summary_path.exists():
            logger.error("Record %s: pipeline returned without a valid output directory", record_id)
            return _failed(
                record_id,
                "pipeline did not produce a valid output directory",
                error_code="missing_output_directory",
            )

        try:
            display_files, compact_summary, metadata = build_display_outputs(
                raw_dir=raw_dir,
                input_image=oss_key,
                target_dir=raw_dir,
                raw_result_files=pipeline_result.get("results", {}),
            )
            raw_result = _upload_result_group(
                record_id=record_id,
                files=pipeline_result.get("results", {}),
                use_basename=False,
            )
            display_result = _upload_result_group(
                record_id=record_id,
                files=display_files,
                use_basename=True,
            )
            medical_v2_additions = _build_medical_v2_additions(
                record_id=record_id,
                output_dir=raw_dir,
            )
        except StorageError as exc:
            return _failed(record_id, exc)
        except Exception as exc:
            logger.exception("Record %s: failed to build display outputs", record_id)
            return _failed(
                record_id,
                f"{type(exc).__name__}: {exc}",
                error_code="display_result_build_failed",
            )

    # algorithm_version 已写死 "1"，不再从环境变量读取
    quant_result = compact_summary["量化结果"]
    logger.info("Record %s: success raw_result=%s display_result=%s", record_id, raw_result, display_result)
    response = {
        "record_id": record_id,
        "status": "success",
        "schema_version": _ACNE_SV,
        "meta_data": {"name": "acne", "version": "1"},
        "raw_result": {
            # OSS key 字段（最终产物文件）
            "acne_summary": raw_result.get("acne_summary"),
            "acne_circles": raw_result.get("acne_circles"),
            "acne_raw_boxes": raw_result.get("acne_raw_boxes"),
            "acne_detections": raw_result.get("acne_detections"),
            "acne_skin_mask": raw_result.get("acne_skin_mask"),
            "acne_forbidden_mask": raw_result.get("acne_forbidden_mask"),
            "acne_standardized": raw_result.get("acne_standardized"),
            "max_recall_heatmap": raw_result.get("max_recall_heatmap"),
            "diffuse_erythema_heatmap": raw_result.get("diffuse_erythema_heatmap"),
            "focal_candidate_heatmap": raw_result.get("focal_candidate_heatmap"),
            "max_recall_circles": raw_result.get("max_recall_circles"),
            "combined_candidate_circles": raw_result.get("combined_candidate_circles"),
            "max_recall_debug": raw_result.get("max_recall_debug"),
            "max_recall_json": raw_result.get("max_recall_json"),
            "original_yolo_circles": raw_result.get("original_yolo_circles"),
            "original_unsupervised_circles": raw_result.get("original_unsupervised_circles"),
            "original_combined_circles": raw_result.get("original_combined_circles"),
            # 结构化数据（前端需要）
            "acne_presence": metadata.get("acne_presence"),
            "acne_count": metadata.get("acne_count"),
            "region_counts": metadata.get("region_counts", []),
            "grading_status": metadata.get("grading_status"),
            "grading_reason": metadata.get("grading_reason"),
            "detector_status": metadata.get("detector_status"),
            "detector_reason": metadata.get("detector_reason"),
            "input_mode": metadata.get("input_mode"),
            "detection_scope": metadata.get("detection_scope"),
            "量化结果": quant_result,
            **medical_v2_additions,
        },
        "debug_info": {
            "display_result": display_result,
            "algorithm_version": "1",
            "elapsed_seconds": metadata.get("elapsed_seconds"),
        },
    }
    return response


def _required_acne_v2_count(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise AcneV2ProjectionError(f"{field_name}必须为非负整数")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise AcneV2ProjectionError(f"{field_name}必须为非负整数") from exc
    if count < 0 or count != value:
        raise AcneV2ProjectionError(f"{field_name}必须为非负整数")
    return count


def _build_acne_v2_response(legacy_response: dict) -> dict:
    raw_result = legacy_response["raw_result"]
    quantification = raw_result["量化结果"]
    total = _required_acne_v2_count(
        quantification["疑似痤疮圈选数量"],
        "疑似痤疮数量.总计",
    )
    if _required_acne_v2_count(raw_result["acne_count"], "acne_count") != total:
        raise AcneV2ProjectionError("acne_count与正式量化总计不一致")

    region_counts = {}
    for item in raw_result["region_counts"]:
        region = item["region"]
        if region in region_counts:
            raise AcneV2ProjectionError(f"重复痤疮分区: {region}")
        region_counts[region] = _required_acne_v2_count(
            item["count"],
            f"region_counts.{region}",
        )
    region_mapping = (
        ("额头", "forehead"),
        ("左脸颊", "subject_left_cheek"),
        ("右脸颊", "subject_right_cheek"),
        ("鼻部", "nose"),
        ("下巴", "chin"),
    )
    counts = {"总计": total}
    for public_name, source_name in region_mapping:
        counts[public_name] = region_counts[source_name]

    debug_info = legacy_response["debug_info"]
    overlay = debug_info["display_result"]["overlay_original"]
    if not isinstance(overlay, str) or not overlay:
        raise AcneV2ProjectionError("缺少正式痤疮原图圈选结果")
    response = {
        "record_id": legacy_response["record_id"],
        "status": "success",
        "schema_version": _ACNE_V2_SV,
        "meta_data": {"name": "acne", "version": "2"},
        "raw_result": {
            "overlay": overlay,
            "metrics": {
                "疑似痤疮数量": counts,
                "痤疮严重程度等级": quantification["痤疮严重程度等级"],
            },
            "quality_score": None,
            "quality_status": None,
            "quality_flags": [],
        },
        "debug_info": {
            "execution_mode": "formal_fast",
            "elapsed_seconds": debug_info["elapsed_seconds"],
        },
    }
    return validate_worker_envelope("acne_v2", response)


@celery_app.task(name="dermavision.analyze_image", queue=_ACNE_V2_QUEUE, bind=True)
def analyze_image_v2(self, task_id, oss_key, oss_result_prefix=None, algorithms=None):
    """运行固定formal-fast痤疮流程并投影严格v2公开合同。"""

    legacy_response = analyze_image.run(
        task_id,
        oss_key,
        oss_result_prefix,
        ["acne", FORMAL_FAST_TOKEN],
    )
    if legacy_response.get("status") != "success":
        return {**legacy_response, "schema_version": _ACNE_V2_SV}
    try:
        return _build_acne_v2_response(legacy_response)
    except (AcneV2ProjectionError, KeyError, TypeError, ValueError) as exc:
        logger.exception("Record %s: acne v2 contract projection failed", task_id)
        return _failed(
            task_id,
            f"{type(exc).__name__}: {exc}",
            schema_version=_ACNE_V2_SV,
            error_code="v2_contract_projection_failed",
        )
