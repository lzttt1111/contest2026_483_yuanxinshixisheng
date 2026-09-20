# -*- coding: utf-8 -*-
"""
Celery 异步任务 Worker — AISIA 单RGB十二项DermaVision入口
═══════════════════════════════════════════════════
本文件中的 ``WORKER_TARGETS`` 是云端11任务/12项结果的唯一注册表。
根 ``celery_app`` 承载DermaVision旧六队列和新增三队列；痤疮与皱纹Worker
分别位于 ``src.acne.worker`` 和 ``src.wrinkle.worker``，并由根目录
``run_cloud_worker.py`` 统一启动。
后端对接契约（ai-skin-backend → Celery）：
  - Queues:      redness / spots / brown / texture / pores / purple /
                 surface_gloss / vascular / contour_firmness
  - Task Name:   dermavision.analyze_image
  - 参数:        task_id (str)
                 oss_key (str)
                 oss_result_prefix (str | None) [已停用] 保留入参兼容 backend 下发，结果路径由 backend 统一管理
                 algorithms (list[str] | None)
  - 正式单算法成功返回六字段：record_id / status / schema_version /
    meta_data / raw_result / debug_info。raw_result按AGENTS.md和严格Pydantic
    保存正式OSS键、精简metrics和质量字段。
  - 多算法兼容返回属于multi_algorithm_compat旧结构，不满足当前后端六字段
    合同；正式后端必须每任务只传一个算法。
                 失败时: {"record_id": task_id, "status": "failed", "error_message": "..."}

流程：下载 OSS 图片 → Pipeline 预处理+检测 → 上传正式结果文件（图片/CSV 报告上传 OSS 返回 oss_key；
      metrics JSON 为结构化量化数据，读取本地文件解析成 dict inline 返回，不上传 OSS）。

启动命令:
  # 单进程开发验证（6 个执行槽）：
  celery -A src.worker worker --loglevel=info \
    --queues=redness,spots,brown,texture,pores,purple --concurrency=6

  # 生产环境使用 deploy/dermavision@.service，11个独立 Worker。
  # purple 同时产生 UV 色斑与紫质；acne v1/v2由互斥target二选一。
"""

import csv
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from aisia_contracts.algorithms.redness.v2 import SCHEMA_VERSION as _REDNESS_SV
from aisia_contracts.algorithms.spots.v2 import SCHEMA_VERSION as _SPOTS_SV
from aisia_contracts.algorithms.brown.v2 import SCHEMA_VERSION as _BROWN_SV
from aisia_contracts.algorithms.texture.v2 import SCHEMA_VERSION as _TEXTURE_SV
from aisia_contracts.algorithms.pores.v2 import SCHEMA_VERSION as _PORES_SV
from aisia_contracts.algorithms.purple.v1 import SCHEMA_VERSION as _PURPLE_SV

# 每算法的数据格式版本(从 contract SCHEMA_VERSION 取,SSOT)
_SCHEMA_VERSIONS = {
    "redness": _REDNESS_SV,
    "spots": _SPOTS_SV,
    "brown": _BROWN_SV,
    "texture": _TEXTURE_SV,
    "pores": _PORES_SV,
    "purple": _PURPLE_SV,
    "surface_gloss": "1",
    "vascular": "1",
    "contour_firmness": "1",
}

from src.core.config import settings
from src.capture_profile import execution_mode_for_profile
from src.core.storage import StorageError, download_via_internal_api, upload_via_internal_api
from src.medical_v2_delivery import load_and_validate

logger = logging.getLogger(__name__)
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (PROJECT_ROOT / "output").resolve()

# 十一个云端 Worker 任务的唯一注册表。只统一发现和启动方式，
# 不合并或改写三类 Worker 已对接的返回合同。
WORKER_TARGETS: dict[str, dict[str, object]] = {
    "redness": {"app": "src.worker:celery_app", "queue": "redness", "task": "dermavision.analyze_image", "algorithms": ("redness",), "results": ("红区",)},
    "spots": {"app": "src.worker:celery_app", "queue": "spots", "task": "dermavision.analyze_image", "algorithms": ("spots",), "results": ("斑点",)},
    "brown": {"app": "src.worker:celery_app", "queue": "brown", "task": "dermavision.analyze_image", "algorithms": ("brown",), "results": ("棕区",)},
    "texture": {"app": "src.worker:celery_app", "queue": "texture", "task": "dermavision.analyze_image", "algorithms": ("texture",), "results": ("纹理",)},
    "pores": {"app": "src.worker:celery_app", "queue": "pores", "task": "dermavision.analyze_image", "algorithms": ("pores",), "results": ("毛孔",)},
    "purple": {"app": "src.worker:celery_app", "queue": "purple", "task": "dermavision.analyze_image", "algorithms": ("purple",), "results": ("UV色斑", "紫质")},
    "acne": {"app": "src.acne.worker:celery_app", "queue": "acne", "task": "acne.analyze_image", "algorithms": ("acne",), "results": ("痤疮",)},
    "wrinkle": {"app": "src.wrinkle.worker:celery_app", "queue": "wrinkle", "task": "wrinkle.analyze_image", "algorithms": ("wrinkle",), "results": ("皱纹",)},
    "surface_gloss": {"app": "src.worker:celery_app", "queue": "surface_gloss", "task": "dermavision.analyze_image", "algorithms": ("surface_gloss",), "results": ("油光",)},
    "vascular": {"app": "src.worker:celery_app", "queue": "vascular", "task": "dermavision.analyze_image", "algorithms": ("vascular",), "results": ("血管样结构",)},
    "contour_firmness": {"app": "src.worker:celery_app", "queue": "contour_firmness", "task": "dermavision.analyze_image", "algorithms": ("contour_firmness",), "results": ("轮廓紧致度",)},
}
WORKER_TARGET_ALIASES = {
    **{f"dermavision-{name}": name for name in (
        "redness", "spots", "brown", "texture", "pores", "purple",
        "surface_gloss", "vascular", "contour_firmness",
    )},
    "acne-detection-worker": "acne",
}
NINE_ANALYSIS_ITEMS = tuple(
    result
    for definition in WORKER_TARGETS.values()
    for result in definition["results"]
)

# Celery 实例化
celery_app = Celery(
    "dermavision",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.worker"],
)

# 任务队列配置（六个独立 queue，每个算法一个）
celery_app.conf.task_queues = {
    "redness": {"exchange": "redness", "exchange_type": "direct", "routing_key": "redness"},
    "spots": {"exchange": "spots", "exchange_type": "direct", "routing_key": "spots"},
    "brown": {"exchange": "brown", "exchange_type": "direct", "routing_key": "brown"},
    "texture": {"exchange": "texture", "exchange_type": "direct", "routing_key": "texture"},
    "pores": {"exchange": "pores", "exchange_type": "direct", "routing_key": "pores"},
    "purple": {"exchange": "purple", "exchange_type": "direct", "routing_key": "purple"},
    "surface_gloss": {"exchange": "surface_gloss", "exchange_type": "direct", "routing_key": "surface_gloss"},
    "vascular": {"exchange": "vascular", "exchange_type": "direct", "routing_key": "vascular"},
    "contour_firmness": {"exchange": "contour_firmness", "exchange_type": "direct", "routing_key": "contour_firmness"},
    **{
        f"consumer_{queue}": {
            "exchange": f"consumer_{queue}",
            "exchange_type": "direct",
            "routing_key": f"consumer_{queue}",
        }
        for queue in (
            "redness",
            "spots",
            "brown",
            "texture",
            "pores",
            "purple",
            "surface_gloss",
            "vascular",
            "contour_firmness",
        )
    },
}
celery_app.conf.task_default_queue = "dermavision"
celery_app.conf.task_default_exchange = "dermavision"
celery_app.conf.task_default_routing_key = "dermavision"

# 可靠性 / 超时 / 序列化
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.celery_worker_concurrency,
    task_soft_time_limit=settings.task_soft_time_limit,
    task_time_limit=settings.task_time_limit,
    result_expires=86400,
    result_serializer="json",
    task_serializer="json",
    accept_content=["json"],
)

# 全局 Pipeline 实例（延迟初始化，避免 import 时加载 MediaPipe 模型）
_pipeline = None

# 以下列表只属于 DermaVision 六队列 Pipeline，不是九项 Worker 总注册表。
# 云端未显式传 algorithms 时，一次预处理后执行历史五项检测。
_DEFAULT_ALGORITHMS = [
    "redness",
    "spots",
    "brown",
    "texture",
    "pores",
]
# 不传 algorithms 时仍保持历史五项默认结果，避免现有前端字段突然增加；
# 紫区由后端显式提交 algorithms=["purple"] 的独立任务。
_SUPPORTED_ALGORITHMS = tuple(_DEFAULT_ALGORITHMS) + (
    "purple", "surface_gloss", "vascular", "contour_firmness",
)

# 对外正式返回。redness/spots 保持旧后端键兼容。
# 图片 / CSV 报告上传 OSS 返回 oss_key；
# metrics 为结构化量化数据，读取本地 JSON 解析成 dict inline 返回（不上传 OSS）。
_RESULT_UPLOAD_KEYS = (
    "redness",
    "red_areas_overlay",
    "redness_report",
    "spots",
    "spots_report",
    "brown",
    "brown_spots_overlay",
    "brown_report",
    "texture",
    "texture_report",
    "pores",
    "pores_report",
    "purple_uv_base",
    "purple_uv_spots_overlay",
    "purple_fluorescence_base",
    "purple_porphyrin_overlay",
    "purple_report",
    "surface_gloss",
    "surface_gloss_report",
    "surface_gloss_medical_report_csv_v2",
    "vascular",
    "vascular_report",
    "vascular_medical_report_csv_v2",
    "contour_firmness",
    "contour_firmness_report",
    "contour_firmness_medical_report_csv_v2",
)
_METRICS_INLINE_KEYS = (
    "redness_metrics",
    "spots_metrics",
    "brown_metrics",
    "texture_metrics",
    "pores_metrics",
    "purple_metrics",
    "surface_gloss_metrics",
    "vascular_metrics",
    "contour_firmness_metrics",
)
_RESULT_KEYS = _RESULT_UPLOAD_KEYS + _METRICS_INLINE_KEYS
_RESULTS_BY_ALGORITHM = {
    "redness": {
        "redness",
        "red_areas_overlay",
        "redness_report",
        "redness_metrics",
    },
    "spots": {"spots", "spots_report", "spots_metrics"},
    "brown": {
        "brown",
        "brown_spots_overlay",
        "brown_report",
        "brown_metrics",
    },
    "texture": {"texture", "texture_report", "texture_metrics"},
    "pores": {"pores", "pores_report", "pores_metrics"},
    "purple": {
        "purple_uv_base",
        "purple_uv_spots_overlay",
        "purple_fluorescence_base",
        "purple_porphyrin_overlay",
        "purple_report",
        "purple_metrics",
    },
    "surface_gloss": {"surface_gloss", "surface_gloss_report", "surface_gloss_metrics", "surface_gloss_medical_report_csv_v2"},
    "vascular": {"vascular", "vascular_report", "vascular_metrics", "vascular_medical_report_csv_v2"},
    "contour_firmness": {"contour_firmness", "contour_firmness_report", "contour_firmness_metrics", "contour_firmness_medical_report_csv_v2"},
}
_RESULT_KEY_TO_ALGO = {
    key: algo for algo, keys in _RESULTS_BY_ALGORITHM.items() for key in keys
}
_METRICS_REPORT_KEYS = {
    "redness_metrics": "redness_report",
    "spots_metrics": "spots_report",
    "brown_metrics": "brown_report",
    "texture_metrics": "texture_report",
    "pores_metrics": "pores_report",
    "purple_metrics": "purple_report",
    "surface_gloss_metrics": "surface_gloss_report",
    "vascular_metrics": "vascular_report",
    "contour_firmness_metrics": "contour_firmness_report",
}


def _get_pipeline():
    """延迟初始化 Pipeline 单例（避免 Worker 启动时就加载 MediaPipe 模型）。"""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import DermaVisionPipeline

        _pipeline = DermaVisionPipeline()
        logger.info("DermaVision Pipeline 实例已初始化。")
    return _pipeline


@worker_process_init.connect
def _warm_worker_process(**_kwargs) -> None:
    """只在 prefork 子进程中初始化 CUDA 和常驻模型。"""
    if os.getenv("DERMAVISION_WARMUP_ON_START", "true").strip().lower() in {
        "0", "false", "no", "off",
    }:
        logger.info("DermaVision 子进程冷启动预热已通过环境变量关闭。")
        return
    started = time.perf_counter()
    metadata = _get_pipeline().warmup()
    logger.info(
        "DermaVision 子进程冷启动完成 elapsed=%.3fs metadata=%s",
        time.perf_counter() - started,
        metadata,
    )


@worker_process_shutdown.connect
def _close_worker_process(**_kwargs) -> None:
    global _pipeline
    if _pipeline is not None:
        _pipeline.close()
        _pipeline = None


def _safe_record_component(record_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(record_id)).strip("._")
    if not safe:
        raise ValueError("task_id 不能用于构造安全的结果路径")
    return safe


def _input_suffix(oss_key: str) -> str:
    suffix = Path(urlparse(str(oss_key)).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return suffix
    return ".jpg"


def _normalize_algorithms(algorithms) -> list[str]:
    """Validate and normalize the backward-compatible algorithms argument."""
    if algorithms is None:
        return list(_DEFAULT_ALGORITHMS)
    if not isinstance(algorithms, (list, tuple)):
        raise ValueError("algorithms 必须是算法名称列表")

    normalized = [str(name).strip() for name in algorithms]
    if not normalized or any(not name for name in normalized):
        raise ValueError("algorithms 不能为空")
    if len(set(normalized)) != len(normalized):
        raise ValueError("algorithms 不能包含重复项目")

    unknown = sorted(set(normalized) - set(_SUPPORTED_ALGORITHMS))
    if unknown:
        raise ValueError(f"不支持的算法: {unknown}")
    return normalized


def _load_and_validate_metrics(
    metrics_path: str,
    report_path: str,
) -> dict[str, int]:
    """恢复原前端合同：棕区/纹理/毛孔使用精简中文整数 dict。"""
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("量化 JSON 必须是非空对象")
    # 本地最终量化 JSON 允许附加医学 V2；旧 raw_result.metrics 必须保持
    # 原字段集合，因此校验与返回前剥离该并列新增节点。
    metrics = dict(metrics)
    metrics.pop("medical_metrics_v2", None)

    normalized_metrics: dict[str, int] = {}
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        if not any("\u4e00" <= character <= "\u9fff" for character in name):
            raise ValueError(f"量化指标名称必须使用中文: {name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"量化指标值必须是整数: {name}={raw_value!r}")
        normalized_metrics[name] = raw_value

    with open(report_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 2 or len(rows[0]) != len(rows[1]):
        raise ValueError("量化 CSV 必须严格为等长的表头行和数据行")
    try:
        csv_metrics = {
            name: int(value) for name, value in zip(rows[0], rows[1])
        }
    except ValueError as exc:
        raise ValueError("量化 CSV 的数据必须是整数") from exc
    if csv_metrics != normalized_metrics:
        raise ValueError("量化 JSON 与 CSV 的中文指标和值不一致")
    return normalized_metrics


_PURPLE_PUBLIC_PROJECTS = {
    "紫外线色斑": "uv_spots",
    "紫质": "porphyrin",
}
_PURPLE_PUBLIC_COLUMNS = {
    "总计": "total",
    "额头": "forehead",
    "左脸颊": "left_cheek",
    "右脸颊": "right_cheek",
    "鼻部": "nose",
    "下巴": "chin",
}


def _load_and_validate_purple_metrics(
    metrics_path: str,
    report_path: str,
) -> dict[str, Any]:
    """校验紫区英文 JSON 与两行中文用户简表严格一致。"""

    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("紫区 metrics 顶层必须是对象")

    with open(report_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 3 or not rows[0] or rows[0][0] != "检测项目":
        raise ValueError("紫区用户 CSV 必须包含表头和两项检测数据")
    headers = rows[0]
    if any(name not in _PURPLE_PUBLIC_COLUMNS for name in headers[1:]):
        raise ValueError("紫区用户 CSV 包含未知分区列")
    if headers[1:] not in (["总计"], list(_PURPLE_PUBLIC_COLUMNS)):
        raise ValueError("紫区用户 CSV 列顺序不符合合同")

    expected: dict[str, int] = {}
    seen_projects: set[str] = set()
    for row in rows[1:]:
        if len(row) != len(headers):
            raise ValueError("紫区用户 CSV 行列数量不一致")
        project = _PURPLE_PUBLIC_PROJECTS.get(row[0])
        if project is None or project in seen_projects:
            raise ValueError("紫区用户 CSV 项目名称重复或未知")
        seen_projects.add(project)
        try:
            expected.update({
                f"{project}_{_PURPLE_PUBLIC_COLUMNS[header]}": int(value)
                for header, value in zip(headers[1:], row[1:])
            })
        except ValueError as exc:
            raise ValueError("紫区用户 CSV 指标必须为整数") from exc

    if any(
        not str(key).isascii()
        or isinstance(value, bool)
        or not isinstance(value, int)
        for key, value in metrics.items()
    ):
        raise ValueError("紫区 metrics 必须是英文键和整数值的扁平对象")
    if metrics != expected:
        raise ValueError("紫区英文 JSON 与用户 CSV 指标不一致")
    return metrics


def _build_medical_v2_additions(
    pipeline_result: dict,
    algorithm: str,
    record_component: str,
) -> dict:
    """校验并上传医学 V2 旁路；任何异常均不影响旧任务成功。"""
    if not settings.enable_medical_metrics_v2:
        return {}
    if algorithm in {"purple", "surface_gloss", "vascular", "contour_firmness"}:
        # 紫区包含两个子项目，医学宽表字段很多。当前云端合同只返回精简
        # metrics 和用户 CSV，医生宽表不作为前端可选字段透传。
        return {}
    try:
        spec = (pipeline_result.get("medical_v2_results") or {}).get(algorithm)
        if not isinstance(spec, dict):
            raise ValueError(f"缺少 {algorithm} 医学 V2 产物")
        json_path = spec.get("json")
        csv_path = Path(str(spec.get("csv") or ""))
        if not json_path or not str(csv_path):
            raise ValueError(f"{algorithm} 医学 V2 路径不完整")
        document = load_and_validate(json_path, csv_path)
        if not csv_path.is_file() or csv_path.stat().st_size <= 0:
            raise ValueError("医学 V2 CSV 不存在或为空")
        object_key = upload_via_internal_api(str(csv_path), record_component, algorithm, csv_path.name)
        return {
            "medical_metrics_v2": document,
            "medical_report_csv_v2": object_key,
        }
    except Exception:
        logger.exception(
            "Record %s: 医学 V2 旁路失败，保留旧合同成功返回",
            record_component,
        )
        return {}


def submit_independent_tasks(
    task_ids: Mapping[str, str],
    oss_key: str,
    oss_result_prefix: str | None = None,
    *,
    app=None,
) -> dict[str, dict[str, str]]:
    """Submit five independent single-algorithm tasks.

    The backend owns the five business record IDs.  This helper deliberately
    does not aggregate results: each Celery task has its own final result.
    """
    if not isinstance(task_ids, Mapping):
        raise ValueError("task_ids 必须是算法到独立 task_id 的映射")

    # 该助手是现有后端的“五项并行”固定合同，新增紫区不能把它从五项
    # 变成六项必填。紫区由后端另行提交 algorithms=["purple"]。
    dispatch_algorithms = tuple(_DEFAULT_ALGORITHMS)
    missing = sorted(set(dispatch_algorithms) - set(task_ids))
    extra = sorted(set(task_ids) - set(dispatch_algorithms))
    if missing or extra:
        raise ValueError(f"task_ids 算法集合不完整: missing={missing}, extra={extra}")
    if not str(oss_key).strip():
        raise ValueError("oss_key 不能为空")

    record_ids = {name: str(task_ids[name]).strip() for name in dispatch_algorithms}
    if any(not value for value in record_ids.values()):
        raise ValueError("五个 task_id 均不能为空")
    if len(set(record_ids.values())) != len(record_ids):
        raise ValueError("五个任务必须使用不同的 task_id")

    safe_components = [_safe_record_component(value) for value in record_ids.values()]
    if len(set(safe_components)) != len(safe_components):
        raise ValueError("task_id 规范化后发生路径冲突，请使用差异更明确的 ID")

    sender = app or celery_app
    submitted: dict[str, dict[str, str]] = {}
    for algorithm in dispatch_algorithms:
        record_id = record_ids[algorithm]
        async_result = sender.send_task(
            "dermavision.analyze_image",
            args=[
                record_id,
                oss_key,
                oss_result_prefix,
                [algorithm],
            ],
            queue=algorithm,
        )
        submitted[algorithm] = {
            "record_id": record_id,
            "celery_task_id": str(async_result.id),
        }
    return submitted


def _cleanup_success_outputs(paths: list[str]) -> None:
    """Delete only files/directories inside this project's output root."""
    unique = sorted({str(path) for path in paths if path}, key=len, reverse=True)
    for raw_path in unique:
        candidate = Path(raw_path).resolve()
        if candidate == OUTPUT_ROOT or OUTPUT_ROOT not in candidate.parents:
            logger.warning("跳过不安全的清理路径: %s", candidate)
            continue
        try:
            if candidate.is_dir():
                shutil.rmtree(candidate)
            elif candidate.is_file():
                candidate.unlink()
        except OSError as exc:
            logger.warning("清理本地结果失败 %s: %s", candidate, exc)


_ENVELOPE_ALGOS = {
    "redness", "spots", "brown", "texture", "pores", "purple",
    "surface_gloss", "vascular", "contour_firmness",
}


def _failed(record_id: object, message: object, *, schema_version: str | None = None, **details: object) -> dict:
    payload = {
        "record_id": str(record_id),
        "status": "failed",
        "error_message": str(message),
        **details,
    }
    if schema_version:
        payload["schema_version"] = schema_version
    return payload


def _build_envelope_response(
    record_id: str,
    algorithms: list[str],
    raw_result: dict,
    metadata: dict,
    medical_v2_additions: dict | None = None,
) -> dict:
    """按AGENTS.md构建单算法六字段信封，算法内部结果保持不变。"""
    algo = algorithms[0]
    if algo == "purple":
        algorithm_result = {
            "uv_base": raw_result.get("purple_uv_base"),
            "uv_spots_overlay": raw_result.get("purple_uv_spots_overlay"),
            "fluorescence_base": raw_result.get("purple_fluorescence_base"),
            "porphyrin_overlay": raw_result.get("purple_porphyrin_overlay"),
            "metrics": raw_result.get("purple_metrics"),
            "quality_score": metadata.get("quality_score"),
            "quality_status": metadata.get("quality_status"),
            "quality_flags": metadata.get("quality_flags", []),
        }
    elif algo in {"surface_gloss", "vascular", "contour_firmness"}:
        algorithm_result = {
            "overlay": raw_result.get(algo),
            "metrics": raw_result.get(f"{algo}_metrics"),
            "medical_report_csv_v2": raw_result.get(f"{algo}_medical_report_csv_v2"),
            "quality_score": metadata.get("quality_score"),
            "quality_status": metadata.get("quality_status"),
            "quality_flags": metadata.get("quality_flags", []),
        }
    else:
        algorithm_result = {
            "overlay": raw_result.get(algo),
            "metrics": raw_result.get(f"{algo}_metrics"),
            "quality_score": metadata.get("quality_score"),
            "quality_status": metadata.get("quality_status"),
            "quality_flags": metadata.get("quality_flags", []),
            **(
                {"brown_spots_overlay": raw_result.get("brown_spots_overlay")}
                if algo == "brown"
                else {}
            ),
            **(
                {"red_areas_overlay": raw_result.get("red_areas_overlay")}
                if algo == "redness"
                else {}
            ),
        }
    if medical_v2_additions:
        algorithm_result.update(medical_v2_additions)
    response = {
        "record_id": record_id,
        "status": "success",
        "schema_version": _SCHEMA_VERSIONS.get(algo, "1"),
        "meta_data": {
            "name": algo,
            "version": "1",
        },
        "raw_result": algorithm_result,
        "debug_info": {
            "report_csv": raw_result.get(f"{algo}_report"),
            "timing_seconds": metadata.get("timing_seconds", {}),
            "execution_mode": metadata.get("execution_mode", ""),
        },
    }
    return response


@celery_app.task(name="dermavision.analyze_image", bind=True)
def analyze_image(self, task_id, oss_key, oss_result_prefix=None, algorithms=None):
    """皮肤检测异步任务。

    Args:
        task_id (str): 任务标识（同时用作 OSS 结果路径组件）
        oss_key (str): 待分析图片的 OSS key
        oss_result_prefix (str | None): [已停用] 保留入参兼容 backend 下发，结果路径由 backend 统一管理（report/{task_id}/{algo}/）
        algorithms (list[str] | None): 算法列表；不传时默认一次执行
            redness、spots、brown、texture、pores。

    Returns:
        dict: 成功时 raw_result 根据 algorithms 返回对应正式图片、
              CSV 的 OSS key，以及与 CSV 一致的 metrics inline dict；
              失败时返回 error_message。
    """
    worker_started_at = time.perf_counter()
    record_id = task_id
    try:
        algorithms = _normalize_algorithms(algorithms)
        record_component = _safe_record_component(record_id)
    except ValueError as exc:
        return _failed(record_id, exc)

    logger.info(
        "Record %s: 开始处理 oss_key=%s algorithms=%s", record_id, oss_key, algorithms,
    )

    # 1. 经 backend internal-api 下载图片
    download_started_at = time.perf_counter()
    try:
        img_bytes = download_via_internal_api(oss_key)
    except StorageError as e:
        logger.exception("Record %s: 下载图片失败", record_id)
        return _failed(record_id, e, schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
    download_seconds = time.perf_counter() - download_started_at

    # 2. 写入临时文件（用 task_id 命名，避免并发任务在共享 output/ 目录写文件碰撞）
    with tempfile.TemporaryDirectory(prefix=f"dermavision_{record_component}_") as tmpdir:
        input_path = os.path.join(tmpdir, f"{record_component}{_input_suffix(oss_key)}")
        with open(input_path, "wb") as f:
            f.write(img_bytes)

        # 3. 运行 Pipeline（纯计算，结果图写到 pipeline 自身的 output/ 目录）
        pipeline_started_at = time.perf_counter()
        try:
            pipeline = _get_pipeline()
            pipeline_result = pipeline.process_single(input_path, algorithms)
        except Exception as e:
            logger.exception("Record %s: Pipeline 执行异常", record_id)
            return _failed(record_id, f"{type(e).__name__}: {e}", schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
        pipeline_seconds = time.perf_counter() - pipeline_started_at

        if pipeline_result.get("status") != "success":
            msg = pipeline_result.get("message", "分析处理失败")
            logger.warning("Record %s: Pipeline 返回失败 - %s", record_id, msg)
            return _failed(record_id, msg, schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))

        local_results = pipeline_result.get("results", {})
        expected_keys = set().union(
            *(
                _RESULTS_BY_ALGORITHM.get(name, set())
                for name in algorithms
            )
        )
        missing_keys = sorted(expected_keys - set(local_results))
        if missing_keys:
            message = f"Pipeline 缺少必需结果字段: {missing_keys}"
            logger.error("Record %s: %s", record_id, message)
            return _failed(record_id, message, schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))

        # 4. 上传前处理量化指标。
        #    红区/色斑：完整英文指标，直接读取 JSON（不过中文/整数校验）
        #    棕区/纹理/毛孔：精简中文指标，校验 JSON/CSV 一致性
        # 红区/斑点恢复同事定义的完整英文 dict；三个逐实例数组为兼容
        # 字段并固定返回空列表，算法内部仍保留实例定位。紫区是后续独立
        # 新增算法，不参与原七项合同。棕区/纹理/毛孔继续走精简中文整数
        # 合同。新增 medical_metrics_v2 的字段名统一为英文。
        _DIRECT_METRICS = {"redness_metrics", "spots_metrics"}
        inline_metrics: dict[str, dict] = {}
        for name in _METRICS_INLINE_KEYS:
            if name not in expected_keys:
                continue
            metrics_path = local_results.get(name)
            if not metrics_path:
                return _failed(record_id, f"量化指标缺少文件路径: {name}", schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
            if name == "purple_metrics":
                report_path = local_results.get("purple_report")
                if not report_path:
                    return _failed(
                        record_id,
                        "量化合同缺少文件路径: purple_metrics/purple_report",
                        schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""),
                    )
                try:
                    inline_metrics[name] = _load_and_validate_purple_metrics(
                        metrics_path,
                        report_path,
                    )
                except (OSError, ValueError) as exc:
                    logger.exception("Record %s: 紫区量化合同校验失败", record_id)
                    return _failed(record_id, f"紫区量化合同校验失败: {exc}", schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
            elif name in _DIRECT_METRICS or name in {
                "surface_gloss_metrics", "vascular_metrics", "contour_firmness_metrics",
            }:
                try:
                    with open(metrics_path, "r", encoding="utf-8") as handle:
                        loaded_metrics = json.load(handle)
                    if not isinstance(loaded_metrics, dict):
                        raise ValueError("metrics 顶层必须是对象")
                    loaded_metrics = dict(loaded_metrics)
                    loaded_metrics.pop("medical_metrics_v2", None)
                    inline_metrics[name] = loaded_metrics
                except (OSError, ValueError) as exc:
                    logger.exception("Record %s: 解析 metrics 失败 %s", record_id, name)
                    return _failed(record_id, f"metrics 解析失败 {name}: {exc}", schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
            else:
                report_name = _METRICS_REPORT_KEYS[name]
                report_path = local_results.get(report_name)
                if not report_path:
                    return _failed(record_id, f"量化合同缺少文件路径: {name}/{report_name}", schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
                try:
                    inline_metrics[name] = _load_and_validate_metrics(
                        metrics_path,
                        report_path,
                    )
                except (OSError, ValueError) as exc:
                    logger.exception("Record %s: 量化合同校验失败 %s", record_id, name)
                    return _failed(record_id, f"量化合同校验失败 {name}: {exc}", schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))

        # 5. 上传正式结果（图片 / CSV 报告 → OSS），OSS 文件名保留中文原名。
        upload_started_at = time.perf_counter()
        raw_result = {}
        medical_v2_additions = {}
        for name in _RESULT_UPLOAD_KEYS:
            if name not in expected_keys:
                continue
            local_path = local_results.get(name)
            if not local_path:
                continue
            if not os.path.exists(local_path):
                logger.warning("Record %s: 结果文件不存在 - %s", record_id, local_path)
                continue
            algo = _RESULT_KEY_TO_ALGO.get(name, "unknown")
            try:
                object_key = upload_via_internal_api(
                    local_path, record_component, algo, Path(local_path).name
                )
            except StorageError as e:
                logger.exception("Record %s: 上传 %s 失败", record_id, name)
                return _failed(record_id, e, schema_version=_SCHEMA_VERSIONS.get(algorithms[0] if algorithms else ""))
            raw_result[name] = object_key
            logger.info("Record %s: 已上传 %s -> %s", record_id, name, object_key)

        # metrics JSON 不上传 OSS，按各算法既有类型原样 inline 返回。
        for name in _METRICS_INLINE_KEYS:
            if name not in expected_keys:
                continue
            raw_result[name] = inline_metrics[name]
            logger.info("Record %s: 已 inline metrics %s", record_id, name)

        if len(algorithms) == 1 and algorithms[0] in _ENVELOPE_ALGOS:
            medical_v2_additions = _build_medical_v2_additions(
                pipeline_result,
                algorithms[0],
                record_component,
            )
        upload_seconds = time.perf_counter() - upload_started_at

        # 6. 全部上传成功后，清理本次任务的预处理文件和引擎子目录。
        _cleanup_success_outputs(list(pipeline_result.get("cleanup_paths", [])))

    metadata = pipeline_result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["execution_mode"] = (
        execution_mode_for_profile(settings.capture_profile, "single_algorithm")
        if len(algorithms) == 1
        else "multi_algorithm_compat"
    )
    pipeline_timing = metadata.get("timing_seconds")
    if not isinstance(pipeline_timing, dict):
        pipeline_timing = {}
    metadata["timing_seconds"] = {
        **pipeline_timing,
        "download": round(download_seconds, 4),
        "pipeline": round(pipeline_seconds, 4),
        "upload": round(upload_seconds, 4),
        "worker_total": round(time.perf_counter() - worker_started_at, 4),
    }
    logger.info("Record %s: 任务完成 raw_result=%s", record_id, raw_result)
    if len(algorithms) == 1 and algorithms[0] in _ENVELOPE_ALGOS:
        return _build_envelope_response(
            record_id,
            algorithms,
            raw_result,
            metadata,
            medical_v2_additions,
        )
    return {
        "record_id": record_id,
        "status": "success",
        "raw_result": raw_result,
        "metadata": metadata,
    }


if __name__ == "__main__":
    # 本地测试：直接调用 Pipeline（不经过 Celery / OSS）
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.pipeline import DermaVisionPipeline

    test_image = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "test_images",
        "正脸.png",
    )
    result = DermaVisionPipeline().process_single(
        test_image,
        list(_DEFAULT_ALGORITHMS),
    )
    print(f"\n📋 返回结果: {result}")
