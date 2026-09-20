#!/usr/bin/env python3
"""在单一来源服务进程中模拟正式 OSS Worker 请求。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cloud_contracts import validate_worker_envelope
from cloud.simulated_contract_versions import install_simulated_contract_versions

DERMAVISION_ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores", "purple",
    "surface_gloss", "vascular", "contour_firmness",
)


def _queue_name(algorithm: str) -> str:
    from src.capture_profile import (
        CaptureProfile,
        capture_profile_from_environment,
    )
    from typing_extensions import assert_never

    profile = capture_profile_from_environment()
    match profile:
        case CaptureProfile.INSTITUTION:
            return algorithm
        case CaptureProfile.CONSUMER:
            return f"consumer_{algorithm}"
        case unreachable:
            assert_never(unreachable)


def _copy_upload(
    local_path: str, report_id: str, algo_name: str, file_name: str, oss_root: Path
) -> str:
    """模拟 backend internal-api 上传：本地目录镜像 report/{report_id}/{algo_name}/ 布局。"""
    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(f"模拟上传源文件不存在: {source}")
    object_key = f"report/{report_id}/{algo_name}/{file_name}"
    destination = oss_root / object_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return object_key


def _run_one(task, *, record_id: str, oss_key: str, algorithm: str | None) -> dict:
    algorithms = [algorithm] if algorithm else None
    return task.run(
        record_id,
        oss_key,
        "cloud-simulation/v1/",
        algorithms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True, choices=("dermavision", "acne", "wrinkle"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oss-root", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    oss_root = args.oss_root.resolve()
    response_path = args.response.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    # 三类 Worker 都从根项目导入，共用根 uv.lock。正式契约包
    # 不可用时，仅在本地模拟子进程注入文档锁定的版本常量。
    install_simulated_contract_versions()
    if args.service == "dermavision":
        from src import worker
    elif args.service == "acne":
        from src.acne import worker
    else:
        from src.wrinkle import worker

    input_bytes = input_path.read_bytes()
    source_key = f"cloud-simulation/input/{input_path.name}"
    # worker 已收敛到 backend internal-api；模拟时直接替换 worker 模块级
    # download/upload 函数（三个 worker 均从 src.core.storage 按名导入），无需真实 backend。
    worker.download_via_internal_api = lambda _oss_key: input_bytes
    worker.upload_via_internal_api = (
        lambda local_path, report_id, algo_name, file_name: _copy_upload(
            local_path, report_id, algo_name, file_name, oss_root
        )
    )

    started = time.perf_counter()
    responses: dict[str, dict] = {}
    if args.service == "dermavision":
        for algorithm in DERMAVISION_ALGORITHMS:
            task_started = time.perf_counter()
            response = _run_one(
                worker.analyze_image,
                record_id=f"visia2-{algorithm}",
                oss_key=source_key,
                algorithm=algorithm,
            )
            response = validate_worker_envelope(algorithm, response)
            responses[algorithm] = {
                "queue": _queue_name(algorithm),
                "task": "dermavision.analyze_image",
                "arguments": [f"visia2-{algorithm}", source_key, "cloud-simulation/v1/", [algorithm]],
                "elapsed_seconds": round(time.perf_counter() - task_started, 6),
                "pydantic_validation": "passed",
                "response": response,
            }
    else:
        if args.service == "acne":
            algorithm_name = "acne_v2"
            service_name = "acne_v2"
            task_name = "dermavision.analyze_image"
            task = worker.analyze_image_v2
        else:
            algorithm_name = args.service
            service_name = worker._SERVICE_NAME
            task_name = f"{service_name}.analyze_image"
            task = worker.analyze_image
        task_started = time.perf_counter()
        response = _run_one(
            task,
            record_id=f"visia2-{algorithm_name}",
            oss_key=source_key,
            algorithm=algorithm_name,
        )
        response = validate_worker_envelope(algorithm_name, response)
        responses[algorithm_name] = {
            "queue": _queue_name(service_name),
            "task": task_name,
            "arguments": [
                f"visia2-{algorithm_name}", source_key,
                "cloud-simulation/v1/", [algorithm_name],
            ],
            "elapsed_seconds": round(time.perf_counter() - task_started, 6),
            "pydantic_validation": "passed",
            "response": response,
        }

    payload = {
        "service": args.service,
        "input": str(input_path),
        "simulated_oss_root": str(oss_root),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "tasks": responses,
    }
    # 与 Celery JSON serializer 一样，强制确认返回可以直接 JSON 编码。
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return 0 if all(item["response"].get("status") == "success" for item in responses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
