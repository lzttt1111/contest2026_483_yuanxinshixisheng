#!/usr/bin/env python3
"""AISIA 单 RGB 十二项云端 Worker 统一启动器。

十一任务profile共用根 ``uv.lock``，以十一个 ``concurrency=1``
独立进程运行。``purple`` 一个任务返回 UV 色斑和紫质，因此十一任务对应
十二项结果；acne v1/v2及institution/consumer是互斥profile，旧任务命令与合同保持兼容。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.capture_profile import (
    CONSUMER_BASE_TARGETS,
    CONSUMER_TARGET_PREFIX,
    CaptureProfile,
)
from src.summary import (
    SUMMARY_APP,
    SUMMARY_CONCURRENCY,
    SUMMARY_QUEUE,
    SUMMARY_TARGET,
    SUMMARY_TASK,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONTRACT_MANIFEST = PROJECT_ROOT / "cloud" / "contracts" / "internal_dev_contracts.json"
DERMAVISION_TARGETS = ("redness", "spots", "brown", "texture", "pores", "purple")


@lru_cache(maxsize=1)
def _registry() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Load the detection registry lazily.

    ``src.worker`` imports cv2/mediapipe/docx; deferring it keeps the CPU-only
    ``--target summary --print-command`` path free of heavy modules.
    """
    from src.worker import WORKER_TARGET_ALIASES, WORKER_TARGETS

    return WORKER_TARGETS, WORKER_TARGET_ALIASES


@dataclass(frozen=True)
class WorkerLaunch:
    target: str
    capture_profile: CaptureProfile
    app: str
    queues: tuple[str, ...]
    concurrency: int
    task: str
    results: tuple[str, ...]


def _canonical_target(target: str) -> str:
    _, aliases = _registry()
    return str(aliases.get(target, target))


def resolve_launch(target: str, concurrency: int | None = None) -> WorkerLaunch:
    """解析十一正式目标，并保留旧 ``dermavision-*`` 别名。"""

    requested_target = target
    capture_profile = CaptureProfile.INSTITUTION
    if target.startswith(CONSUMER_TARGET_PREFIX):
        candidate = target.removeprefix(CONSUMER_TARGET_PREFIX)
        if candidate not in CONSUMER_BASE_TARGETS:
            raise ValueError(f"未知 Worker target: {requested_target}")
        target = candidate
        capture_profile = CaptureProfile.CONSUMER

    if target == SUMMARY_TARGET:
        # CPU-only aggregation: resolved without importing src.worker.
        launch = WorkerLaunch(
            target=SUMMARY_TARGET,
            capture_profile=capture_profile,
            app=SUMMARY_APP,
            queues=(SUMMARY_QUEUE,),
            concurrency=SUMMARY_CONCURRENCY,
            task=SUMMARY_TASK,
            results=(),
        )
    elif target == "dermavision-all":
        launch = WorkerLaunch(
            target=target,
            capture_profile=capture_profile,
            app="src.worker:celery_app",
            queues=DERMAVISION_TARGETS,
            concurrency=6,
            task="dermavision.analyze_image",
            results=("红区", "斑点", "棕区", "纹理", "毛孔", "UV色斑", "紫质"),
        )
    elif target == "acne_v2":
        launch = WorkerLaunch(
            target=requested_target,
            capture_profile=capture_profile,
            app="src.acne.worker:celery_app",
            queues=(requested_target,),
            concurrency=1,
            task="dermavision.analyze_image",
            results=("痤疮",),
        )
    else:
        worker_targets, worker_aliases = _registry()
        canonical = _canonical_target(target)
        definition = worker_targets.get(canonical)
        if definition is None:
            valid = [
                *worker_targets,
                *worker_aliases,
                "acne_v2",
                "dermavision-all",
            ]
            raise ValueError(f"未知 Worker target: {target}; 可用值: {', '.join(valid)}")
        launch = WorkerLaunch(
            target=(requested_target if capture_profile is CaptureProfile.CONSUMER else canonical),
            capture_profile=capture_profile,
            app=str(definition["app"]),
            queues=(
                requested_target
                if capture_profile is CaptureProfile.CONSUMER
                else str(definition["queue"]),
            ),
            concurrency=int(definition.get("concurrency", 1)),
            task=str(definition["task"]),
            results=tuple(str(item) for item in definition["results"]),
        )

    if concurrency is None:
        return launch
    if concurrency < 1:
        raise ValueError("concurrency 必须大于等于 1")
    return WorkerLaunch(
        target=launch.target,
        capture_profile=launch.capture_profile,
        app=launch.app,
        queues=launch.queues,
        concurrency=concurrency,
        task=launch.task,
        results=launch.results,
    )


def build_command(launch: WorkerLaunch, loglevel: str, pool: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "celery",
        "-A",
        launch.app,
        "worker",
        f"--loglevel={loglevel}",
        f"--queues={','.join(launch.queues)}",
        f"--concurrency={launch.concurrency}",
        f"--pool={pool}",
    ]


def validate_layout(launch: WorkerLaunch) -> None:
    module_name = launch.app.split(":", 1)[0]
    module_path = PROJECT_ROOT / Path(*module_name.split(".")).with_suffix(".py")
    base_target = launch.target.removeprefix(CONSUMER_TARGET_PREFIX)
    if base_target in {"acne", "acne_v2"}:
        agents_path = PROJECT_ROOT / "src" / "acne" / "WORKER_CONTRACT.md"
    elif base_target == "wrinkle":
        agents_path = PROJECT_ROOT / "src" / "wrinkle" / "WORKER_CONTRACT.md"
    else:
        agents_path = PROJECT_ROOT / "WORKER_CONTRACT.md"
    required = (module_path, agents_path, CONTRACT_MANIFEST, PROJECT_ROOT / "uv.lock")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("云端 Worker 必需文件缺失: " + ", ".join(missing))


def describe(launch: WorkerLaunch, command: list[str]) -> dict[str, object]:
    return {
        "target": launch.target,
        "capture_profile": launch.capture_profile.value,
        "app": launch.app,
        "queues": list(launch.queues),
        "task": launch.task,
        "results": list(launch.results),
        "concurrency": launch.concurrency,
        "working_directory": str(PROJECT_ROOT),
        "python": sys.executable,
        "command": command,
        "command_text": shlex.join(command),
        "contract_manifest": str(CONTRACT_MANIFEST),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用根 uv.lock 启动机构目标或加法consumer目标"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="原十一目标、acne_v2，或加法consumer_*目标",
    )
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--loglevel", default="info")
    parser.add_argument("--pool", default="prefork")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="只打印解析后的 App、Queue、Task 和命令，不连接 Redis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        launch = resolve_launch(args.target, args.concurrency)
        validate_layout(launch)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    command = build_command(launch, args.loglevel, args.pool)
    if args.print_command:
        print(json.dumps(describe(launch, command), ensure_ascii=False, indent=2))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["DERMAVISION_CAPTURE_PROFILE"] = launch.capture_profile.value
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    os.chdir(PROJECT_ROOT)
    os.execve(sys.executable, command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
