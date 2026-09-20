"""九项单仓库的统一项目根目录映射。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_NAMES = ("dermavision", "acne", "wrinkle")


def service_root(service: str) -> Path:
    """三类分析共用根 ``uv.lock`` 和根工作目录。"""
    if service in SERVICE_NAMES:
        return PROJECT_ROOT
    raise ValueError(f"未知服务: {service}")
