"""为本地 GPU 云端模拟注入锁定的契约版本常量。

正式 Worker 仍必须使用 ``aisia-contracts@v0.1.0``。本模块只在
``cloud/simulate_service_task.py`` 启动的本地子进程中生效，不修改
Worker 入口或部署依赖。
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from cloud_contracts import SCHEMA_VERSIONS


_MODULES = {
    "aisia_contracts.algorithms.redness.v3": SCHEMA_VERSIONS["redness"],
    "aisia_contracts.algorithms.spots.v3": SCHEMA_VERSIONS["spots"],
    "aisia_contracts.algorithms.brown.v3": SCHEMA_VERSIONS["brown"],
    "aisia_contracts.algorithms.texture.v3": SCHEMA_VERSIONS["texture"],
    "aisia_contracts.algorithms.pores.v3": SCHEMA_VERSIONS["pores"],
    "aisia_contracts.algorithms.purple.v2": SCHEMA_VERSIONS["purple"],
    "aisia_contracts.algorithms.acne.v1": SCHEMA_VERSIONS["acne"],
    "aisia_contracts.algorithms.wrinkle.v2": SCHEMA_VERSIONS["wrinkle"],
    "aisia_contracts.algorithms.surface_gloss.v1": SCHEMA_VERSIONS["surface_gloss"],
    "aisia_contracts.algorithms.vascular.v1": SCHEMA_VERSIONS["vascular"],
    "aisia_contracts.algorithms.contour_firmness.v1": SCHEMA_VERSIONS["contour_firmness"],
}


def _package(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def install_simulated_contract_versions() -> bool:
    """安装本地模拟常量；真实契约模块始终优先。"""

    try:
        importlib.import_module("aisia_contracts")
    except ModuleNotFoundError:
        _package("aisia_contracts")

    _package("aisia_contracts.algorithms")
    installed = False
    for full_name, version in _MODULES.items():
        try:
            importlib.import_module(full_name)
            continue
        except ModuleNotFoundError:
            pass
        parts = full_name.split(".")
        for index in range(3, len(parts)):
            _package(".".join(parts[:index]))
        module = ModuleType(full_name)
        module.SCHEMA_VERSION = version  # type: ignore[attr-defined]
        sys.modules[full_name] = module
        installed = True
    return installed
