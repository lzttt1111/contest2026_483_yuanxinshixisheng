from __future__ import annotations

"""Fill explanation metadata after current-run metrics replace frozen values."""

from collections.abc import Iterator
from typing import TypeAlias

from .formal_view import _metric_explanation


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _metric_rows(module: JsonObject) -> Iterator[JsonObject]:
    core = module.get("核心指标")
    if isinstance(core, list):
        for metric in core:
            if isinstance(metric, dict):
                yield metric
    groups = module.get("医生结果分组")
    if not isinstance(groups, list):
        return
    for group in groups:
        if not isinstance(group, dict):
            continue
        metrics = group.get("metrics")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if isinstance(metric, dict):
                yield metric


def attach_missing_metric_explanations(payload: JsonObject) -> None:
    """Add explanations only where controlled truth replacement left them blank."""

    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        return
    for module in modules:
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("模块编号", ""))
        for metric in _metric_rows(module):
            if str(metric.get("explanation") or "").strip():
                continue
            metric["explanation"] = _metric_explanation(metric, module_id=module_id)


__all__ = ["attach_missing_metric_explanations"]
