from __future__ import annotations

"""Project current redness truth into the formal report tables."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ControlledRednessError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(name: str, value: Any, unit: str, display_type: str) -> dict[str, Any]:
    return {
        "name": name,
        "medical_name": name,
        "value": value,
        "unit": unit,
        "display_type": display_type,
        "availability": "available" if value not in (None, "不可评估") else "unavailable",
    }


def _module(payload: dict[str, Any]) -> dict[str, Any]:
    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        raise ControlledRednessError("正式报告缺少检测模块")
    for module in modules:
        if isinstance(module, dict) and module.get("模块编号") == "04":
            return module
    raise ControlledRednessError("正式报告缺少泛红模块")


def _detector(complete: dict[str, Any]) -> dict[str, Any]:
    detectors = _dictionary(complete.get("detector_results"))
    row = _dictionary(detectors.get("redness"))
    return _dictionary(row.get("metrics"))


def apply_current_redness(
    payload: dict[str, Any],
    complete: dict[str, Any],
) -> None:
    """Replace every displayed redness value with this run's detector truth."""

    module = _module(payload)
    raw = _detector(complete)
    medical = _dictionary(raw.get("medical_metrics_v2")) or raw
    overall = _dictionary(medical.get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    scope = _dictionary(core.get("scope_and_burden"))
    auxiliary = _dictionary(core.get("auxiliary_statistics"))
    analysis = _dictionary(
        _dictionary(overall.get("auxiliary_metrics")).get("analysis_scope")
    )
    valid_area = analysis.get("valid_skin_area_px")
    max_area = scope.get("max_continuous_region_area_px")
    max_ratio = (
        float(max_area) / float(valid_area)
        if isinstance(max_area, (int, float))
        and isinstance(valid_area, (int, float))
        and valid_area > 0
        else None
    )
    metrics = {
        "coverage": [
            _metric("弥漫红区面积占比", scope.get("diffuse_red_area_ratio"), "%", "PERCENTAGE"),
            _metric("高红度响应区域面积占比", raw.get("high_red_area_ratio"), "%", "PERCENTAGE"),
        ],
        "intensity": [
            _metric("平均红度", raw.get("mean_redness"), "0～1", "INTENSITY_ENGINEERING"),
            _metric("P90红度", raw.get("p90_redness"), "0～1", "INTENSITY_ENGINEERING"),
        ],
        "continuity": [
            _metric("最大连续红区面积占比", max_ratio, "%", "PERCENTAGE"),
            _metric("红区连续性", scope.get("redness_continuity"), "0～1", "ENGINEERING_0_1"),
        ],
        "boundary": [
            _metric("边界渐变", auxiliary.get("redness_boundary_gradient"), "0～1", "ENGINEERING_0_1"),
            _metric("弥漫红区均匀度", auxiliary.get("diffuse_redness_uniformity"), "0～1", "ENGINEERING_0_1"),
            _metric("局灶红色实例数量", raw.get("red_feature_count"), "个", "COUNT"),
        ],
    }
    module["核心指标"] = [metric for group in metrics.values() for metric in group]
    module["医生结果分组"] = [
        {"title": "泛红覆盖", "metrics": metrics["coverage"]},
        {"title": "泛红强度", "metrics": metrics["intensity"]},
        {"title": "泛红连续性", "metrics": metrics["continuity"]},
        {"title": "边界与均匀性", "metrics": metrics["boundary"]},
    ]
    diffuse = scope.get("diffuse_red_area_ratio")
    module["结果摘要"] = (
        f"本次弥漫红区面积占比为{float(diffuse) * 100:.2f}%，"
        f"检出局灶红色实例{raw.get('red_feature_count', '不可评估')}个。"
        if isinstance(diffuse, (int, float))
        else "本次泛红结果缺少可用面积分母，标记为不可评估。"
    )
    module["分区指标"] = []
    for row in medical.get("region_metrics", []):
        if not isinstance(row, dict):
            continue
        row_core = _dictionary(row.get("core_metrics"))
        row_scope = _dictionary(row_core.get("scope_and_burden"))
        row_signal = _dictionary(row_core.get("signal_intensity"))
        module["分区指标"].append(
            {
                "分区名称": str(row.get("analysis_region", "未知分区")),
                "状态": str(row.get("evaluation_status", "不可评估")),
                "指标": {
                    "弥漫红区面积占比": row_scope.get("diffuse_red_area_ratio"),
                    "平均红度": row_signal.get("mean_intensity"),
                    "P90红度": row_signal.get("p90_intensity"),
                },
            }
        )


__all__ = ["ControlledRednessError", "apply_current_redness"]
