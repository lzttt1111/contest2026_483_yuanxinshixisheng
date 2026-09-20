from __future__ import annotations

"""Project current visible, UV and Brown truth into five formal groups."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ControlledPigmentationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        current = _dictionary(current).get(key)
    return current


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
        raise ControlledPigmentationError("正式报告缺少检测模块")
    for module in modules:
        if isinstance(module, dict) and module.get("模块编号") == "03":
            return module
    raise ControlledPigmentationError("正式报告缺少综合色素模块")


def _detector(complete: dict[str, Any], item_id: str) -> dict[str, Any]:
    detectors = _dictionary(complete.get("detector_results"))
    return _dictionary(_dictionary(detectors.get(item_id)).get("metrics"))


def _medical(raw: dict[str, Any]) -> dict[str, Any]:
    return _dictionary(raw.get("medical_metrics_v2")) or raw


def _spot_values(raw: dict[str, Any]) -> dict[str, Any]:
    overall = _dictionary(_medical(raw).get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    auxiliary = _dictionary(overall.get("auxiliary_metrics"))
    scope = _dictionary(core.get("scope_and_burden"))
    counts = _dictionary(auxiliary.get("count_and_density"))
    aux_scope = _dictionary(auxiliary.get("scope_and_burden"))
    signal = _dictionary(core.get("signal_intensity"))
    aux_signal = _dictionary(auxiliary.get("signal_intensity"))
    distribution = _dictionary(auxiliary.get("distribution"))
    count = raw.get("spot_count")
    if not isinstance(count, (int, float)):
        count = sum(
            int(counts.get(key) or 0)
            for key in (
                "punctate_spot_count",
                "patch_spot_count",
                "confluent_candidate_count",
            )
        )
    return {
        "count": count,
        "density": scope.get("feature_density_per_100k_skin_px"),
        "area_ratio": raw.get("spot_area_ratio") or (
            float(scope.get("punctate_spot_area_ratio") or 0)
            + float(scope.get("patch_spot_area_ratio") or 0)
        ),
        "p50_area": aux_scope.get("p50_instance_area_px"),
        "p90_area": aux_scope.get("p90_instance_area_px"),
        "mean_delta": raw.get("mean_deltaE") or aux_signal.get("p50_delta_e"),
        "p90_delta": signal.get("p90_delta_e"),
        "punctate": counts.get("punctate_spot_count"),
        "patch": counts.get("patch_spot_count"),
        "primary": distribution.get("primary_concentration_region"),
    }


def _uv_values(raw: dict[str, Any]) -> dict[str, Any]:
    chinese = _dictionary(raw.get("uv_spots"))
    if chinese:
        total = _dictionary(chinese.get("总体指标"))
        core = _dictionary(total.get("核心指标"))
        auxiliary = _dictionary(total.get("辅助指标"))
        return {
            "count": _nested(auxiliary, "数量与密度", "特征数量（个）"),
            "density": _nested(core, "范围与负担", "单位面积密度（个/10万有效皮肤像素）"),
            "area_ratio": _nested(core, "范围与负担", "特征面积占比"),
            "p90": _nested(core, "信号强度", "P90强度（0～1）"),
        }
    overall = _dictionary(raw.get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    auxiliary = _dictionary(overall.get("auxiliary_metrics"))
    count = _nested(core, "count_and_density", "feature_count")
    valid = _nested(auxiliary, "analysis_scope", "valid_skin_area_px")
    density = (
        float(count) * 100000.0 / float(valid)
        if isinstance(count, (int, float))
        and isinstance(valid, (int, float))
        and valid > 0
        else None
    )
    return {
        "count": count,
        "density": density,
        "area_ratio": _nested(core, "scope_and_morphology", "feature_area_ratio"),
        "p90": _nested(core, "signal_intensity", "p90_intensity"),
    }


def _brown_values(raw: dict[str, Any]) -> dict[str, Any]:
    overall = _dictionary(_medical(raw).get("overall_metrics"))
    core = _dictionary(overall.get("core_metrics"))
    auxiliary = _dictionary(overall.get("auxiliary_metrics"))
    scope = _dictionary(auxiliary.get("scope_and_burden"))
    signal = _dictionary(auxiliary.get("signal_intensity"))
    distribution = _dictionary(auxiliary.get("distribution"))
    return {
        "count": raw.get("总计"),
        "density": scope.get("feature_density_per_100k_skin_px"),
        "area_ratio": scope.get("brown_spot_area_ratio"),
        "mean": signal.get("mean_intensity"),
        "p90": _nested(core, "signal_intensity", "p90_intensity"),
        "primary": distribution.get("primary_concentration_region"),
    }


def apply_current_pigmentation(
    payload: dict[str, Any],
    complete: dict[str, Any],
) -> None:
    """Replace every pigmentation number; derived groups remain image-only."""

    module = _module(payload)
    spots = _spot_values(_detector(complete, "spots"))
    uv = _uv_values(_detector(complete, "uv_spots"))
    brown = _brown_values(_detector(complete, "brown"))
    visible_metrics = [
        _metric("可见斑点数量", spots["count"], "个", "COUNT"),
        _metric("可见斑点密度", spots["density"], "个/10万有效皮肤像素", "DENSITY"),
        _metric("可见斑点面积占比", spots["area_ratio"], "%", "PERCENTAGE"),
        _metric("可见斑点P50面积", spots["p50_area"], "像素", "PIXEL"),
        _metric("可见斑点P90面积", spots["p90_area"], "像素", "PIXEL"),
        _metric("P90综合色差ΔE", spots["p90_delta"], "ΔE", "DELTA_E"),
        _metric("点状斑点数量", spots["punctate"], "个", "COUNT"),
        _metric("片状斑点数量", spots["patch"], "个", "COUNT"),
    ]
    uv_metrics = [
        _metric("UV色斑数量", uv["count"], "个", "COUNT"),
        _metric("UV色斑密度", uv["density"], "个/10万有效皮肤像素", "DENSITY"),
        _metric("UV色斑面积占比", uv["area_ratio"], "%", "PERCENTAGE"),
        _metric("UV色斑P90强度", uv["p90"], "0～1", "INTENSITY_ENGINEERING"),
    ]
    brown_metrics = [
        _metric("Brown综合色素目标数量", brown["count"], "个", "COUNT"),
        _metric("Brown综合色素密度", brown["density"], "个/10万有效皮肤像素", "DENSITY"),
        _metric("Brown综合色素面积占比", brown["area_ratio"], "%", "PERCENTAGE"),
        _metric("Brown综合色素平均响应", brown["mean"], "0～1", "INTENSITY_ENGINEERING"),
        _metric("Brown综合色素P90响应", brown["p90"], "0～1", "INTENSITY_ENGINEERING"),
    ]
    primary = spots["primary"] or brown["primary"] or "不可评估"
    module["核心指标"] = [
        visible_metrics[0],
        brown_metrics[0],
        _metric("可见色斑主要集中区域", primary, "", "TEXT"),
        uv_metrics[0],
        uv_metrics[2],
        uv_metrics[3],
    ]
    module["医生结果分组"] = [
        {"title": "可见色斑", "metrics": visible_metrics},
        {"title": "UV色斑", "metrics": uv_metrics},
        {"title": "Brown综合色素", "metrics": brown_metrics},
        {"title": "红褐混合印记", "metrics": [], "unavailable": True},
        {"title": "重点色斑区域", "metrics": [], "unavailable": True},
    ]
    module["结果摘要"] = (
        f"本次检出可见斑点{spots['count']}个、UV色斑{uv['count']}个、"
        f"Brown综合色素目标{brown['count']}个，主要集中于{primary}。"
    )


__all__ = ["ControlledPigmentationError", "apply_current_pigmentation"]
