from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


_INTERNAL_TERMS = (
    "official", "shadow", "candidate", "ECDF", "GAP", "Mask", "回灌",
    "参考样本数", "工程默认等权", "验收", "生产分",
)

_SOURCE_NAMES = {
    "01": ["标准白光图像"],
    "02": ["标准白光图像", "UV图像"],
    "03": ["标准白光图像", "UV图像", "Brown综合色素图", "Red红色图"],
    "04": ["标准白光图像", "Red红色图"],
    "05": ["标准白光图像", "Red红色图"],
    "06": ["标准白光图像"],
    "07": ["标准白光图像"],
    "08": ["标准白光图像"],
    "09": ["标准白光图像"],
    "10": ["标准白光图像"],
    "11": ["标准白光图像"],
}

_DRIVER_NAMES = {
    "01": {
        "density": "毛孔密度", "coverage": "毛孔面积覆盖", "size": "典型毛孔面积",
        "large_pores": "偏大毛孔尺度", "shape": "毛孔形态不规则度",
    },
    "03": {
        "visible_spots": "可见色斑负担", "uv_spots": "UV色斑负担", "brown": "Brown综合色素负担",
    },
    "04": {
        "coverage": "泛红覆盖", "intensity": "泛红强度", "continuity": "泛红连续性",
        "boundary_uniformity": "边界与均匀性",
    },
}

_REGION_METRIC_TYPES = {
    "油光面积占比": "PERCENTAGE",
    "高强度油光面积占比": "PERCENTAGE",
    "最大连续油光区域面积占比": "PERCENTAGE",
    "P90油光强度": "INTENSITY_ENGINEERING",
    "可见斑点特征数量（个）": "COUNT",
    "可见斑点单位面积密度（个/10万有效皮肤像素）": "DENSITY",
    "可见斑点特征面积占比": "PERCENTAGE",
    "可见斑点P90强度（0～1）": "INTENSITY_ENGINEERING",
    "棕色实例特征数量（个）": "COUNT",
    "棕色实例单位面积密度（个/10万有效皮肤像素）": "DENSITY",
    "棕色实例特征面积占比": "PERCENTAGE",
    "棕色实例P90强度（0～1）": "INTENSITY_ENGINEERING",
    "综合色素联合区域数量": "COUNT",
    "综合色素联合区域面积占比": "PERCENTAGE",
    "有效皮肤面积（像素）": "PIXEL",
    "弥漫红区面积占比": "PERCENTAGE",
    "平均强度（0～1）": "INTENSITY_ENGINEERING",
    "P90强度（0～1）": "INTENSITY_ENGINEERING",
    "P95强度（0～1）": "INTENSITY_ENGINEERING",
    "高强度区域面积占比": "PERCENTAGE",
    "特征数量（个）": "COUNT",
    "单位面积密度（个/10万有效皮肤像素）": "DENSITY",
    "红度综合负担": "ENGINEERING_0_1",
    "候选数量": "COUNT",
    "候选总长度（像素）": "PIXEL",
    "候选面积（像素）": "PIXEL",
    "线状结构数量": "COUNT",
    "线状结构总长度（像素）": "PIXEL",
    "线状结构面积（像素）": "PIXEL",
    "痤疮样特征数量（个）": "COUNT",
}

_REGION_METRIC_UNITS = {
    "有效皮肤面积（像素）": "像素",
    "特征数量（个）": "个",
    "单位面积密度（个/10万有效皮肤像素）": "个/10万有效皮肤像素",
    "可见斑点特征数量（个）": "个",
    "可见斑点单位面积密度（个/10万有效皮肤像素）": "个/10万有效皮肤像素",
    "棕色实例特征数量（个）": "个",
    "棕色实例单位面积密度（个/10万有效皮肤像素）": "个/10万有效皮肤像素",
    "综合色素联合区域数量": "个",
    "候选数量": "个",
    "候选总长度（像素）": "像素",
    "候选面积（像素）": "像素",
    "线状结构数量": "个",
    "线状结构总长度（像素）": "像素",
    "线状结构面积（像素）": "像素",
    "痤疮样特征数量（个）": "个",
}

# 正式报告字段类型必须按字段语义登记，不允许从数值大小推测。
_FULL_METRIC_OVERRIDES: dict[tuple[str, str] | str, tuple[str, str]] = {
    # 01 可见毛孔
    ("01", "可见毛孔数量"): ("COUNT", "个"),
    ("01", "单位面积密度"): ("DENSITY_100K", "个/10万有效皮肤像素"),
    ("01", "毛孔面积占比"): ("PERCENTAGE", "%"),
    ("01", "P50单体面积"): ("AREA_PIXEL", "像素²"),
    ("01", "P90单体面积"): ("AREA_PIXEL", "像素²"),
    ("01", "最大单体面积"): ("AREA_PIXEL", "像素²"),
    ("01", "P50圆度"): ("NORMALIZED_0_1", "无量纲（0～1）"),
    ("01", "P50长宽比"): ("RATIO", "无量纲（比值）"),
    ("01", "低圆度毛孔比例"): ("PERCENTAGE", "%"),
    ("01", "拉长样毛孔比例"): ("PERCENTAGE", "%"),
    # 02 油脂分泌倾向
    ("02", "表面油光面积占比"): ("PERCENTAGE", "%"),
    ("02", "高强度油光面积占比"): ("PERCENTAGE", "%"),
    ("02", "P50油光强度"): ("NORMALIZED_0_1", "无量纲（0～1）"),
    ("02", "P90油光强度"): ("NORMALIZED_0_1", "无量纲（0～1）"),
    ("02", "最大连续油光区域面积占比"): ("PERCENTAGE", "%"),
    ("02", "紫质目标数量"): ("COUNT", "个"),
    ("02", "紫质单位面积密度"): ("DENSITY_100K", "个/10万有效皮肤像素"),
    ("02", "紫质面积占比"): ("PERCENTAGE", "%"),
    ("02", "紫质P50强度"): ("NORMALIZED_0_1", "无量纲（0～1）"),
    ("02", "紫质P90强度"): ("NORMALIZED_0_1", "无量纲（0～1）"),
    "分区P90视觉对比度最大值": ("NORMALIZED_0_1", "无量纲（0～1）"),
    # 03 综合色素
    "可见斑点数量": ("COUNT", "个"),
    "可见斑点密度": ("DENSITY_100K", "个/10万有效皮肤像素"),
    "可见斑点面积占比": ("PERCENTAGE", "%"),
    "可见斑点P50面积": ("AREA_PIXEL", "像素²"),
    "可见斑点P90面积": ("AREA_PIXEL", "像素²"),
    "可见斑点最大面积": ("AREA_PIXEL", "像素²"),
    "平均综合色差ΔE": ("DELTA_E", "ΔE"),
    "P90综合色差ΔE": ("DELTA_E", "ΔE"),
    "点状斑点数量": ("COUNT", "个"),
    "片状斑点数量": ("COUNT", "个"),
    "融合样候选数量": ("COUNT", "个"),
    "UV色斑数量": ("COUNT", "个"),
    "UV色斑密度": ("DENSITY_100K", "个/10万有效皮肤像素"),
    "UV色斑面积占比": ("PERCENTAGE", "%"),
    "UV色斑P90强度": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "UV全部目标面积占比": ("PERCENTAGE", "%"),
    "UV下更明显区域数量": ("COUNT", "个"),
    "UV下更明显区域面积占比": ("PERCENTAGE", "%"),
    "棕色实例数量": ("COUNT", "个"),
    "棕色实例密度": ("DENSITY_100K", "个/10万有效皮肤像素"),
    "棕色实例面积占比": ("PERCENTAGE", "%"),
    "棕色实例P50面积": ("AREA_PIXEL", "像素²"),
    "棕色实例P90面积": ("AREA_PIXEL", "像素²"),
    "P90棕色强度": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "连续棕色色素覆盖占比": ("PERCENTAGE", "%"),
    "最大连续棕色区域面积": ("AREA_PIXEL", "像素²"),
    "棕色色素综合负担": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "红褐混合目标数量": ("COUNT", "个"),
    "红褐混合总面积占比": ("PERCENTAGE", "%"),
    "红褐混合面积占可见色斑面积比例": ("PERCENTAGE", "%"),
    "疑似纯红区域面积占比": ("PERCENTAGE", "%"),
    "综合色素联合区域数量": ("COUNT", "个"),
    "综合色素联合区域面积占比": ("PERCENTAGE", "%"),
    "可见色斑主要集中区域": ("TEXT", "—"),
    # 04 弥漫性泛红
    "弥漫红区面积占比": ("PERCENTAGE", "%"),
    "高红度响应区域面积占比": ("PERCENTAGE", "%"),
    "平均红度": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "P90红度": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "最大连续红区面积占比": ("PERCENTAGE", "%"),
    "红区连续性": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "边界渐变": ("NORMALIZED_0_1", "无量纲（0～1）"),
    "弥漫红区均匀度": ("NORMALIZED_0_1", "无量纲（0～1）"),
    # 05 血管样结构候选观察
    "血管样线状结构数量": ("COUNT", "个"),
    "线状结构总长度": ("LENGTH_PIXEL", "像素"),
    "线状结构长度密度": ("LENGTH_DENSITY_10K", "像素/1万有效皮肤像素"),
    "线状结构面积占比": ("PERCENTAGE", "%"),
    "线状结构分支点数量": ("COUNT", "个"),
    "线状结构分支点密度": ("DENSITY_10K", "个/1万有效皮肤像素"),
    "线状结构网络比例": ("PERCENTAGE", "%"),
    # 06 痤疮样活动
    ("06", "有效皮肤面积"): ("AREA_PIXEL", "像素²"),
    ("06", "痤疮样特征数量"): ("COUNT", "个"),
    ("06", "单位面积密度"): ("DENSITY_100K", "个/10万有效皮肤像素"),
    ("06", "特征框总面积"): ("AREA_PIXEL", "像素²"),
    ("06", "特征框面积占比"): ("PERCENTAGE", "%"),
    ("06", "P50特征框面积"): ("AREA_PIXEL", "像素²"),
    ("06", "P90特征框面积"): ("AREA_PIXEL", "像素²"),
    ("06", "主要集中区域"): ("TEXT", "—"),
}

_GROUP_TITLES = {
    "A. Visible可见色斑": "可见色斑",
    "B. UV下更明显的色斑": "UV色斑与UV下更明显区域",
    "C. Brown综合色素": "Brown综合色素",
    "D. 红褐混合印记（辅助空间结果）": "红褐混合印记",
    "E. 重点色斑区域（辅助空间结果）": "重点色斑区域",
    "A. 候选数量与长度": "线状结构数量与长度",
    "B. 候选覆盖与宽度": "线状结构覆盖",
    "C. 局部红度工程响应": "局部红度相对响应",
    "D. 候选分支与网络形态": "线状结构分支与网络形态",
}


def _formal_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"(?i)V\d+(?:\.\d+)*", "", text)
    for term in _INTERNAL_TERMS:
        text = re.sub(re.escape(term), "", text, flags=re.IGNORECASE)
    replacements = {
        "当前评分口径": "评分口径", "工程归一化值": "相对强度", "工程值": "相对值",
        "工程响应": "相对响应", "候选结构": "线状结构", "候选线状结构": "线状结构",
        "候选线": "线", "候选分支": "分支", "候选网络": "网络", "候选数量": "数量",
        "候选": "",
        "当前版本": "本次检测", "当前正式": "当前", "正式": "",
        "算法": "检测", "VALID": "有效", "valid": "有效",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("；；", "；").replace("。。", "。")
    return text.strip(" ；，")


def _metric_value(module: dict[str, Any], name: str, default: Any = None) -> Any:
    for metric in module.get("核心指标", []) or []:
        if metric.get("name") == name:
            return metric.get("value")
    for group in module.get("医生结果分组", []) or []:
        for metric in group.get("metrics", []) or []:
            if metric.get("name") == name:
                return metric.get("value")
    return default


def _percent(value: Any) -> str:
    return f"{float(value or 0) * 100:.2f}%"


def _reliable_region(value: Any) -> str | None:
    text = _formal_text(value)
    if not text:
        return None
    invalid = ("未检出", "不可评估", "尚未形成可靠量化", "全面部")
    return None if any(token in text for token in invalid) else text


def _metric_explanation(metric: dict[str, Any], *, module_id: str = "") -> str:
    name = _formal_text(metric.get("name"))
    display = metric.get("display_type", "TEXT")
    if name == "主要集中区域":
        return {
            "06": "痤疮样特征数量最多的有效面部分区。",
            "07": "双眼下二维可见细纹相对响应较高的分区。",
            "08": "额头、眉间及鱼尾纹二维可见响应较高的分区。",
            "10": "二维表面纹理特征数量或负担较集中的有效面部分区。",
        }.get(module_id, "当前模块特征数量或负担较集中的有效面部分区。")
    exact = {
        "可见毛孔数量": "有效分析区域内检出的可见毛孔实例总数。",
        "毛孔面积占比": "全部毛孔实例面积之和占有效皮肤面积的比例。",
        "P50单体面积": "已检出毛孔实例面积的中位数。",
        "P90单体面积": "已检出毛孔实例面积的第90百分位，用于描述偏大毛孔尺度。",
        "最大单体面积": "本次检出的单个毛孔实例最大面积。",
        "P50圆度": "毛孔圆度的中位数，数值越接近1，形态越接近圆形。",
        "P50长宽比": "毛孔长轴与短轴比值的中位数；1表示长短轴相同。",
        "表面油光面积占比": "表面油光区域面积占有效皮肤面积的比例。",
        "高强度油光面积占比": "高强度油光区域面积占有效皮肤面积的比例。",
        "P50油光强度": "油光区域相对强度的中位数。",
        "P90油光强度": "油光区域相对强度的第90百分位。",
        "最大连续油光区域面积占比": "最大连通油光区域面积占有效皮肤面积的比例。",
        "紫质目标数量": "有效分析区域内检出的紫质实例总数。",
        "紫质单位面积密度": "按10万有效皮肤像素标准化后的紫质实例数量。",
        "紫质面积占比": "紫质实例面积之和占有效皮肤面积的比例。",
        "紫质P50强度": "紫质实例相对强度的中位数。",
        "紫质P90强度": "紫质实例相对强度的第90百分位。",
        "P90综合色差ΔE": "可见色斑综合色差的第90百分位。",
        "平均综合色差ΔE": "可见色斑实例的平均综合色差。",
        "可见色斑主要集中区域": "可见色斑实例数量或负担最集中的面部区域。",
        "UV全部目标面积占比": "UV图像中全部UV色素目标面积占有效皮肤面积的比例。",
        "UV下更明显区域数量": "经普通白光图像与UV图像空间比较后，在UV图像中表现更突出的区域数量。",
        "UV下更明显区域面积占比": "经普通白光图像与UV图像空间比较后，更突出区域的面积占比。",
        "连续棕色色素覆盖占比": "连续Brown综合色素区域面积占有效皮肤面积的比例。",
        "红褐混合总面积占比": "红色与棕色色素空间重叠区域面积占有效皮肤面积的比例。",
        "红褐混合面积占可见色斑面积比例": "红褐混合区域面积占可见色斑总面积的比例。",
        "疑似纯红区域面积占比": "辅助识别的纯红色区域面积占有效皮肤面积的比例。",
        "综合色素联合区域数量": "Visible、UV与Brown结果空间融合后形成的联合区域数量。",
        "综合色素联合区域面积占比": "综合色素联合区域面积占对应有效分区面积的比例。",
        "弥漫红区面积占比": "扣除局灶红色实例后，连续弥漫红区面积占有效皮肤面积的比例。",
        "高红度响应区域面积占比": "有效皮肤内达到高红度阈值的区域面积占比；该阈值区域未扣除局灶红色实例。",
        "平均红度": "有效分析区域内红度响应的平均水平。",
        "P90红度": "有效分析区域内红度响应的第90百分位。",
        "最大连续红区面积占比": "最大连通弥漫红区面积占有效皮肤面积的比例。",
        "红区连续性": "最大连通弥漫红区面积占全部弥漫红区面积的比例。",
        "边界渐变": "弥漫红区边界内外红度变化的相对梯度。",
        "弥漫红区均匀度": "弥漫红区内部红度分布的均匀程度，数值越高表示越均匀。",
        "血管样线状结构数量": "有效区域内检出的血管样线状候选结构数量。",
        "线状结构总长度": "全部血管样线状候选结构的骨架长度总和。",
        "线状结构长度密度": "每1万有效皮肤像素对应的候选线状结构总长度。",
        "线状结构面积占比": "血管样线状候选结构面积占有效皮肤面积的比例。",
        "线状结构分支点数量": "候选线状结构骨架中的分支点总数。",
        "线状结构分支点密度": "每1万有效皮肤像素对应的候选分支点数量。",
        "线状结构网络比例": "形成分支网络的候选线状结构占比。",
        "单位面积长度密度": "每1万有效皮肤像素对应的血管样线状结构总长度。",
        "最大连续网络长度": "本次血管样候选中最大连续网络的骨架长度。",
        "眼下有效皮肤面积": "参与双眼下细纹统计的有效皮肤像素面积。",
        "稳定区域有效皮肤面积": "参与稳定性皱纹统计的有效皮肤像素面积。",
        "眼下单位面积纹路长度密度": "每1万眼下有效皮肤像素对应的细纹中心线总长度。",
        "单位面积纹路长度密度": "每1万对应分区有效皮肤像素的纹路中心线总长度。",
        "有效皮肤面积": "本次参与痤疮样特征统计的有效皮肤面积。",
        "痤疮样特征数量": "有效分析区域内检出的痤疮样特征数量。",
        "特征框总面积": "全部痤疮样特征框在标准化图像中的面积总和。",
        "特征框面积占比": "全部痤疮样特征框面积占有效皮肤面积的比例。",
        "P50特征框面积": "痤疮样特征框面积的中位数。",
        "P90特征框面积": "痤疮样特征框面积的第90百分位。",
    }
    if name in exact:
        return exact[name]
    if name == "低圆度毛孔比例":
        return "低圆度毛孔实例数占有效毛孔实例总数的比例。"
    if name == "拉长样毛孔比例":
        return "拉长样毛孔实例数占有效毛孔实例总数的比例。"
    if name in {"偏大毛孔尺度", "P90单体面积"}:
        return "表示已检出毛孔实例面积的较高水平，用于描述偏大毛孔尺度。"
    if name == "线状结构长度密度":
        return "单位有效皮肤面积内检出的血管样线状结构总长度。"
    if name == "红褐混合面积占可见色斑面积比例":
        return "红褐混合区域面积占可见色斑总面积的比例。"
    if "P50" in name:
        return "表示该指标在已检出目标中的中位水平。"
    if "P90" in name:
        return "表示该指标在已检出目标中的较高水平。"
    if "密度" in name:
        return "按有效皮肤面积标准化后的目标数量。"
    if display == "PERCENTAGE" or "面积占比" in name or "覆盖" in name:
        return "该类特征面积占有效分析区域的比例。"
    if display == "COUNT" or "数量" in name:
        return "有效分析区域内检出的特征数量。"
    if "左右" in name:
        return "画面左右对应区域的定量比较。"
    if "连续" in name:
        return "表示相邻特征在有效区域内的连续程度。"
    if "均匀" in name:
        return "表示特征在有效区域内的分布均匀程度。"
    if "强度" in name or "红度" in name or "响应" in name:
        return "表示该图像特征的相对显著程度。"
    if display == "AREA_PIXEL" or "面积" in name:
        return "标准化图像中对应目标的像素面积统计。"
    if display == "LENGTH_PIXEL" or "长度" in name:
        return "标准化图像中对应线状结构的长度统计。"
    if display == "RATIO" or "比" in name:
        return "用于描述两个同类量之间的无量纲比值。"
    return f"用于描述{name}的本次检测结果。"


def _formal_metric(metric: dict[str, Any], *, module_id: str = "", source_path: str = "") -> dict[str, Any]:
    clean_name = _formal_text(metric.get("name"))
    display_type = metric.get("display_type", "TEXT")
    unit = _formal_text(metric.get("unit", ""))
    override = _FULL_METRIC_OVERRIDES.get((module_id, clean_name))
    if override is None:
        override = _FULL_METRIC_OVERRIDES.get(clean_name)
    if override is not None:
        display_type, unit = override
    if not unit:
        unit = {
            "PERCENTAGE": "%",
            "NORMALIZED_0_1": "无量纲（0～1）",
            "ENGINEERING_0_1": "无量纲（0～1）",
            "INTENSITY_ENGINEERING": "无量纲（0～1）",
            "RATIO": "无量纲（比值）",
            "COUNT": "个",
            "AREA_PIXEL": "像素²",
            "LENGTH_PIXEL": "像素",
        }.get(str(display_type), "—" if display_type == "TEXT" else "")
    result = {
        "name": clean_name,
        "value": metric.get("value"),
        "unit": unit,
        "display_type": display_type,
        "source_path": source_path or f"检测模块[{module_id}].指标.{metric.get('name', '')}",
        "availability": "available" if metric.get("value") not in (None, "", "不可评估") else "unavailable",
    }
    result["medical_name"] = result["name"]
    result["explanation"] = _metric_explanation(result, module_id=module_id)
    return result


def _refresh_metric_metadata(metric: dict[str, Any], *, module_id: str) -> None:
    override = _FULL_METRIC_OVERRIDES.get((module_id, str(metric.get("name"))))
    if override is None:
        override = _FULL_METRIC_OVERRIDES.get(str(metric.get("name")))
    if override is not None:
        metric["display_type"], metric["unit"] = override
    metric["medical_name"] = metric.get("name")
    metric["explanation"] = _metric_explanation(metric, module_id=module_id)


def _ratio_value(value: Any) -> float | None:
    if value in (None, "", "不可评估"):
        return None
    if isinstance(value, str) and value.strip().endswith("%"):
        try:
            return float(value.strip()[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _region_metric(name: str, value: Any, *, module_id: str, region_name: str) -> dict[str, Any]:
    display_type = _REGION_METRIC_TYPES.get(name, "TEXT")
    normalized = _ratio_value(value) if display_type == "PERCENTAGE" else value
    unit = _REGION_METRIC_UNITS.get(name, "")
    if not unit:
        unit = {
            "PERCENTAGE": "%",
            "ENGINEERING_0_1": "无量纲（0～1）",
            "INTENSITY_ENGINEERING": "无量纲（0～1）",
            "COUNT": "个",
            "AREA_PIXEL": "像素²",
            "LENGTH_PIXEL": "像素",
        }.get(display_type, "")
    metric = {
        "name": _formal_text(name),
        "medical_name": _formal_text(name),
        "value": normalized,
        "unit": unit,
        "display_type": display_type,
        "source_path": f"检测模块[{module_id}].分区指标[{region_name}].{name}",
        "availability": "available" if normalized not in (None, "", "不可评估") else "unavailable",
    }
    metric["explanation"] = _metric_explanation(metric, module_id=module_id)
    return metric


def _formal_regions(regions: list[dict[str, Any]], *, module_id: str) -> list[dict[str, Any]]:
    output = []
    for region in regions or []:
        region_name = _formal_region_name(region.get("分区名称"))
        metrics = {}
        for key, value in (region.get("指标") or {}).items():
            clean_key = _formal_text(key)
            if clean_key and value not in (None, ""):
                metrics[clean_key] = _region_metric(str(key), value, module_id=module_id, region_name=region_name)
        if metrics:
            output.append({
                "分区名称": region_name,
                "状态": "有效" if str(region.get("状态", region.get("评估状态", ""))).upper() == "VALID" else _formal_text(region.get("状态", region.get("评估状态", "有效"))),
                "指标": metrics,
            })
    return output


def _formal_region_name(value: Any) -> str:
    """Normalize all report regions to the image-left/image-right convention."""
    return (
        _formal_text(value)
        .replace("受检者左", "画面右")
        .replace("受检者右", "画面左")
    )


def _formal_score_explanation(module: dict[str, Any]) -> dict[str, Any] | None:
    explanation = module.get("评分解释") or {}
    drivers = []
    for row in explanation.get("drivers", []) or []:
        identifier = str(row.get("id", ""))
        drivers.append({
            "name": _DRIVER_NAMES.get(str(module.get("模块编号")), {}).get(identifier, _formal_text(row.get("name"))),
            "weight": row.get("weight"), "score": row.get("score"),
            "weighted_contribution": row.get("weighted_contribution"),
        })
    if not drivers:
        return None
    score = float(module.get("综合得分"))
    dominant = max(drivers, key=lambda row: float(row.get("weighted_contribution") or 0))
    return {
        "summary_text": f"本项问题负担分为{score:.2f}分，评分综合考虑下列因素，其中{dominant['name']}的贡献最高。",
        "drivers": drivers,
    }


def _candidate_root_from_images(module: dict[str, Any]) -> Path | None:
    for value in module.get("结果图", []) or []:
        path = Path(value)
        if "七项检测" in path.parts:
            index = path.parts.index("七项检测")
            return Path(*path.parts[:index])
    return None


def _region_source_map(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _formal_text(row.get("分区名称")): row
        for row in module.get("分区指标", []) or []
        if _formal_text(row.get("分区名称"))
    }


def _region_is_valid(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    status = str(row.get("状态", row.get("评估状态", ""))).upper()
    return status in {"VALID", "有效", "可评估", "检测完成", "候选检测结果"}


def _region_raw_value(row: dict[str, Any] | None, key: str) -> float | None:
    if not _region_is_valid(row):
        return None
    return _ratio_value((row.get("指标") or {}).get(key))


def _direction_text(
    label: str,
    left: float | None,
    right: float | None,
    *,
    percentage: bool,
    unit: str = "",
) -> str:
    if left is None and right is None:
        return ""
    if left is None:
        return f"{label}仅画面右侧具备有效结果，暂不作左右高低比较"
    if right is None:
        return f"{label}仅画面左侧具备有效结果，暂不作左右高低比较"
    difference = left - right
    if percentage:
        amount = abs(difference) * 100.0
        if amount < 0.005:
            return f"{label}画面左右基本相近"
        side = "左侧" if difference > 0 else "右侧"
        return f"{label}{side}高{amount:.2f}个百分点"
    amount = abs(difference)
    if amount < 0.00005:
        return f"{label}画面左右基本相近"
    side = "左侧" if difference > 0 else "右侧"
    return f"{label}{side}高{amount:.2f}{unit}"


def _oil_distribution(module: dict[str, Any]) -> tuple[str, str, str, str, str]:
    rows = _region_source_map(module)
    leaf_names = [
        "额部", "鼻部", "画面左鼻旁", "画面右鼻旁", "画面左内侧面颊",
        "画面右内侧面颊", "画面左外侧面颊", "画面右外侧面颊", "下巴",
    ]
    candidates = [
        (name, _region_raw_value(rows.get(name), "油光面积占比")) for name in leaf_names
    ]
    candidates = [(name, value) for name, value in candidates if value is not None]
    dominant = max(candidates, key=lambda item: item[1]) if candidates else ("当前可见区域", 0.0)
    has_detectable_gloss = bool(candidates and dominant[1] > 0.0)
    t_zone = _region_raw_value(rows.get("T区"), "油光面积占比")
    cheeks = _region_raw_value(rows.get("面颊区"), "油光面积占比")
    zone_text = ""
    if t_zone is not None and cheeks is not None:
        zone_text = f"T区油光覆盖{t_zone * 100:.2f}%，面颊区{cheeks * 100:.2f}%"

    root = _candidate_root_from_images(module)
    porphyrin_text = ""
    porphyrin_region = ""
    if root:
        core_path = root / "九项核心量化指标.json"
        if core_path.is_file():
            document = json.loads(core_path.read_text(encoding="utf-8"))
            values = (((document.get("九项") or {}).get("porphyrin") or {}).get("核心总体指标") or {})
            regions = {
                "额头": values.get("porphyrin_forehead"),
                "画面左面颊": values.get("porphyrin_left_cheek"),
                "画面右面颊": values.get("porphyrin_right_cheek"),
                "鼻部": values.get("porphyrin_nose"),
                "下巴": values.get("porphyrin_chin"),
            }
            available = [(name, int(value)) for name, value in regions.items() if isinstance(value, (int, float))]
            if available:
                name, count = max(available, key=lambda item: item[1])
                porphyrin_region = name
                porphyrin_text = f"紫质目标以{name}数量最多（{count}个）"

    lr_parts = []
    pairs = (
        ("鼻旁油光覆盖", "画面左鼻旁", "画面右鼻旁"),
        ("内侧面颊油光覆盖", "画面左内侧面颊", "画面右内侧面颊"),
        ("外侧面颊油光覆盖", "画面左外侧面颊", "画面右外侧面颊"),
    )
    for label, left_name, right_name in pairs:
        text = _direction_text(
            label,
            _region_raw_value(rows.get(left_name), "油光面积占比"),
            _region_raw_value(rows.get(right_name), "油光面积占比"),
            percentage=True,
        )
        if text:
            lr_parts.append(text)
    relationship = ""
    if has_detectable_gloss and porphyrin_region:
        oil_region = dominant[0]
        same_region = (
            oil_region == porphyrin_region
            or (oil_region in {"鼻部", "画面左鼻旁", "画面右鼻旁"} and porphyrin_region == "鼻部")
        )
        if same_region:
            relationship = f"表面油光与紫质目标均在{porphyrin_region}相对集中"
        else:
            relationship = (
                f"表面油光以{oil_region}覆盖较高，紫质目标以{porphyrin_region}数量较多，"
                "两类信号呈不同的区域分布特点"
            )
    return (
        (
            f"{dominant[0]}（{dominant[1] * 100:.2f}%）"
            if has_detectable_gloss
            else "未检出明确油光区域"
        ),
        zone_text,
        porphyrin_text,
        "；".join(lr_parts),
        relationship,
    )


def _bilateral_gate_from_payload(payload: dict[str, Any]) -> tuple[bool | None, str]:
    """Use real valid-area evidence to decide whether bilateral numbers are fair."""
    module = next(
        (row for row in payload.get("检测模块", []) or [] if str(row.get("模块编号")) == "03"),
        None,
    )
    rows = ((module or {}).get("空间融合证据") or {}).get("region_distribution") or []
    by_id = {str(row.get("region_id")): row for row in rows if row.get("available")}
    left_ids = ("image_left_nasal", "image_left_zygomatic", "image_left_cheek", "image_left_jaw")
    right_ids = ("image_right_nasal", "image_right_zygomatic", "image_right_cheek", "image_right_jaw")

    def collect(ids: tuple[str, ...]) -> tuple[int, float]:
        selected = [
            by_id[key] for key in ids
            if key in by_id and float(by_id[key].get("valid_area_px") or 0) >= 2000
        ]
        return len(selected), sum(float(row.get("valid_area_px") or 0) for row in selected)

    left_count, left_area = collect(left_ids)
    right_count, right_area = collect(right_ids)
    if not rows:
        return None, ""
    enough = (
        left_count >= 3
        and right_count >= 3
        and min(left_area, right_area) >= 2000
        and min(left_area, right_area) / max(left_area, right_area) >= 0.5
    )
    if enough:
        return True, ""
    return False, "当前拍摄角度下双侧有效皮肤区域差异较大，本次不进行左右定量比较。"


def _aggregate_medical_side(rows: list[dict[str, Any]], prefix: str) -> tuple[float, float, float] | None:
    valid_area = feature_area = estimated_count = 0.0
    for row in rows:
        name = str(row.get("analysis_region", ""))
        if not name.startswith(prefix) or str(row.get("evaluation_status", "")) != "可评估":
            continue
        area = float(row.get("valid_skin_area_px") or 0)
        scope = ((row.get("core_metrics") or {}).get("scope_and_burden") or {})
        density = float(scope.get("feature_density_per_100k_skin_px") or 0)
        ratio = float(scope.get("feature_area_ratio") or 0)
        if area <= 0:
            continue
        valid_area += area
        estimated_count += density * area / 100000.0
        feature_area += ratio * area
    if valid_area <= 0:
        return None
    return estimated_count * 100000.0 / valid_area, feature_area / valid_area, valid_area


def _pore_distribution(module: dict[str, Any]) -> tuple[str | None, str]:
    root = _candidate_root_from_images(module)
    path = root / "七项检测" / "毛孔" / "毛孔量化指标.json" if root else None
    if not path or not path.is_file():
        return None, ""
    document = json.loads(path.read_text(encoding="utf-8"))
    medical = document.get("medical_metrics_v2") or {}
    auxiliary = ((medical.get("overall_metrics") or {}).get("auxiliary_metrics") or {})
    dominant = ((auxiliary.get("distribution") or {}).get("primary_concentration_region"))
    rows = medical.get("region_metrics") or []
    left = _aggregate_medical_side(rows, "画面左")
    right = _aggregate_medical_side(rows, "画面右")
    comparison = ""
    if left and right:
        density_text = _direction_text(
            "毛孔密度", left[0], right[0], percentage=False,
            unit="个/10万有效皮肤像素",
        )
        area_text = _direction_text("毛孔面积占比", left[1], right[1], percentage=True)
        comparison = f"{density_text}；{area_text}。"
    elif left or right:
        comparison = "仅单侧具备有效毛孔分区结果，暂不作左右高低比较。"
    return _reliable_region(dominant), comparison


def _pigmentation_distribution(module: dict[str, Any]) -> tuple[str | None, str, list[dict[str, Any]]]:
    fusion = module.get("空间融合证据") or {}
    priority = fusion.get("priority_pigment_regions") or []
    dominant = _reliable_region(priority[0].get("region_name")) if priority else None
    rows = [row for row in fusion.get("region_distribution", []) or [] if row.get("available")]
    formal_regions = []
    for row in rows:
        formal_regions.append({
            "分区名称": _formal_region_name(row.get("region_name")),
            "状态": "有效",
            "指标": {
                "综合色素联合区域数量": _region_metric(
                    "综合色素联合区域数量", row.get("component_count"), module_id="03", region_name=str(row.get("region_name"))
                ),
                "综合色素联合区域面积占比": _region_metric(
                    "综合色素联合区域面积占比", row.get("feature_area_ratio"), module_id="03", region_name=str(row.get("region_name"))
                ),
            },
        })
    by_id = {str(row.get("region_id")): row for row in rows}
    left_ids = ("image_left_nasal", "image_left_zygomatic", "image_left_cheek", "image_left_jaw")
    right_ids = ("image_right_nasal", "image_right_zygomatic", "image_right_cheek", "image_right_jaw")

    def aggregate(ids: tuple[str, ...]) -> tuple[float, int] | None:
        selected = [by_id[key] for key in ids if key in by_id]
        if not selected:
            return None
        valid = sum(float(row.get("valid_area_px") or 0) for row in selected)
        feature = sum(float(row.get("feature_area_px") or 0) for row in selected)
        count = sum(int(row.get("component_count") or 0) for row in selected)
        return (feature / valid, count) if valid > 0 else None

    left, right = aggregate(left_ids), aggregate(right_ids)
    if left and right:
        ratio_text = _direction_text("综合色素联合区域面积占比", left[0], right[0], percentage=True)
        count_text = _direction_text(
            "综合色素联合区域数量", float(left[1]), float(right[1]),
            percentage=False, unit="个",
        )
        comparison = f"{ratio_text}；{count_text}。"
    elif left or right:
        comparison = "仅单侧具备有效综合色素分区结果，暂不作左右高低比较。"
    else:
        comparison = ""
    return dominant, comparison, formal_regions


def _redness_distribution(module: dict[str, Any]) -> tuple[str | None, str]:
    rows = [row for row in module.get("分区指标", []) or [] if _region_is_valid(row)]
    burden_rows = []
    for row in rows:
        burden = _region_raw_value(row, "红度综合负担")
        if burden is not None:
            burden_rows.append((_formal_region_name(row.get("分区名称")), burden))
    dominant = _reliable_region(max(burden_rows, key=lambda item: item[1])[0]) if burden_rows else None

    def aggregate(prefix: str) -> float | None:
        valid_area = red_area = 0.0
        for row in rows:
            if not _formal_region_name(row.get("分区名称", "")).startswith(prefix):
                continue
            area = _region_raw_value(row, "有效皮肤面积（像素）")
            ratio = _region_raw_value(row, "弥漫红区面积占比")
            if area is None or ratio is None or area <= 0:
                continue
            valid_area += area
            red_area += area * ratio
        return red_area / valid_area if valid_area > 0 else None

    left, right = aggregate("画面左"), aggregate("画面右")
    comparison = _direction_text("弥漫红区面积占比", left, right, percentage=True)
    if comparison:
        comparison += "。"
    return dominant, comparison


_REDNESS_METRIC_META = {
    "diffuse_red_area_ratio": ("弥漫红区面积占比", "PERCENTAGE", ""),
    "high_red_area_ratio": ("高红度响应区域面积占比", "PERCENTAGE", ""),
    "mean_redness": ("平均红度", "INTENSITY_ENGINEERING", "0～1相对强度"),
    "p90_redness": ("P90红度", "INTENSITY_ENGINEERING", "0～1相对强度"),
    "max_continuous_region_ratio": ("最大连续红区面积占比", "PERCENTAGE", ""),
    "redness_continuity": ("红区连续性", "ENGINEERING_0_1", "0～1"),
    "boundary_gradient": ("边界渐变", "ENGINEERING_0_1", "0～1"),
    "uniformity": ("弥漫红区均匀度", "ENGINEERING_0_1", "0～1"),
}


def _official_redness_groups(module: dict[str, Any]) -> list[dict[str, Any]]:
    titles = {
        "coverage": "泛红覆盖", "intensity": "泛红强度",
        "continuity": "泛红连续性", "boundary_uniformity": "边界与均匀性",
    }
    groups = []
    for group in module.get("评分追溯", []) or []:
        group_id = str(group.get("group_id", ""))
        if group_id not in titles:
            continue
        metrics = []
        for source in group.get("metrics", []) or []:
            metric_id = str(source.get("metric_id", ""))
            if metric_id not in _REDNESS_METRIC_META:
                continue
            name, display_type, unit = _REDNESS_METRIC_META[metric_id]
            metrics.append(_formal_metric({
                "name": name, "value": source.get("raw_value"), "unit": unit,
                "display_type": display_type,
            }, module_id="04", source_path=f"评分追溯.{group_id}.{metric_id}.raw_value"))
        groups.append({"title": titles[group_id], "summary": "", "metrics": metrics})
    return groups


def _doctor_groups(module: dict[str, Any]) -> list[dict[str, Any]]:
    module_id = str(module.get("模块编号"))
    if module_id == "04":
        return _official_redness_groups(module)
    groups = []
    for group in module.get("医生结果分组", []) or []:
        metrics = []
        for metric in group.get("metrics", []) or []:
            name = str(metric.get("name", ""))
            if module_id == "02" and name in {"紫质高强度目标比例", "紫质高强度区域面积占比"}:
                continue
            if module_id == "05" and name in {
                "P50候选宽度", "P90候选宽度",
                "P50局部红度工程响应", "P90局部红度工程响应",
            }:
                continue
            formal_metric = _formal_metric(metric, module_id=module_id)
            if module_id == "05":
                replacements = {
                    "血管样结构数量": "血管样线状结构数量",
                    "线长度密度": "线状结构长度密度",
                    "结构面积占比": "线状结构面积占比",
                    "分支点数量": "线状结构分支点数量",
                    "分支点密度": "线状结构分支点密度",
                    "网络比例": "线状结构网络比例",
                }
                formal_metric["name"] = replacements.get(formal_metric["name"], formal_metric["name"])
                _refresh_metric_metadata(formal_metric, module_id=module_id)
            metrics.append(formal_metric)
        title = _GROUP_TITLES.get(str(group.get("title")), _formal_text(group.get("title")))
        summary = _formal_text(group.get("summary"))
        if module_id == "03" and title == "UV色斑与UV下更明显区域":
            summary = "UV下更明显区域表示相较普通白光图像，在UV图像中表现更突出的色素相关区域。"
        if metrics:
            groups.append({"title": title, "summary": summary, "metrics": metrics})

    images = [Path(value) for value in module.get("结果图", []) or [] if Path(value).is_file()]
    technical = [Path(value) for value in module.get("技术附录结果图", []) or [] if Path(value).is_file()]
    if module_id == "02" and len(groups) >= 2:
        if images:
            groups[0]["images"] = [{"path": str(images[0]), "caption": "表面油光分布"}]
        if len(images) > 1:
            groups[1]["images"] = [{"path": str(images[1]), "caption": "紫质分布"}]
    elif module_id == "03" and groups:
        root = _candidate_root_from_images(module)
        uv_images = []
        if root:
            uv_images = sorted((root / "七项检测" / "紫区").glob("*UV*检测结果*.jpg"))
            if not uv_images:
                uv_images = sorted((root / "七项检测" / "紫区").glob("*紫外线色斑*.jpg"))
        if len(images) > 1:
            groups[0]["images"] = [{"path": str(images[1]), "caption": "可见色斑结果"}]
        if uv_images:
            groups[1]["images"] = [{"path": str(uv_images[0]), "caption": "UV色斑结果"}]
        if len(images) > 2:
            groups[2]["images"] = [{"path": str(images[2]), "caption": "Brown综合色素结果"}]
        red_brown = next((path for path in technical if "红褐混合" in path.name), None)
        priority = next((path for path in technical if "重点色斑" in path.name), None)
        if red_brown:
            groups[3]["images"] = [{"path": str(red_brown), "caption": "红褐混合印记"}]
        if priority:
            groups[4]["images"] = [{"path": str(priority), "caption": "重点色斑区域"}]
    return groups


def _formal_summary(module: dict[str, Any]) -> str:
    module_id = str(module.get("模块编号"))
    score = module.get("综合得分")
    grade = module.get("程度等级")
    if module_id == "01":
        count = int(_metric_value(module, "可见毛孔数量", 0) or 0)
        area = _percent(_metric_value(module, "毛孔面积占比", 0))
        region, _ = _pore_distribution(module)
        where = f"，主要分布于{region}" if region else "，整体呈散在分布"
        if not isinstance(score, (int, float)):
            return f"检测到{count}个可见毛孔，面积占比{area}{where}。"
        drivers = (_formal_score_explanation(module) or {}).get("drivers", [])
        driver = max(drivers, key=lambda row: float(row.get("weighted_contribution") or 0)).get("name") if drivers else "毛孔密度"
        return f"检测到{count}个可见毛孔，面积占比{area}{where}。问题负担分{float(score):.2f}分，程度为{grade}，主要受{driver}影响。"
    if module_id == "02":
        coverage = _percent(_metric_value(module, "表面油光面积占比", 0))
        continuity = _percent(_metric_value(module, "最大连续油光区域面积占比", 0))
        count = int(_metric_value(module, "紫质目标数量", 0) or 0)
        dominant, zone_text, porphyrin_text, _, relationship = _oil_distribution(module)
        if dominant == "未检出明确油光区域":
            parts = ["本次未检出明确表面油光区域"]
        else:
            parts = [f"表面油光覆盖{coverage}，最大连续油光区域占比{continuity}，主要分布于{dominant}"]
        if zone_text:
            parts.append(zone_text)
        parts.append(f"检测到紫质目标{count}个")
        if porphyrin_text:
            parts.append(porphyrin_text)
        if relationship:
            parts.append(relationship)
        return "；".join(parts) + "。"
    if module_id == "03":
        if not isinstance(score, (int, float)):
            region = _reliable_region((module.get("medical_summary") or {}).get("main_regions"))
            region_text = f"，重点区域为{region}" if region else ""
            return f"已完成Visible、UV与Brown综合色素检测{region_text}。"
        explanation = _formal_score_explanation(module) or {}
        drivers = explanation.get("drivers") or []
        driver = max(drivers, key=lambda row: float(row.get("weighted_contribution") or 0)).get("name") if drivers else "可见色斑负担"
        region = _reliable_region((module.get("medical_summary") or {}).get("main_regions"))
        region_text = f"，综合色素重点区域为{region}" if region else ""
        return f"综合色素问题负担分{float(score):.2f}分，程度为{grade}，主要受{driver}影响{region_text}。"
    if module_id == "04":
        official_groups = _official_redness_groups(module)
        official_module = {"医生结果分组": official_groups}
        coverage = _percent(_metric_value(official_module, "弥漫红区面积占比", 0))
        region, _ = _redness_distribution(module)
        region_text = f"，综合泛红表现以{region}最明显" if region else ""
        if not isinstance(score, (int, float)):
            return f"弥漫红区覆盖{coverage}{region_text}。"
        explanation = _formal_score_explanation(module) or {}
        drivers = explanation.get("drivers") or []
        driver = max(drivers, key=lambda row: float(row.get("weighted_contribution") or 0)).get("name") if drivers else "泛红覆盖"
        scores = {str(row.get("name")): float(row.get("score") or 0) for row in drivers}
        return (
            f"弥漫红区覆盖{coverage}{region_text}。"
            f"覆盖、强度、连续性、边界与均匀性分项得分分别为"
            f"{scores.get('泛红覆盖', 0):.2f}、{scores.get('泛红强度', 0):.2f}、"
            f"{scores.get('泛红连续性', 0):.2f}、{scores.get('边界与均匀性', 0):.2f}；"
            f"综合问题负担分{float(score):.2f}分，程度为{grade}，其中{driver}贡献最高。"
        )
    if module_id == "05":
        count = int(_metric_value(module, "血管样候选结构数量", 0) or 0)
        length = float(_metric_value(module, "候选线状结构总长度", 0) or 0)
        regions = module.get("分区指标", []) or []
        dominant = None
        if regions:
            dominant_row = max(regions, key=lambda row: float((row.get("指标") or {}).get("候选总长度（像素）", 0) or 0))
            dominant = _reliable_region(dominant_row.get("分区名称"))
        where = f"，主要分布于{dominant}" if dominant else "，整体呈散在分布"
        return f"本次观察到{count}个血管样线状结构，总长度约{length:.0f}像素{where}。"
    if module_id == "06":
        count = int(_metric_value(module, "痤疮样特征数量", 0) or 0)
        if count == 0:
            return "本次未检出符合当前检测标准的痤疮样特征。"
        region = _reliable_region(_metric_value(module, "主要集中区域"))
        if region and region != "局部可见区域":
            return f"检测到{count}个痤疮样特征，主要分布于{region}。"
        return f"在当前可见区域检出{count}个痤疮样特征。"
    return _formal_text(module.get("结果摘要"))


def _report_date(value: date | str | None, *, field: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field}必须为YYYY-MM-DD日期: {value}") from exc


def build_formal_report_view(
    payload: dict[str, Any],
    *,
    generated_on: date | str | None = None,
    collected_on: date | str | None = None,
) -> dict[str, Any]:
    """Build a report-only whitelist projection; internal trace remains in ``payload``."""
    generated = _report_date(generated_on, field="报告日期") or date.today()
    subject = payload.get("受检者信息") or {}
    collected = (
        _report_date(collected_on, field="采集日期")
        or _report_date(subject.get("采集日期"), field="采集日期")
        or generated
    )
    report_id = str((payload.get("报告信息") or {}).get("报告编号", ""))
    formal: dict[str, Any] = {
        "报告信息": {"报告编号": report_id, "报告日期": generated.isoformat()},
        "受检者信息": {
            "姓名或编号": subject.get("姓名或编号", "匿名受检者"),
            "采集日期": collected.isoformat(),
        },
        "检测模块": [],
    }
    bilateral_allowed, bilateral_gate_text = _bilateral_gate_from_payload(payload)
    for source in payload.get("检测模块", []) or []:
        module_id = str(source.get("模块编号", ""))
        score = source.get("综合得分")
        advice = {
            "01": "建议温和清洁、适度保湿并做好日常防晒。",
            "02": "建议注意T区与面颊的清洁和清爽护理，避免过度去油。",
            "03": "建议做好日常防晒，减少长时间强光暴露。",
            "04": "建议使用温和护肤方式，减少骤冷骤热和过度摩擦。",
            "05": "建议避免骤冷骤热和反复摩擦，注意观察局部红色线状结构的变化。",
            "06": "建议保持温和清洁并避免挤压。",
            "07": "建议加强保湿与屏障护理，减少过度清洁。",
            "08": "建议注意保湿和防晒，减少反复牵拉局部皮肤。",
            "09": "建议保持稳定的保湿与防晒护理。",
            "10": "建议温和清洁、适度保湿，减少对皮肤表面的反复刺激。",
            "11": "建议保持规律作息、防晒和日常保湿护理。",
        }.get(module_id, "建议保持温和、稳定的日常护理。")
        module: dict[str, Any] = {
            "模块编号": module_id,
            "模块名称": "血管样结构观察" if module_id == "05" else _formal_text(source.get("模块名称")),
            "数据来源": _SOURCE_NAMES.get(module_id, ["标准白光图像"]),
            "综合得分": score if isinstance(score, (int, float)) else None,
            "程度等级": source.get("程度等级") if isinstance(score, (int, float)) else None,
            "结果摘要": _formal_summary(source),
            "核心指标": [_formal_metric(row, module_id=module_id) for row in source.get("核心指标", []) or [] if row.get("value") not in (None, "", "不可评估")],
            "分区指标": _formal_regions(source.get("分区指标", []), module_id=module_id),
            "结果图": [str(path) for path in [Path(value) for value in source.get("结果图", []) or []] if path.is_file()],
            "左右比较摘要": _formal_text(source.get("左右比较摘要", "")),
            "日常管理提示": advice,
            "用户评估状态": "血管样结构观察结果" if module_id == "05" else ("检测结果" if score is None else "检测完成"),
            "医生评估状态": "血管样结构观察结果" if module_id == "05" else ("检测结果" if score is None else "检测完成"),
            "医生结果分组": _doctor_groups(source),
            "正式评分说明": _formal_score_explanation(source),
        }
        if module_id == "02":
            porphyrin = next((group for group in module["医生结果分组"] if group.get("title") == "紫质：现有量化结果"), None)
            if porphyrin:
                selected = {"紫质目标数量", "紫质单位面积密度", "紫质面积占比", "紫质P90强度"}
                module["核心指标"] = module["核心指标"][:3] + [row for row in porphyrin["metrics"] if row["name"] in selected][:1]
            _, _, _, lr_text, _ = _oil_distribution(source)
            module["左右比较摘要"] = lr_text
        elif module_id == "03":
            for row in module["核心指标"]:
                if row["name"] == "主要集中区域":
                    row["name"] = "可见色斑主要集中区域"
                    row["medical_name"] = row["name"]
                    row["display_type"], row["unit"] = _FULL_METRIC_OVERRIDES[row["name"]]
                    row["explanation"] = _metric_explanation(row)
            selected_names = {"可见斑点数量", "UV色斑数量", "棕色实例数量", "可见色斑主要集中区域"}
            selected = [row for row in module["核心指标"] if row["name"] in selected_names]
            module["核心指标"] = selected or module["核心指标"][:4]
            dominant, comparison, regions = _pigmentation_distribution(source)
            module["分区指标"] = regions
            module["左右比较摘要"] = comparison
        elif module_id == "04":
            official_groups = module["医生结果分组"]
            representative_names = {"弥漫红区面积占比", "P90红度", "红区连续性", "弥漫红区均匀度"}
            module["核心指标"] = [
                metric for group in official_groups for metric in group.get("metrics", [])
                if metric.get("name") in representative_names
            ]
            module["医生版隐藏全面部核心指标"] = True
            _, comparison = _redness_distribution(source)
            module["左右比较摘要"] = comparison
        elif module_id == "05":
            module["结果图"] = []
            replacements = {
                "血管样结构数量": "血管样线状结构数量", "线状结构总长度": "线状结构总长度",
                "结构面积占比": "线状结构面积占比", "分支点数量": "线状结构分支点数量",
            }
            module["核心指标"] = [
                row for row in module["核心指标"]
                if row["name"] not in {"P90宽度", "P90局部红度相对响应"}
            ]
            for row in module["核心指标"]:
                row["name"] = replacements.get(row["name"], _formal_text(row["name"]))
                _refresh_metric_metadata(row, module_id=module_id)
            module["分区指标"] = [
                {
                    **region,
                    "指标": {
                        {"数量": "线状结构数量", "总长度（像素）": "线状结构总长度（像素）", "面积（像素）": "线状结构面积（像素）"}.get(key, key): value
                        for key, value in region.get("指标", {}).items()
                    },
                }
                for region in module["分区指标"]
            ]
        elif module_id == "06":
            count = int(_metric_value(source, "痤疮样特征数量", 0) or 0)
            allowed = {"有效皮肤面积", "痤疮样特征数量", "单位面积密度", "特征框总面积", "特征框面积占比", "P50特征框面积", "P90特征框面积", "主要集中区域"}
            module["核心指标"] = [row for row in module["核心指标"] if row["name"] in allowed]
            if count == 0:
                module["核心指标"] = [row for row in module["核心指标"] if row["name"] == "痤疮样特征数量"]
                module["分区指标"] = []
        if module_id == "01":
            _, comparison = _pore_distribution(source)
            module["左右比较摘要"] = comparison
        if module_id in {"01", "02", "03", "04"} and bilateral_allowed is False:
            module["左右比较摘要"] = bilateral_gate_text
        formal["检测模块"].append(module)
    return copy.deepcopy(formal)


__all__ = ["build_formal_report_view"]
