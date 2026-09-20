"""medical_metrics_v1 量化指标的共享 Pydantic 模型工厂。

dermavision 所有算法自 commit 7ade530 起统一返回 medical_metrics_v1 医学宽表:
    {"检测项目": str, "指标版本": str, "总体指标": Row, "分区指标": list[Row]}

每行字段 = 19 个公共字段 + 各算法专属字段。行字段名含全角括号（）、～ 等
非 Python 标识符字符,故用 Field(alias=...) 映射到合法 Python 名。
extra="forbid" 确保 worker 增减字段时 drift 检测能立即发现。

注:medical_metrics_to_list(转扁平 list)是 backend to_response 侧逻辑,不属于契约,留 backend。
"""
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

# medical_metrics_v1 行值类型:
# - 可用区域为 int(计数/面积)或 float(占比/强度/分位数)
# - 不可用区域为空字符串 "" 或 "不可评估"
MetricValue = int | float | str

# 19 个公共行字段(所有算法共有,来自 _csv_columns common)
_COMMON_ROW_KEYS: list[str] = [
    "检测范围", "评估状态", "有效皮肤面积（像素）", "特征数量（个）",
    "单位面积密度（个/10万有效皮肤像素）", "特征总面积（像素）",
    "特征面积占比", "P50单体面积（像素）", "P90单体面积（像素）",
    "最大单体面积（像素）", "平均强度（0～1）", "P50强度（0～1）",
    "P90强度（0～1）", "P95强度（0～1）",
    "实例P50强度（0～1）", "实例P90强度（0～1）", "主要集中区域",
    "画面左右面颊数量密度差异", "画面左右面颊面积占比差异",
]

# 始终为字符串的字段(区域名 / 评估状态 / 主要集中区域 / 主要问题类型)
_STRING_ROW_KEYS: set[str] = {
    "检测范围", "评估状态", "主要集中区域", "主要问题类型",
}


def _py_ident(key: str) -> str:
    r"""把含全角括号/斜杠/波浪号的中文 key 转成合法 Python 标识符。

    re \w 匹配 Unicode 字母(含汉字) + ASCII 字母数字下划线,
    只把 （）～/ 等非 \w 字符替换为 _。
    """
    return re.sub(r"[^\w]", "_", key, flags=re.UNICODE)


def make_metrics_model(class_name: str, extra_row_keys: list[str]) -> type[BaseModel]:
    """创建 medical_metrics_v1 顶层 payload 模型。

    Args:
        class_name: 模型类名(如 "RednessV2Metrics")
        extra_row_keys: 算法专属行字段名(来自 _csv_columns extras)

    Returns:
        Pydantic 模型类,含 4 个顶层字段(检测项目/指标版本/总体指标/分区指标),
        总体指标和分区指标用 strict row 模型(extra="forbid" + 全字段 alias)校验。
    """
    all_row_keys = _COMMON_ROW_KEYS + extra_row_keys
    row_fields: dict[str, tuple] = {}
    for key in all_row_keys:
        py_name = _py_ident(key)
        ftype = str if key in _STRING_ROW_KEYS else MetricValue
        row_fields[py_name] = (ftype, Field(alias=key))

    row_model = create_model(
        f"_{class_name}Row",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **row_fields,
    )

    return create_model(
        class_name,
        __config__=ConfigDict(extra="forbid"),
        检测项目=(str, ...),
        指标版本=(str, ...),
        总体指标=(row_model, ...),
        分区指标=(list[row_model], ...),
    )


# ── Purple 专属 ──────────────────────────────────────────────
# Purple 行字段 = 3 共享 + 32 前缀(紫外线色斑{metric} + 紫质{metric}),
# 与其他 5 项算法的 19 公共字段不同,故单独建模型。
_PURPLE_SHARED_KEYS: list[str] = ["检测范围", "评估状态", "有效皮肤面积（像素）"]
_PURPLE_METRIC_COLUMNS: list[str] = [
    "特征数量（个）", "单位面积密度（个/10万有效皮肤像素）",
    "特征总面积（像素）", "特征面积占比",
    "P50单体面积（像素）", "P90单体面积（像素）", "最大单体面积（像素）",
    "平均强度（0～1）", "P50强度（0～1）", "P90强度（0～1）", "P95强度（0～1）",
    "实例P50强度（0～1）", "实例P90强度（0～1）",
    "主要集中区域", "画面左右面颊数量密度差异", "画面左右面颊面积占比差异",
]
_PURPLE_STRING_KEYS: set[str] = {
    "检测范围", "评估状态",
    "紫外线色斑主要集中区域", "紫质主要集中区域",
}


def make_purple_metrics_model(class_name: str) -> type[BaseModel]:
    """创建 purple 专属 medical_metrics_v1 模型。

    行字段 = 3 共享 + 32 前缀(紫外线色斑{col} + 紫质{col}),extra="forbid"。
    另含可选 medical_metrics_v2 嵌套实验文档(dermavision embed_medical_metrics_v2)。
    """
    row_keys = _PURPLE_SHARED_KEYS + [
        f"{prefix}{col}"
        for prefix in ("紫外线色斑", "紫质")
        for col in _PURPLE_METRIC_COLUMNS
    ]
    row_fields: dict[str, tuple] = {}
    for key in row_keys:
        py_name = _py_ident(key)
        ftype = str if key in _PURPLE_STRING_KEYS else MetricValue
        row_fields[py_name] = (ftype, Field(alias=key))

    row_model = create_model(
        f"_{class_name}Row",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **row_fields,
    )

    return create_model(
        class_name,
        __config__=ConfigDict(extra="forbid"),
        检测项目=(str, ...),
        指标版本=(str, ...),
        总体指标=(row_model, ...),
        分区指标=(list[row_model], ...),
        medical_metrics_v2=(dict[str, Any] | None, None),
    )
