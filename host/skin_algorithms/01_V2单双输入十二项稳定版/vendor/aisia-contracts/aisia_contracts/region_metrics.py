"""公开分区量化模型:新增三项(surface_gloss/vascular)共用的六分区结构。

镜像 dermavision cloud_contracts/models.py(HEAD 0f6bba0)的
PublicRegionRatios/PublicRegionCounts/PublicRegionLengths。
字段名、中文 alias、类型与 dermavision 定义完全一致;
title/description/unit(json_schema_extra)取源模型 _metric_field 语义,
契约仓不强制 examples。

发布即冻结铁律适用:已发布的 v{N}.py 不可修改,字段变动必须新建版本。
"""
from pydantic import ConfigDict, Field

from aisia_contracts.base import ContractModel

_REGION_CONFIG = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


def _region_field(*, alias: str, title: str, description: str, unit: str):
    """登记中文名、口径和单位(镜像 dermavision _metric_field,省略 examples)。"""

    return Field(
        alias=alias,
        title=title,
        description=description,
        json_schema_extra={"unit": unit},
    )


class PublicRegionRatios(ContractModel):
    """六个面部分区的面积比例(0～1)。"""

    model_config = _REGION_CONFIG

    total: float = _region_field(
        alias="总计",
        title="全面部比例",
        description="全面部有效分析区域内的面积比例。",
        unit="比例（0～1）",
    )
    forehead: float = _region_field(
        alias="额头",
        title="额头比例",
        description="额头区域内的面积比例。",
        unit="比例（0～1）",
    )
    left_cheek: float = _region_field(
        alias="左脸颊",
        title="画面左脸颊比例",
        description="画面左侧脸颊区域内的面积比例。",
        unit="比例（0～1）",
    )
    right_cheek: float = _region_field(
        alias="右脸颊",
        title="画面右脸颊比例",
        description="画面右侧脸颊区域内的面积比例。",
        unit="比例（0～1）",
    )
    nose: float = _region_field(
        alias="鼻部",
        title="鼻部比例",
        description="鼻部区域内的面积比例。",
        unit="比例（0～1）",
    )
    chin: float = _region_field(
        alias="下巴",
        title="下巴比例",
        description="下巴区域内的面积比例。",
        unit="比例（0～1）",
    )


class PublicRegionCounts(ContractModel):
    """六个面部分区的目标数量。"""

    model_config = _REGION_CONFIG

    total: int = _region_field(
        alias="总计",
        title="全面部数量",
        description="全面部有效分析区域内的目标数量。",
        unit="个",
    )
    forehead: int = _region_field(
        alias="额头",
        title="额头数量",
        description="额头区域内的目标数量。",
        unit="个",
    )
    left_cheek: int = _region_field(
        alias="左脸颊",
        title="画面左脸颊数量",
        description="画面左侧脸颊区域内的目标数量。",
        unit="个",
    )
    right_cheek: int = _region_field(
        alias="右脸颊",
        title="画面右脸颊数量",
        description="画面右侧脸颊区域内的目标数量。",
        unit="个",
    )
    nose: int = _region_field(
        alias="鼻部",
        title="鼻部数量",
        description="鼻部区域内的目标数量。",
        unit="个",
    )
    chin: int = _region_field(
        alias="下巴",
        title="下巴数量",
        description="下巴区域内的目标数量。",
        unit="个",
    )


class PublicRegionLengths(ContractModel):
    """六个面部分区检出结构在1024标准化图像中的总长度。"""

    model_config = _REGION_CONFIG

    total: float = _region_field(
        alias="总计",
        title="全面部总长度",
        description="全面部检出结构在1024标准化图像中的总长度。",
        unit="标准化图像像素",
    )
    forehead: float = _region_field(
        alias="额头",
        title="额头总长度",
        description="额头区域内检出结构的总长度。",
        unit="标准化图像像素",
    )
    left_cheek: float = _region_field(
        alias="左脸颊",
        title="画面左脸颊总长度",
        description="画面左侧脸颊区域内检出结构的总长度。",
        unit="标准化图像像素",
    )
    right_cheek: float = _region_field(
        alias="右脸颊",
        title="画面右脸颊总长度",
        description="画面右侧脸颊区域内检出结构的总长度。",
        unit="标准化图像像素",
    )
    nose: float = _region_field(
        alias="鼻部",
        title="鼻部总长度",
        description="鼻部及鼻旁区域内检出结构的总长度。",
        unit="标准化图像像素",
    )
    chin: float = _region_field(
        alias="下巴",
        title="下巴总长度",
        description="下巴及下颌区域内检出结构的总长度。",
        unit="标准化图像像素",
    )
