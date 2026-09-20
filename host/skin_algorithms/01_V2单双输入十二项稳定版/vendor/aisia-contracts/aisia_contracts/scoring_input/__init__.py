"""版本化评分输入(scoring_input)契约包。

依据《十二项总体评分聚合接入方案_20260911.md》r3 §5.2 与
《十二项评分输入字段证据矩阵_20260912.md》§4/§5(存储=inline、evidence 镜像
detector_results 子集、契约只定义容器)。
发布即冻结铁律适用:已发布的 schema_version 不可修改,字段变动必须新建 v2。
"""
from aisia_contracts.scoring_input.v1 import (
    SCORING_INPUT_SCHEMA_VERSION,
    InputQualityGate,
    ScoringInputQuality,
    ScoringInputV1,
)

__all__ = [
    "SCORING_INPUT_SCHEMA_VERSION",
    "InputQualityGate",
    "ScoringInputQuality",
    "ScoringInputV1",
]
