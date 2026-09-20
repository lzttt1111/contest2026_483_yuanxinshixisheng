"""acne v2 契约。

镜像 dermavision cloud_contracts/models.py(HEAD 0f6bba0)的
AcneV2Counts/AcneV2Metrics/AcneV2RawResult(SCHEMA_VERSIONS["acne_v2"]="2")。
v1 已发布且不得修改:本文件只新增 v2 模型并注册 ("acne", "2")。

严重程度子模型直接复用 acne v1 的 AcneSeverityLevel
(等级:int|None + 注释:str),其字段形状与 dermavision AcneSeverityResult
完全一致;v2 不重复定义,避免两处结构漂移。
发布即冻结铁律适用:已发布的 v2 不可修改,字段变动必须新建 v3。
"""
from pydantic import ConfigDict, Field

from aisia_contracts.algorithms.acne.v1 import AcneSeverityLevel
from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class AcneV2Counts(ContractModel):
    """疑似痤疮数量:总计 + 五个固定面部分区,均非负。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    total: int = Field(alias="总计", ge=0, description="正式痤疮圈选结果中的疑似痤疮总数")
    forehead: int = Field(alias="额头", ge=0, description="额头区域疑似痤疮数量")
    left_cheek: int = Field(alias="左脸颊", ge=0, description="受检者左脸颊疑似痤疮数量")
    right_cheek: int = Field(alias="右脸颊", ge=0, description="受检者右脸颊疑似痤疮数量")
    nose: int = Field(alias="鼻部", ge=0, description="鼻部区域疑似痤疮数量")
    chin: int = Field(alias="下巴", ge=0, description="下巴区域疑似痤疮数量")


class AcneV2Metrics(ContractModel):
    """痤疮 v2 中文精简量化指标:疑似痤疮数量 + 严重程度等级。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    疑似痤疮数量: AcneV2Counts = Field(
        alias="疑似痤疮数量",
        description="正式痤疮圈选结果的总计和五个固定面部分区数量",
    )
    痤疮严重程度等级: AcneSeverityLevel = Field(
        alias="痤疮严重程度等级",
        description="Acne-LDS四级严重程度结果及中文说明",
    )


class AcneV2RawResult(ContractModel):
    """痤疮 v2 的 raw_result 结构:overlay 必填,质量信息可空。"""

    overlay: str
    metrics: AcneV2Metrics
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []
    # 未发布版本:第三段直接纳入版本化评分输入(inline),见证据矩阵 §6。
    scoring_input: ScoringInputV1 | None = None


class _AcneV2Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 acne v2 的返回结构。"""

    raw_result: AcneV2RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "2"
register("acne", SCHEMA_VERSION, _AcneV2Envelope)
