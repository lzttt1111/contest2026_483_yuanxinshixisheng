"""pores v1 契约,源自 ai-skin-backend service/algorithm/implements/pores/v1.py。
metrics 改用 dermavision PoresPublicMetrics/PoresPartialMetrics 强结构(替代 dict[str,int])。"""
from pydantic import ConfigDict, Field

from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class PoresPublicMetrics(ContractModel):
    """pores 用户简表(完整脸):6 分区计数。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    total: int = Field(alias="总计")
    forehead: int = Field(alias="额头")
    left_cheek: int = Field(alias="左脸颊")
    right_cheek: int = Field(alias="右脸颊")
    nose: int = Field(alias="鼻部")
    chin: int = Field(alias="下巴")


class PoresPartialMetrics(ContractModel):
    """pores 用户简表(局部脸):仅总计。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    total: int = Field(alias="总计")


class PoresV1RawResult(ContractModel):
    """pores v1 的 raw_result 结构。"""

    overlay: str | None = None
    metrics: PoresPublicMetrics | PoresPartialMetrics
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []


class _PoresV1Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 pores v1 的返回结构。"""

    raw_result: PoresV1RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("pores", SCHEMA_VERSION, _PoresV1Envelope)
