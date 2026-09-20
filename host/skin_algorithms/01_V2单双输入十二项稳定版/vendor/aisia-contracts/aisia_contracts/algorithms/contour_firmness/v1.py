"""contour_firmness v1 契约。

镜像 dermavision cloud_contracts/models.py(HEAD 0f6bba0)的
ContourFirmnessPublicMetrics/ContourFirmnessRawResult:
公开 metrics 为三个 2D/2.5D 轮廓紧致度代理指标(中面部曲面连续性/
下颌缘连续性/左右轮廓差异),
raw_result 为新增三项加法合同(overlay/medical_report_csv_v2/quality_*)。
发布即冻结铁律适用:已发布的 v1 不可修改,字段变动必须新建 v2。
"""
from pydantic import ConfigDict, Field

from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class ContourFirmnessPublicMetrics(ContractModel):
    """contour_firmness 用户简表:三个轮廓紧致度相对代理值(0～1)。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    中面部曲面连续性: float = Field(
        alias="中面部曲面连续性",
        title="中面部曲面连续性",
        description="从颗部至下颊的2.5D相对曲面连续性代理;越高表示相对连续性越好。",
        json_schema_extra={"unit": "相对代理值（0～1）"},
    )
    下颌缘连续性: float = Field(
        alias="下颌缘连续性",
        title="下颌缘连续性",
        description="下颌缘曲线平滑和连续程度的2D几何代理;越高表示轮廓越连续。",
        json_schema_extra={"unit": "相对代理值（0～1）"},
    )
    左右轮廓差异: float = Field(
        alias="左右轮廓差异",
        title="左右轮廓差异",
        description="左右下颌弧长和轮廓几何差异的相对代理;越高表示左右差异越大。",
        json_schema_extra={"unit": "相对代理值（0～1）"},
    )


class ContourFirmnessV1RawResult(ContractModel):
    """contour_firmness v1 的 raw_result 结构。"""

    overlay: str | None = None
    medical_report_csv_v2: str | None = None
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []
    metrics: ContourFirmnessPublicMetrics
    # 未发布版本:第三段直接纳入版本化评分输入(inline),见证据矩阵 §6。
    scoring_input: ScoringInputV1 | None = None


class _ContourFirmnessV1Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 contour_firmness v1 的返回结构。"""

    raw_result: ContourFirmnessV1RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("contour_firmness", SCHEMA_VERSION, _ContourFirmnessV1Envelope)
