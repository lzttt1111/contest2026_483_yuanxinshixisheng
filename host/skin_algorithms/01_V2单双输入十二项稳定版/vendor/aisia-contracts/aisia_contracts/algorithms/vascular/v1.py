"""vascular v1 契约。

镜像 dermavision cloud_contracts/models.py(HEAD 0f6bba0)的
VascularPublicMetrics/VascularRawResult:
公开 metrics 为中文精简分区指标(血管样结构数量 + 血管样结构总长度),
raw_result 为新增三项加法合同(overlay/medical_report_csv_v2/quality_*)。
发布即冻结铁律适用:已发布的 v1 不可修改,字段变动必须新建 v2。
"""
from pydantic import ConfigDict, Field

from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.region_metrics import PublicRegionCounts, PublicRegionLengths
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class VascularPublicMetrics(ContractModel):
    """vascular 用户简表:血管样结构数量 + 总长度(各六分区)。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    血管样结构数量: PublicRegionCounts = Field(
        alias="血管样结构数量",
        title="分区血管样结构数量",
        description="各分区内符合当前检测标准的血管样线状结构数量。",
    )
    血管样结构总长度: PublicRegionLengths = Field(
        alias="血管样结构总长度",
        title="分区血管样结构总长度",
        description="各分区内血管样结构中心线在1024标准化图像中的总长度。",
    )


class VascularV1RawResult(ContractModel):
    """vascular v1 的 raw_result 结构。"""

    overlay: str | None = None
    medical_report_csv_v2: str | None = None
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []
    metrics: VascularPublicMetrics
    # 未发布版本:第三段直接纳入版本化评分输入(inline),见证据矩阵 §6。
    scoring_input: ScoringInputV1 | None = None


class _VascularV1Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 vascular v1 的返回结构。"""

    raw_result: VascularV1RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("vascular", SCHEMA_VERSION, _VascularV1Envelope)
