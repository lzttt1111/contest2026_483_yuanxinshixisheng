"""surface_gloss v1 契约。

镜像 dermavision cloud_contracts/models.py(HEAD 0f6bba0)的
SurfaceGlossPublicMetrics/SurfaceGlossRawResult:
公开 metrics 为中文精简分区指标(油光面积占比 + 油光区域数量),
raw_result 为新增三项加法合同(overlay/medical_report_csv_v2/quality_*)。
发布即冻结铁律适用:已发布的 v1 不可修改,字段变动必须新建 v2。
"""
from pydantic import ConfigDict, Field

from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.region_metrics import PublicRegionCounts, PublicRegionRatios
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class SurfaceGlossPublicMetrics(ContractModel):
    """surface_gloss 用户简表:油光面积占比 + 油光区域数量(各六分区)。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    油光面积占比: PublicRegionRatios = Field(
        alias="油光面积占比",
        title="分区油光面积占比",
        description="油光区域面积占各分区有效皮肤面积的比例。",
    )
    油光区域数量: PublicRegionCounts = Field(
        alias="油光区域数量",
        title="分区油光区域数量",
        description="各分区内独立油光区域的数量。",
    )


class SurfaceGlossV1RawResult(ContractModel):
    """surface_gloss v1 的 raw_result 结构。"""

    overlay: str | None = None
    medical_report_csv_v2: str | None = None
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []
    metrics: SurfaceGlossPublicMetrics
    # 未发布版本:第三段直接纳入版本化评分输入(inline),见证据矩阵 §6。
    scoring_input: ScoringInputV1 | None = None


class _SurfaceGlossV1Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 surface_gloss v1 的返回结构。"""

    raw_result: SurfaceGlossV1RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("surface_gloss", SCHEMA_VERSION, _SurfaceGlossV1Envelope)
