"""purple v1 契约。
metrics 改用 dermavision PurpleFullMetrics/PurplePartialMetrics 精简英文计数
(原误用 medical_metrics_v1 中文宽表, 但 worker 实际返回 public_metrics 英文计数)。
purple 无 medical_metrics_v2 envelope 字段(医学统计走独立文件不进 metrics)。"""
from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class PurplePartialMetrics(ContractModel):
    """purple 用户简表(局部脸):UV色斑/紫质总数。"""

    uv_spots_total: int
    porphyrin_total: int


class PurpleFullMetrics(PurplePartialMetrics):
    """purple 用户简表(完整脸):UV色斑/紫质总数 + 5 分区计数。"""

    uv_spots_forehead: int
    uv_spots_left_cheek: int
    uv_spots_right_cheek: int
    uv_spots_nose: int
    uv_spots_chin: int
    porphyrin_forehead: int
    porphyrin_left_cheek: int
    porphyrin_right_cheek: int
    porphyrin_nose: int
    porphyrin_chin: int


class PurpleV1RawResult(ContractModel):
    """purple v1 的 raw_result 结构。

    紫区为独立算法，返回四张图（UV 底图、UV 色斑实例图、荧光底图、紫质实例图），
    无统一 overlay 字段。metrics 为精简英文计数(public_metrics)。
    """

    uv_base: str | None = None
    uv_spots_overlay: str | None = None
    fluorescence_base: str | None = None
    porphyrin_overlay: str | None = None
    metrics: PurpleFullMetrics | PurplePartialMetrics
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []


class _PurpleV1Envelope(AlgorithmEnvelope):
    """drift 检测：校验 dermavision worker 对 purple v1 的返回结构。"""

    raw_result: PurpleV1RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("purple", SCHEMA_VERSION, _PurpleV1Envelope)
