"""wrinkle v1 契约,源自 ai-skin-backend service/algorithm/implements/wrinkle_v1.py。
max_segment_length 改 int|float(对齐 dermavision)+ 加 medical_metrics_v2/medical_report_csv_v2 Optional。"""
from typing import Any

from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class WrinkleRegionMetric(ContractModel):
    """summary.json 中 region_metrics 的每个条目。"""
    region_key: str
    region_name: str
    short_name: str
    relative_score: float
    segment_count: int
    wrinkle_pixels: int
    mean_segment_length: float
    max_segment_length: int | float
    density_per_10k: float
    share_pct: float
    area_px: int


class WrinkleRawResult(ContractModel):
    """wrinkle v1 信封的 raw_result 结构。"""

    # 13 个 OSS key 字段
    analysis_face: str | None = None
    preprocessed_face: str | None = None
    stage1_candidates: str | None = None
    vote_heatmap: str | None = None
    stage2_overlay: str | None = None
    stage2_centerline: str | None = None
    face_filter_debug: str | None = None
    texture_reference: str | None = None
    comparison: str | None = None
    region_overlay: str | None = None
    region_tiles: str | None = None
    region_metrics_csv: str | None = None
    summary_json: str | None = None

    # 结构化数据（前端需要）
    region_metrics: list[WrinkleRegionMetric] = []
    run_preset: str | None = None
    device: str | None = None
    successful_runs: int = 0
    failed_runs: int = 0
    region_analysis_status: str | None = None
    # medical_metrics_v2 (dermavision worker 已返回,Optional 兼容旧数据)
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _WrinkleV1Envelope(AlgorithmEnvelope):
    """drift 检测：校验 wrinkle worker 对 v1 的返回结构。"""
    raw_result: WrinkleRawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("wrinkle", SCHEMA_VERSION, _WrinkleV1Envelope)
