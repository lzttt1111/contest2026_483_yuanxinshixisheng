from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class RenderPreset(ContractModel):
    """render_presets 中每个预设的渲染参数。"""
    grain: float
    contrast: float
    saturation: float
    tone_weight: float
    texture_weight: float


class RegionStat(ContractModel):
    """region_statistics 中每个区域的 redness 统计。"""
    area: int
    max_redness: float
    p50_redness: float
    p90_redness: float
    p95_redness: float
    mean_redness: float
    red_area_ratio: float
    redness_burden: float
    high_red_area_ratio: float


class RedFeatureRegionStat(ContractModel):
    """red_feature_region_distribution 中每个区域的红色特征统计。"""
    area: int
    count: int
    area_ratio: float


class RedFeatureLocation(ContractModel):
    """red_feature_locations 中每个红色特征的位置信息。"""
    area: int
    bbox: list[int]
    region: str
    local_z: float
    centroid: list[float]
    confidence: float
    feature_id: int
    scale_vote: float
    mean_redness: float
    area_ratio_analysis_zone: float


class RednessFilterReasons(ContractModel):
    """red_feature_filter_reasons: 局灶红色候选按原因过滤的数量统计。"""

    eyes: int
    eyebrows: int
    nostrils: int
    lips: int
    boundary: int
    nasolabial: int


class RednessV1Metrics(ContractModel):
    """redness v1 metrics 完整结构（41 字段）。"""
    algorithm: str
    disclaimer: str
    max_redness: float
    p50_redness: float
    p90_redness: float
    p95_redness: float
    source_name: str
    mean_redness: float
    eye_rendering: str
    quality_flags: list[str]
    quality_score: float
    metric_version: str
    quality_status: str
    red_area_ratio: float
    redness_burden: float
    render_default: str
    render_presets: dict[str, RenderPreset]
    face_stats_area: int
    reference_image: str | None = None
    face_mask_source: str
    raw_redness_mean: float
    red_feature_area: int
    red_feature_count: int
    redness_base_mean: float
    region_statistics: dict[str, RegionStat]
    red_area_threshold: float
    high_red_area_ratio: float
    redness_detail_mean: float
    color_transfer_scope: str
    skin_foreground_area: int
    red_feature_locations: list[RedFeatureLocation]
    color_transfer_enabled: bool
    red_feature_area_ratio: float
    color_transfer_strength: float
    high_red_area_threshold: float
    red_feature_marker_count: int
    red_feature_overlay_base: str
    red_feature_filter_reasons: RednessFilterReasons
    red_feature_filtered_count: int
    red_feature_pre_filter_count: int
    red_feature_region_distribution: dict[str, RedFeatureRegionStat]


class RednessV1RawResult(ContractModel):
    """redness v1 的 raw_result 结构（信封格式）。metrics 为完整英文 dict，透传不转 list。"""
    overlay: str | None = None
    red_areas_overlay: str | None = None
    metrics: RednessV1Metrics
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []


class _RednessV1Envelope(AlgorithmEnvelope):
    """drift 检测：校验 dermavision worker 对 redness v1 的返回结构。"""
    raw_result: RednessV1RawResult


from aisia_contracts.registry import register
SCHEMA_VERSION = "1"
register("redness", SCHEMA_VERSION, _RednessV1Envelope)