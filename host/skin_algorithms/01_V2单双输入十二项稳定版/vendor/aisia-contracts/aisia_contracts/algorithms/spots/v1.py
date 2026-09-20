from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class SpotsParameters(ContractModel):
    """spots v1 metrics 的 parameters 子结构。"""
    local_sigmas: list[float]
    minimum_area_px: float
    scale_z_threshold: float
    large_local_sigmas: list[float]
    maximum_area_ratio: float
    minimum_confidence: float
    minimum_scale_votes: int
    salient_core_sigmas: list[float]
    large_minimum_area_px: float
    large_scale_z_threshold: float
    large_maximum_area_ratio: float
    large_minimum_confidence: float
    large_minimum_scale_votes: int
    salient_minimum_confidence: float
    salient_minimum_scale_votes: int


class SpotRegionStat(ContractModel):
    """region_distribution 中每个区域的 spot 统计。"""
    count: int
    area_ratio: float


class SpotLocation(ContractModel):
    """spot_locations / large_spot_locations 中每个斑点的位置信息（36 字段）。"""
    area: float
    bbox: list[int]
    extent: float
    region: str
    score_z: float
    spot_id: int
    centroid: list[float]
    solidity: float
    spot_type: str
    confidence: float
    scale_vote: float
    circularity: float
    mean_deltaE: float
    aspect_ratio: float
    line_response: float
    local_contrast: float
    max_scale_vote: int
    patch_contrast: float
    salient_deltaE: float
    area_ratio_skin: float
    brownness_score: float
    hair_scale_vote: float
    regional_deltaE: float
    color_difference: float
    uniformity_score: float
    salient_supported: bool
    prominent_red_rescue: bool
    distance_to_hair_mask: float
    nostril_overlap_ratio: float
    hair_line_response_p90: float
    salient_red_difference: float
    nostril_enclosure_ratio: float
    occlusion_overlap_ratio: float
    salient_dark_difference: float
    distance_to_feature_mask: float
    salient_support_fraction: float
    salient_yellow_difference: float


class SpotOcclusionStatistics(ContractModel):
    """occlusion_filter_statistics: 斑点遮挡和线状误检过滤统计。"""

    hair: int
    eyebrow_eyelash_feature: int
    facial_hair: int
    nostril: int
    nasolabial_shadow: int
    line_like: int
    total: int
    small_suppressed_by_large: int
    multipeak_parents_split: int


class SpotsV1Metrics(ContractModel):
    """spots v1 metrics 完整结构（24 字段）。"""
    definition: str
    parameters: SpotsParameters
    spot_count: int
    mean_deltaE: float
    mean_spot_area: float
    spot_locations: list[SpotLocation]
    spot_area_ratio: float
    spot_confidence: float
    large_spot_count: int
    median_spot_area: float
    small_spot_count: int
    merged_spot_count: int
    analysis_zone_area: int
    salient_spot_count: int
    hair_filtered_count: int
    region_distribution: dict[str, SpotRegionStat]
    large_spot_locations: list[SpotLocation]
    large_spot_area_ratio: float
    nostril_filtered_count: int
    large_spot_recall_notes: list[str]
    pre_occlusion_spot_count: int
    post_split_instance_count: int
    occlusion_filter_statistics: SpotOcclusionStatistics
    pre_split_large_component_count: int


class SpotsV1RawResult(ContractModel):
    """spots v1 的 raw_result 结构（信封格式）。metrics 为完整英文 dict，透传不转 list。"""
    overlay: str | None = None
    metrics: SpotsV1Metrics
    quality_score: float | None = None
    quality_status: str | None = None
    quality_flags: list[str] = []


class _SpotsV1Envelope(AlgorithmEnvelope):
    """drift 检测：校验 dermavision worker 对 spots v1 的返回结构。"""
    raw_result: SpotsV1RawResult


from aisia_contracts.registry import register
SCHEMA_VERSION = "1"
register("spots", SCHEMA_VERSION, _SpotsV1Envelope)