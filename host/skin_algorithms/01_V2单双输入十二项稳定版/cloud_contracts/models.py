"""DermaVision 单RGB十二项云端返回合同。

这里的模型只负责校验 Worker 已经构造好的字典。校验成功后始终返回原始
``dict``，不调用 ``model_dump``，因此不会改变旧字段、字段顺序、中文键、
``None`` 或数值类型。
"""

from __future__ import annotations

import types
from typing import Any, Generic, Literal, TypeVar, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
)

from aisia_contracts.scoring_input.v1 import ScoringInputV1


AlgorithmName = Literal[
    "redness", "spots", "brown", "texture", "pores", "purple", "acne", "wrinkle",
    "surface_gloss", "vascular", "contour_firmness",
]

SCHEMA_VERSIONS: dict[str, str] = {
    "redness": "3",
    "spots": "3",
    "brown": "3",
    "texture": "3",
    "pores": "3",
    "purple": "2",
    "acne": "1",
    "acne_v2": "2",
    "wrinkle": "2",
    "surface_gloss": "1",
    "vascular": "1",
    "contour_firmness": "1",
}

SCORING_INPUT_DESCRIPTION = (
    "版本化评分输入（scoring_input_v1）：聚合侧据此重建 detector_results；"
    "检测失败或字段缺失时诚实降级，不补零。"
)


class StrictContractModel(BaseModel):
    """禁止未登记字段，防止 Worker 与后端合同静默漂移。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


def _metric_field(
    title: str,
    description: str,
    unit: str,
    example: Any,
    *,
    alias: str | None = None,
):
    """统一给公开量化字段登记中文名、口径、单位和示例。"""

    return Field(
        alias=alias,
        title=title,
        description=description,
        examples=[example],
        json_schema_extra={"unit": unit},
    )


class BrownPublicMetrics(StrictContractModel):
    total: int = _metric_field("棕色斑总数", "有效分析区域内检测到的棕色斑实例总数。", "个", 234, alias="总计")
    forehead: int = _metric_field("额头棕色斑数量", "额头区域检测到的棕色斑实例数。", "个", 57, alias="额头")
    left_cheek: int = _metric_field("画面左脸颊棕色斑数量", "画面左侧脸颊区域检测到的棕色斑实例数。", "个", 61, alias="左脸颊")
    right_cheek: int = _metric_field("画面右脸颊棕色斑数量", "画面右侧脸颊区域检测到的棕色斑实例数。", "个", 83, alias="右脸颊")
    nose: int = _metric_field("鼻部棕色斑数量", "鼻部区域检测到的棕色斑实例数。", "个", 7, alias="鼻部")
    chin: int = _metric_field("下巴棕色斑数量", "下巴区域检测到的棕色斑实例数。", "个", 26, alias="下巴")


class BrownPartialMetrics(StrictContractModel):
    total: int = _metric_field("棕色斑总数", "局部脸有效分析区域内检测到的棕色斑实例总数。", "个", 80, alias="总计")


class PoresPublicMetrics(StrictContractModel):
    total: int = _metric_field("可见毛孔总数", "有效分析区域内检测到的可见毛孔实例总数。", "个", 936, alias="总计")
    forehead: int = _metric_field("额头可见毛孔数量", "额头区域检测到的可见毛孔实例数。", "个", 218, alias="额头")
    left_cheek: int = _metric_field("画面左脸颊可见毛孔数量", "画面左侧脸颊区域检测到的可见毛孔实例数。", "个", 231, alias="左脸颊")
    right_cheek: int = _metric_field("画面右脸颊可见毛孔数量", "画面右侧脸颊区域检测到的可见毛孔实例数。", "个", 309, alias="右脸颊")
    nose: int = _metric_field("鼻部可见毛孔数量", "鼻部区域检测到的可见毛孔实例数。", "个", 91, alias="鼻部")
    chin: int = _metric_field("下巴可见毛孔数量", "下巴区域检测到的可见毛孔实例数。", "个", 87, alias="下巴")


class PoresPartialMetrics(StrictContractModel):
    total: int = _metric_field("可见毛孔总数", "局部脸有效分析区域内检测到的可见毛孔实例总数。", "个", 320, alias="总计")


class TexturePublicMetrics(StrictContractModel):
    total: int = _metric_field("纹理特征总数", "有效分析区域内凸起样与凹陷样二维纹理特征总数。", "个", 971, alias="总计")
    raised: int = _metric_field("凸起样纹理数量", "结果图中以黄色标注的凸起样二维纹理特征数。", "个", 598, alias="凸起样（黄）")
    depressed: int = _metric_field("凹陷样纹理数量", "结果图中以蓝色标注的凹陷样二维纹理特征数。", "个", 373, alias="凹陷样（蓝）")
    forehead: int = _metric_field("额头纹理数量", "额头区域检测到的两类纹理特征总数。", "个", 174, alias="额头")
    left_cheek: int = _metric_field("画面左脸颊纹理数量", "画面左侧脸颊区域检测到的两类纹理特征总数。", "个", 255, alias="左脸颊")
    right_cheek: int = _metric_field("画面右脸颊纹理数量", "画面右侧脸颊区域检测到的两类纹理特征总数。", "个", 324, alias="右脸颊")
    nose: int = _metric_field("鼻部纹理数量", "鼻部区域检测到的两类纹理特征总数。", "个", 80, alias="鼻部")
    chin: int = _metric_field("下巴纹理数量", "下巴区域检测到的两类纹理特征总数。", "个", 138, alias="下巴")


class TexturePartialMetrics(StrictContractModel):
    total: int = _metric_field("纹理特征总数", "局部脸有效分析区域内检测到的二维纹理特征总数。", "个", 280, alias="总计")


class PurplePartialMetrics(StrictContractModel):
    uv_spots_total: int = _metric_field("UV色斑总数", "有效分析区域内检测到的UV色斑实例总数。", "个", 506)
    porphyrin_total: int = _metric_field("紫质总数", "有效分析区域内检测到的紫质实例总数。", "个", 447)


class PurpleFullMetrics(PurplePartialMetrics):
    uv_spots_forehead: int = _metric_field("额头UV色斑数量", "额头区域检测到的UV色斑实例数。", "个", 87)
    uv_spots_left_cheek: int = _metric_field("画面左脸颊UV色斑数量", "画面左侧脸颊区域检测到的UV色斑实例数。", "个", 134)
    uv_spots_right_cheek: int = _metric_field("画面右脸颊UV色斑数量", "画面右侧脸颊区域检测到的UV色斑实例数。", "个", 184)
    uv_spots_nose: int = _metric_field("鼻部UV色斑数量", "鼻部区域检测到的UV色斑实例数。", "个", 30)
    uv_spots_chin: int = _metric_field("下巴UV色斑数量", "下巴区域检测到的UV色斑实例数。", "个", 71)
    porphyrin_forehead: int = _metric_field("额头紫质数量", "额头区域检测到的紫质实例数。", "个", 26)
    porphyrin_left_cheek: int = _metric_field("画面左脸颊紫质数量", "画面左侧脸颊区域检测到的紫质实例数。", "个", 121)
    porphyrin_right_cheek: int = _metric_field("画面右脸颊紫质数量", "画面右侧脸颊区域检测到的紫质实例数。", "个", 216)
    porphyrin_nose: int = _metric_field("鼻部紫质数量", "鼻部区域检测到的紫质实例数。", "个", 51)
    porphyrin_chin: int = _metric_field("下巴紫质数量", "下巴区域检测到的紫质实例数。", "个", 33)


class MedicalMetricsV2(StrictContractModel):
    metrics_version: str = Field(description="医学量化结构版本，例如medical_metrics_v2_20260728")
    scoring_status: str = Field(description="评分标定状态")
    units: dict[str, Any] = Field(description="指标单位、成像口径和画面左右定义")
    overall_metrics: dict[str, Any] = Field(default_factory=dict, description="全面部医学量化指标")
    region_metrics: list[dict[str, Any]] = Field(default_factory=list, description="面部分区医学量化指标")
    left_right_comparison: dict[str, Any] = Field(default_factory=dict, description="画面左右区域比较")
    medical_limitations: list[str] = Field(default_factory=list, description="医学解释边界和成像限制")


class FullMedicalMetricsV2(StrictContractModel):
    """医学量化生成器的完整文档结构；历史合同夹具仍会直接注入该结构。"""

    project: str = Field(description="医学量化检测项目名称")
    metrics_version: str = Field(description="医学量化结构版本，例如medical_metrics_v2_20260728")
    scoring_status: str = Field(description="评分标定状态")
    imaging_and_units: dict[str, Any] = Field(description="指标单位和成像口径说明")
    overall_metrics: dict[str, Any] = Field(default_factory=dict, description="全面部医学量化指标")
    region_metrics: list[dict[str, Any]] = Field(default_factory=list, description="面部分区医学量化指标")
    left_right_comparison: dict[str, Any] = Field(default_factory=dict, description="画面左右区域比较")
    quality_control: dict[str, Any] = Field(description="图像质量和算法质量控制信息")
    medical_limitations: list[str] = Field(default_factory=list, description="医学解释边界和成像限制")


class RednessRegionStatistics(StrictContractModel):
    area: int = _metric_field("分区统计面积", "当前面部分区参与红度统计的像素面积。", "像素", 120378)
    red_area_ratio: float = _metric_field("分区红区面积占比", "当前分区内达到红区阈值的面积占比。", "比例（0～1）", 0.31)
    high_red_area_ratio: float = _metric_field("分区高红区面积占比", "当前分区内达到高红度阈值的面积占比。", "比例（0～1）", 0.12)
    mean_redness: float = _metric_field("分区平均红度", "当前分区红度响应的平均值。", "0～1工程归一化值", 0.52)
    p50_redness: float = _metric_field("分区P50红度", "当前分区红度响应的中位数。", "0～1工程归一化值", 0.5)
    p90_redness: float = _metric_field("分区P90红度", "当前分区90%分位红度响应。", "0～1工程归一化值", 0.83)
    p95_redness: float = _metric_field("分区P95红度", "当前分区95%分位红度响应。", "0～1工程归一化值", 0.91)
    max_redness: float = _metric_field("分区最大红度", "当前分区红度响应最大值。", "0～1工程归一化值", 0.97)
    redness_burden: float = _metric_field("分区红度负担", "当前分区综合色度范围和强度形成的工程负担值。", "工程指标", 0.52)


class RedFeatureRegionStatistics(StrictContractModel):
    count: int = _metric_field("分区局灶红色实例数", "当前分区内局灶性红色实例数量。", "个", 22)
    area: int = _metric_field("分区局灶红色实例面积", "当前分区内局灶性红色实例像素面积之和。", "像素", 1199)
    area_ratio: float = _metric_field("分区局灶红色实例面积占比", "局灶性红色实例面积占当前分区统计面积的比例。", "比例（0～1）", 0.01)


class RednessRenderPreset(StrictContractModel):
    contrast: int | float = Field(description="红区结果图显示对比度参数")
    texture_weight: int | float = Field(description="红区结果图纹理融合权重")
    saturation: int | float = Field(description="红区结果图饱和度参数")
    tone_weight: int | float = Field(description="红区结果图底色融合权重")
    grain: int | float = Field(description="红区结果图颗粒显示参数")


class RednessFilterReasons(StrictContractModel):
    eyes: int = Field(description="因眼部安全区被过滤的候选数量")
    eyebrows: int = Field(description="因眉毛区域被过滤的候选数量")
    nostrils: int = Field(description="因鼻孔区域被过滤的候选数量")
    lips: int = Field(description="因嘴唇区域被过滤的候选数量")
    boundary: int = Field(description="因有效分析边缘被过滤的候选数量")
    nasolabial: int = Field(description="因法令纹暗线区域被过滤的候选数量")


class RednessPublicMetrics(StrictContractModel):
    algorithm: str = Field(description="红区算法名称")
    metric_version: str = Field(description="红区工程量化指标版本")
    source_name: str = Field(description="输入图片来源名称")
    reference_image: str | None = Field(description="颜色参考图名称；未使用时为空")
    color_transfer_enabled: bool = Field(description="是否启用颜色迁移校正")
    color_transfer_strength: int | float = Field(description="颜色迁移校正强度")
    red_area_threshold: int | float = _metric_field("红区阈值", "连续红度响应进入红区统计的工程阈值。", "0～1", 0.45)
    high_red_area_threshold: int | float = _metric_field("高红区阈值", "连续红度响应进入高红区统计的工程阈值。", "0～1", 0.7)
    skin_foreground_area: int = _metric_field("皮肤前景面积", "预处理获得的皮肤前景像素面积。", "像素", 628123)
    face_stats_area: int = _metric_field("有效红度统计面积", "参与红度统计的有效面部像素面积。", "像素", 478979)
    red_area_ratio: int | float = _metric_field("红区面积占比", "有效统计区域内达到红区阈值的面积占比。", "比例（0～1）", 0.64)
    high_red_area_ratio: int | float = _metric_field("高红区面积占比", "有效统计区域内达到高红度阈值的面积占比。", "比例（0～1）", 0.25)
    mean_redness: int | float = _metric_field("平均红度", "有效统计区域红度响应平均值。", "0～1工程归一化值", 0.5)
    p50_redness: int | float = _metric_field("P50红度", "有效统计区域红度响应中位数。", "0～1工程归一化值", 0.54)
    p90_redness: int | float = _metric_field("P90红度", "有效统计区域90%分位红度响应。", "0～1工程归一化值", 0.86)
    p95_redness: int | float = _metric_field("P95红度", "有效统计区域95%分位红度响应。", "0～1工程归一化值", 0.94)
    max_redness: int | float = _metric_field("最大红度", "有效统计区域红度响应最大值。", "0～1工程归一化值", 0.98)
    redness_burden: int | float = _metric_field("红度综合负担", "综合色度范围和强度形成的工程负担值。", "工程指标", 0.5)
    region_statistics: dict[str, RednessRegionStatistics] = Field(description="额头、画面左右脸颊、鼻部和下巴的红度统计")
    red_feature_count: int = _metric_field("局灶红色实例总数", "有效分析区域内局灶性红色实例数量。", "个", 95)
    red_feature_area: int = _metric_field("局灶红色实例总面积", "全部局灶性红色实例的像素面积之和。", "像素", 8672)
    red_feature_area_ratio: int | float = _metric_field("局灶红色实例面积占比", "局灶性红色实例总面积占有效分析面积的比例。", "比例（0～1）", 0.018)
    red_feature_locations: list[dict[str, Any]] = Field(description="兼容保留的局灶红色实例坐标数组；正式公开返回为空数组")
    red_feature_region_distribution: dict[str, RedFeatureRegionStatistics] = Field(description="局灶红色实例的面部分区数量和面积统计")
    quality_score: int | float = Field(description="输入图片质量评分")
    quality_status: str = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    disclaimer: str = Field(description="红区工程量化口径说明")
    redness_base_mean: int | float = Field(description="红度低频基础响应平均值")
    redness_detail_mean: int | float = Field(description="红度细节响应平均值")
    raw_redness_mean: int | float = Field(description="原始红度响应平均值")
    render_default: str = Field(description="默认红区结果图显示预设名称")
    render_presets: dict[str, RednessRenderPreset] = Field(description="红区结果图显示预设参数")
    color_transfer_scope: str = Field(description="颜色迁移处理范围")
    eye_rendering: str = Field(description="眼部底图渲染处理方式")
    face_mask_source: str = Field(description="人脸皮肤Mask来源")
    red_feature_overlay_base: str = Field(description="局灶红色实例图使用的底图预设")
    red_feature_marker_count: int = Field(description="最终绘制的局灶红色实例标记数量")
    red_feature_pre_filter_count: int = Field(description="后处理过滤前的局灶红色候选数量")
    red_feature_filtered_count: int = Field(description="后处理删除的局灶红色候选数量")
    red_feature_filter_reasons: RednessFilterReasons = Field(description="局灶红色候选按原因过滤的数量统计")


class SpotRegionStatistics(StrictContractModel):
    count: int = _metric_field("分区斑点数量", "当前分区检测到的可见斑点实例数。", "个", 20)
    area_ratio: int | float = _metric_field("分区斑点面积占比", "当前分区斑点实例面积占有效分析面积的比例。", "比例（0～1）", 0.002)


class SpotOcclusionStatistics(StrictContractModel):
    hair: int = Field(description="毛发遮挡过滤数量")
    eyebrow_eyelash_feature: int = Field(description="眉毛和睫毛区域过滤数量")
    facial_hair: int = Field(description="胡须等面部毛发过滤数量")
    nostril: int = Field(description="鼻孔区域过滤数量")
    nasolabial_shadow: int = Field(description="法令纹阴影过滤数量")
    line_like: int = Field(description="线状结构过滤数量")
    total: int = Field(description="全部遮挡和误检过滤数量")
    small_suppressed_by_large: int = Field(description="被大片状候选覆盖的小斑点数量")
    multipeak_parents_split: int = Field(description="因包含多个峰值而被拆分的父级候选数量")


class SpotsParameters(StrictContractModel):
    local_sigmas: list[int | float] = Field(description="小型斑点检测使用的高斯尺度")
    large_local_sigmas: list[int | float] = Field(description="大片状斑点检测使用的高斯尺度")
    salient_core_sigmas: list[int | float] = Field(description="显著颜色异常检测使用的高斯尺度")
    scale_z_threshold: int | float = Field(description="小型斑点多尺度响应阈值")
    large_scale_z_threshold: int | float = Field(description="大片状斑点多尺度响应阈值")
    minimum_scale_votes: int = Field(description="小型斑点最少尺度投票数")
    large_minimum_scale_votes: int = Field(description="大片状斑点最少尺度投票数")
    salient_minimum_scale_votes: int = Field(description="显著异常最少尺度投票数")
    minimum_area_px: int | float = Field(description="候选最小像素面积")
    maximum_area_ratio: int | float = Field(description="小型斑点最大面积比例")
    large_minimum_area_px: int | float = Field(description="大片状斑点最小像素面积")
    large_maximum_area_ratio: int | float = Field(description="大片状斑点最大面积比例")
    minimum_confidence: int | float = Field(description="小型斑点最低工程置信度")
    large_minimum_confidence: int | float = Field(description="大片状斑点最低工程置信度")
    salient_minimum_confidence: int | float = Field(description="显著颜色异常最低工程置信度")


class SpotsPublicMetrics(StrictContractModel):
    definition: str = Field(description="可见斑点的工程检测定义")
    spot_count: int = _metric_field("可见斑点总数", "有效分析区域内合并去重后的可见斑点实例总数。", "个", 137)
    spot_area_ratio: int | float = _metric_field("可见斑点面积占比", "全部可见斑点实例面积占有效分析区域的比例。", "比例（0～1）", 0.021)
    spot_confidence: int | float = _metric_field("平均工程置信度", "全部可见斑点实例的平均工程置信度。", "0～1", 0.58)
    mean_deltaE: int | float = _metric_field("平均综合色差", "可见斑点相对周围皮肤的平均综合色差。", "Delta E工程值", 4.42)
    pre_occlusion_spot_count: int = Field(description="遮挡过滤前的可见斑点候选数量")
    hair_filtered_count: int = Field(description="毛发及相关遮挡后处理删除的候选数量")
    small_spot_count: int = _metric_field("小型斑点分支数量", "小型可见斑点检测分支产生的实例数。", "个", 33)
    large_spot_count: int = _metric_field("大片状斑点分支数量", "大片状可见色差检测分支产生的实例数。", "个", 23)
    merged_spot_count: int = _metric_field("融合后斑点数量", "多路候选融合、拆分和去重后的斑点数量。", "个", 137)
    large_spot_area_ratio: int | float = _metric_field("大片状斑点面积占比", "大片状斑点实例面积占有效分析区域的比例。", "比例（0～1）", 0.0075)
    large_spot_locations: list[dict[str, Any]] = Field(description="兼容保留的大片状斑点坐标数组；正式公开返回为空数组")
    large_spot_recall_notes: list[str] = Field(description="大片状斑点召回分支的工程说明")
    salient_spot_count: int = _metric_field("显著颜色异常候选数", "显著颜色异常分支产生的候选数量。", "个", 120)
    nostril_filtered_count: int = Field(description="鼻孔区域后处理删除的候选数量")
    analysis_zone_area: int = _metric_field("斑点有效分析面积", "用于斑点检测和密度统计的有效像素面积。", "像素", 476501)
    mean_spot_area: int | float = _metric_field("平均斑点面积", "单个可见斑点实例面积的平均值。", "像素", 72.88)
    median_spot_area: int | float = _metric_field("P50斑点面积", "单个可见斑点实例面积的中位数。", "像素", 62.5)
    pre_split_large_component_count: int = Field(description="分水岭拆分前的大片状连通域数量")
    post_split_instance_count: int = Field(description="分水岭拆分后的最终实例数量")
    occlusion_filter_statistics: SpotOcclusionStatistics = Field(description="斑点遮挡和线状误检过滤统计")
    spot_locations: list[dict[str, Any]] = Field(description="兼容保留的斑点坐标数组；正式公开返回为空数组")
    region_distribution: dict[str, SpotRegionStatistics] = Field(description="斑点在各面部分区的数量与面积占比")
    parameters: SpotsParameters = Field(description="当前斑点算法工程参数快照")


class AlgorithmMetaData(StrictContractModel):
    name: AlgorithmName = Field(description="算法名称；紫区一个任务同时产生UV色斑和紫质结果")
    version: Literal["1"] = Field(description="后端AlgorithmRegistry接口版本，当前固定为字符串1")


class DermavisionDebugInfo(StrictContractModel):
    report_csv: str | None = Field(description="用户量化CSV在OSS中的对象键")
    timing_seconds: dict[str, Any] = Field(
        description="下载、预处理、各算法、上传和总耗时，单位秒；algorithms节点按算法名分组"
    )
    execution_mode: str = Field(description="执行模式，正式单项任务为single_algorithm")


class AcneDebugInfo(StrictContractModel):
    display_result: Any = Field(description="痤疮前端展示结果索引，保持旧合同原结构")
    algorithm_version: Literal["1"] = Field(description="痤疮接口版本，固定为字符串1")
    elapsed_seconds: int | float | None = Field(description="痤疮任务总耗时，单位秒")


class WrinkleDebugInfo(StrictContractModel):
    display_result: Any = Field(description="皱纹前端展示结果索引，保持旧合同原结构")
    algorithm_version: Literal["1"] = Field(description="皱纹接口版本，固定为字符串1")
    elapsed_seconds: int | float | None = Field(description="皱纹任务总耗时，单位秒")
    output_dir: str | None = Field(description="Worker内部排障目录；不作为用户结果文件")


class MedicalV2OptionalFields(StrictContractModel):
    """仅用作字段声明基类，不单独实例化。"""

    medical_metrics_v2: MedicalMetricsV2 | FullMedicalMetricsV2 | None = Field(
        default=None, description="可选医学V2结构化量化指标；关闭开关时省略"
    )
    medical_report_csv_v2: str | None = Field(
        default=None, description="可选医学V2 CSV在OSS中的对象键；关闭开关时省略"
    )


class RednessRawResult(MedicalV2OptionalFields):
    overlay: str | None = Field(description="红区主结果图OSS对象键")
    metrics: RednessPublicMetrics = Field(description="红区旧版完整英文工程量化指标")
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    red_areas_overlay: str | None = Field(description="局灶性红色实例结果图OSS对象键")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class SpotsRawResult(MedicalV2OptionalFields):
    overlay: str | None = Field(description="可见斑点结果图OSS对象键")
    metrics: SpotsPublicMetrics = Field(description="斑点旧版完整英文工程量化指标")
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class BrownRawResult(MedicalV2OptionalFields):
    overlay: str | None = Field(description="RBX棕区结果图OSS对象键")
    metrics: BrownPublicMetrics | BrownPartialMetrics = Field(description="棕区用户简表；完整脸返回分区计数，局部脸只返回总计")
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    brown_spots_overlay: str | None = Field(description="VISIA式棕色斑实例结果图OSS对象键")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class TextureRawResult(MedicalV2OptionalFields):
    overlay: str | None = Field(description="二维可见纹理结果图OSS对象键")
    metrics: TexturePublicMetrics | TexturePartialMetrics = Field(description="纹理用户简表；完整脸返回分区和双类型计数，局部脸只返回总计")
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class PoresRawResult(MedicalV2OptionalFields):
    overlay: str | None = Field(description="可见毛孔结果图OSS对象键")
    metrics: PoresPublicMetrics | PoresPartialMetrics = Field(description="毛孔用户简表；完整脸返回分区计数，局部脸只返回总计")
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class PurpleRawResult(StrictContractModel):
    uv_base: str | None = Field(description="UV色斑标准化底图OSS对象键")
    uv_spots_overlay: str | None = Field(description="UV色斑实例结果图OSS对象键")
    fluorescence_base: str | None = Field(description="紫质荧光风格底图OSS对象键")
    porphyrin_overlay: str | None = Field(description="紫质实例结果图OSS对象键")
    metrics: PurpleFullMetrics | PurplePartialMetrics = Field(
        description="紫区前端精简英文计数：UV色斑和紫质的总数及五个常用分区"
    )
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class PublicRegionRatios(StrictContractModel):
    total: float = _metric_field("全面部比例", "全面部有效分析区域内的面积比例。", "比例（0～1）", 0.04, alias="总计")
    forehead: float = _metric_field("额头比例", "额头区域内的面积比例。", "比例（0～1）", 0.03, alias="额头")
    left_cheek: float = _metric_field("画面左脸颊比例", "画面左侧脸颊区域内的面积比例。", "比例（0～1）", 0.02, alias="左脸颊")
    right_cheek: float = _metric_field("画面右脸颊比例", "画面右侧脸颊区域内的面积比例。", "比例（0～1）", 0.02, alias="右脸颊")
    nose: float = _metric_field("鼻部比例", "鼻部区域内的面积比例。", "比例（0～1）", 0.12, alias="鼻部")
    chin: float = _metric_field("下巴比例", "下巴区域内的面积比例。", "比例（0～1）", 0.14, alias="下巴")


class PublicRegionCounts(StrictContractModel):
    total: int = _metric_field("全面部数量", "全面部有效分析区域内的目标数量。", "个", 32, alias="总计")
    forehead: int = _metric_field("额头数量", "额头区域内的目标数量。", "个", 8, alias="额头")
    left_cheek: int = _metric_field("画面左脸颊数量", "画面左侧脸颊区域内的目标数量。", "个", 13, alias="左脸颊")
    right_cheek: int = _metric_field("画面右脸颊数量", "画面右侧脸颊区域内的目标数量。", "个", 18, alias="右脸颊")
    nose: int = _metric_field("鼻部数量", "鼻部区域内的目标数量。", "个", 7, alias="鼻部")
    chin: int = _metric_field("下巴数量", "下巴区域内的目标数量。", "个", 2, alias="下巴")


class PublicRegionLengths(StrictContractModel):
    total: float = _metric_field("全面部总长度", "全面部检出结构在1024标准化图像中的总长度。", "标准化图像像素", 1095.0, alias="总计")
    forehead: float = _metric_field("额头总长度", "额头区域内检出结构的总长度。", "标准化图像像素", 82.0, alias="额头")
    left_cheek: float = _metric_field("画面左脸颊总长度", "画面左侧脸颊区域内检出结构的总长度。", "标准化图像像素", 283.0, alias="左脸颊")
    right_cheek: float = _metric_field("画面右脸颊总长度", "画面右侧脸颊区域内检出结构的总长度。", "标准化图像像素", 160.0, alias="右脸颊")
    nose: float = _metric_field("鼻部总长度", "鼻部及鼻旁区域内检出结构的总长度。", "标准化图像像素", 248.0, alias="鼻部")
    chin: float = _metric_field("下巴总长度", "下巴及下颌区域内检出结构的总长度。", "标准化图像像素", 322.0, alias="下巴")


class SurfaceGlossPublicMetrics(StrictContractModel):
    area_ratio: PublicRegionRatios = Field(alias="油光面积占比", title="分区油光面积占比", description="油光区域面积占各分区有效皮肤面积的比例。")
    patch_count: PublicRegionCounts = Field(alias="油光区域数量", title="分区油光区域数量", description="各分区内独立油光区域的数量。")


class VascularPublicMetrics(StrictContractModel):
    count: PublicRegionCounts = Field(alias="血管样结构数量", title="分区血管样结构数量", description="各分区内符合当前检测标准的血管样线状结构数量。")
    total_length: PublicRegionLengths = Field(alias="血管样结构总长度", title="分区血管样结构总长度", description="各分区内血管样结构中心线在1024标准化图像中的总长度。")


class ContourFirmnessPublicMetrics(StrictContractModel):
    midface_continuity: float = _metric_field("中面部曲面连续性", "从颗部至下颊的2.5D相对曲面连续性代理；越高表示相对连续性越好。", "相对代理值（0～1）", 0.82, alias="中面部曲面连续性")
    jaw_continuity: float = _metric_field("下颌缘连续性", "下颌缘曲线平滑和连续程度的2D几何代理；越高表示轮廓越连续。", "相对代理值（0～1）", 0.76, alias="下颌缘连续性")
    laterality_difference: float = _metric_field("左右轮廓差异", "左右下颌弧长和轮廓几何差异的相对代理；越高表示左右差异越大。", "相对代理值（0～1）", 0.08, alias="左右轮廓差异")


class AddedAlgorithmRawResultBase(StrictContractModel):
    """新增三项的v1加法合同；旧八个Worker字段不变。"""

    overlay: str | None = Field(description="新增检测主结果图OSS对象键")
    medical_report_csv_v2: str | None = Field(description="详细医学宽表CSV OSS对象键")
    quality_score: int | float | None = Field(description="输入图片质量评分")
    quality_status: str | None = Field(description="输入图片质量状态")
    quality_flags: list[str] = Field(description="输入图片质量提示代码")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class SurfaceGlossRawResult(AddedAlgorithmRawResultBase):
    metrics: SurfaceGlossPublicMetrics = Field(description="表面油光的前端精简分区指标")


class VascularRawResult(AddedAlgorithmRawResultBase):
    metrics: VascularPublicMetrics = Field(description="血管样结构的前端精简分区指标")


class ContourFirmnessRawResult(AddedAlgorithmRawResultBase):
    metrics: ContourFirmnessPublicMetrics = Field(description="轮廓紧致度的前端精简2D/2.5D代理指标")


class AcnePresence(StrictContractModel):
    status: str = Field(description="痤疮检出状态")
    message: str = Field(description="痤疮检出状态中文说明")


class AcneRegionCount(StrictContractModel):
    region: str = Field(description="面部分区名称")
    count: int | float = Field(description="该分区疑似痤疮候选数量")


class AcneSeverityResult(StrictContractModel):
    level: int | None = _metric_field(
        "痤疮严重程度等级",
        "Acne-LDS完成评级时返回1～4级；当前图片不满足评级条件时为空。",
        "级",
        2,
        alias="等级",
    )
    note: str = Field(alias="注释", description="痤疮严重程度等级的中文解释或未评级原因")


class AcneQuantificationResult(StrictContractModel):
    candidate_count: int | None = _metric_field(
        "疑似痤疮圈选数量",
        "正式痤疮圈选结果中保留的疑似痤疮候选数量；无法完成检测时为空。",
        "个",
        13,
        alias="疑似痤疮圈选数量",
    )
    severity: AcneSeverityResult = Field(
        alias="痤疮严重程度等级",
        description="Acne-LDS四级严重程度结果及中文说明",
    )


class AcneV2Counts(StrictContractModel):
    total: int = Field(
        alias="总计", ge=0, description="正式痤疮圈选结果中的疑似痤疮总数"
    )
    forehead: int = Field(alias="额头", ge=0, description="额头区域疑似痤疮数量")
    left_cheek: int = Field(
        alias="左脸颊", ge=0, description="受检者左脸颊疑似痤疮数量"
    )
    right_cheek: int = Field(
        alias="右脸颊", ge=0, description="受检者右脸颊疑似痤疮数量"
    )
    nose: int = Field(alias="鼻部", ge=0, description="鼻部区域疑似痤疮数量")
    chin: int = Field(alias="下巴", ge=0, description="下巴区域疑似痤疮数量")


class AcneV2Metrics(StrictContractModel):
    counts: AcneV2Counts = Field(
        alias="疑似痤疮数量",
        description="正式痤疮圈选结果的总计和五个固定面部分区数量",
    )
    severity: AcneSeverityResult = Field(
        alias="痤疮严重程度等级",
        description="Acne-LDS四级严重程度结果及中文说明",
    )


class AcneRawResult(MedicalV2OptionalFields):
    acne_summary: str | None = Field(description="痤疮摘要文件OSS对象键")
    acne_circles: str | None = Field(description="痤疮正式圈选结果OSS对象键")
    acne_raw_boxes: str | None = Field(description="YOLO原始候选框结果OSS对象键")
    acne_detections: str | None = Field(description="痤疮候选明细OSS对象键")
    acne_skin_mask: str | None = Field(description="痤疮有效皮肤Mask OSS对象键")
    acne_forbidden_mask: str | None = Field(description="痤疮五官和无效区域Mask OSS对象键")
    acne_standardized: str | None = Field(description="痤疮标准化人脸图OSS对象键")
    max_recall_heatmap: str | None = Field(description="高召回候选热力图OSS对象键")
    diffuse_erythema_heatmap: str | None = Field(description="弥漫泛红热力图OSS对象键")
    focal_candidate_heatmap: str | None = Field(description="局灶候选热力图OSS对象键")
    max_recall_circles: str | None = Field(description="高召回候选圈选图OSS对象键")
    combined_candidate_circles: str | None = Field(description="融合候选圈选图OSS对象键")
    max_recall_debug: str | None = Field(description="高召回调试结果OSS对象键")
    max_recall_json: str | None = Field(description="高召回候选JSON OSS对象键")
    original_yolo_circles: str | None = Field(description="原始YOLO圈选图OSS对象键")
    original_unsupervised_circles: str | None = Field(description="原始无监督圈选图OSS对象键")
    original_combined_circles: str | None = Field(description="原始融合圈选图OSS对象键")
    acne_presence: AcnePresence | None = Field(description="是否检出疑似痤疮及中文提示")
    acne_count: int | float | None = Field(description="疑似痤疮候选数量")
    region_counts: list[AcneRegionCount] = Field(description="各面部分区疑似痤疮数量")
    grading_status: str | None = Field(description="Acne-LDS分级状态")
    grading_reason: str | None = Field(description="Acne-LDS未分级或异常原因")
    detector_status: str | None = Field(description="痤疮检测器执行状态")
    detector_reason: str | None = Field(description="痤疮检测器未执行或异常原因")
    input_mode: str | None = Field(description="痤疮输入图模式")
    detection_scope: str | None = Field(description="痤疮检测范围")
    quantification_result: AcneQuantificationResult = Field(
        alias="量化结果", description="痤疮旧版中文用户量化结果"
    )


class AcneV2RawResult(StrictContractModel):
    overlay: str = Field(description="痤疮正式原图圈选结果的OSS对象键")
    metrics: AcneV2Metrics = Field(description="痤疮v2中文精简量化指标")
    quality_score: int | float | None = Field(
        description="输入图片质量评分；当前无可信来源时为空"
    )
    quality_status: str | None = Field(
        description="输入图片质量状态；当前无可信来源时为空"
    )
    quality_flags: list[str] = Field(
        description="输入图片质量提示代码；当前无可信来源时为空数组"
    )
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


class WrinkleRegionMetric(StrictContractModel):
    region_key: str = Field(description="皱纹分区稳定英文标识")
    region_name: str = Field(description="皱纹分区中文名称")
    short_name: str = Field(description="皱纹分区英文缩写")
    relative_score: int | float = _metric_field("分区相对得分", "该分区相对本图最高问题分区的响应得分。", "0～100", 45.12)
    segment_count: int = _metric_field("分区纹路线段数", "当前分区最终保留的皱纹中心线段数量。", "条", 2)
    wrinkle_pixels: int = _metric_field("分区皱纹像素数", "当前分区最终皱纹中心线包含的像素数。", "像素", 124)
    mean_segment_length: int | float = _metric_field("平均线段长度", "当前分区皱纹中心线段的平均长度。", "像素", 62.0)
    max_segment_length: int | float = _metric_field("最大线段长度", "当前分区最长皱纹中心线段长度。", "像素", 100)
    density_per_10k: int | float = _metric_field("分区皱纹密度", "每1万分区有效像素对应的皱纹中心线像素数。", "像素/1万分区像素", 172.27)
    share_pct: int | float = _metric_field("分区皱纹占比", "当前分区皱纹像素占全脸皱纹像素的百分比。", "%", 13.14)
    area_px: int = _metric_field("分区有效面积", "当前皱纹分区参与统计的像素面积。", "像素", 7198)


class WrinkleRawResult(MedicalV2OptionalFields):
    analysis_face: str | None = Field(description="皱纹分析人脸图OSS对象键")
    preprocessed_face: str | None = Field(description="皱纹预处理人脸图OSS对象键")
    stage1_candidates: str | None = Field(description="皱纹第一阶段候选图OSS对象键")
    vote_heatmap: str | None = Field(description="皱纹多次推理投票热力图OSS对象键")
    stage2_overlay: str | None = Field(description="皱纹第二阶段结果图OSS对象键")
    stage2_centerline: str | None = Field(description="皱纹中心线结果图OSS对象键")
    face_filter_debug: str | None = Field(description="皱纹人脸过滤调试图OSS对象键")
    texture_reference: str | None = Field(description="皱纹纹理参考图OSS对象键")
    comparison: str | None = Field(description="皱纹结果对比图OSS对象键")
    region_overlay: str | None = Field(description="皱纹全脸分区结果图OSS对象键")
    region_tiles: str | None = Field(description="皱纹分区拼图OSS对象键")
    region_metrics_csv: str | None = Field(description="皱纹分区量化CSV OSS对象键")
    summary_json: str | None = Field(description="皱纹原始摘要JSON OSS对象键")
    region_metrics: list[WrinkleRegionMetric] = Field(description="皱纹旧版分区结构化指标")
    run_preset: str | None = Field(description="皱纹运行预设，正式模式保持balanced")
    device: str | None = Field(description="皱纹算法实际推理设备")
    successful_runs: int | None = Field(description="皱纹多视图成功推理次数")
    failed_runs: int | None = Field(description="皱纹多视图失败推理次数")
    region_analysis_status: str | None = Field(description="皱纹分区分析状态")
    scoring_input: ScoringInputV1 | None = Field(
        default=None, description=SCORING_INPUT_DESCRIPTION
    )


RawResultT = TypeVar("RawResultT", bound=BaseModel)
DebugInfoT = TypeVar("DebugInfoT", bound=BaseModel)


class SuccessEnvelope(StrictContractModel, Generic[RawResultT, DebugInfoT]):
    record_id: str = Field(description="业务任务追踪ID，同时用于OSS结果路径")
    status: Literal["success"] = Field(description="任务成功状态")
    schema_version: str = Field(description="契约仓定义的算法数据格式版本")
    meta_data: AlgorithmMetaData = Field(description="算法名称和后端接口版本")
    raw_result: RawResultT = Field(description="算法正式OSS产物和结构化量化结果")
    debug_info: DebugInfoT = Field(description="CSV、展示索引和运行耗时等调试信息")


class AcneV2MetaData(StrictContractModel):
    name: Literal["acne"] = Field(description="算法名称，固定为acne")
    version: Literal["2"] = Field(description="后端AlgorithmRegistry接口版本，固定为字符串2")


class AcneV2DebugInfo(StrictContractModel):
    execution_mode: Literal["formal_fast"] = Field(
        description="痤疮v2固定正式快速执行模式"
    )
    elapsed_seconds: int | float | None = Field(description="痤疮任务总耗时，单位秒")


class AcneV2SuccessEnvelope(StrictContractModel):
    record_id: str = Field(description="业务任务追踪ID，同时用于OSS结果路径")
    status: Literal["success"] = Field(description="任务成功状态")
    schema_version: Literal["2"] = Field(
        description="痤疮v2成功信封结构版本，固定为字符串2"
    )
    meta_data: AcneV2MetaData = Field(description="痤疮算法名称和v2后端接口版本")
    raw_result: AcneV2RawResult = Field(
        description="痤疮v2正式结果图、量化指标和质量信息"
    )
    debug_info: AcneV2DebugInfo = Field(description="痤疮v2执行模式和运行耗时")


class FailureEnvelope(StrictContractModel):
    record_id: str = Field(description="业务任务追踪ID")
    status: Literal["failed"] = Field(description="任务失败状态")
    schema_version: str | None = Field(default=None, description="可选的算法数据格式版本")
    error_message: str = Field(description="可用于日志和任务失败提示的错误信息")
    error_code: str | None = Field(default=None, description="可选稳定错误代码；当前痤疮任务使用")
    error_details: dict[str, Any] | None = Field(
        default=None, description="可选结构化诊断信息；当前皱纹任务在部分失败路径使用"
    )


ALGORITHM_RAW_RESULT_MODELS: dict[str, type[BaseModel]] = {
    "redness": RednessRawResult,
    "spots": SpotsRawResult,
    "brown": BrownRawResult,
    "texture": TextureRawResult,
    "pores": PoresRawResult,
    "purple": PurpleRawResult,
    "acne": AcneRawResult,
    "acne_v2": AcneV2RawResult,
    "wrinkle": WrinkleRawResult,
    "surface_gloss": SurfaceGlossRawResult,
    "vascular": VascularRawResult,
    "contour_firmness": ContourFirmnessRawResult,
}

PUBLIC_METRICS_MODELS: dict[str, type[BaseModel]] = {
    "redness": RednessPublicMetrics,
    "spots": SpotsPublicMetrics,
    "brown": BrownPublicMetrics,
    "texture": TexturePublicMetrics,
    "pores": PoresPublicMetrics,
    "purple": PurpleFullMetrics,
    "acne": AcneQuantificationResult,
    "acne_v2": AcneV2Metrics,
    "wrinkle": WrinkleRegionMetric,
    "surface_gloss": SurfaceGlossPublicMetrics,
    "vascular": VascularPublicMetrics,
    "contour_firmness": ContourFirmnessPublicMetrics,
}

_DEBUG_MODELS: dict[str, type[BaseModel]] = {
    **{name: DermavisionDebugInfo for name in (
        "redness", "spots", "brown", "texture", "pores", "purple",
        "surface_gloss", "vascular", "contour_firmness",
    )},
    "acne": AcneDebugInfo,
    "acne_v2": AcneV2DebugInfo,
    "wrinkle": WrinkleDebugInfo,
}

WORKER_ENVELOPE_TYPES: dict[str, Any] = {
    name: SuccessEnvelope[raw_model, _DEBUG_MODELS[name]] | FailureEnvelope
    for name, raw_model in ALGORITHM_RAW_RESULT_MODELS.items()
    if name != "acne_v2"
}
WORKER_ENVELOPE_TYPES["acne_v2"] = AcneV2SuccessEnvelope | FailureEnvelope

WORKER_ENVELOPE_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    name: TypeAdapter(envelope_type)
    for name, envelope_type in WORKER_ENVELOPE_TYPES.items()
}


def validate_worker_envelope(algorithm: str, payload: dict[str, Any]) -> dict[str, Any]:
    """严格校验并原样返回 Worker 字典。"""

    if algorithm not in WORKER_ENVELOPE_ADAPTERS:
        raise ValueError(f"没有登记Pydantic合同的算法: {algorithm}")
    WORKER_ENVELOPE_ADAPTERS[algorithm].validate_python(payload, strict=True)
    if payload.get("status") == "success":
        actual_name = (payload.get("meta_data") or {}).get("name")
        expected_name = "acne" if algorithm == "acne_v2" else algorithm
        if actual_name != expected_name:
            raise ValueError(f"meta_data.name不匹配: expected={expected_name} actual={actual_name}")
        actual_schema = payload.get("schema_version")
        expected_schema = SCHEMA_VERSIONS[algorithm]
        if actual_schema != expected_schema:
            raise ValueError(
                "schema_version不匹配: "
                f"expected={expected_schema} actual={actual_schema}"
            )
    return payload


def validate_failure_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """校验三个服务共用的失败信封并原样返回。"""

    TypeAdapter(FailureEnvelope).validate_python(payload, strict=True)
    return payload


def contract_json_schema(algorithm: str) -> dict[str, Any]:
    """返回单个算法成功/失败信封的JSON Schema。"""

    try:
        return WORKER_ENVELOPE_ADAPTERS[algorithm].json_schema()
    except KeyError as exc:
        raise ValueError(f"未知算法: {algorithm}") from exc


def contract_catalog() -> dict[str, dict[str, Any]]:
    """返回网页和文档使用的十二项合同目录。"""

    return {
        name: {
            "algorithm": name,
            "result_items": (
                ["uv_spots", "porphyrin"]
                if name == "purple"
                else ["acne"] if name == "acne_v2" else [name]
            ),
            "schema": contract_json_schema(name),
        }
        for name in ALGORITHM_RAW_RESULT_MODELS
    }


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _base_model_type(annotation: Any, value: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin in (types.UnionType, getattr(__import__("typing"), "Union")):
        for candidate in get_args(annotation):
            model = _base_model_type(candidate, value)
            if model is None:
                continue
            try:
                model.model_validate(value, strict=True)
            except Exception:
                continue
            return model
    return None


def _field_row(path: str, field: Any, value: Any) -> dict[str, Any]:
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
    description = field.description or ""
    fallback_title = description.split("；", 1)[0].split("。", 1)[0]
    return {
        "json_path": path,
        "title": field.title or fallback_title or str(field.alias or path.rsplit(".", 1)[-1]),
        "value": value,
        "value_type": _value_type(value),
        "unit": str(extra.get("unit", "—")),
        "description": description,
        "required": bool(field.is_required()),
    }


def _walk_documented_model(
    model_type: type[BaseModel],
    value: dict[str, Any],
    path: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_name, field in model_type.model_fields.items():
        public_name = str(field.alias or field_name)
        if public_name not in value:
            continue
        item = value[public_name]
        item_path = f"{path}.{public_name}"
        nested_model = _base_model_type(field.annotation, item)
        if nested_model is not None and isinstance(item, dict):
            rows.extend(_walk_documented_model(nested_model, item, item_path))
            continue
        origin = get_origin(field.annotation)
        args = get_args(field.annotation)
        if origin is dict and isinstance(item, dict) and len(args) == 2:
            child_model = _base_model_type(args[1], None)
            if child_model is not None and item:
                for key, child in item.items():
                    if isinstance(child, dict):
                        rows.extend(
                            _walk_documented_model(
                                child_model, child, f"{item_path}.{key}"
                            )
                        )
                continue
        if origin is list and isinstance(item, list) and args:
            child_model = _base_model_type(args[0], None)
            if child_model is not None and item:
                for index, child in enumerate(item):
                    if isinstance(child, dict):
                        rows.extend(
                            _walk_documented_model(
                                child_model, child, f"{item_path}[{index}]"
                            )
                        )
                continue
        rows.append(_field_row(item_path, field, item))
    return rows


def public_metric_documentation_rows(
    algorithm: str,
    raw_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """根据实际返回值生成前端量化指标的逐字段中文说明。"""

    if algorithm == "acne":
        value = raw_result.get("量化结果")
        path = "raw_result.量化结果"
        model_type: type[BaseModel] = AcneQuantificationResult
    elif algorithm == "wrinkle":
        values = raw_result.get("region_metrics")
        if not isinstance(values, list):
            return []
        rows: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            if isinstance(value, dict):
                rows.extend(
                    _walk_documented_model(
                        WrinkleRegionMetric,
                        value,
                        f"raw_result.region_metrics[{index}]",
                    )
                )
        return rows
    else:
        value = raw_result.get("metrics")
        path = "raw_result.metrics"
        model_type = PUBLIC_METRICS_MODELS[algorithm]
        if algorithm == "purple" and isinstance(value, dict):
            model_type = (
                PurpleFullMetrics if len(value) == 12 else PurplePartialMetrics
            )
    if not isinstance(value, dict):
        return []
    return _walk_documented_model(model_type, value, path)


def worker_field_documentation_rows(
    algorithm: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """只列出本次响应实际出现的信封、raw_result和debug_info字段。"""

    descriptions = {
        "record_id": "业务任务追踪ID，同时用于OSS结果路径。",
        "status": "任务状态，成功为success，失败为failed。",
        "schema_version": "算法成功信封结构版本。",
        "meta_data": "算法名称和后端接口版本。",
        "raw_result": "算法正式结果文件、量化指标和质量信息。",
        "debug_info": "CSV、展示索引和运行耗时等调试信息。",
    }
    rows: list[dict[str, Any]] = []
    for key in payload:
        value = payload[key]
        if key in {"raw_result", "debug_info", "meta_data"} and isinstance(value, dict):
            if key == "raw_result":
                model_type = ALGORITHM_RAW_RESULT_MODELS[algorithm]
            elif key == "debug_info":
                model_type = _DEBUG_MODELS[algorithm]
            else:
                model_type = AcneV2MetaData if algorithm == "acne_v2" else AlgorithmMetaData
            for field_name, field in model_type.model_fields.items():
                public_name = str(field.alias or field_name)
                if public_name in value:
                    rows.append(_field_row(f"{key}.{public_name}", field, value[public_name]))
            continue
        rows.append(
            {
                "json_path": key,
                "title": {"record_id": "任务追踪ID", "status": "任务状态"}.get(key, key),
                "value": value,
                "value_type": _value_type(value),
                "unit": "—",
                "description": descriptions.get(key, ""),
                "required": True,
            }
        )
    return rows
