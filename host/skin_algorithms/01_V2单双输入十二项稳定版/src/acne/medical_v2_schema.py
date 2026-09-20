from __future__ import annotations

"""医学 V2 JSON 的稳定英文字段协议。

CSV 继续使用医生可读的中文表头；本模块只转换 JSON 对象的键名，不改
数值和中文说明文本。映射必须保持一一对应，便于 Worker 校验 CSV/JSON。
"""

from typing import Any


KEYS_ZH_TO_EN: dict[str, str] = {
    # 固定文档结构
    "检测项目": "project",
    "指标版本": "metrics_version",
    "评分状态": "scoring_status",
    "成像与单位说明": "imaging_and_units",
    "总体指标": "overall_metrics",
    "分区指标": "region_metrics",
    "左右比较": "left_right_comparison",
    "质量控制": "quality_control",
    "医学局限性": "medical_limitations",
    "核心指标": "core_metrics",
    "辅助指标": "auxiliary_metrics",
    "分析范围": "analysis_scope",
    "检测范围": "analysis_region",
    "评估状态": "evaluation_status",
    "有效皮肤面积（像素）": "valid_skin_area_px",
    "成像类型": "imaging_type",
    "面积单位": "area_unit",
    "强度单位": "intensity_unit",
    "强度说明": "intensity_note",
    "密度单位": "density_unit",
    "左右定义": "laterality_definition",
    "子项目": "subprojects",
    "说明": "description",
    "九项医学量化": "nine_analysis_metrics",
    "分区明细文件": "region_metrics_file",

    # 指标分组
    "范围与负担": "scope_and_burden",
    "范围与形态": "scope_and_morphology",
    "信号强度": "signal_intensity",
    "数量与密度": "count_and_density",
    "形态": "morphology",
    "分布": "distribution",
    "辅助统计": "auxiliary_statistics",
    "模型分级": "model_grading",

    # 通用数量、面积与强度
    "特征数量（个）": "feature_count",
    "单位面积密度（个/10万有效皮肤像素）": "feature_density_per_100k_skin_px",
    "特征总面积（像素）": "total_feature_area_px",
    "特征面积占比": "feature_area_ratio",
    "P50单体面积（像素）": "p50_instance_area_px",
    "P90单体面积（像素）": "p90_instance_area_px",
    "最大单体面积（像素）": "max_instance_area_px",
    "平均强度（0～1）": "mean_intensity",
    "P50强度（0～1）": "p50_intensity",
    "P90强度（0～1）": "p90_intensity",
    "P95强度（0～1）": "p95_intensity",
    "实例平均强度（0～1）": "mean_instance_intensity",
    "实例P50强度（0～1）": "p50_instance_intensity",
    "实例P90强度（0～1）": "p90_instance_intensity",
    "高强度区域面积占比": "high_intensity_area_ratio",
    "连续异常面积（像素）": "continuous_abnormal_area_px",
    "连续异常面积占比": "continuous_abnormal_area_ratio",
    "连续区域数量（个）": "continuous_region_count",
    "最大连续区域面积（像素）": "max_continuous_region_area_px",
    "主要集中区域": "primary_concentration_region",
    "画面左右面颊数量密度差异": "image_cheek_density_difference",
    "画面左右面颊面积占比差异": "image_cheek_area_ratio_difference",
    "画面左右面颊P90强度差异": "image_cheek_p90_intensity_difference",

    # 红区
    "弥漫红区面积（像素）": "diffuse_red_area_px",
    "弥漫红区面积占比": "diffuse_red_area_ratio",
    "红区连续性（0～1）": "redness_continuity",
    "红区边界渐变度（0～1）": "redness_boundary_gradient",
    "弥漫红区均匀度（0～1）": "diffuse_redness_uniformity",
    "红度综合负担": "redness_burden",
    "局灶红色实例数量（个）": "focal_red_feature_count",
    "局灶红色实例面积占比": "focal_red_feature_area_ratio",
    "中央面部实例集中比例": "central_face_feature_concentration",
    "实例过滤前候选数量": "pre_filter_candidate_count",
    "实例过滤数量": "filtered_instance_count",
    "实例过滤原因统计": "instance_filter_reason_counts",
    "颜色映射参考图": "color_transfer_reference",
    "颜色映射是否启用": "color_transfer_enabled",

    # 可见斑点
    "平均综合色差ΔE": "mean_delta_e",
    "P50综合色差ΔE": "p50_delta_e",
    "P90综合色差ΔE": "p90_delta_e",
    "平均工程置信度（0～1）": "mean_engineering_confidence",
    "点状斑点数量（个）": "punctate_spot_count",
    "点状斑点面积占比": "punctate_spot_area_ratio",
    "片状斑点数量（个）": "patch_spot_count",
    "片状斑点面积占比": "patch_spot_area_ratio",
    "融合样候选数量（个）": "confluent_candidate_count",
    "融合样候选面积占比": "confluent_candidate_area_ratio",
    "小型候选数量（个）": "small_candidate_count",
    "大型候选数量（个）": "large_candidate_count",
    "显著颜色异常数量（个）": "salient_color_anomaly_count",
    "边界清晰度中位数（0～1）": "median_boundary_sharpness",
    "颜色均匀性中位数（0～1）": "median_color_uniformity",
    "遮挡过滤前候选数量": "pre_occlusion_candidate_count",
    "毛发和五官过滤数量": "hair_feature_filtered_count",
    "鼻孔过滤数量": "nostril_filtered_count",
    "过滤原因统计": "filter_reason_counts",
    "拆分前大连通块数量": "pre_split_large_component_count",
    "拆分后实例数量": "post_split_instance_count",

    # 棕区
    "棕色斑数量（个）": "brown_spot_count",
    "棕色斑面积占比": "brown_spot_area_ratio",
    "连续棕色色素覆盖面积（像素）": "continuous_brown_coverage_area_px",
    "连续棕色色素覆盖占比": "continuous_brown_coverage_ratio",
    "棕色色素综合负担": "brown_pigmentation_burden",
    "点状棕色目标数量（个）": "punctate_brown_target_count",
    "片状棕色目标数量（个）": "patch_brown_target_count",
    "点状目标数量（个）": "punctate_target_count",
    "点状目标面积占比": "punctate_target_area_ratio",
    "片状目标数量（个）": "patch_target_count",
    "片状目标面积占比": "patch_target_area_ratio",

    # 纹理
    "凸起样数量（个）": "raised_like_count",
    "凹陷样数量（个）": "depressed_like_count",
    "凸起样比例": "raised_like_ratio",
    "凹陷样比例": "depressed_like_ratio",
    "凸起样面积占比": "raised_like_area_ratio",
    "凹陷样面积占比": "depressed_like_area_ratio",
    "聚集区域数量（个）": "cluster_region_count",
    "最大聚集区域特征数（个）": "max_cluster_feature_count",
    "最大聚集区域面积（像素）": "max_cluster_area_px",
    "主要问题类型": "primary_issue_type",

    # 毛孔
    "平均中心—环带视觉对比度（0～1）": "mean_center_ring_contrast",
    "P50中心—环带视觉对比度（0～1）": "p50_center_ring_contrast",
    "P90中心—环带视觉对比度（0～1）": "p90_center_ring_contrast",
    "P50等效直径（像素）": "p50_equivalent_diameter_px",
    "P90等效直径（像素）": "p90_equivalent_diameter_px",
    "P50圆度（0～1）": "p50_circularity",
    "P50长宽比": "p50_aspect_ratio",
    "低圆度毛孔比例": "low_circularity_pore_ratio",
    "椭圆形毛孔比例": "elliptical_pore_ratio",
    "拉长样毛孔比例": "elongated_pore_ratio",

    # 紫区
    "高强度目标比例": "high_intensity_target_ratio",
    "标准化UV紫外线色斑工程代理": "uv_spots",
    "标准化荧光UV紫质工程代理": "porphyrin",
    "紫区（紫外线色斑与紫质）": "purple",

    # 痤疮
    "疑似痤疮候选数量（个）": "suspected_acne_candidate_count",
    "候选框总面积（像素）": "candidate_box_total_area_px",
    "候选框面积占比": "candidate_box_area_ratio",
    "P50候选框面积（像素）": "p50_candidate_box_area_px",
    "P90候选框面积（像素）": "p90_candidate_box_area_px",
    "平均置信度（0～1）": "mean_confidence",
    "P90置信度（0～1）": "p90_confidence",
    "Acne-LDS评估状态": "acne_lds_evaluation_status",
    "Acne-LDS等级": "acne_lds_level",
    "输入类型": "input_type",
    "人脸数量": "face_count",
    "分级不可评估原因": "grading_unavailable_reason",

    # 皱纹
    "单位面积密度（皱纹像素/1万分区像素）": "wrinkle_pixel_density_per_10k_region_px",
    "纹路线段数量（段）": "wrinkle_segment_count",
    "皱纹中心线像素（像素）": "wrinkle_centerline_length_px",
    "皱纹面积占比": "wrinkle_area_ratio",
    "平均线段长度（像素）": "mean_segment_length_px",
    "最大线段长度（像素）": "max_segment_length_px",
    "总纹路长度（中心线像素）": "total_wrinkle_length_px",
    "P50线段长度（像素，分区均值代理）": "p50_segment_length_px_proxy",
    "P90线段长度（像素，分区均值代理）": "p90_segment_length_px_proxy",
    "平均可见宽度（像素，面积/中心线代理）": "mean_visible_width_px_proxy",
    "P90视觉对比度（0～1，响应代理）": "p90_visual_contrast_proxy",
    "纹路连续性（0～1）": "wrinkle_continuity",
    "相对响应（0～100）": "relative_response_0_100",
    "占全部皱纹像素比例": "share_of_total_wrinkle_pixels",
    "运行模式": "run_preset",
    "成功推理次数": "successful_run_count",
    "失败推理次数": "failed_run_count",
    "人脸过滤状态": "face_filter_status",
    "语义皮肤分割状态": "semantic_skin_status",

    # 质量控制
    "图像质量状态": "image_quality_status",
    "图像质量分数": "image_quality_score",
    "图像质量提示": "image_quality_flags",
    "是否局部脸": "is_partial_face",
    "局部脸": "partial_face",
}


KEYS_EN_TO_ZH = {value: key for key, value in KEYS_ZH_TO_EN.items()}
if len(KEYS_EN_TO_ZH) != len(KEYS_ZH_TO_EN):
    raise RuntimeError("医学 V2 中英字段映射存在重复英文键")


def _translate_keys(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            mapping.get(str(key), str(key)): _translate_keys(child, mapping)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_translate_keys(child, mapping) for child in value]
    return value


def _assert_no_chinese_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            if any("\u4e00" <= character <= "\u9fff" for character in name):
                raise ValueError(f"医学 V2 JSON 存在未映射中文字段: {path}.{name}")
            _assert_no_chinese_keys(child, f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_chinese_keys(child, f"{path}[{index}]")


def to_english_document(document: dict[str, Any]) -> dict[str, Any]:
    translated = _translate_keys(document, KEYS_ZH_TO_EN)
    _assert_no_chinese_keys(translated)
    return translated


def to_chinese_document(document: dict[str, Any]) -> dict[str, Any]:
    return _translate_keys(document, KEYS_EN_TO_ZH)


def is_english_document(document: Any) -> bool:
    return isinstance(document, dict) and "metrics_version" in document


INCREMENTAL_OVERALL_METRICS = frozenset({"feature_density_per_100k_skin_px", "candidate_box_total_area_px", "candidate_box_area_ratio", "p50_candidate_box_area_px", "p90_candidate_box_area_px", "mean_confidence", "p90_confidence", "primary_concentration_region"})


def _prune_metric_groups(groups: Any) -> dict[str, Any]:
    if not isinstance(groups, dict): return {}
    output: dict[str, Any] = {}
    for level_name in ("core_metrics", "auxiliary_metrics"):
        level = groups.get(level_name)
        if not isinstance(level, dict): continue
        dimensions: dict[str, Any] = {}
        for dimension, metrics in level.items():
            if not isinstance(metrics, dict): continue
            if dimension == "analysis_scope":
                identity = {key: metrics[key] for key in ("analysis_region", "evaluation_status", "valid_skin_area_px") if key in metrics}
                if identity: dimensions[dimension] = identity
                continue
            selected = {key: value for key, value in metrics.items() if key in INCREMENTAL_OVERALL_METRICS}
            if selected: dimensions[dimension] = selected
        if dimensions: output[level_name] = dimensions
    return output


def to_incremental_document(document: dict[str, Any], project: str | None = None) -> dict[str, Any]:
    public = document if is_english_document(document) else to_english_document(document)
    regions = []
    for row in public.get("region_metrics", []):
        if not isinstance(row, dict): continue
        selected = {key: row[key] for key in ("analysis_region", "evaluation_status", "valid_skin_area_px") if key in row}
        selected.update(_prune_metric_groups(row))
        metric_groups = {level: {dimension: metrics for dimension, metrics in selected.get(level, {}).items() if dimension != "analysis_scope"} for level in ("core_metrics", "auxiliary_metrics")}
        if any(groups for groups in metric_groups.values()): regions.append(selected)
    result = {"metrics_version": public.get("metrics_version", "medical_metrics_v2_20260728"), "scoring_status": public.get("scoring_status", "uncalibrated"), "units": public.get("imaging_and_units", {}), "overall_metrics": _prune_metric_groups(public.get("overall_metrics", {})), "region_metrics": regions, "left_right_comparison": {}, "medical_limitations": list(public.get("medical_limitations") or [])}
    _assert_no_chinese_keys(result)
    return result
