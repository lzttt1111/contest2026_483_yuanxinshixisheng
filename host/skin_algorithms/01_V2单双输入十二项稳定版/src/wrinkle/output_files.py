"""Stable result keys and user-readable output filenames."""

from __future__ import annotations


# Result keys are part of the Worker contract and remain English for compatibility.
# Only the actual local/OSS filenames are localized.
OUTPUT_IMAGE_FILES: dict[str, str] = {
    "analysis_face": "00_标准化人脸分析图.png",
    "preprocessed_face": "00_标准化人脸展示图.png",
    "stage1_candidates": "01_第一阶段候选皱纹图.jpg",
    "vote_heatmap": "06_皱纹投票热力图.jpg",
    "stage2_overlay": "08_最终皱纹检测图.jpg",
    "stage2_centerline": "09_最终皱纹中心线.png",
    "face_filter_debug": "12_人脸区域过滤调试图.jpg",
    "texture_reference": "20_皮肤纹理参考图.png",
    "comparison": "21_皱纹检测阶段对比图.jpg",
    "region_overlay": "23_全脸皱纹分区结果图.jpg",
    "region_tiles": "24_皱纹分区汇总图.jpg",
}

OUTPUT_END_TEXTURE_FILENAME = OUTPUT_IMAGE_FILES["texture_reference"]
OUTPUT_END_REGION_OVERLAY_FILENAME = OUTPUT_IMAGE_FILES["region_overlay"]
