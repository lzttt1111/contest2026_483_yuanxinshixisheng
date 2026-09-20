# -*- coding: utf-8 -*-
"""VISIA-like visible skin-texture feature detector."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import peak_local_max

from src.engines.visia_regions import (
    REGION_LABELS,
    REGION_ORDER,
    VISIA_BOUNDARY_COLOR,
    assign_region,
    build_nasolabial_shadow_mask,
    build_moustache_feature_exclusion_mask,
    build_visia_regions,
    compact_region_counts,
    draw_region_boundaries,
    masked_gaussian,
    regional_positive_z,
    robust_unit_map,
)
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.utils.detailed_metrics import (
    build_medical_payload,
    embed_medical_metrics_v2,
    write_medical_metrics,
    try_write_medical_metrics_v2,
)
from src.utils.io_utils import cv_imwrite
from src.utils.analysis_debug_outputs import (
    capture_preprocess_debug,
    write_analysis_debug_outputs,
)
from src.utils.gpu_backend import get_cuda_backend


class Config:
    """Texture V2 defaults for a 1024×1024 aligned face.

    Texture V2 keeps the sign of the local luminance residual:

    - positive residual: locally brighter, raised-like appearance;
    - negative residual: locally darker, depressed-like appearance.

    These are appearance cues from a single RGB photograph, not physical
    height measurements.  Controlled lighting is still required for
    longitudinal comparison.
    """

    FINE_SIGMA = 1.2
    MEDIUM_SIGMA = 3.5
    BACKGROUND_SIGMA = 12.0
    # 黄/蓝两路分别寻找局部峰；7px 间距避免同一微纹理相邻像素重复计数。
    PEAK_MIN_DISTANCE = 7
    RAISED_PEAK_THRESHOLD = 0.60
    DEPRESSED_PEAK_THRESHOLD = 0.60
    MAX_FEATURES = 6000
    FEATURE_RADIUS = 1
    # 只用于纹理特征点的五官安全区，不改变底图颜色。
    FEATURE_EXCLUSION_MARGIN_PX = 30
    # 法令纹属于长条凹陷/皱纹，不应被拆成大量蓝色微纹理点。
    # 这里只屏蔽“解剖走廊内且有真实暗线证据”的窄区域，不删除整块口鼻皮肤。
    NASOLABIAL_CORRIDOR_RADIUS_PX = 25
    NASOLABIAL_SHADOW_MARGIN_PX = 5

    FINE_SIGNED_WEIGHT = 0.68
    MEDIUM_SIGNED_WEIGHT = 0.32
    SIGNED_RESPONSE_SCALE = 0.020
    LOCAL_STD_SUPPORT_WEIGHT = 0.22
    GRADIENT_SUPPORT_WEIGHT = 0.12
    SUPPORT_LIMIT = 1.8
    ROBUST_SIGMA_FLOOR = 0.055
    ROBUST_LOW_PERCENTILE = 1.0
    ROBUST_HIGH_PERCENTILE = 99.3

    # 极端高光和深阴影不作为表面起伏证据。OpenCV HSV 值域归一化到 0-1。
    HIGHLIGHT_VALUE = 0.97
    HIGHLIGHT_MAX_SATURATION = 0.18
    DEEP_SHADOW_VALUE = 0.12

    # OpenCV 使用 BGR：黄色表示凸起样，蓝色表示凹陷样。
    RAISED_COLOR = (0, 225, 255)
    DEPRESSED_COLOR = (255, 105, 35)
    REGION_COLOR = VISIA_BOUNDARY_COLOR


@dataclass
class TextureResult:
    texture_feature_count: int
    raised_like_count: int
    depressed_like_count: int
    texture_feature_locations: list[dict]
    region_distribution: dict[str, dict]
    texture_area_ratio: float
    mean_texture: float
    p90_texture: float
    texture_score_map: np.ndarray
    raised_score_map: np.ndarray
    depressed_score_map: np.ndarray
    feature_mask: np.ndarray
    raised_mask: np.ndarray
    depressed_mask: np.ndarray
    filtered_hair_mask: np.ndarray
    overlay: np.ndarray
    partial_face: bool
    quality_score: float
    quality_status: str
    quality_flags: list[str]

    def metrics(self) -> dict:
        return {
            "algorithm": "VISIA-like Visible Texture V2",
            "texture_feature_count": self.texture_feature_count,
            "raised_like_count": self.raised_like_count,
            "depressed_like_count": self.depressed_like_count,
            "texture_feature_locations": self.texture_feature_locations,
            "region_distribution": self.region_distribution,
            "texture_area_ratio": self.texture_area_ratio,
            "mean_texture": self.mean_texture,
            "p90_texture": self.p90_texture,
            "partial_face": self.partial_face,
            "quality_score": self.quality_score,
            "quality_status": self.quality_status,
            "quality_flags": self.quality_flags,
            "disclaimer": (
                "当前结果是普通RGB照片的可见皮肤纹理工程量化，"
                "不等价于VISIA常模百分位或医学诊断。"
            ),
        }


class TextureAnalyzer:
    @staticmethod
    def _texture_scores(
        image: np.ndarray,
        regions,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = regions.analysis_mask
        safe_mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        )
        l_channel = (
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            / 255.0
        )
        gaussian_maps = get_cuda_backend().masked_gaussian_batch(
            (l_channel, l_channel * l_channel),
            mask,
            (Config.FINE_SIGMA, Config.MEDIUM_SIGMA, Config.BACKGROUND_SIGMA),
        )
        fine = gaussian_maps[float(Config.FINE_SIGMA)][0]
        medium = gaussian_maps[float(Config.MEDIUM_SIGMA)][0]
        background = gaussian_maps[float(Config.BACKGROUND_SIGMA)][0]
        signed_response = (
            Config.FINE_SIGNED_WEIGHT * (fine - medium)
            + Config.MEDIUM_SIGNED_WEIGHT * (medium - background)
        )

        local_mean, local_second = gaussian_maps[float(Config.MEDIUM_SIGMA)]
        local_std = np.sqrt(np.maximum(local_second - local_mean * local_mean, 0.0))

        gx = cv2.Sobel(fine, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(fine, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        support = np.clip(
            Config.LOCAL_STD_SUPPORT_WEIGHT * local_std / 0.035
            + Config.GRADIENT_SUPPORT_WEIGHT * gradient / 0.10,
            0.0,
            Config.SUPPORT_LIMIT,
        )
        raised_raw = (
            np.maximum(signed_response, 0.0)
            / Config.SIGNED_RESPONSE_SCALE
            * (1.0 + support)
        )
        depressed_raw = (
            np.maximum(-signed_response, 0.0)
            / Config.SIGNED_RESPONSE_SCALE
            * (1.0 + support)
        )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        highlight = (
            (value >= Config.HIGHLIGHT_VALUE)
            & (saturation <= Config.HIGHLIGHT_MAX_SATURATION)
        )
        deep_shadow = value <= Config.DEEP_SHADOW_VALUE
        raised_raw[highlight] = 0.0
        depressed_raw[deep_shadow] = 0.0
        raised_raw[safe_mask == 0] = 0.0
        depressed_raw[safe_mask == 0] = 0.0

        raised_robust = regional_positive_z(
            raised_raw,
            regions,
            sigma_floor=Config.ROBUST_SIGMA_FLOOR,
        )
        depressed_robust = regional_positive_z(
            depressed_raw,
            regions,
            sigma_floor=Config.ROBUST_SIGMA_FLOOR,
        )
        raised_score = robust_unit_map(
            raised_robust,
            safe_mask,
            Config.ROBUST_LOW_PERCENTILE,
            Config.ROBUST_HIGH_PERCENTILE,
        )
        depressed_score = robust_unit_map(
            depressed_robust,
            safe_mask,
            Config.ROBUST_LOW_PERCENTILE,
            Config.ROBUST_HIGH_PERCENTILE,
        )
        raised_score[(safe_mask == 0) | (signed_response <= 0.0)] = 0.0
        depressed_score[(safe_mask == 0) | (signed_response >= 0.0)] = 0.0
        combined_score = np.maximum(raised_score, depressed_score)
        return raised_score, depressed_score, combined_score

    @staticmethod
    def _features(
        raised_score: np.ndarray,
        depressed_score: np.ndarray,
        regions,
        postprocess_exclusion: np.ndarray,
    ) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        locations: list[dict] = []
        raised_mask = np.zeros(raised_score.shape, dtype=np.uint8)
        depressed_mask = np.zeros(depressed_score.shape, dtype=np.uint8)
        filtered_hair_mask = np.zeros(depressed_score.shape, dtype=np.uint8)
        labels = (regions.analysis_mask > 0).astype(np.uint8)
        feature_specs = (
            (
                "raised_like",
                "凸起样",
                raised_score,
                Config.RAISED_PEAK_THRESHOLD,
                raised_mask,
            ),
            (
                "depressed_like",
                "凹陷样",
                depressed_score,
                Config.DEPRESSED_PEAK_THRESHOLD,
                depressed_mask,
            ),
        )
        for texture_type, type_label, score, threshold, output_mask in feature_specs:
            coordinates = peak_local_max(
                score,
                min_distance=Config.PEAK_MIN_DISTANCE,
                threshold_abs=threshold,
                labels=labels,
                exclude_border=False,
                num_peaks=Config.MAX_FEATURES,
            )
            for y, x in coordinates:
                if postprocess_exclusion[int(y), int(x)] > 0:
                    value = float(score[y, x])
                    radius = Config.FEATURE_RADIUS + int(value >= 0.82)
                    cv2.circle(
                        filtered_hair_mask,
                        (int(x), int(y)),
                        radius,
                        255,
                        -1,
                    )
                    continue
                region = assign_region(float(x), float(y), regions.regions)
                if region == "other":
                    continue
                value = float(score[y, x])
                radius = Config.FEATURE_RADIUS + int(value >= 0.82)
                support_radius = max(4, Config.PEAK_MIN_DISTANCE)
                y0 = max(0, int(y) - support_radius)
                y1 = min(score.shape[0], int(y) + support_radius + 1)
                x0 = max(0, int(x) - support_radius)
                x1 = min(score.shape[1], int(x) + support_radius + 1)
                support = (
                    score[y0:y1, x0:x1]
                    >= max(float(threshold), 0.62 * value)
                ).astype(np.uint8)
                _, support_labels = cv2.connectedComponents(support, connectivity=8)
                local_label = int(support_labels[int(y) - y0, int(x) - x0])
                measurement_area = (
                    int(np.count_nonzero(support_labels == local_label))
                    if local_label > 0 else 1
                )
                cv2.circle(output_mask, (int(x), int(y)), radius, 255, -1)
                locations.append(
                    {
                        "id": len(locations) + 1,
                        "centroid": [int(x), int(y)],
                        "region": region,
                        "texture_type": texture_type,
                        "texture_type_label": type_label,
                        "texture_score": round(value, 5),
                        "measurement_area": measurement_area,
                        "confidence": round(
                            float(np.clip(0.35 + 0.65 * value, 0, 1)),
                            4,
                        ),
                    }
                )
        locations.sort(
            key=lambda item: (
                int(item["centroid"][1]),
                int(item["centroid"][0]),
                str(item["texture_type"]),
            )
        )
        for feature_id, item in enumerate(locations, 1):
            item["id"] = feature_id
        feature_mask = cv2.bitwise_or(raised_mask, depressed_mask)
        return (
            locations,
            raised_mask,
            depressed_mask,
            feature_mask,
            filtered_hair_mask,
        )

    def detect_texture(self, preprocess_result: PreprocessResultV2) -> TextureResult:
        image = np.asarray(preprocess_result.analysis_image)
        regions = build_visia_regions(
            image,
            preprocess_result.skin_mask,
            preprocess_result.landmarks,
            preprocess_result.quality_flags,
            include_chin=True,
            mode="full",
            feature_margin_px=Config.FEATURE_EXCLUSION_MARGIN_PX,
        )
        raised_score, depressed_score, score = self._texture_scores(image, regions)
        nasolabial_shadow = build_nasolabial_shadow_mask(
            image,
            preprocess_result.landmarks,
            regions.analysis_mask,
            radius_px=Config.NASOLABIAL_CORRIDOR_RADIUS_PX,
            output_margin_px=Config.NASOLABIAL_SHADOW_MARGIN_PX,
            restrict_frangi_to_corridor=True,
        )
        moustache_exclusion = build_moustache_feature_exclusion_mask(
            image.shape[:2],
            preprocess_result.landmarks,
            regions.analysis_mask,
            margin_px=12,
        )
        postprocess_exclusion = cv2.bitwise_or(
            nasolabial_shadow,
            moustache_exclusion,
        )
        score = np.maximum(raised_score, depressed_score)
        (
            locations,
            raised_mask,
            depressed_mask,
            feature_mask,
            filtered_hair_mask,
        ) = self._features(
            raised_score,
            depressed_score,
            regions,
            postprocess_exclusion,
        )
        raised_locations = [
            item for item in locations if item["texture_type"] == "raised_like"
        ]
        depressed_locations = [
            item for item in locations if item["texture_type"] == "depressed_like"
        ]
        analysis_area = max(int(np.count_nonzero(regions.analysis_mask)), 1)
        area_ratio = float(np.count_nonzero(feature_mask) / analysis_area)
        valid = score[regions.analysis_mask > 0]
        mean_value = float(np.mean(valid)) if valid.size else 0.0
        p90 = float(np.percentile(valid, 90)) if valid.size else 0.0

        distribution = {}
        for name in REGION_ORDER:
            local = [item for item in locations if item["region"] == name]
            distribution[name] = {
                "label": REGION_LABELS[name],
                "count": len(local),
                "raised_like_count": sum(
                    item["texture_type"] == "raised_like" for item in local
                ),
                "depressed_like_count": sum(
                    item["texture_type"] == "depressed_like" for item in local
                ),
            }

        overlay = image.copy()
        for item in locations:
            x, y = item["centroid"]
            radius = Config.FEATURE_RADIUS + int(item["texture_score"] >= 0.82)
            color = (
                Config.RAISED_COLOR
                if item["texture_type"] == "raised_like"
                else Config.DEPRESSED_COLOR
            )
            cv2.circle(
                overlay,
                (x, y),
                radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
        overlay = draw_region_boundaries(
            overlay,
            regions.display_regions,
            Config.REGION_COLOR,
            thickness=3,
            closed_contour=regions.display_contour,
            separator_contour=regions.display_separator,
        )
        result = TextureResult(
            texture_feature_count=len(locations),
            raised_like_count=len(raised_locations),
            depressed_like_count=len(depressed_locations),
            texture_feature_locations=locations,
            region_distribution=distribution,
            texture_area_ratio=round(area_ratio, 8),
            mean_texture=round(mean_value, 6),
            p90_texture=round(p90, 6),
            texture_score_map=score,
            raised_score_map=raised_score,
            depressed_score_map=depressed_score,
            feature_mask=feature_mask,
            raised_mask=raised_mask,
            depressed_mask=depressed_mask,
            filtered_hair_mask=filtered_hair_mask,
            overlay=overlay,
            partial_face=regions.partial_face,
            quality_score=float(preprocess_result.quality_score),
            quality_status=str(preprocess_result.quality_status),
            quality_flags=list(preprocess_result.quality_flags),
        )
        result._medical_analysis_mask = regions.analysis_mask.copy()
        result._medical_landmarks = np.asarray(
            preprocess_result.landmarks, dtype=np.float32
        ).copy()
        result._common_debug_masks = capture_preprocess_debug(preprocess_result)
        return result

    @staticmethod
    def save_result(
        result: TextureResult,
        output_dir: str,
        base_name: str,
    ) -> dict[str, str]:
        sample_dir = os.path.join(output_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)
        paths = {
            "overlay": os.path.join(sample_dir, "01_纹理检测结果图.jpg"),
            "score": os.path.join(sample_dir, "02_纹理强度图.png"),
            "mask": os.path.join(sample_dir, "03_纹理特征Mask.png"),
            "csv": os.path.join(sample_dir, "纹理量化指标.csv"),
            "json": os.path.join(sample_dir, "纹理量化指标.json"),
            "medical_csv": os.path.join(sample_dir, "纹理医学量化指标.csv"),
            "medical_json": os.path.join(sample_dir, "纹理医学量化指标.json"),
            "medical_v2_csv": os.path.join(sample_dir, "纹理医学量化指标_V2.csv"),
            "medical_v2_json": os.path.join(sample_dir, "纹理医学量化指标_V2.json"),
        }
        cv_imwrite(paths["overlay"], result.overlay)
        score_visual = np.zeros((*result.texture_score_map.shape, 3), dtype=np.uint8)
        score_visual = np.maximum(
            score_visual,
            np.rint(
                result.raised_score_map[:, :, None]
                * np.asarray(Config.RAISED_COLOR, dtype=np.float32)[None, None, :]
            ).astype(np.uint8),
        )
        score_visual = np.maximum(
            score_visual,
            np.rint(
                result.depressed_score_map[:, :, None]
                * np.asarray(Config.DEPRESSED_COLOR, dtype=np.float32)[None, None, :]
            ).astype(np.uint8),
        )
        mask_visual = np.zeros_like(score_visual)
        mask_visual[result.raised_mask > 0] = Config.RAISED_COLOR
        mask_visual[result.depressed_mask > 0] = Config.DEPRESSED_COLOR
        cv_imwrite(paths["score"], score_visual)
        cv_imwrite(paths["mask"], mask_visual)
        write_analysis_debug_outputs(
            sample_dir,
            getattr(result, "_common_debug_masks", None),
            result.filtered_hair_mask,
        )
        instances = []
        for item in result.texture_feature_locations:
            instance = dict(item)
            instance["intensity"] = item.get("texture_score", 0.0)
            instance["morphology"] = item.get("texture_type_label", "二维纹理")
            instances.append(instance)
        detailed = build_medical_payload(
            project="texture",
            project_label="二维可见表面纹理代理",
            analysis_mask=result._medical_analysis_mask,
            landmarks=result._medical_landmarks,
            instances=instances,
            score_map=result.texture_score_map,
            instance_mask=result.feature_mask,
            quality_control={
                "图像质量状态": result.quality_status,
                "图像质量分数": result.quality_score,
                "图像质量提示": result.quality_flags,
                "是否局部脸": result.partial_face,
            },
            limitations=[
                "单张白光照片只能反映二维明暗起伏代理。",
                "不能输出真实高度、深度、体积或三维平整度。",
                "凸起样和凹陷样是图像外观分类，不是医学病变诊断。",
            ],
        )
        write_medical_metrics(
            paths["medical_json"], paths["medical_csv"], detailed, "texture"
        )
        medical_v2 = try_write_medical_metrics_v2(
            paths["medical_v2_json"], paths["medical_v2_csv"], detailed, "texture"
        )
        headers, values = compact_region_counts(
            result.texture_feature_locations, result.partial_face
        )
        if not result.partial_face:
            headers = [
                headers[0], "凸起样（黄）", "凹陷样（蓝）", *headers[1:]
            ]
            values = [
                values[0], result.raised_like_count, result.depressed_like_count,
                *values[1:],
            ]
        compact_metrics = {
            header: int(value) for header, value in zip(headers, values)
        }
        with open(paths["csv"], "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerow(values)
        with open(paths["json"], "w", encoding="utf-8") as handle:
            json.dump(compact_metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        embed_medical_metrics_v2(
            paths["json"],
            medical_v2,
            cleanup_json_paths=(paths["medical_json"], paths["medical_v2_json"]),
        )
        return paths

    def process_preprocess_result(
        self,
        preprocess_result: PreprocessResultV2,
        output_dir: str,
        source_name: str,
    ) -> str | None:
        if preprocess_result.quality_status == "REJECT":
            return None
        base_name = os.path.splitext(os.path.basename(source_name))[0]
        result = self.detect_texture(preprocess_result)
        return self.save_result(result, output_dir, base_name)["overlay"]


def detect_texture(preprocess_result: PreprocessResultV2) -> TextureResult:
    return TextureAnalyzer().detect_texture(preprocess_result)
