# -*- coding: utf-8 -*-
"""VISIA-like visible pore detector for standardized RGB face images."""

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
    """Pores V1 defaults for a 1024×1024 aligned face."""

    BLACKHAT_KERNEL_SIZES = (5, 9, 13)
    PEAK_MIN_DISTANCE = 4
    PEAK_THRESHOLD = 0.38
    MAX_FEATURES = 5000
    MIN_CENTER_RING_CONTRAST = 0.014
    MAX_PORE_RADIUS = 6
    # 只用于毛孔特征点的五官安全区，不改变底图颜色。
    FEATURE_EXCLUSION_MARGIN_PX = 30
    # 仅排除实际检测到的法令纹暗线，避免把线状阴影误认为一串圆形毛孔。
    NASOLABIAL_CORRIDOR_RADIUS_PX = 25
    NASOLABIAL_SHADOW_MARGIN_PX = 4

    HIGHLIGHT_VALUE = 0.96
    HIGHLIGHT_MAX_SATURATION = 0.22
    SHADOW_VALUE = 0.16

    FEATURE_COLOR = (145, 30, 105)
    REGION_COLOR = VISIA_BOUNDARY_COLOR


@dataclass
class PoresResult:
    pore_count: int
    pore_locations: list[dict]
    region_distribution: dict[str, dict]
    pore_area_ratio: float
    mean_pore_score: float
    median_pore_diameter_px: float
    pore_score_map: np.ndarray
    pore_mask: np.ndarray
    filtered_hair_mask: np.ndarray
    overlay: np.ndarray
    partial_face: bool
    quality_score: float
    quality_status: str
    quality_flags: list[str]

    def metrics(self) -> dict:
        return {
            "algorithm": "VISIA-like Visible Pores V1",
            "pore_count": self.pore_count,
            "pore_locations": self.pore_locations,
            "region_distribution": self.region_distribution,
            "pore_area_ratio": self.pore_area_ratio,
            "mean_pore_score": self.mean_pore_score,
            "median_pore_diameter_px": self.median_pore_diameter_px,
            "partial_face": self.partial_face,
            "quality_score": self.quality_score,
            "quality_status": self.quality_status,
            "quality_flags": self.quality_flags,
            "disclaimer": (
                "当前结果是普通RGB照片中的可见毛孔工程量化，"
                "不等价于VISIA常模百分位或医学诊断。"
            ),
        }


class PoresAnalyzer:
    @staticmethod
    def _pore_maps(image: np.ndarray, regions) -> tuple[np.ndarray, np.ndarray]:
        mask = regions.analysis_mask
        safe_mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )
        l_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        filled = l_channel.copy()
        valid_values = l_channel[mask > 0]
        fill_value = int(np.median(valid_values)) if valid_values.size else 128
        filled[mask == 0] = fill_value

        response_stack = get_cuda_backend().elliptical_blackhat_stack(
            filled.astype(np.float32) / 255.0,
            Config.BLACKHAT_KERNEL_SIZES,
        )
        raw = 0.70 * np.max(response_stack, axis=0) + 0.30 * np.mean(
            response_stack, axis=0
        )
        raw[safe_mask == 0] = 0.0
        robust = regional_positive_z(raw, regions, sigma_floor=0.005)
        score = robust_unit_map(robust, safe_mask, 1.0, 99.5)
        score[safe_mask == 0] = 0.0
        return score, response_stack

    @staticmethod
    def _center_ring_contrast(
        l_float: np.ndarray,
        x: int,
        y: int,
    ) -> float:
        height, width = l_float.shape
        radius = 5
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = l_float[y0:y1, x0:x1]
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return 0.0
        yy, xx = np.indices(patch.shape)
        cx, cy = x - x0, y - y0
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        center = patch[distance <= 1.5]
        ring = patch[(distance >= 3.0) & (distance <= 5.0)]
        if center.size == 0 or ring.size == 0:
            return 0.0
        return float(np.median(ring) - np.median(center))

    @staticmethod
    def _features(
        image: np.ndarray,
        score: np.ndarray,
        response_stack: np.ndarray,
        regions,
        postprocess_exclusion: np.ndarray,
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:
        coordinates = peak_local_max(
            score,
            min_distance=Config.PEAK_MIN_DISTANCE,
            threshold_abs=Config.PEAK_THRESHOLD,
            labels=(regions.analysis_mask > 0).astype(np.uint8),
            exclude_border=False,
            num_peaks=Config.MAX_FEATURES,
        )
        l_float = (
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            / 255.0
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        locations: list[dict] = []
        pore_mask = np.zeros(score.shape, dtype=np.uint8)
        filtered_hair_mask = np.zeros(score.shape, dtype=np.uint8)
        kernel_radii = [(size - 1) // 2 for size in Config.BLACKHAT_KERNEL_SIZES]

        for _, (y, x) in enumerate(coordinates, 1):
            if postprocess_exclusion[y, x] > 0:
                cv2.circle(filtered_hair_mask, (int(x), int(y)), 2, 255, -1)
                continue
            if value[y, x] < Config.SHADOW_VALUE:
                continue
            if (
                value[y, x] > Config.HIGHLIGHT_VALUE
                and saturation[y, x] < Config.HIGHLIGHT_MAX_SATURATION
            ):
                continue
            contrast = PoresAnalyzer._center_ring_contrast(l_float, int(x), int(y))
            if contrast < Config.MIN_CENTER_RING_CONTRAST:
                continue
            region = assign_region(float(x), float(y), regions.regions)
            if region == "other":
                continue
            scale_index = int(np.argmax(response_stack[:, y, x]))
            radius = int(np.clip(kernel_radii[scale_index] // 2, 2, Config.MAX_PORE_RADIUS))
            value_score = float(score[y, x])
            measure_radius = max(5, radius + 2)
            y0, y1 = max(0, int(y) - measure_radius), min(score.shape[0], int(y) + measure_radius + 1)
            x0, x1 = max(0, int(x) - measure_radius), min(score.shape[1], int(x) + measure_radius + 1)
            response_patch = response_stack[scale_index, y0:y1, x0:x1]
            peak_response = float(response_stack[scale_index, y, x])
            support = (
                response_patch >= max(0.006, 0.45 * peak_response)
            ).astype(np.uint8)
            _, support_labels = cv2.connectedComponents(support, connectivity=8)
            local_label = int(support_labels[int(y) - y0, int(x) - x0])
            component = (
                (support_labels == local_label).astype(np.uint8)
                if local_label > 0 else np.zeros_like(support)
            )
            measurement_area = max(1, int(np.count_nonzero(component)))
            measurement_circularity = 0.0
            measurement_aspect = 1.0
            contours, _ = cv2.findContours(
                component * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour_area = float(cv2.contourArea(contour))
                perimeter = float(cv2.arcLength(contour, True))
                if perimeter > 0:
                    measurement_circularity = float(np.clip(
                        4.0 * np.pi * contour_area / (perimeter * perimeter),
                        0.0, 1.0,
                    ))
                rotated_w, rotated_h = cv2.minAreaRect(contour)[1]
                if min(rotated_w, rotated_h) > 0:
                    measurement_aspect = float(
                        max(rotated_w, rotated_h) / min(rotated_w, rotated_h)
                    )
            cv2.circle(pore_mask, (int(x), int(y)), radius, 255, -1)
            locations.append(
                {
                    "id": len(locations) + 1,
                    "centroid": [int(x), int(y)],
                    "radius_px": radius,
                    "diameter_px": 2 * radius,
                    "region": region,
                    "center_ring_contrast": round(contrast, 6),
                    "pore_score": round(value_score, 5),
                    "measurement_area": measurement_area,
                    "measurement_circularity": round(measurement_circularity, 5),
                    "measurement_aspect_ratio": round(measurement_aspect, 5),
                    "confidence": round(
                        float(
                            np.clip(
                                0.35 + 0.45 * value_score + 0.20 * contrast / 0.04,
                                0,
                                1,
                            )
                        ),
                        4,
                    ),
                }
            )
        return locations, pore_mask, filtered_hair_mask

    def detect_pores(self, preprocess_result: PreprocessResultV2) -> PoresResult:
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
        score, response_stack = self._pore_maps(image, regions)
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
        locations, pore_mask, filtered_hair_mask = self._features(
            image,
            score,
            response_stack,
            regions,
            postprocess_exclusion,
        )
        analysis_area = max(int(np.count_nonzero(regions.analysis_mask)), 1)
        area_ratio = float(np.count_nonzero(pore_mask) / analysis_area)
        distribution = {}
        for name in REGION_ORDER:
            local = [item for item in locations if item["region"] == name]
            distribution[name] = {
                "label": REGION_LABELS[name],
                "count": len(local),
            }
        scores = [item["pore_score"] for item in locations]
        diameters = [item["diameter_px"] for item in locations]

        overlay = image.copy()
        for item in locations:
            x, y = item["centroid"]
            cv2.circle(
                overlay,
                (x, y),
                max(1, min(2, int(item["radius_px"]))),
                Config.FEATURE_COLOR,
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
        result = PoresResult(
            pore_count=len(locations),
            pore_locations=locations,
            region_distribution=distribution,
            pore_area_ratio=round(area_ratio, 8),
            mean_pore_score=round(float(np.mean(scores)) if scores else 0.0, 6),
            median_pore_diameter_px=round(
                float(np.median(diameters)) if diameters else 0.0, 3
            ),
            pore_score_map=score,
            pore_mask=pore_mask,
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
        result: PoresResult,
        output_dir: str,
        base_name: str,
    ) -> dict[str, str]:
        sample_dir = os.path.join(output_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)
        paths = {
            "overlay": os.path.join(sample_dir, "01_毛孔检测结果图.jpg"),
            "score": os.path.join(sample_dir, "02_毛孔强度图.png"),
            "mask": os.path.join(sample_dir, "03_毛孔实例Mask.png"),
            "csv": os.path.join(sample_dir, "毛孔量化指标.csv"),
            "json": os.path.join(sample_dir, "毛孔量化指标.json"),
            "medical_csv": os.path.join(sample_dir, "毛孔医学量化指标.csv"),
            "medical_json": os.path.join(sample_dir, "毛孔医学量化指标.json"),
            "medical_v2_csv": os.path.join(sample_dir, "毛孔医学量化指标_V2.csv"),
            "medical_v2_json": os.path.join(sample_dir, "毛孔医学量化指标_V2.json"),
        }
        cv_imwrite(paths["overlay"], result.overlay)
        cv_imwrite(paths["score"], np.rint(result.pore_score_map * 255).astype(np.uint8))
        cv_imwrite(paths["mask"], result.pore_mask)
        write_analysis_debug_outputs(
            sample_dir,
            getattr(result, "_common_debug_masks", None),
            result.filtered_hair_mask,
        )
        instances = []
        for item in result.pore_locations:
            instance = dict(item)
            instance["intensity"] = item.get("pore_score", 0.0)
            instance["morphology"] = "可见毛孔"
            instances.append(instance)
        detailed = build_medical_payload(
            project="pores",
            project_label="可见毛孔负担",
            analysis_mask=result._medical_analysis_mask,
            landmarks=result._medical_landmarks,
            instances=instances,
            score_map=result.pore_score_map,
            instance_mask=result.pore_mask,
            quality_control={
                "图像质量状态": result.quality_status,
                "图像质量分数": result.quality_score,
                "图像质量提示": result.quality_flags,
                "是否局部脸": result.partial_face,
            },
            limitations=[
                "仅量化标准化白光照片中可见的毛孔外观。",
                "不能测量毛孔深度、堵塞、角栓、皮脂量或真实毛囊总数。",
                "大毛孔阈值尚未经过医生标注与人群数据标定，故不输出大毛孔比例。",
            ],
        )
        write_medical_metrics(
            paths["medical_json"], paths["medical_csv"], detailed, "pores"
        )
        medical_v2 = try_write_medical_metrics_v2(
            paths["medical_v2_json"], paths["medical_v2_csv"], detailed, "pores"
        )
        headers, values = compact_region_counts(
            result.pore_locations, result.partial_face
        )
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
        result = self.detect_pores(preprocess_result)
        return self.save_result(result, output_dir, base_name)["overlay"]


def detect_pores(preprocess_result: PreprocessResultV2) -> PoresResult:
    return PoresAnalyzer().detect_pores(preprocess_result)
