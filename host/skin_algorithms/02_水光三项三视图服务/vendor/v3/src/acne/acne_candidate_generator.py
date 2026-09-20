from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .artifact_policy import AcneArtifactPolicy, AcnePhaseTimings
from .detector_postprocess import draw_circles
from .image_io import write_image


@dataclass(frozen=True)
class MaxRecallConfig:
    hsv_redness_weight: float = 0.15
    lab_a_redness_weight: float = 0.20
    clahe_texture_weight: float = 0.20
    blackhat_weight: float = 0.20
    tophat_weight: float = 0.10
    local_contrast_weight: float = 0.15

    low_threshold: float = 0.28
    medium_threshold: float = 0.45
    high_threshold: float = 0.70
    local_mean_kernel: int = 41
    local_texture_kernel: int = 17
    diffuse_close_kernel: int = 31
    blackhat_kernel: int = 13
    tophat_kernel: int = 9
    line_kernel: int = 13
    dog_sigma_small: float = 1.0
    dog_sigma_large: float = 3.0

    min_area_px: int = 10
    min_area_ratio_of_skin: float = 0.000035
    max_area_ratio_of_skin: float = 0.012
    min_equivalent_diameter: float = 4.5
    max_equivalent_diameter: float = 90.0
    max_aspect_ratio: float = 4.5
    min_solidity: float = 0.15
    min_circularity: float = 0.025
    max_forbidden_overlap: float = 0.12
    forbidden_margin_px: int = 31
    skin_boundary_margin_px: int = 7

    diffuse_min_area_ratio_of_skin: float = 0.008
    diffuse_structure_ceiling: float = 0.42
    max_normality_suppression: float = 0.22
    highlight_suppression_strength: float = 0.85
    line_suppression_strength: float = 0.75
    repetitive_texture_suppression_strength: float = 0.45
    watershed_peak_ratio: float = 0.52
    watershed_max_peaks: int = 12

    combined_iou_threshold: float = 0.10
    combined_center_distance_px: float = 12.0
    explosion_candidate_count: int = 220
    explosion_candidates_per_10k_skin: float = 4.0
    explosion_heatmap_coverage_ratio: float = 0.25


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _valid_mask(skin_mask: np.ndarray, forbidden_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    skin = _resize_mask(skin_mask, shape)
    forbidden = _resize_mask(forbidden_mask, shape)
    return ((skin > 0) & (forbidden == 0)).astype(np.uint8)


def _masked_gaussian_mean(
    channel: np.ndarray,
    valid: np.ndarray,
    kernel_size: int | None = None,
    sigma: float = 0.0,
) -> np.ndarray:
    """Mask-aware normalized convolution; invalid pixels never enter the mean."""
    channel_f = channel.astype(np.float32)
    valid_f = valid.astype(np.float32)
    if kernel_size is None:
        ksize = (0, 0)
    else:
        kernel = _odd(kernel_size)
        ksize = (kernel, kernel)
    numerator = cv2.GaussianBlur(channel_f * valid_f, ksize, sigma)
    denominator = cv2.GaussianBlur(valid_f, ksize, sigma)
    fallback = float(np.median(channel_f[valid > 0])) if np.any(valid > 0) else float(np.median(channel_f))
    mean = np.full_like(channel_f, fallback, dtype=np.float32)
    np.divide(numerator, denominator, out=mean, where=denominator > 1e-5)
    return mean


def _masked_local_mad(channel: np.ndarray, valid: np.ndarray, kernel_size: int) -> tuple[np.ndarray, np.ndarray]:
    center = _masked_gaussian_mean(channel, valid, kernel_size=kernel_size)
    absolute_deviation = np.abs(channel.astype(np.float32) - center)
    mad = _masked_gaussian_mean(absolute_deviation, valid, kernel_size=kernel_size)
    return center, mad


def _normalize(score: np.ndarray, valid: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 98.0) -> np.ndarray:
    score = score.astype(np.float32)
    out = np.zeros_like(score, dtype=np.float32)
    values = score[valid > 0]
    if values.size == 0:
        return out
    lo = float(np.percentile(values, lower_percentile))
    hi = float(np.percentile(values, upper_percentile))
    if hi <= lo + 1e-6:
        return out
    out = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
    out[valid == 0] = 0.0
    return out.astype(np.float32)


def _amplitude_gate(response: np.ndarray, noise_floor: float, full_strength: float, valid: np.ndarray) -> np.ndarray:
    """Prevent per-image normalization from promoting tiny sensor/compression noise."""
    denominator = max(float(full_strength) - float(noise_floor), 1e-6)
    gate = np.clip((response.astype(np.float32) - float(noise_floor)) / denominator, 0.0, 1.0)
    gate[valid == 0] = 0.0
    return gate.astype(np.float32)


def _robust_positive_score(channel: np.ndarray, valid: np.ndarray, kernel_size: int) -> np.ndarray:
    center, mad = _masked_local_mad(channel, valid, kernel_size)
    robust_z = np.maximum(channel.astype(np.float32) - center, 0.0) / np.maximum(1.4826 * mad, 1e-3)
    score = np.clip(robust_z / 4.0, 0.0, 1.0)
    score[valid == 0] = 0.0
    return score.astype(np.float32)


def _robust_absolute_score(channel: np.ndarray, valid: np.ndarray, kernel_size: int) -> np.ndarray:
    center, mad = _masked_local_mad(channel, valid, kernel_size)
    robust_z = np.abs(channel.astype(np.float32) - center) / np.maximum(1.4826 * mad, 1e-3)
    score = np.clip(robust_z / 4.0, 0.0, 1.0)
    score[valid == 0] = 0.0
    return score.astype(np.float32)


def _fill_invalid(channel: np.ndarray, valid: np.ndarray, kernel_size: int) -> np.ndarray:
    local = _masked_gaussian_mean(channel, valid, kernel_size=kernel_size)
    filled = channel.astype(np.float32).copy()
    filled[valid == 0] = local[valid == 0]
    return np.clip(filled, 0, 255).astype(np.uint8)


def _tier(score: float, evidence_count: int, config: MaxRecallConfig) -> str:
    if score >= config.high_threshold and evidence_count >= 2:
        return "high"
    if score >= config.medium_threshold:
        return "medium"
    return "low"


def _colorize_score(score: np.ndarray) -> np.ndarray:
    image = np.clip(score * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def _score_overlay(image_bgr: np.ndarray, score: np.ndarray) -> np.ndarray:
    color = _colorize_score(score)
    active = score > 0.01
    overlay = image_bgr.copy()
    blended = cv2.addWeighted(image_bgr, 0.45, color, 0.55, 0)
    overlay[active] = blended[active]
    return overlay


def _channel_statistics(score: np.ndarray, valid: np.ndarray, active_threshold: float = 0.28) -> dict[str, float]:
    values = score[valid > 0].astype(np.float32)
    if values.size == 0:
        return {key: 0.0 for key in ("mean", "p90", "p95", "p99", "max", "active_pixel_ratio")}
    return {
        "mean": round(float(values.mean()), 6),
        "p90": round(float(np.percentile(values, 90)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "p99": round(float(np.percentile(values, 99)), 6),
        "max": round(float(values.max()), 6),
        "active_pixel_ratio": round(float(np.mean(values >= active_threshold)), 6),
    }


def _component_geometry(component: np.ndarray) -> dict[str, float | list[int]]:
    ys, xs = np.where(component > 0)
    if xs.size == 0:
        return {
            "bbox": [0, 0, 0, 0],
            "area": 0,
            "equivalent_diameter": 0.0,
            "aspect_ratio": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
        }
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    area = int(xs.size)
    bw, bh = x2 - x1, y2 - y1
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None and len(contour) >= 3 else 0.0
    return {
        "bbox": [x1, y1, x2, y2],
        "area": area,
        "equivalent_diameter": round(math.sqrt(4.0 * area / math.pi), 4),
        "aspect_ratio": round(max(bw / max(bh, 1), bh / max(bw, 1)), 4),
        "solidity": round(area / hull_area, 4) if hull_area > 0 else 0.0,
        "circularity": round(4.0 * math.pi * area / (perimeter * perimeter), 4) if perimeter > 0 else 0.0,
    }


def _component_masks(binary: np.ndarray) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return [(labels == label_id).astype(np.uint8) for label_id in range(1, count)]


def _split_component_with_watershed(
    component: np.ndarray,
    image_bgr: np.ndarray,
    min_area: int,
    config: MaxRecallConfig,
) -> list[np.ndarray]:
    area = int(component.sum())
    if area < max(4 * min_area, 36):
        return [component]
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak < 2.5:
        return [component]
    local_max = distance >= cv2.dilate(distance, np.ones((9, 9), np.uint8)) - 1e-6
    seeds = (local_max & (distance >= peak * config.watershed_peak_ratio) & (component > 0)).astype(np.uint8)
    seeds = cv2.dilate(seeds, np.ones((3, 3), np.uint8), iterations=1)
    seed_count, seed_labels = cv2.connectedComponents(seeds, connectivity=8)
    peak_count = seed_count - 1
    if peak_count < 2 or peak_count > config.watershed_max_peaks:
        return [component]

    markers = np.ones(component.shape, dtype=np.int32)
    markers[component > 0] = 0
    for seed_id in range(1, seed_count):
        markers[seed_labels == seed_id] = seed_id + 1
    cv2.watershed(image_bgr, markers)
    parts = [(markers == marker_id).astype(np.uint8) for marker_id in range(2, seed_count + 1)]
    parts = [part for part in parts if int(part.sum()) >= min_area]
    return parts if len(parts) >= 2 else [component]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = [float(value) for value in a["center_xy"]]
    bx, by = [float(value) for value in b["center_xy"]]
    return math.hypot(ax - bx, ay - by)


def _candidate_from_yolo(detection: dict[str, Any]) -> dict[str, Any]:
    item = dict(detection)
    x1, y1, x2, y2 = [float(value) for value in item["bbox_xyxy"]]
    item.setdefault("center_xy", [round((x1 + x2) / 2.0, 4), round((y1 + y2) / 2.0, 4)])
    item.setdefault("area", round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 4))
    item.setdefault("candidate_score", float(item.get("confidence", 0.0)))
    item["source_label_zh"] = item.get("label_zh", "疑似痤疮病灶")
    item["source"] = "yolo"
    item["label"] = "acne_candidate"
    item["label_zh"] = "疑似异常皮肤区域"
    return item


def combine_candidates(
    yolo_candidates: list[dict[str, Any]],
    unsupervised_focal_candidates: list[dict[str, Any]],
    config: MaxRecallConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    yolo = [_candidate_from_yolo(item) for item in yolo_candidates]
    unsupervised = [dict(item) for item in unsupervised_focal_candidates]
    used_unsupervised: set[int] = set()
    combined: list[dict[str, Any]] = []

    for yolo_item in yolo:
        best_index: int | None = None
        best_match_score = -1.0
        for index, unsupervised_item in enumerate(unsupervised):
            if index in used_unsupervised:
                continue
            overlap = _iou(yolo_item["bbox_xyxy"], unsupervised_item["bbox_xyxy"])
            distance = _center_distance(yolo_item, unsupervised_item)
            yolo_size = math.sqrt(max(float(yolo_item.get("area", 0.0)), 1.0))
            unsupervised_size = math.sqrt(max(float(unsupervised_item.get("area", 0.0)), 1.0))
            distance_limit = max(config.combined_center_distance_px, 0.55 * max(yolo_size, unsupervised_size))
            if overlap < config.combined_iou_threshold and distance > distance_limit:
                continue
            match_score = overlap + max(0.0, 1.0 - distance / max(distance_limit, 1.0))
            if match_score > best_match_score:
                best_index = index
                best_match_score = match_score
        if best_index is None:
            item = dict(yolo_item)
            item["combined_source"] = "yolo_only"
            item["source_members"] = ["yolo"]
            combined.append(item)
            continue
        used_unsupervised.add(best_index)
        matched = unsupervised[best_index]
        item = dict(yolo_item)
        item["combined_source"] = "yolo_and_unsupervised"
        item["source"] = "yolo_and_unsupervised"
        item["source_members"] = ["yolo", "unsupervised_max_recall"]
        item["unsupervised_candidate_id"] = matched.get("id")
        item["unsupervised_candidate_score"] = matched.get("final_candidate_score")
        item["evidence_channels"] = matched.get("evidence_channels", [])
        item["candidate_branch"] = matched.get("candidate_branch")
        combined.append(item)

    for index, unsupervised_item in enumerate(unsupervised):
        if index in used_unsupervised:
            continue
        item = dict(unsupervised_item)
        item["combined_source"] = "unsupervised_only"
        item["source_members"] = ["unsupervised_max_recall"]
        combined.append(item)

    combined.sort(key=lambda item: float(item.get("candidate_score", item.get("final_candidate_score", 0.0))), reverse=True)
    for index, item in enumerate(combined, start=1):
        item["combined_id"] = index
    source_counts = {"yolo_only": 0, "unsupervised_only": 0, "yolo_and_unsupervised": 0}
    for item in combined:
        source_counts[item["combined_source"]] += 1
    return combined, source_counts


class AcneCandidateGenerator:
    def __init__(self, config: MaxRecallConfig | None = None) -> None:
        self.config = config or MaxRecallConfig()

    def _compute_scores(self, image_bgr: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.config
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bgr_f = image_bgr.astype(np.float32) / 255.0

        hue = hsv[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        lab_a = lab[:, :, 1].astype(np.float32)
        lab_b = lab[:, :, 2].astype(np.float32)
        lightness = lab[:, :, 0]

        hue_distance = np.minimum(hue, 180.0 - hue)
        red_hue_response = np.clip(1.0 - hue_distance / 22.0, 0.0, 1.0)
        hsv_redness_signal = red_hue_response * saturation * np.sqrt(np.maximum(value, 0.0))
        hsv_redness = _robust_positive_score(hsv_redness_signal, valid, cfg.local_mean_kernel)
        hsv_center = _masked_gaussian_mean(hsv_redness_signal, valid, kernel_size=cfg.local_mean_kernel)
        hsv_delta = np.maximum(hsv_redness_signal - hsv_center, 0.0)
        hsv_redness *= _amplitude_gate(hsv_delta, 0.008, 0.080, valid)
        lab_a_redness = _robust_positive_score(lab_a, valid, cfg.local_mean_kernel)

        lab_a_center, lab_a_mad = _masked_local_mad(lab_a, valid, cfg.local_mean_kernel)
        lab_b_center, lab_b_mad = _masked_local_mad(lab_b, valid, cfg.local_mean_kernel)
        lab_a_delta = np.maximum(lab_a - lab_a_center, 0.0)
        lab_a_redness *= _amplitude_gate(lab_a_delta, 0.8, 8.0, valid)
        absolute_color_distance = np.sqrt((lab_a - lab_a_center) ** 2 + (lab_b - lab_b_center) ** 2)
        color_distance = np.sqrt(
            ((lab_a - lab_a_center) / np.maximum(1.4826 * lab_a_mad, 1.0)) ** 2
            + ((lab_b - lab_b_center) / np.maximum(1.4826 * lab_b_mad, 1.0)) ** 2
        )
        local_color_anomaly = np.clip(color_distance / 5.0, 0.0, 1.0).astype(np.float32)
        local_color_anomaly *= _amplitude_gate(absolute_color_distance, 1.2, 10.0, valid)
        local_color_anomaly[valid == 0] = 0.0

        filled_lightness = _fill_invalid(lightness, valid, cfg.local_mean_kernel)
        clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filled_lightness)
        clahe_texture = _robust_absolute_score(clahe_image, valid, cfg.local_texture_kernel)
        clahe_center = _masked_gaussian_mean(clahe_image, valid, kernel_size=cfg.local_texture_kernel)
        clahe_texture *= _amplitude_gate(np.abs(clahe_image.astype(np.float32) - clahe_center), 1.5, 12.0, valid)
        local_contrast = _robust_absolute_score(filled_lightness, valid, cfg.local_texture_kernel)
        lightness_center = _masked_gaussian_mean(filled_lightness, valid, kernel_size=cfg.local_texture_kernel)
        local_contrast *= _amplitude_gate(np.abs(filled_lightness.astype(np.float32) - lightness_center), 1.2, 10.0, valid)

        filled_gray = _fill_invalid(gray, valid, cfg.local_mean_kernel)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.blackhat_kernel), _odd(cfg.blackhat_kernel)))
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.tophat_kernel), _odd(cfg.tophat_kernel)))
        blackhat = cv2.morphologyEx(filled_gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        tophat = cv2.morphologyEx(filled_gray, cv2.MORPH_TOPHAT, tophat_kernel)
        blackhat_before = _normalize(blackhat, valid) * _amplitude_gate(blackhat, 1.0, 10.0, valid)
        tophat_before = _normalize(tophat, valid) * _amplitude_gate(tophat, 1.0, 10.0, valid)

        line_binary = (blackhat_before >= 0.35).astype(np.uint8)
        line_size = _odd(cfg.line_kernel)
        horizontal = cv2.morphologyEx(line_binary, cv2.MORPH_OPEN, np.ones((1, line_size), np.uint8))
        vertical = cv2.morphologyEx(line_binary, cv2.MORPH_OPEN, np.ones((line_size, 1), np.uint8))
        diagonal_kernel = np.eye(line_size, dtype=np.uint8)
        anti_diagonal_kernel = np.fliplr(diagonal_kernel)
        diagonal = cv2.morphologyEx(line_binary, cv2.MORPH_OPEN, diagonal_kernel)
        anti_diagonal = cv2.morphologyEx(line_binary, cv2.MORPH_OPEN, anti_diagonal_kernel)
        linear_structure = cv2.dilate(
            np.maximum.reduce([horizontal, vertical, diagonal, anti_diagonal]),
            np.ones((3, 3), np.uint8),
            iterations=1,
        ).astype(np.float32)
        linear_structure[valid == 0] = 0.0
        blackhat_after = blackhat_before * (1.0 - cfg.line_suppression_strength * linear_structure)

        rgb_min = np.min(bgr_f, axis=2)
        highlight_luma = np.clip((value - 0.65) / 0.28, 0.0, 1.0)
        highlight_low_saturation = np.clip((0.50 - saturation) / 0.50, 0.0, 1.0)
        highlight_rgb = np.clip((rgb_min - 0.45) / 0.45, 0.0, 1.0)
        specular_highlight = (highlight_luma * highlight_low_saturation * highlight_rgb).astype(np.float32)
        specular_highlight[valid == 0] = 0.0
        specular_highlight_mask = ((specular_highlight >= 0.14) & (valid > 0)).astype(np.float32)
        highlight_factor = 1.0 - cfg.highlight_suppression_strength * specular_highlight
        tophat_after = np.clip(tophat_before * highlight_factor, 0.0, 1.0)

        gray_f = filled_gray.astype(np.float32)
        dog_small = _masked_gaussian_mean(gray_f, valid, sigma=cfg.dog_sigma_small)
        dog_large = _masked_gaussian_mean(gray_f, valid, sigma=cfg.dog_sigma_large)
        blob_amplitude = np.abs(dog_small - dog_large)
        blob_before = _normalize(blob_amplitude, valid) * _amplitude_gate(blob_amplitude, 0.6, 5.0, valid)
        blob_after = np.clip(blob_before * highlight_factor, 0.0, 1.0)

        redness = np.maximum(hsv_redness, lab_a_redness)
        repetitive_seed = ((blackhat_after >= 0.32) & (local_contrast >= 0.25) & (valid > 0)).astype(np.float32)
        repetitive_density = _masked_gaussian_mean(repetitive_seed, valid, kernel_size=41)
        repetitive_texture = np.clip((repetitive_density - 0.03) / 0.18, 0.0, 1.0)
        normal_color_context = 1.0 - np.maximum(redness, local_color_anomaly)
        repetitive_texture *= np.clip(normal_color_context, 0.0, 1.0)
        repetitive_factor = 1.0 - cfg.repetitive_texture_suppression_strength * repetitive_texture
        blackhat_after = np.clip(blackhat_after * repetitive_factor, 0.0, 1.0)
        blob_after = np.clip(blob_after * (1.0 - 0.15 * repetitive_texture), 0.0, 1.0)

        raw_heatmap = (
            cfg.hsv_redness_weight * hsv_redness
            + cfg.lab_a_redness_weight * lab_a_redness
            + cfg.clahe_texture_weight * clahe_texture
            + cfg.blackhat_weight * blackhat_after
            + cfg.tophat_weight * tophat_after
            + cfg.local_contrast_weight * local_contrast
        )
        raw_heatmap = np.clip(raw_heatmap, 0.0, 1.0)

        texture_evidence = np.maximum.reduce([clahe_texture, blackhat_after, tophat_after, local_contrast])
        abnormal_evidence = np.maximum.reduce([redness, texture_evidence, blob_after, local_color_anomaly])
        normality = np.maximum(np.clip(1.0 - abnormal_evidence, 0.0, 1.0), repetitive_texture)
        normality_suppression = cfg.max_normality_suppression * normality
        final_heatmap = np.clip(raw_heatmap * (1.0 - normality_suppression), 0.0, 1.0)
        final_heatmap[valid == 0] = 0.0

        red_halo = _masked_gaussian_mean(redness, valid, kernel_size=17)
        branch_raw = {
            "inflammatory_branch": np.clip(0.58 * redness + 0.42 * np.maximum(blob_after, local_contrast), 0.0, 1.0),
            "dark_comedonal_branch": np.clip(0.58 * blackhat_after + 0.27 * local_contrast + 0.15 * blob_after, 0.0, 1.0),
            "bright_pustular_branch": np.clip(0.42 * tophat_after + 0.28 * red_halo + 0.30 * blob_after, 0.0, 1.0),
            "texture_branch": np.clip(
                0.52 * clahe_texture
                + 0.18 * local_contrast
                + 0.30 * np.maximum.reduce([redness, blob_after, local_color_anomaly]),
                0.0,
                1.0,
            ),
        }
        for score in branch_raw.values():
            score[valid == 0] = 0.0

        branch_masks = {
            "inflammatory_branch": (redness >= 0.38) & (np.maximum(blob_after, local_contrast) >= 0.30),
            "dark_comedonal_branch": (blackhat_after >= 0.45) & (local_contrast >= 0.30) & (linear_structure < 0.45),
            "bright_pustular_branch": (
                (tophat_after >= 0.50)
                & (red_halo >= 0.16)
                & (blob_after >= 0.38)
                & (specular_highlight < 0.35)
                & (linear_structure < 0.55)
            ),
            "texture_branch": (
                (clahe_texture >= 0.45)
                & (np.maximum.reduce([redness, blob_after, local_color_anomaly]) >= 0.32)
                & (linear_structure < 0.55)
            ),
        }
        branch_final: dict[str, np.ndarray] = {}
        branch_binary: dict[str, np.ndarray] = {}
        for name, raw_score in branch_raw.items():
            final_score = np.clip(raw_score * (1.0 - normality_suppression), 0.0, 1.0)
            final_score[valid == 0] = 0.0
            branch_final[name] = final_score
            branch_binary[name] = ((branch_masks[name]) & (final_score >= cfg.low_threshold) & (valid > 0)).astype(np.uint8)

        focal_raw = np.maximum.reduce(list(branch_raw.values()))
        focal_heatmap = np.maximum.reduce(list(branch_final.values()))
        focal_binary = np.maximum.reduce(list(branch_binary.values())).astype(np.uint8)

        return {
            "hsv_local_redness": hsv_redness,
            "lab_a_local_redness": lab_a_redness,
            "redness_only": redness,
            "local_color_anomaly": local_color_anomaly,
            "clahe_texture": clahe_texture,
            "texture_only": np.maximum(clahe_texture, local_contrast),
            "blackhat_before_suppression": blackhat_before,
            "blackhat_after_line_suppression": blackhat_after,
            "linear_dark_structure": linear_structure,
            "repetitive_normal_texture": repetitive_texture.astype(np.float32),
            "tophat_before_suppression": tophat_before,
            "tophat_after_suppression": tophat_after,
            "local_contrast": local_contrast,
            "blob_before_suppression": blob_before,
            "blob_score": blob_after,
            "specular_highlight_score": specular_highlight,
            "specular_highlight_mask": specular_highlight_mask,
            "normality_suppression": normality_suppression.astype(np.float32),
            "raw_candidate_heatmap": raw_heatmap,
            "acne_candidate_heatmap": final_heatmap.astype(np.float32),
            "focal_raw_heatmap": focal_raw.astype(np.float32),
            "focal_candidate_heatmap": focal_heatmap.astype(np.float32),
            "focal_binary": focal_binary,
            "branch_raw": branch_raw,
            "branch_final": branch_final,
            "branch_binary": branch_binary,
        }

    def _detect_diffuse_regions(
        self,
        scores: dict[str, Any],
        valid: np.ndarray,
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
        cfg = self.config
        redness = scores["redness_only"]
        structure = np.maximum.reduce([scores["blob_score"], scores["local_contrast"], scores["clahe_texture"]])
        smoothed_redness = _masked_gaussian_mean(redness, valid, kernel_size=31)
        valid_redness = smoothed_redness[valid > 0]
        adaptive_threshold = max(0.18, float(np.percentile(valid_redness, 95))) if valid_redness.size else 0.18
        red_binary = ((smoothed_redness >= adaptive_threshold) & (valid > 0)).astype(np.uint8)
        red_binary = cv2.morphologyEx(red_binary, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_odd(cfg.diffuse_close_kernel), _odd(cfg.diffuse_close_kernel)),
        )
        red_binary = cv2.morphologyEx(red_binary, cv2.MORPH_CLOSE, close_kernel)
        red_binary[valid == 0] = 0
        skin_area = max(int(valid.sum()), 1)
        minimum_area = max(64, int(round(skin_area * cfg.diffuse_min_area_ratio_of_skin)))
        regions: list[dict[str, Any]] = []
        diffuse_mask = np.zeros_like(valid, dtype=np.uint8)

        for component in _component_masks(red_binary):
            geometry = _component_geometry(component)
            area = int(geometry["area"])
            if area < minimum_area:
                continue
            active = component > 0
            redness_score = float(np.mean(redness[active]))
            structure_score = float(np.mean(structure[active]))
            area_ratio = area / skin_area
            if structure_score > cfg.diffuse_structure_ceiling and area_ratio < 0.025:
                continue
            diffuse_mask[active] = 1
            regions.append(
                {
                    "id": len(regions) + 1,
                    "bbox_xyxy": [float(value) for value in geometry["bbox"]],
                    "area": area,
                    "skin_coverage_ratio": round(area_ratio, 6),
                    "redness_score": round(redness_score, 4),
                    "structural_evidence_score": round(structure_score, 4),
                    "source": "unsupervised_diffuse_erythema",
                    "label": "diffuse_erythema",
                    "label_zh": "弥漫性泛红区域",
                    "countable": False,
                }
            )
        diffuse_heatmap = redness * diffuse_mask.astype(np.float32)
        return regions, diffuse_mask, diffuse_heatmap

    def _extract_focal_candidates(
        self,
        image_bgr: np.ndarray,
        scores: dict[str, Any],
        valid: np.ndarray,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        cfg = self.config
        skin_area = max(int(valid.sum()), 1)
        min_area = max(cfg.min_area_px, int(math.ceil(skin_area * cfg.min_area_ratio_of_skin)))
        max_area = int(math.floor(skin_area * cfg.max_area_ratio_of_skin))
        focal_binary = scores["focal_binary"].astype(np.uint8)
        focal_binary = cv2.morphologyEx(focal_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        focal_binary[valid == 0] = 0

        parts: list[np.ndarray] = []
        for component in _component_masks(focal_binary):
            parts.extend(_split_component_with_watershed(component, image_bgr, min_area, cfg))

        candidates: list[dict[str, Any]] = []
        accepted_mask = np.zeros_like(valid, dtype=np.uint8)
        channel_thresholds = {
            "hsv_local_redness": 0.34,
            "lab_a_local_redness": 0.34,
            "local_color_anomaly": 0.32,
            "clahe_texture": 0.34,
            "blackhat_after_line_suppression": 0.34,
            "tophat_after_suppression": 0.34,
            "local_contrast": 0.30,
            "blob_score": 0.30,
        }

        for component in parts:
            geometry = _component_geometry(component)
            area = int(geometry["area"])
            equivalent_diameter = float(geometry["equivalent_diameter"])
            aspect_ratio = float(geometry["aspect_ratio"])
            solidity = float(geometry["solidity"])
            circularity = float(geometry["circularity"])
            if area < min_area or area > max_area:
                continue
            if equivalent_diameter < cfg.min_equivalent_diameter or equivalent_diameter > cfg.max_equivalent_diameter:
                continue
            if aspect_ratio > cfg.max_aspect_ratio or solidity < cfg.min_solidity or circularity < cfg.min_circularity:
                continue
            active = component > 0
            if not np.any(active):
                continue

            branch_scores = {
                name: float(np.percentile(score[active], 90))
                for name, score in scores["branch_raw"].items()
            }
            active_branches = [name for name, score in branch_scores.items() if score >= cfg.low_threshold]
            if not active_branches:
                continue
            candidate_branch = max(branch_scores, key=branch_scores.get)
            evidence_values = {
                name: float(np.percentile(scores[name][active], 90))
                for name in channel_thresholds
            }
            evidence_channels = [
                name for name, threshold in channel_thresholds.items() if evidence_values[name] >= threshold
            ]
            evidence_groups = []
            if any(name in evidence_channels for name in ("hsv_local_redness", "lab_a_local_redness", "local_color_anomaly")):
                evidence_groups.append("color")
            if "blob_score" in evidence_channels:
                evidence_groups.append("blob")
            if any(name in evidence_channels for name in ("clahe_texture", "local_contrast")):
                evidence_groups.append("texture")
            if "blackhat_after_line_suppression" in evidence_channels:
                evidence_groups.append("dark_morphology")
            if "tophat_after_suppression" in evidence_channels:
                evidence_groups.append("bright_morphology")

            raw_score = float(np.percentile(scores["focal_raw_heatmap"][active], 95))
            normality_suppression = float(np.mean(scores["normality_suppression"][active]))
            final_score = float(np.percentile(scores["focal_candidate_heatmap"][active], 95))
            if final_score < cfg.low_threshold:
                continue
            highlight_suppression = float(
                np.mean(scores["tophat_before_suppression"][active] - scores["tophat_after_suppression"][active])
            )
            line_suppression = float(
                np.mean(scores["blackhat_before_suppression"][active] - scores["blackhat_after_line_suppression"][active])
            )
            x1, y1, x2, y2 = [int(value) for value in geometry["bbox"]]
            ys, xs = np.where(active)
            center_x = float(xs.mean())
            center_y = float(ys.mean())
            candidate = {
                "id": len(candidates) + 1,
                "bbox": [x1, y1, x2, y2],
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "area": area,
                "area_ratio_of_skin": round(area / skin_area, 7),
                "center_xy": [round(center_x, 4), round(center_y, 4)],
                "equivalent_diameter": equivalent_diameter,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "circularity": circularity,
                "redness_score": round(float(np.mean(scores["redness_only"][active])), 4),
                "texture_score": round(float(np.mean(scores["texture_only"][active])), 4),
                "blob_score": round(float(np.mean(scores["blob_score"][active])), 4),
                "raw_candidate_score": round(raw_score, 4),
                "normality_suppression": round(normality_suppression, 4),
                "highlight_suppression": round(highlight_suppression, 4),
                "line_suppression": round(line_suppression, 4),
                "final_candidate_score": round(final_score, 4),
                "candidate_score": round(final_score, 4),
                "tier": _tier(final_score, len(evidence_groups), cfg),
                "evidence_channels": evidence_channels,
                "evidence_groups": evidence_groups,
                "evidence_count": len(evidence_groups),
                "candidate_branch": candidate_branch,
                "active_branches": active_branches,
                "branch_scores": {name: round(value, 4) for name, value in branch_scores.items()},
                "source": "unsupervised_max_recall",
                "label": "acne_candidate",
                "label_zh": "疑似异常皮肤区域",
                "countable": True,
            }
            candidates.append(candidate)
            accepted_mask[active] = 1

        candidates.sort(key=lambda item: float(item["final_candidate_score"]), reverse=True)
        for index, candidate in enumerate(candidates, start=1):
            candidate["id"] = index
        return candidates, accepted_mask

    def generate(
        self,
        image_bgr: np.ndarray,
        skin_mask: np.ndarray,
        forbidden_mask: np.ndarray,
    ) -> dict[str, Any]:
        h, w = image_bgr.shape[:2]
        cfg = self.config
        skin = _resize_mask(skin_mask, (h, w))
        forbidden = _resize_mask(forbidden_mask, (h, w))
        skin_margin = max(0, int(cfg.skin_boundary_margin_px))
        if skin_margin > 0:
            skin = cv2.erode(skin, np.ones((_odd(skin_margin), _odd(skin_margin)), np.uint8), iterations=1)
        margin = max(0, int(cfg.forbidden_margin_px))
        if margin > 0:
            kernel_size = _odd(margin)
            forbidden = cv2.dilate(forbidden, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        valid = ((skin > 0) & (forbidden == 0)).astype(np.uint8)
        scores = self._compute_scores(image_bgr, valid)
        diffuse_regions, diffuse_mask, diffuse_heatmap = self._detect_diffuse_regions(scores, valid)
        focal_candidates, focal_mask = self._extract_focal_candidates(image_bgr, scores, valid)
        scores["diffuse_erythema_heatmap"] = diffuse_heatmap.astype(np.float32)
        scores["diffuse_erythema_mask"] = diffuse_mask.astype(np.float32)
        scores["accepted_focal_mask"] = focal_mask.astype(np.float32)

        tier_counts = {"high": 0, "medium": 0, "low": 0}
        for candidate in focal_candidates:
            tier_counts[candidate["tier"]] += 1

        valid_pixels = max(int(valid.sum()), 1)
        heatmap_coverage = float(np.mean(scores["acne_candidate_heatmap"][valid > 0] >= cfg.low_threshold))
        diffuse_coverage = float(diffuse_mask.sum() / valid_pixels)
        candidates_per_10k = len(focal_candidates) * 10000.0 / valid_pixels
        highlight_reduced = sum(float(item["highlight_suppression"]) > 0.01 for item in focal_candidates)
        normality_reduced = sum(float(item["normality_suppression"]) > 0.01 for item in focal_candidates)
        explosion_warning = bool(
            len(focal_candidates) > cfg.explosion_candidate_count
            or candidates_per_10k > cfg.explosion_candidates_per_10k_skin
            or heatmap_coverage > cfg.explosion_heatmap_coverage_ratio
        )

        statistic_channels = {
            name: score
            for name, score in scores.items()
            if isinstance(score, np.ndarray) and score.ndim == 2 and name not in {"focal_binary"}
        }
        channel_statistics = {
            name: _channel_statistics(score.astype(np.float32), valid)
            for name, score in statistic_channels.items()
        }
        return {
            "status": "ok",
            "config": asdict(cfg),
            "valid_pixels": valid_pixels,
            "scores": scores,
            "focal_acne_candidates": focal_candidates,
            "unsupervised_focal_candidates": focal_candidates,
            "diffuse_erythema_regions": diffuse_regions,
            "tier_counts": tier_counts,
            "channel_statistics": channel_statistics,
            "statistics": {
                "focal_candidate_count": len(focal_candidates),
                "diffuse_erythema_region_count": len(diffuse_regions),
                "heatmap_skin_coverage_ratio": round(heatmap_coverage, 6),
                "focal_candidates_per_10k_skin_pixels": round(candidates_per_10k, 6),
                "diffuse_erythema_coverage_ratio": round(diffuse_coverage, 6),
                "highlight_suppressed_candidate_count": int(highlight_reduced),
                "normality_suppressed_candidate_count": int(normality_reduced),
                "candidate_explosion_warning": explosion_warning,
            },
        }


def save_candidate_outputs(
    output_dir: str | Path,
    image_bgr: np.ndarray,
    candidate_result: dict[str, Any],
    yolo_candidates: list[dict[str, Any]],
    combined_candidates: list[dict[str, Any]],
    combined_source_counts: dict[str, int],
    artifact_policy: AcneArtifactPolicy = AcneArtifactPolicy.FORMAL,
    timings: AcnePhaseTimings | None = None,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    debug_dir = out_dir / "23_debug_channels"
    scores = candidate_result["scores"]
    focal_candidates = candidate_result["focal_acne_candidates"]
    paths: dict[str, Path] = {
        "candidate_heatmap": out_dir / "19_candidate_heatmap.jpg",
        "diffuse_erythema_heatmap": out_dir / "19a_diffuse_erythema_heatmap.jpg",
        "focal_candidate_heatmap": out_dir / "19b_focal_candidate_heatmap.jpg",
        "candidate_circle_overlay": out_dir / "20_candidate_circle_overlay.jpg",
        "combined_candidate_overlay": out_dir / "20a_combined_candidate_overlay.jpg",
        "candidate_score_debug": out_dir / "21_candidate_score_debug.jpg",
        "candidate_json": out_dir / "22_candidate.json",
        "debug_channels": debug_dir,
    }

    debug_paths: dict[str, str] = {}
    if artifact_policy.includes_debug_images:
        image_scope = timings.measure_image_encoding() if timings else nullcontext()
        with image_scope:
            debug_dir.mkdir(parents=True, exist_ok=True)
            write_image(paths["candidate_heatmap"], _score_overlay(image_bgr, scores["acne_candidate_heatmap"]))
            write_image(paths["diffuse_erythema_heatmap"], _score_overlay(image_bgr, scores["diffuse_erythema_heatmap"]))
            write_image(paths["focal_candidate_heatmap"], _score_overlay(image_bgr, scores["focal_candidate_heatmap"]))
            write_image(paths["candidate_circle_overlay"], draw_circles(image_bgr, focal_candidates))
            write_image(paths["combined_candidate_overlay"], draw_circles(image_bgr, combined_candidates))

            debug_channels = {
                "redness_only": scores["redness_only"],
                "texture_only": scores["texture_only"],
                "blob_only": scores["blob_score"],
                "blackhat_only": scores["blackhat_after_line_suppression"],
                "linear_dark_structure": scores["linear_dark_structure"],
                "repetitive_normal_texture": scores["repetitive_normal_texture"],
                "tophat_before_highlight_suppression": scores["tophat_before_suppression"],
                "tophat_after_highlight_suppression": scores["tophat_after_suppression"],
                "highlight_mask": scores["specular_highlight_mask"],
                "normality_suppression": scores["normality_suppression"],
                "diffuse_erythema": scores["diffuse_erythema_heatmap"],
                "focal_candidates": scores["focal_candidate_heatmap"],
                "final_heatmap": scores["acne_candidate_heatmap"],
            }
            tiles: list[np.ndarray] = []
            for name, score in debug_channels.items():
                color = _colorize_score(score)
                cv2.putText(color, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                path = debug_dir / f"{name}.jpg"
                write_image(path, color)
                debug_paths[name] = str(path)
                tiles.append(cv2.resize(color, (320, 320), interpolation=cv2.INTER_AREA))
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)]
            write_image(paths["candidate_score_debug"], np.vstack(rows))

    payload = {
        "status": "ok",
        "label": "acne_candidate",
        "label_zh": "疑似异常皮肤区域",
        "threshold_calibration_status": "initial_unvalidated_thresholds",
        "yolo_candidates": [_candidate_from_yolo(item) for item in yolo_candidates],
        "unsupervised_focal_candidates": focal_candidates,
        "focal_acne_candidates": focal_candidates,
        "diffuse_erythema_regions": candidate_result["diffuse_erythema_regions"],
        "combined_candidates": combined_candidates,
        "counts": {
            "yolo": len(yolo_candidates),
            "unsupervised_focal": len(focal_candidates),
            "diffuse_erythema_regions": len(candidate_result["diffuse_erythema_regions"]),
            "combined": len(combined_candidates),
        },
        "combined_source_counts": combined_source_counts,
        "tier_counts": candidate_result["tier_counts"],
        "statistics": candidate_result["statistics"],
        "channel_statistics": candidate_result["channel_statistics"],
        "config": candidate_result["config"],
        "debug_outputs": debug_paths,
        "notes": [
            "所有输出均为疑似异常皮肤区域，不是痤疮确诊结果。",
            "弥漫性泛红只报告区域和覆盖率，不进入局灶候选计数或 combined 候选池。",
            "YOLO、无监督局灶和 combined 分开保留；combined 仅表示候选池。",
            "high/medium/low 阈值是未经医学标注集校准的初始工程值。",
        ],
    }
    structured_scope = timings.measure_structured_persistence() if timings else nullcontext()
    with structured_scope:
        paths["candidate_json"].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {key: str(path) for key, path in paths.items() if path.exists()}
