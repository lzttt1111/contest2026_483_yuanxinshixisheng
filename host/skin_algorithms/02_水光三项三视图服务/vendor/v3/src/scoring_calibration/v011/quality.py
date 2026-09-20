from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from src.preprocess.image_preprocessor import ImagePreprocessor


GateStatus = Literal["PASS", "REVIEW", "REJECT"]

REJECT_CODES = {
    "decode_failed", "no_face", "multiple_faces", "multi_face", "collage",
    "multi_panel", "heatmap", "pseudo_color", "result_image", "severe_mosaic",
}
REVIEW_CODES = {
    "strong_filter", "beauty_filter", "skin_smoothing", "unstable_lighting",
    "partial_occlusion", "partial_face", "low_contrast", "blur",
}


def evaluate_quality(observation: dict[str, Any]) -> dict[str, Any]:
    gate = observation.get("input_quality_gate") or observation.get("quality_gate")
    quality = observation.get("quality") or {}
    if not isinstance(gate, dict):
        # 九项算法自身的quality_status只能描述算法输入质量，不能替代
        # 评分常模要求的原图派生图/拼接图门禁。
        return {
            "gate_version": "input_quality_gate_v0.1.1_integrity",
            "status": "REVIEW",
            "reference_eligible": False,
            "formal_score_eligible": False,
            "reason_codes": ["missing_input_quality_gate"],
            "algorithm_quality": quality,
        }
    reasons = set(gate.get("reason_codes") or [])
    declared = str(gate.get("status") or "").upper()
    if reasons & REJECT_CODES or declared == "REJECT":
        status: GateStatus = "REJECT"
    elif reasons & REVIEW_CODES or declared == "REVIEW":
        status = "REVIEW"
    elif declared == "PASS" and observation.get("status", "success") in {"success", "partial_success"}:
        status = "PASS"
    else:
        status = "REVIEW"
        reasons.add("missing_quality_gate")
    return {
        "gate_version": "input_quality_gate_v0.1.1_integrity",
        "status": status,
        "reference_eligible": status == "PASS",
        "formal_score_eligible": status == "PASS",
        "reason_codes": sorted(reasons),
    }


def region_is_valid(region: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    valid_pixels = float(region.get("valid_pixels") or region.get("valid_area_px") or 0)
    mask_pixels = float(region.get("anatomical_mask_pixels") or region.get("mask_area_px") or 0)
    if valid_pixels < 2000:
        reasons.append("valid_pixels_lt_2000")
    if mask_pixels <= 0 or valid_pixels / mask_pixels < .5:
        reasons.append("coverage_lt_50_percent")
    return not reasons, reasons


def instance_metric_is_valid(metric_kind: str, instance_count: int) -> tuple[bool, str | None]:
    if metric_kind in {"p50", "p90", "morphology"} and instance_count < 5:
        return False, "instance_count_lt_5"
    if metric_kind in {"p90", "ratio", "morphology"} and instance_count < 10:
        return False, "instance_count_lt_10"
    if 10 <= instance_count < 20:
        return True, "low_stability_n_lt_20"
    return True, None


def _resize(image: np.ndarray, limit: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, limit / max(height, width))
    if scale >= 1:
        return image.copy()
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _image_derived_features(image: np.ndarray) -> dict[str, Any]:
    small = _resize(image)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    vertical = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2)
    horizontal = np.linalg.norm(lab[1:] - lab[:-1], axis=2)
    vp = np.percentile(vertical, 65, axis=0)
    hp = np.percentile(horizontal, 65, axis=1)
    vlo, vhi = int(len(vp) * .3), int(len(vp) * .7)
    hlo, hhi = int(len(hp) * .3), int(len(hp) * .7)
    centre_vertical = float(np.max(vp[vlo:vhi]) / max(np.median(vp), 1e-6))
    centre_horizontal = float(np.max(hp[hlo:hhi]) / max(np.median(hp), 1e-6))
    vertical_index = vlo + int(np.argmax(vp[vlo:vhi]))
    horizontal_index = hlo + int(np.argmax(hp[hlo:hhi]))
    vertical_median = max(float(np.median(vertical)), 1e-6)
    horizontal_median = max(float(np.median(horizontal)), 1e-6)
    vertical_coverage = float(
        np.mean(vertical[:, vertical_index] > max(4 * vertical_median, 10))
    )
    horizontal_coverage = float(
        np.mean(horizontal[horizontal_index, :] > max(4 * horizontal_median, 10))
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    # 黑色仪器背景中的低亮度色度噪声在 HSV 中会表现为高饱和度，
    # 但它并不是可见的伪彩热图。热图门禁只统计有可见亮度的高饱和像素。
    high_sat = (saturation > .62) & (value >= 48)
    hist = np.histogram(hue[high_sat], bins=18, range=(0, 180))[0] if np.any(high_sat) else np.zeros(18)
    occupied_hues = int(np.count_nonzero(hist > max(5, int(np.sum(hist) * .015))))
    canvas = (value < 48) | ((value > 247) & (hsv[:, :, 1] < 18))
    border = max(2, min(small.shape[:2]) // 40)
    border_canvas = np.concatenate((canvas[:border].ravel(), canvas[-border:].ravel(), canvas[:, :border].ravel(), canvas[:, -border:].ravel()))
    foreground = np.where(~canvas)
    bbox_ratio = 0.0
    if foreground[0].size:
        bbox_ratio = float(
            (foreground[1].max() - foreground[1].min() + 1)
            * (foreground[0].max() - foreground[0].min() + 1)
            / canvas.size
        )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dhash = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (dhash[:, 1:] > dhash[:, :-1]).ravel()
    phash = f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"
    return {
        "central_vertical_seam_score": round(centre_vertical, 6),
        "central_horizontal_seam_score": round(centre_horizontal, 6),
        "central_vertical_seam_location": round(
            float((vertical_index + 0.5) / max(vertical.shape[1], 1)),
            6,
        ),
        "central_horizontal_seam_location": round(
            float((horizontal_index + 0.5) / max(horizontal.shape[0], 1)),
            6,
        ),
        "central_vertical_seam_coverage": round(vertical_coverage, 6),
        "central_horizontal_seam_coverage": round(horizontal_coverage, 6),
        "high_saturation_ratio": round(float(np.mean(high_sat)), 6),
        "high_saturation_hue_bins": occupied_hues,
        "canvas_background_ratio": round(float(np.mean(canvas)), 6),
        "canvas_border_ratio": round(float(np.mean(border_canvas)), 6),
        "content_bbox_ratio": round(bbox_ratio, 6),
        "perceptual_hash": phash,
    }


def _has_collage_seam(features: dict[str, Any]) -> bool:
    vertical = (
        features["central_vertical_seam_score"] > 4.5
        and 0.43 <= features["central_vertical_seam_location"] <= 0.57
        and features["central_vertical_seam_coverage"] > 0.18
    )
    horizontal = (
        features["central_horizontal_seam_score"] > 4.5
        and 0.43 <= features["central_horizontal_seam_location"] <= 0.57
        and features["central_horizontal_seam_coverage"] > 0.18
    )
    return vertical or horizontal


def evaluate_image_path(path: str | Path, preprocessor: ImagePreprocessor | None = None) -> dict[str, Any]:
    """对进入常模的原图做保守门禁；只拒绝明确派生图或不可分析输入。"""
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError:
        payload = b""
    source_hash = hashlib.sha256(payload).hexdigest()
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
    if image is None:
        return {
            "gate_version": "input_quality_gate_v0.1.1_integrity", "status": "REJECT",
            "reference_eligible": False, "formal_score_eligible": False,
            "reason_codes": ["decode_failed"], "quality_features": {},
            "source_sha256": source_hash,
        }
    features = _image_derived_features(image)
    reasons: set[str] = set()
    pre = (preprocessor or ImagePreprocessor()).preprocess_image(image)
    flags = {str(flag).lower() for flag in pre.quality_flags}
    if "no_face" in flags:
        reasons.add("no_face")
    if "multiple_faces" in flags or "multi_face" in flags:
        reasons.add("multiple_faces")
    if _has_collage_seam(features):
        reasons.add("collage")
    if (
        features["canvas_background_ratio"] > .34
        and features["canvas_border_ratio"] > .72
        and features["content_bbox_ratio"] < .84
    ):
        reasons.add("multi_panel")
    if features["high_saturation_ratio"] > .34 and features["high_saturation_hue_bins"] >= 7:
        reasons.add("heatmap")
    status: GateStatus = "REJECT" if reasons & REJECT_CODES else "PASS"
    return {
        "gate_version": "input_quality_gate_v0.1.1_integrity",
        "status": status, "reference_eligible": status == "PASS",
        "formal_score_eligible": status == "PASS", "reason_codes": sorted(reasons),
        "quality_features": {**features, "preprocess_quality_status": pre.quality_status, "preprocess_quality_flags": sorted(pre.quality_flags)},
        "source_sha256": source_hash, "perceptual_hash": features["perceptual_hash"],
    }
