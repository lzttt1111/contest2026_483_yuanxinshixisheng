# -*- coding: utf-8 -*-
"""DermaVision Vascular Structure Engine V2.1.

目标
----
从标准白光人脸图中检测“红色线状/分支状/网状血管样结构”，用于医生报告
05 血管样结构。V2.1 重点修复 V1/V2 的眼睑、眉毛、鼻孔、唇缘、头发、
发际线和胡须误检。

工程边界
--------
- 输出是图像中的血管样结构候选，不是真实血管数量或血管直径。
- 不做疾病诊断。
- 本地 DermaVision 中应优先调用 detect_from_preprocess_result()：它会直接
  使用 MediaPipe 对齐后的 landmarks，以及 Preprocessor V2 已生成的
  feature_exclusions / nostril_mask / hair_mask / facial_hair_mask。
- 云端独立演示缺少 mediapipe，因此使用 fallback 人脸/五官几何，只用于
  V2.1 算法自查；本地正式接入时会自动走 landmarks 精确屏蔽路径。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import csv
import json
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects


LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE_POLYGON = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 98, 327, 326]


@dataclass
class VascularInstance:
    id: int
    bbox: list[int]
    centroid: list[float]
    area_px: int
    skeleton_length_px: int
    mean_width_px: float
    p90_width_px: float
    elongation: float
    p50_redness: float
    p90_redness: float
    mean_line_response: float
    branch_points: int
    region: str


@dataclass
class VascularMetrics:
    algorithm: str
    mode: str
    input_width: int
    input_height: int
    face_pixels: int
    valid_skin_pixels: int
    exclusion_pixels: int
    vascular_count: int
    vascular_area_px: int
    vascular_area_ratio: float
    vascular_total_length_px: float
    vascular_line_density_per_10k_skin_px: float
    p50_width_px: float
    p90_width_px: float
    p50_redness: float
    p90_redness: float
    branch_point_count: int
    branch_point_density_per_10k_skin_px: float
    network_ratio: float
    region_distribution: dict[str, dict[str, float | int]]
    self_check: dict[str, Any]
    limitations: list[str]


@dataclass
class VascularResult:
    face_mask: np.ndarray
    exclusion_mask: np.ndarray
    valid_mask: np.ndarray
    redness_map: np.ndarray
    line_response: np.ndarray
    vascular_mask: np.ndarray
    skeleton_mask: np.ndarray
    branch_mask: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    instances: list[VascularInstance]
    metrics: VascularMetrics


class VascularStructureAnalyzer:
    ALGORITHM = 'VascularStructure-V2.1'

    # --------------------------
    # Project-mode extra safety margins (actual outward pixel radii)
    # --------------------------
    EYE_EXTRA_PX = 8
    EYEBROW_EXTRA_PX = 8
    LIP_EXTRA_PX = 8
    NOSTRIL_EXTRA_PX = 6
    HAIR_EXTRA_PX = 7
    FACIAL_HAIR_EXTRA_PX = 5
    FACE_BOUNDARY_ERODE_PX = 6

    # --------------------------
    # Detection thresholds
    # --------------------------
    MIN_COMPONENT_AREA = 10
    MAX_COMPONENT_AREA = 850
    MIN_SKELETON_LENGTH = 12
    MIN_ELONGATION = 2.15
    MAX_MEAN_WIDTH = 6.0
    MAX_P90_WIDTH = 8.0
    MIN_MEAN_LINE_RESPONSE = 0.045
    MIN_REDNESS_Z = 0.82
    RED_GATE_QUANTILE = 74.0
    LINE_GATE_QUANTILE = 94.5
    STRONG_LINE_QUANTILE = 98.3
    MIN_NETWORK_SKELETON = 18

    # fallback only
    FALLBACK_FEATURE_DILATE_PX = 7
    FALLBACK_TOP_HAIR_BAND_RATIO = 0.14

    def __init__(self) -> None:
        self.last_debug: dict[str, np.ndarray] = {}

    # ==================================================================
    # Public APIs
    # ==================================================================
    def detect_from_preprocess_result(self, preprocess_result: Any) -> VascularResult:
        """Formal DermaVision integration path.

        MediaPipe itself does not need to be imported here.  The upstream
        Preprocessor V2 already provides 478 aligned landmarks and exact masks.
        """
        image = np.asarray(preprocess_result.analysis_image)
        skin_mask = self._binary(getattr(preprocess_result, 'skin_mask'))
        landmarks = np.asarray(getattr(preprocess_result, 'landmarks', np.empty((0, 2))), dtype=np.float32)
        debug = getattr(preprocess_result, '_debug_masks', {}) or {}

        display_face = debug.get('display_face_mask')
        geometry = debug.get('face_geometry_mask')
        if isinstance(display_face, np.ndarray) and np.count_nonzero(display_face):
            face_mask = self._binary(display_face)
        elif isinstance(geometry, np.ndarray) and np.count_nonzero(geometry):
            face_mask = self._binary(geometry)
        else:
            # skin_mask already excludes five features; recover outer face only for display.
            face_mask = self._recover_outer_face_from_skin(skin_mask)

        exclusion = self._build_project_exclusion(image, face_mask, landmarks, debug)
        # Do not heal the upstream holes: skin_mask is the algorithm-safe base.
        valid = cv2.bitwise_and(skin_mask, cv2.bitwise_not(exclusion))
        valid = self._erode_px(valid, self.FACE_BOUNDARY_ERODE_PX)
        return self._detect_core(image, face_mask, valid, exclusion, landmarks, mode='project_mediapipe')

    def detect_from_bgr(self, image_bgr: np.ndarray) -> VascularResult:
        """Standalone fallback used in this cloud demo (no mediapipe dependency)."""
        face = self._fallback_face_mask(image_bgr)
        exclusion = self._fallback_exclusion(image_bgr, face)
        valid = cv2.bitwise_and(self._erode_px(face, self.FACE_BOUNDARY_ERODE_PX), cv2.bitwise_not(exclusion))
        return self._detect_core(image_bgr, face, valid, exclusion, None, mode='cloud_fallback')

    # ==================================================================
    # Core detection
    # ==================================================================
    def _detect_core(
        self,
        image: np.ndarray,
        face_mask: np.ndarray,
        valid_mask: np.ndarray,
        exclusion_mask: np.ndarray,
        landmarks: Optional[np.ndarray],
        mode: str,
    ) -> VascularResult:
        valid = self._binary(valid_mask)
        if np.count_nonzero(valid) < 200:
            return self._empty(image, face_mask, exclusion_mask, valid, mode, 'valid skin area too small')

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_star = lab[:, :, 1] - 128.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # Two independent redness cues.  Both are local-residual based so diffuse
        # background redness alone cannot create a vascular candidate.
        local_a = np.maximum(a_star - gaussian_filter(a_star, sigma=5.0), 0.0)
        log_rg = np.log((r + 1e-4) / (g + 1e-4))
        local_rg = np.maximum(log_rg - gaussian_filter(log_rg, sigma=5.0), 0.0)

        a_z = self._robust_positive_z(local_a, valid)
        rg_z = self._robust_positive_z(local_rg, valid)
        redness = np.clip(0.58 * a_z + 0.42 * rg_z, 0.0, 8.0).astype(np.float32)
        redness[valid == 0] = 0.0

        # Line detection is applied to the redness residual, not to gray/dark edges.
        red_unit = self._robust_unit(redness, valid, 20.0, 99.6)
        line_response = frangi(red_unit, sigmas=(1, 2, 3), black_ridges=False).astype(np.float32)
        line_response = np.nan_to_num(line_response, nan=0.0, posinf=0.0, neginf=0.0)
        line_response[valid == 0] = 0.0

        line_vals = line_response[valid > 0]
        red_vals = redness[valid > 0]
        line_thr = float(np.percentile(line_vals, self.LINE_GATE_QUANTILE))
        strong_line_thr = float(np.percentile(line_vals, self.STRONG_LINE_QUANTILE))
        red_thr = max(float(np.percentile(red_vals, self.RED_GATE_QUANTILE)), self.MIN_REDNESS_Z)

        # Soft line candidates require redness; exceptionally strong line response
        # still needs a lower-but-positive redness gate.
        candidate = (
            ((line_response >= line_thr) & (redness >= red_thr))
            | ((line_response >= strong_line_thr) & (redness >= max(0.55 * red_thr, self.MIN_REDNESS_Z)))
        ) & (valid > 0)

        candidate = remove_small_objects(candidate.astype(bool), min_size=self.MIN_COMPONENT_AREA)
        raw_candidate = candidate.astype(np.uint8) * 255
        final_mask, instances = self._filter_components(raw_candidate, redness, line_response, valid, landmarks)
        final_mask = cv2.bitwise_and(final_mask, valid)
        skeleton = skeletonize(final_mask > 0).astype(np.uint8) * 255
        branch_mask = self._branch_points(skeleton)

        # Self-check: final output must never re-enter exclusions or invalid skin.
        overlap_exclusion = int(np.count_nonzero((final_mask > 0) & (exclusion_mask > 0)))
        outside_valid = int(np.count_nonzero((final_mask > 0) & (valid == 0)))
        binary_ok = set(np.unique(final_mask)).issubset({0, 255}) and set(np.unique(skeleton)).issubset({0, 255})

        vascular_area = int(np.count_nonzero(final_mask))
        valid_area = int(np.count_nonzero(valid))
        skel_len = int(np.count_nonzero(skeleton))
        branch_count = int(np.count_nonzero(branch_mask))

        widths = self._skeleton_widths(final_mask, skeleton)
        red_skel = redness[skeleton > 0]
        network_skeleton = 0
        for inst in instances:
            if inst.branch_points > 0 and inst.skeleton_length_px >= self.MIN_NETWORK_SKELETON:
                network_skeleton += inst.skeleton_length_px

        region_distribution = self._aggregate_regions(instances)
        self_check = {
            'mask_binary': bool(binary_ok),
            'overlap_with_exclusion_pixels': overlap_exclusion,
            'outside_valid_mask_pixels': outside_valid,
            'candidate_count_nonnegative': len(instances) >= 0,
            'passed': bool(binary_ok and overlap_exclusion == 0 and outside_valid == 0),
        }

        metrics = VascularMetrics(
            algorithm=self.ALGORITHM,
            mode=mode,
            input_width=int(image.shape[1]),
            input_height=int(image.shape[0]),
            face_pixels=int(np.count_nonzero(face_mask)),
            valid_skin_pixels=valid_area,
            exclusion_pixels=int(np.count_nonzero(exclusion_mask)),
            vascular_count=len(instances),
            vascular_area_px=vascular_area,
            vascular_area_ratio=float(vascular_area / max(valid_area, 1)),
            vascular_total_length_px=float(skel_len),
            vascular_line_density_per_10k_skin_px=float(skel_len * 10000.0 / max(valid_area, 1)),
            p50_width_px=float(np.percentile(widths, 50)) if widths.size else 0.0,
            p90_width_px=float(np.percentile(widths, 90)) if widths.size else 0.0,
            p50_redness=float(np.percentile(red_skel, 50)) if red_skel.size else 0.0,
            p90_redness=float(np.percentile(red_skel, 90)) if red_skel.size else 0.0,
            branch_point_count=branch_count,
            branch_point_density_per_10k_skin_px=float(branch_count * 10000.0 / max(valid_area, 1)),
            network_ratio=float(network_skeleton / max(skel_len, 1)),
            region_distribution=region_distribution,
            self_check=self_check,
            limitations=[
                '仅表示白光图像中的红色线状血管样结构候选，不代表真实血管数量。',
                '像素宽度不是实际血管直径。',
                '普通RGB受白平衡、曝光、妆容和压缩影响。',
                'cloud_fallback结果只用于算法自查；本地正式运行应使用MediaPipe landmarks与Preprocessor V2屏蔽Mask。',
            ],
        )

        overlay = self._overlay(image, face_mask, final_mask, skeleton)
        heatmap = self._heatmap(red_unit, line_response, final_mask)
        self.last_debug = {
            'face_mask': face_mask,
            'exclusion_mask': exclusion_mask,
            'valid_mask': valid,
            'redness_map': redness,
            'line_response': line_response,
            'raw_candidate_mask': raw_candidate,
            'vascular_mask': final_mask,
            'skeleton_mask': skeleton,
            'branch_mask': branch_mask,
        }
        return VascularResult(
            face_mask=face_mask,
            exclusion_mask=exclusion_mask,
            valid_mask=valid,
            redness_map=redness,
            line_response=line_response,
            vascular_mask=final_mask,
            skeleton_mask=skeleton,
            branch_mask=branch_mask,
            heatmap=heatmap,
            overlay=overlay,
            instances=instances,
            metrics=metrics,
        )

    # ==================================================================
    # Project-mode exclusions (MediaPipe landmarks are already supplied upstream)
    # ==================================================================
    def _build_project_exclusion(
        self,
        image: np.ndarray,
        face_mask: np.ndarray,
        landmarks: np.ndarray,
        debug: dict[str, Any],
    ) -> np.ndarray:
        shape = face_mask.shape
        exclusion = np.zeros(shape, np.uint8)

        # Exact masks from Preprocessor V2.
        for key, extra in (
            ('feature_exclusions', 0),
            ('nostril_mask', self.NOSTRIL_EXTRA_PX),
            ('hair_mask', self.HAIR_EXTRA_PX),
            ('facial_hair_mask', self.FACIAL_HAIR_EXTRA_PX),
        ):
            arr = debug.get(key)
            if isinstance(arr, np.ndarray):
                arr = self._binary(arr)
                if extra > 0:
                    arr = self._dilate_px(arr, extra)
                exclusion = cv2.bitwise_or(exclusion, arr)

        # Rebuild individual anatomical masks from MediaPipe points so different
        # safety margins can be applied per feature instead of one global margin.
        if landmarks.ndim == 2 and landmarks.shape[0] > 400:
            eye = np.zeros(shape, np.uint8)
            brow = np.zeros(shape, np.uint8)
            lips = np.zeros(shape, np.uint8)
            self._fill_landmark_hull(eye, landmarks, LEFT_EYE)
            self._fill_landmark_hull(eye, landmarks, RIGHT_EYE)
            self._fill_landmark_hull(brow, landmarks, LEFT_EYEBROW)
            self._fill_landmark_hull(brow, landmarks, RIGHT_EYEBROW)
            self._fill_landmark_hull(lips, landmarks, LIPS)
            eye = self._dilate_px(eye, self.EYE_EXTRA_PX)
            brow = self._dilate_px(brow, self.EYEBROW_EXTRA_PX)
            lips = self._dilate_px(lips, self.LIP_EXTRA_PX)
            nostril = self._landmark_nostril_mask(image, landmarks)
            nostril = self._dilate_px(nostril, self.NOSTRIL_EXTRA_PX)
            for arr in (eye, brow, lips, nostril):
                exclusion = cv2.bitwise_or(exclusion, arr)

        # Hair evidence maps can remove thin strands even if binary hair_mask is conservative.
        hair_line = debug.get('hair_line_response')
        hair_votes = debug.get('hair_scale_vote')
        if isinstance(hair_line, np.ndarray):
            hv = hair_line.astype(np.float32)
            hv_valid = hv[face_mask > 0]
            if hv_valid.size:
                line_thr = float(np.percentile(hv_valid, 92.0))
                thin_hair = hv >= line_thr
                if isinstance(hair_votes, np.ndarray):
                    thin_hair &= np.asarray(hair_votes) >= 1
                thin_hair = thin_hair.astype(np.uint8) * 255
                thin_hair = self._dilate_px(thin_hair, 3)
                exclusion = cv2.bitwise_or(exclusion, thin_hair)

        return cv2.bitwise_and(self._binary(exclusion), face_mask)

    @staticmethod
    def _fill_landmark_hull(mask: np.ndarray, points: np.ndarray, indices: list[int]) -> None:
        valid = [i for i in indices if i < len(points)]
        if len(valid) < 3:
            return
        poly = cv2.convexHull(np.rint(points[valid, :2]).astype(np.int32))
        cv2.fillConvexPoly(mask, poly, 255)

    def _landmark_nostril_mask(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape[:2], np.uint8)
        if len(points) <= 327:
            return mask
        left_outer = points[98, :2].astype(np.float32)
        right_outer = points[327, :2].astype(np.float32)
        nose_base = points[2, :2].astype(np.float32)
        span = max(float(np.linalg.norm(right_outer - left_outer)), 1.0)
        rx = int(np.clip(round(0.18 * span), 10, 28))
        ry = int(np.clip(round(0.11 * span), 7, 18))
        centres = (0.68 * left_outer + 0.32 * nose_base, 0.68 * right_outer + 0.32 * nose_base)
        L = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        h, w = mask.shape
        for cf in centres:
            cx, cy = np.rint(cf).astype(int)
            cx = int(np.clip(cx, 0, w - 1)); cy = int(np.clip(cy, 0, h - 1))
            geom = np.zeros_like(mask)
            cv2.ellipse(geom, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            x0, x1 = max(0, cx - 2 * rx), min(w, cx + 2 * rx + 1)
            y0, y1 = max(0, cy - 2 * ry), min(h, cy + 2 * ry + 1)
            roi = L[y0:y1, x0:x1]
            if roi.size:
                med = float(np.median(roi)); mad = float(np.median(np.abs(roi.astype(np.float32) - med)))
                thr = med - max(6.0, 1.8 * 1.4826 * mad)
                dark = (roi.astype(np.float32) < thr).astype(np.uint8)
                dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                local = np.zeros_like(mask)
                local[y0:y1, x0:x1] = dark * 255
                geom = cv2.bitwise_or(geom, local)
            mask = cv2.bitwise_or(mask, geom)
        return self._dilate_px(mask, 2)

    # ==================================================================
    # Standalone fallback masks (only for cloud demo)
    # ==================================================================
    def _fallback_face_mask(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        grab = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.08), int(h * 0.03), int(w * 0.84), int(h * 0.93))
        bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, grab, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            person = np.where((grab == 1) | (grab == 3), 255, 0).astype(np.uint8)
        except Exception:
            person = np.ones((h, w), np.uint8) * 255
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        skin = ((cr > 132) & (cr < 182) & (cb > 76) & (cb < 136) & (y > 28)).astype(np.uint8) * 255
        face = cv2.bitwise_and(person, skin)
        face = cv2.morphologyEx(face, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        return self._largest_component(face)

    def _fallback_exclusion(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        ys, xs = np.where(face > 0)
        if ys.size == 0:
            return np.zeros_like(face)
        y0, y1 = int(ys.min()), int(ys.max()); x0, x1 = int(xs.min()), int(xs.max())
        fw, fh = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
        e = np.zeros_like(face)

        # conservative frontal geometry
        cx1, cx2 = int(x0 + 0.30 * fw), int(x0 + 0.70 * fw)
        eye_y, brow_y = int(y0 + 0.42 * fh), int(y0 + 0.32 * fh)
        cv2.ellipse(e, (cx1, eye_y), (int(0.18 * fw), int(0.075 * fh)), 0, 0, 360, 255, -1)
        cv2.ellipse(e, (cx2, eye_y), (int(0.18 * fw), int(0.075 * fh)), 0, 0, 360, 255, -1)
        cv2.ellipse(e, (cx1, brow_y), (int(0.20 * fw), int(0.065 * fh)), 0, 0, 360, 255, -1)
        cv2.ellipse(e, (cx2, brow_y), (int(0.20 * fw), int(0.065 * fh)), 0, 0, 360, 255, -1)

        # mouth + moustache zone
        mouth_y = int(y0 + 0.73 * fh)
        cv2.ellipse(e, (int((x0 + x1) / 2), mouth_y), (int(0.26 * fw), int(0.10 * fh)), 0, 0, 360, 255, -1)
        cv2.rectangle(e, (int(x0 + 0.26 * fw), int(y0 + 0.63 * fh)), (int(x0 + 0.74 * fw), int(y0 + 0.82 * fh)), 255, -1)

        # nostril region: keep nose wings visible but remove lower central apertures
        nose_y = int(y0 + 0.585 * fh)
        cv2.ellipse(e, (int(x0 + 0.42 * fw), nose_y), (int(0.06 * fw), int(0.04 * fh)), 0, 0, 360, 255, -1)
        cv2.ellipse(e, (int(x0 + 0.58 * fw), nose_y), (int(0.06 * fw), int(0.04 * fh)), 0, 0, 360, 255, -1)

        # top hairline band + photo-fixture edge
        cv2.rectangle(e, (x0, y0), (x1, int(y0 + self.FALLBACK_TOP_HAIR_BAND_RATIO * fh)), 255, -1)

        # Side-face: enlarge visible eye/brow guard.
        rel = float((xs.mean() - (x0 + x1) * 0.5) / fw)
        if abs(rel) > 0.035:
            side_x = int(x0 + fw * (0.72 if rel < 0 else 0.28))
            cv2.ellipse(e, (side_x, int(y0 + 0.39 * fh)), (int(0.23 * fw), int(0.13 * fh)), 0, 0, 360, 255, -1)
            cv2.ellipse(e, (side_x, int(y0 + 0.30 * fh)), (int(0.25 * fw), int(0.10 * fh)), 0, 0, 360, 255, -1)

        # Photometric dark hair/eyebrow-like lines near upper face are removed.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        upper = np.zeros_like(face); cv2.rectangle(upper, (x0, y0), (x1, int(y0 + 0.43 * fh)), 255, -1)
        upper = cv2.bitwise_and(upper, face)
        if np.count_nonzero(upper):
            L = gray.astype(np.float32)
            bg = cv2.GaussianBlur(L, (0, 0), 9.0)
            dark = np.maximum(bg - L, 0.0)
            vals = dark[upper > 0]
            if vals.size:
                thr = float(np.percentile(vals, 94.0))
                dark_lines = ((dark >= thr) & (upper > 0)).astype(np.uint8) * 255
                dark_lines = self._dilate_px(dark_lines, 2)
                e = cv2.bitwise_or(e, dark_lines)

        e = self._dilate_px(e, self.FALLBACK_FEATURE_DILATE_PX)
        return cv2.bitwise_and(e, face)

    # ==================================================================
    # Component/graph analysis
    # ==================================================================
    def _filter_components(
        self,
        candidate: np.ndarray,
        redness: np.ndarray,
        line_response: np.ndarray,
        valid: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> tuple[np.ndarray, list[VascularInstance]]:
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
        kept = np.zeros_like(candidate)
        instances: list[VascularInstance] = []
        if num <= 1:
            return kept, instances

        red_valid = redness[valid > 0]
        red_floor = max(float(np.percentile(red_valid, 60.0)), self.MIN_REDNESS_Z) if red_valid.size else self.MIN_REDNESS_Z

        for lab_id in range(1, num):
            x, y, w, h, area = [int(v) for v in stats[lab_id]]
            if area < self.MIN_COMPONENT_AREA or area > self.MAX_COMPONENT_AREA:
                continue
            comp = labels == lab_id
            skel = skeletonize(comp)
            skel_len = int(np.count_nonzero(skel))
            if skel_len < self.MIN_SKELETON_LENGTH:
                continue

            pts = np.column_stack(np.where(comp)[::-1]).astype(np.float32)
            if pts.shape[0] < 5:
                continue
            cov = np.cov(pts, rowvar=False)
            eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-6))
            elong = float(np.sqrt(eig[-1] / eig[0]))
            if elong < self.MIN_ELONGATION:
                continue

            local_mask = comp.astype(np.uint8) * 255
            dt = cv2.distanceTransform(local_mask, cv2.DIST_L2, 3)
            widths = dt[skel] * 2.0
            mean_width = float(widths.mean()) if widths.size else 0.0
            p90_width = float(np.percentile(widths, 90)) if widths.size else 0.0
            if mean_width > self.MAX_MEAN_WIDTH or p90_width > self.MAX_P90_WIDTH:
                continue

            red_values = redness[skel]
            mean_red = float(red_values.mean()) if red_values.size else 0.0
            p50_red = float(np.percentile(red_values, 50)) if red_values.size else 0.0
            p90_red = float(np.percentile(red_values, 90)) if red_values.size else 0.0
            mean_line = float(line_response[skel].mean()) if np.any(skel) else 0.0
            if mean_red < red_floor or mean_line < self.MIN_MEAN_LINE_RESPONSE:
                continue

            branch = self._branch_points(skel.astype(np.uint8) * 255)
            branch_count = int(np.count_nonzero(branch))
            cx, cy = [float(v) for v in centroids[lab_id]]
            region = self._assign_region(cx, cy, valid, landmarks)

            kept[comp] = 255
            instances.append(VascularInstance(
                id=len(instances) + 1,
                bbox=[x, y, w, h],
                centroid=[round(cx, 2), round(cy, 2)],
                area_px=area,
                skeleton_length_px=skel_len,
                mean_width_px=round(mean_width, 3),
                p90_width_px=round(p90_width, 3),
                elongation=round(elong, 3),
                p50_redness=round(p50_red, 4),
                p90_redness=round(p90_red, 4),
                mean_line_response=round(mean_line, 6),
                branch_points=branch_count,
                region=region,
            ))
        return kept, instances

    def _assign_region(self, cx: float, cy: float, valid: np.ndarray, landmarks: Optional[np.ndarray]) -> str:
        ys, xs = np.where(valid > 0)
        if ys.size == 0:
            return 'other'
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        fw, fh = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
        nx, ny = (cx - x0) / fw, (cy - y0) / fh
        side = 'left' if nx < 0.5 else 'right'
        if ny < 0.25:
            return 'forehead'
        if 0.36 <= nx <= 0.64 and 0.32 <= ny <= 0.66:
            return 'nose_alar_nasal_side'
        if ny < 0.48:
            return f'{side}_periocular_zygoma'
        if ny < 0.72:
            return f'{side}_cheek'
        return f'{side}_jaw'

    @staticmethod
    def _aggregate_regions(instances: list[VascularInstance]) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for inst in instances:
            row = out.setdefault(inst.region, {'count': 0, 'total_length_px': 0.0, 'total_area_px': 0})
            row['count'] = int(row['count']) + 1
            row['total_length_px'] = float(row['total_length_px']) + inst.skeleton_length_px
            row['total_area_px'] = int(row['total_area_px']) + inst.area_px
        return out

    @staticmethod
    def _branch_points(skeleton_u8: np.ndarray) -> np.ndarray:
        sk = (skeleton_u8 > 0).astype(np.uint8)
        neigh = cv2.filter2D(sk, cv2.CV_16S, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - sk.astype(np.int16)
        return ((sk > 0) & (neigh >= 3)).astype(np.uint8) * 255

    @staticmethod
    def _skeleton_widths(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        if not np.any(skeleton):
            return np.empty(0, np.float32)
        dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        return (2.0 * dt[skeleton > 0]).astype(np.float32)

    # ==================================================================
    # Visuals / utilities
    # ==================================================================
    def _overlay(self, image: np.ndarray, face_mask: np.ndarray, vascular: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        out = image.copy()
        contours, _ = cv2.findContours(self._binary(face_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        vp = vascular > 0
        if np.any(vp):
            out[vp] = np.clip(0.45 * out[vp].astype(np.float32) + 0.55 * np.array([0, 0, 255], np.float32), 0, 255).astype(np.uint8)
        out[skeleton > 0] = (0, 210, 255)
        return out

    @staticmethod
    def _heatmap(red_unit: np.ndarray, line_response: np.ndarray, vascular: np.ndarray) -> np.ndarray:
        score = np.clip(0.55 * red_unit + 0.45 * np.clip(line_response / max(float(line_response.max()), 1e-6), 0, 1), 0, 1)
        heat = cv2.applyColorMap(np.rint(score * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat[score <= 0] = 0
        heat[vascular > 0] = (0, 0, 255)
        return heat

    def _empty(self, image, face, exclusion, valid, mode, reason):
        z = np.zeros_like(valid)
        metrics = VascularMetrics(
            algorithm=self.ALGORITHM, mode=mode, input_width=image.shape[1], input_height=image.shape[0],
            face_pixels=int(np.count_nonzero(face)), valid_skin_pixels=int(np.count_nonzero(valid)), exclusion_pixels=int(np.count_nonzero(exclusion)),
            vascular_count=0, vascular_area_px=0, vascular_area_ratio=0.0, vascular_total_length_px=0.0, vascular_line_density_per_10k_skin_px=0.0,
            p50_width_px=0.0, p90_width_px=0.0, p50_redness=0.0, p90_redness=0.0, branch_point_count=0, branch_point_density_per_10k_skin_px=0.0,
            network_ratio=0.0, region_distribution={}, self_check={'passed': True, 'reason': reason}, limitations=['empty result'])
        return VascularResult(face, exclusion, valid, z.astype(np.float32), z.astype(np.float32), z, z, z, z, self._overlay(image, face, z, z), [], metrics)

    @staticmethod
    def _robust_positive_z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = values[mask > 0]
        if x.size < 100:
            return np.zeros_like(values, np.float32)
        med = float(np.median(x)); mad = float(np.median(np.abs(x - med))); sigma = max(1.4826 * mad, 1e-4)
        z = np.clip((values - med) / sigma, 0.0, 8.0).astype(np.float32)
        z[mask == 0] = 0.0
        return z

    @staticmethod
    def _robust_unit(values: np.ndarray, mask: np.ndarray, low: float, high: float) -> np.ndarray:
        x = values[mask > 0]
        if x.size < 32:
            return np.zeros_like(values, np.float32)
        lo, hi = np.percentile(x, [low, high]); hi = max(float(hi), float(lo) + 1e-6)
        out = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        out[mask == 0] = 0.0
        return out

    @staticmethod
    def _binary(mask: np.ndarray) -> np.ndarray:
        return (np.asarray(mask) > 0).astype(np.uint8) * 255

    @staticmethod
    def _dilate_px(mask: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return (mask > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.dilate((mask > 0).astype(np.uint8) * 255, k)

    @staticmethod
    def _erode_px(mask: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return (mask > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.erode((mask > 0).astype(np.uint8) * 255, k)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        b = (mask > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
        if n <= 1:
            return np.zeros_like(b)
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(b); out[labels == idx] = 255
        return out

    @staticmethod
    def _recover_outer_face_from_skin(skin: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours((skin > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(skin)
        if contours:
            cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
        return out


def _left_right_summary(region_distribution: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    """仅基于已有真实分区统计做确定性汇总，不补造隐藏分区。"""
    summary: dict[str, dict[str, float | int]] = {
        "left": {"count": 0, "total_length_px": 0.0, "total_area_px": 0},
        "right": {"count": 0, "total_length_px": 0.0, "total_area_px": 0},
    }
    for region, values in region_distribution.items():
        side = "left" if region.startswith("left_") else "right" if region.startswith("right_") else None
        if side is None:
            continue
        summary[side]["count"] = int(summary[side]["count"]) + int(values.get("count", 0))
        summary[side]["total_length_px"] = float(summary[side]["total_length_px"]) + float(values.get("total_length_px", 0.0))
        summary[side]["total_area_px"] = int(summary[side]["total_area_px"]) + int(values.get("total_area_px", 0))
    left = float(summary["left"]["total_length_px"])
    right = float(summary["right"]["total_length_px"])
    denominator = max(left + right, 1.0)
    return {
        **summary,
        "length_asymmetry_ratio": abs(left - right) / denominator,
        "dominant_side": "left" if left > right else "right" if right > left else "balanced",
    }


def vascular_candidate_metrics(result: VascularResult) -> dict[str, Any]:
    payload = asdict(result.metrics)
    payload.update({
        "capability_status": "candidate",
        "evidence_status": "candidate_algorithm",
        "score_status": "reference_not_ready",
        "vascular_line_density": result.metrics.vascular_line_density_per_10k_skin_px,
        "branch_point_density": result.metrics.branch_point_density_per_10k_skin_px,
        "left_right_summary": _left_right_summary(result.metrics.region_distribution),
    })
    available = [
        "vascular_count", "vascular_area_px", "vascular_area_ratio",
        "vascular_total_length_px", "vascular_line_density",
        "p50_width_px", "p90_width_px", "p50_redness", "p90_redness",
        "branch_point_count", "branch_point_density", "network_ratio",
        "region_distribution", "left_right_summary",
    ]
    payload["available_metrics"] = available
    payload["missing_metrics"] = []
    return payload


def save_vascular_result(result: VascularResult, output_dir: str | Path, input_name: str) -> dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {
        'overlay': out / '05_血管样结构检测结果图.jpg',
        'heatmap': out / '02_血管样结构热力图.jpg',
        'vascular_mask': out / '03_血管样结构Mask.png',
        'skeleton': out / '04_血管骨架Mask.png',
        'branch': out / '05_分支点Mask.png',
        'exclusion': out / '06_屏蔽区域Mask.png',
        'valid': out / '07_有效分析区域Mask.png',
        'metrics': out / '血管样结构量化指标.json',
        'csv': out / '血管样结构量化指标.csv',
        'medical_v2_csv': out / '血管样结构医学量化指标_V2.csv',
        'instances': out / '血管样结构实例.json',
    }
    cv2.imwrite(str(paths['overlay']), result.overlay)
    cv2.imwrite(str(paths['heatmap']), result.heatmap)
    cv2.imwrite(str(paths['vascular_mask']), result.vascular_mask)
    cv2.imwrite(str(paths['skeleton']), result.skeleton_mask)
    cv2.imwrite(str(paths['branch']), result.branch_mask)
    cv2.imwrite(str(paths['exclusion']), result.exclusion_mask)
    cv2.imwrite(str(paths['valid']), result.valid_mask)
    payload = vascular_candidate_metrics(result); payload['input_name'] = input_name
    paths['metrics'].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    scalar_rows = [
        (key, value, 'engineering_value')
        for key, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for csv_key in ('csv', 'medical_v2_csv'):
        with paths[csv_key].open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('指标名称', '检测结果', '单位'))
            writer.writerows(scalar_rows)
    paths['instances'].write_text(json.dumps([asdict(x) for x in result.instances], ensure_ascii=False, indent=2), encoding='utf-8')
    return {k: str(v) for k, v in paths.items()}


__all__ = [
    'VascularStructureAnalyzer', 'VascularResult', 'VascularMetrics',
    'VascularInstance', 'vascular_candidate_metrics', 'save_vascular_result',
]
