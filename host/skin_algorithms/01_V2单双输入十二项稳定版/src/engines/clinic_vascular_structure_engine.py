# -*- coding: utf-8 -*-
"""Clinic four-light vascular-like linear-structure detector.

CP is the only measurement image.  RED is a CP-derived proposal/display image,
RGB is an optional visibility check, and PP is used only to reject surface
lines.  The result is an engineering observation, not a vessel diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

from src.engines.vascular_structure_engine import (
    LEFT_EYE,
    LEFT_EYEBROW,
    LIPS,
    RIGHT_EYE,
    RIGHT_EYEBROW,
    VascularStructureAnalyzer,
)
from src.preprocess.analysis_mask_bundle import COMMON_HAIR_SAFETY_MARGIN_PX


@dataclass(frozen=True)
class ClinicVascularResult:
    face_mask: np.ndarray
    exclusion_mask: np.ndarray
    valid_mask: np.ndarray
    hemoglobin_response: np.ndarray
    vesselness: np.ndarray
    vascular_mask: np.ndarray
    skeleton_mask: np.ndarray
    branch_mask: np.ndarray
    rgb_visibility_mask: np.ndarray | None
    overlay: np.ndarray
    metrics: dict[str, Any]
    instances: tuple[dict[str, Any], ...]


class ClinicVascularStructureAnalyzer:
    """VascularStructure-V3-Clinic, isolated from the single-RGB V2.1 path."""

    ALGORITHM = "VascularStructure-V3-Clinic"
    ANALYSIS_SPACE = "aligned_1024_cp"
    OUTER_FACE_GUARD_PX = 20
    FEATURE_EXCLUSION_MARGIN_PX = 3
    HAIR_GUARD_PX = COMMON_HAIR_SAFETY_MARGIN_PX
    EYE_GUARD_PX = 8
    EYE_CANDIDATE_GUARD_PX = 45
    EYEBROW_GUARD_PX = 8
    LIP_GUARD_PX = 8
    EXTREME_SATURATION_P90 = 180.0
    EXTREME_SATURATION_THRESHOLD = 160
    EXTREME_SATURATION_AREA_RATIO = 0.10
    MIN_VALID_FACE_RATIO = 0.18

    @classmethod
    def _valid_analysis_mask(
        cls,
        face: np.ndarray,
        exclusion: np.ndarray,
    ) -> np.ndarray:
        """Restrict vascular findings to skin safely inside the face contour."""

        inner_face = cls._erode(face, cls.OUTER_FACE_GUARD_PX)
        valid = cv2.bitwise_and(inner_face, cv2.bitwise_not(exclusion))
        return cls._erode(valid, cls.FEATURE_EXCLUSION_MARGIN_PX)

    def detect(
        self,
        cp: Any,
        *,
        rgb: Any | None = None,
        pp: Any | None = None,
        red_display: np.ndarray | None = None,
    ) -> ClinicVascularResult:
        image = np.asarray(cp.analysis_image)
        if image.shape[:2] != (1024, 1024):
            raise ValueError("Clinic vascular input must use aligned 1024 CP")
        face = self._face_mask(cp)
        exclusion = self._clinic_exclusion(image, face, cp)
        valid = self._valid_analysis_mask(face, exclusion)
        valid_pixels = int(np.count_nonzero(valid))
        face_pixels = int(np.count_nonzero(face))
        if valid_pixels < max(
            20_000,
            int(face_pixels * self.MIN_VALID_FACE_RATIO),
        ):
            return self._empty(
                image,
                face,
                exclusion,
                valid,
                red_display,
                "INSUFFICIENT_GEOMETRIC_FACE_DOMAIN",
            )
        consumer_rgb_proxy = rgb is cp
        if consumer_rgb_proxy and self._has_extreme_chroma_occlusion(image, valid):
            return self._empty(
                image,
                face,
                exclusion,
                valid,
                red_display,
                "EXTREME_CHROMA_OCCLUSION",
            )

        cp_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = (cp_rgb[:, :, index] for index in range(3))
        eps = 1.0 / 255.0
        ei1 = np.log((r + eps) / (g + eps))
        ei2 = np.log((np.sqrt(np.maximum(r * b, 0.0)) + eps) / (g + eps))
        local1 = np.maximum(ei1 - gaussian_filter(ei1, 5.0), 0.0)
        local2 = np.maximum(ei2 - gaussian_filter(ei2, 5.0), 0.0)
        broad1 = np.maximum(ei1 - gaussian_filter(ei1, 11.0), 0.0)
        hemoglobin = np.clip(
            0.45 * self._robust_z(local1, valid)
            + 0.30 * self._robust_z(local2, valid)
            + 0.25 * self._robust_z(broad1, valid),
            0.0,
            8.0,
        ).astype(np.float32)
        hemoglobin[valid == 0] = 0.0

        cp_unit = self._robust_unit(hemoglobin, valid, 15.0, 99.5)
        cp_line = frangi(cp_unit, sigmas=(1, 2, 3, 4, 5), black_ridges=False)
        cp_line = np.nan_to_num(cp_line, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cp_line = self._robust_unit(cp_line, valid, 20.0, 99.8)

        red_aligned = self._aligned_optional(red_display, image.shape)
        if red_aligned is not None:
            red_gray = cv2.cvtColor(red_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            red_dark = np.maximum(gaussian_filter(red_gray, 5.0) - red_gray, 0.0)
            red_line = frangi(
                self._robust_unit(red_dark, valid, 35.0, 99.7),
                sigmas=(1, 2, 3, 4, 5),
                black_ridges=False,
            )
            red_line = self._robust_unit(np.nan_to_num(red_line).astype(np.float32), valid, 25.0, 99.8)
        else:
            red_line = np.zeros_like(cp_line)

        vesselness = np.clip(0.72 * cp_line + 0.28 * red_line, 0.0, 1.0).astype(np.float32)
        values = vesselness[valid > 0]
        line_threshold = max(float(np.percentile(values, 94.0)), 0.08)

        # Local colour-drop support rejects achromatic dark creases and hair.
        local_r = gaussian_filter(r, 4.0) - r
        local_g = gaussian_filter(g, 4.0) - g
        local_b = gaussian_filter(b, 4.0) - b
        colour_drop = self._chromatic_colour_drop(local_r, local_g, local_b)
        colour_z = self._robust_z(np.maximum(colour_drop, 0.0), valid)
        colour_support = (colour_z >= 0.45) | (
            (hemoglobin >= 2.0) & (colour_z >= 0.10)
        )

        surface_reject = np.zeros_like(valid, dtype=bool)
        if pp is not None and np.asarray(pp.analysis_image).shape[:2] == image.shape[:2]:
            pp_gray = cv2.cvtColor(pp.analysis_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            pp_dark = np.maximum(gaussian_filter(pp_gray, 3.0) - pp_gray, 0.0)
            pp_z = self._robust_z(pp_dark, valid)
            surface_reject = (pp_z >= 2.2) & (hemoglobin < 1.35)
            # This wider band is only where PP may classify a line as a
            # surface eyelid/eyelash structure; it does not remove the band
            # from the CP measurement domain.  The anatomical exclusion below
            # remains the required 3px.
            eye_zone = self._landmark_zone(cp.landmarks, image.shape[:2], (LEFT_EYE, RIGHT_EYE), 20)
            # Eyelashes and eyelid folds are unambiguously surface structures:
            # in their narrow search band PP is allowed to veto them even when
            # CP colour residual is high.  The geometric exclusion itself stays
            # at 3px, so the surrounding periocular skin remains measurable.
            surface_reject |= (pp_z >= 1.0) & (eye_zone > 0)

        candidate = (
            (vesselness >= line_threshold)
            & (hemoglobin >= 1.0)
            & colour_support
            & (~surface_reject)
            & (valid > 0)
        )
        # Closed eyelids make the lid fold and eyelashes look like strong
        # vessels in CP/RED.  Keep the general geometric domain at the required
        # 3px margin, but fail closed in this dedicated presentation guard so
        # surface eyelid structures cannot be promoted as vessels.  Periocular
        # skin outside the narrow closed-eye band remains measurable.
        eyelash_guard = self._landmark_zone(
            cp.landmarks,
            image.shape[:2],
            (LEFT_EYE, RIGHT_EYE),
            self.EYE_CANDIDATE_GUARD_PX,
        )
        candidate &= eyelash_guard == 0
        rgb_visible: np.ndarray | None = None
        if consumer_rgb_proxy:
            lab_a = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1].astype(
                np.float32
            )
            local_red = np.maximum(
                lab_a - gaussian_filter(lab_a, 7.0),
                0.0,
            )
            rgb_visible = self._robust_z(local_red, valid) >= 0.7
            rgb_visible = self._dilate(
                rgb_visible.astype(np.uint8) * 255,
                2,
            ) > 0
            candidate &= rgb_visible
        candidate = remove_small_objects(candidate, min_size=4)
        candidate_u8 = candidate.astype(np.uint8) * 255
        face_width = self._face_width(face)
        mask, instances = self._filter_components(
            candidate_u8,
            hemoglobin,
            vesselness,
            cp_line,
            red_line,
            valid,
            face_width,
        )
        mask = cv2.bitwise_and(mask, valid)
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        branch = self._qualified_branch_points(mask, skeleton)

        rgb_visibility: np.ndarray | None = None
        rgb_support_ratio: float | None = None
        if rgb_visible is not None:
            rgb_visibility = (
                (skeleton > 0) & rgb_visible
            ).astype(np.uint8) * 255
            rgb_support_ratio = float(
                np.count_nonzero(rgb_visibility)
                / max(np.count_nonzero(skeleton), 1)
            )
        elif rgb is not None and np.asarray(rgb.analysis_image).shape[:2] == image.shape[:2]:
            rgb_bgr = np.asarray(rgb.analysis_image)
            lab_a = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
            local = np.maximum(lab_a - gaussian_filter(lab_a, 7.0), 0.0)
            visible = self._robust_z(local, valid) >= 0.7
            rgb_visibility = ((skeleton > 0) & visible).astype(np.uint8) * 255
            rgb_support_ratio = float(np.count_nonzero(rgb_visibility) / max(np.count_nonzero(skeleton), 1))

        metrics = self._metrics(
            face,
            exclusion,
            valid,
            mask,
            skeleton,
            branch,
            hemoglobin,
            instances,
            rgb_support_ratio,
        )
        base = red_aligned if red_aligned is not None else image
        overlay = self.render_overlay(base, skeleton, branch)
        return ClinicVascularResult(
            face_mask=face,
            exclusion_mask=exclusion,
            valid_mask=valid,
            hemoglobin_response=hemoglobin,
            vesselness=vesselness,
            vascular_mask=mask,
            skeleton_mask=skeleton,
            branch_mask=branch,
            rgb_visibility_mask=rgb_visibility,
            overlay=overlay,
            metrics=metrics,
            instances=tuple(instances),
        )

    @staticmethod
    def render_overlay(base: np.ndarray, skeleton: np.ndarray, branch: np.ndarray) -> np.ndarray:
        output = np.asarray(base).copy()
        line = cv2.dilate((skeleton > 0).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        # A deeper orange remains visible on the pale RED display without the
        # unrelated VISIA region frame competing with the actual findings.
        output[line] = (0, 105, 205)  # deep orange, BGR
        count, _, _, centres = cv2.connectedComponentsWithStats((branch > 0).astype(np.uint8), 8)
        for index in range(1, count):
            x, y = np.rint(centres[index]).astype(int)
            cv2.circle(output, (int(x), int(y)), 3, (15, 20, 170), -1, cv2.LINE_AA)
        return output

    def _face_mask(self, cp: Any) -> np.ndarray:
        debug = getattr(cp, "_debug_masks", {}) or {}
        for key in ("display_face_mask", "face_geometry_mask"):
            value = debug.get(key)
            if isinstance(value, np.ndarray) and np.count_nonzero(value):
                return self._binary(value)
        return VascularStructureAnalyzer._recover_outer_face_from_skin(cp.skin_mask)

    @staticmethod
    def _landmark_zone(
        points: np.ndarray,
        shape: tuple[int, int],
        groups: tuple[list[int], ...],
        radius: int,
    ) -> np.ndarray:
        zone = np.zeros(shape, np.uint8)
        points = np.asarray(points, dtype=np.float32)
        for indices in groups:
            feature = np.zeros(shape, np.uint8)
            VascularStructureAnalyzer._fill_landmark_hull(feature, points, indices)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            zone = cv2.bitwise_or(zone, cv2.dilate(feature, kernel))
        return zone

    def _clinic_exclusion(self, image: np.ndarray, face: np.ndarray, cp: Any) -> np.ndarray:
        points = np.asarray(cp.landmarks, dtype=np.float32)
        debug = getattr(cp, "_debug_masks", {}) or {}
        exclusion = np.zeros_like(face)
        for key in ("hair_mask", "facial_hair_mask"):
            value = debug.get(key)
            if isinstance(value, np.ndarray):
                exclusion = cv2.bitwise_or(
                    exclusion,
                    self._dilate(self._binary(value), self.HAIR_GUARD_PX),
                )
        eye_hulls: list[np.ndarray] = []
        brow_points = np.concatenate(
            (
                points[np.asarray(LEFT_EYEBROW, dtype=np.int32), :2],
                points[np.asarray(RIGHT_EYEBROW, dtype=np.int32), :2],
            ),
            axis=0,
        )
        if brow_points.size:
            forehead_end = int(
                np.clip(
                    round(float(np.max(brow_points[:, 1]))) + 12,
                    0,
                    face.shape[0],
                )
            )
            exclusion[:forehead_end] = 255
        for indices, radius in (
            (LEFT_EYE, self.EYE_GUARD_PX),
            (RIGHT_EYE, self.EYE_GUARD_PX),
            (LEFT_EYEBROW, self.EYEBROW_GUARD_PX),
            (RIGHT_EYEBROW, self.EYEBROW_GUARD_PX),
            (LIPS, self.LIP_GUARD_PX),
        ):
            feature = np.zeros_like(face)
            VascularStructureAnalyzer._fill_landmark_hull(feature, points, indices)
            if indices is LEFT_EYE or indices is RIGHT_EYE:
                eye_hulls.append(feature)
            exclusion = cv2.bitwise_or(exclusion, self._dilate(feature, radius))
        # Keep the anatomical safety margin at 3px, but remove the actual dark
        # eyelash/eyelid pixels within a narrow photometric search band.  This
        # is more precise than expanding every eye exclusion by 15--17px.
        lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        local_l = gaussian_filter(lab_l, 4.0)
        dark_drop = np.maximum(local_l - lab_l, 0.0)
        for eye in eye_hulls:
            band = self._dilate(eye, 18)
            values = dark_drop[(band > 0) & (face > 0)]
            if values.size:
                threshold = max(float(np.percentile(values, 78.0)), 7.0)
                lashes = ((dark_drop >= threshold) & (band > 0)).astype(np.uint8) * 255
                exclusion = cv2.bitwise_or(exclusion, self._dilate(lashes, 2))
        nostril = VascularStructureAnalyzer()._landmark_nostril_mask(image, points)
        exclusion = cv2.bitwise_or(exclusion, self._dilate(nostril, 2))
        return cv2.bitwise_and(exclusion, face)

    def _filter_components(
        self,
        candidate: np.ndarray,
        hemoglobin: np.ndarray,
        vesselness: np.ndarray,
        cp_line: np.ndarray,
        red_line: np.ndarray,
        valid: np.ndarray,
        face_width: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        count, labels, stats, centres = cv2.connectedComponentsWithStats(candidate, 8)
        output = np.zeros_like(candidate)
        rows: list[dict[str, Any]] = []
        minimum_length = max(6, int(round(face_width * 0.008)))
        maximum_width = max(10.0, face_width * 0.018)
        for label in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[label])
            if area < 4 or area > 1800:
                continue
            component = labels == label
            skeleton = skeletonize(component)
            length = int(np.count_nonzero(skeleton))
            if length < minimum_length:
                continue
            branch = self._branch_points(skeleton.astype(np.uint8) * 255)
            branch_count = int(np.count_nonzero(branch))
            points = np.column_stack(np.where(component)[::-1]).astype(np.float32)
            if len(points) < 3:
                continue
            covariance = np.cov(points, rowvar=False)
            eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-6))
            elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
            if elongation < 1.6 and branch_count == 0:
                continue
            distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
            widths = distance[skeleton] * 2.0
            p90_width = float(np.percentile(widths, 90)) if widths.size else 0.0
            if p90_width > maximum_width:
                continue
            response = hemoglobin[skeleton]
            line_values = vesselness[skeleton]
            if not response.size or float(np.mean(response)) < 1.0 or float(np.mean(line_values)) < 0.08:
                continue
            mean_cp_line = float(np.mean(cp_line[skeleton]))
            mean_red_line = float(np.mean(red_line[skeleton]))
            mean_hemo = float(np.mean(response))
            high_confidence = mean_cp_line >= 0.50 and mean_red_line >= 0.35 and mean_hemo >= 1.80
            medium_confidence = mean_cp_line >= 0.68 and mean_hemo >= 2.10
            if not (high_confidence or medium_confidence):
                # Single-response structures remain visible in vesselness debug
                # maps but are not promoted to the formal mask.
                continue
            if length < 10 and not (
                mean_cp_line >= 0.75
                and mean_red_line >= 0.65
                and mean_hemo >= 2.50
            ):
                # Six-pixel structures are accepted only with exceptional
                # three-cue support; this preserves short true candidates
                # without promoting granular RED texture to the formal mask.
                continue
            endpoints = self._endpoints(skeleton.astype(np.uint8) * 255)
            endpoint_points = np.column_stack(np.where(endpoints > 0)[::-1])
            if len(endpoint_points) >= 2:
                span = max(
                    float(np.linalg.norm(endpoint_points[i] - endpoint_points[j]))
                    for i in range(len(endpoint_points))
                    for j in range(i + 1, len(endpoint_points))
                )
            else:
                span = float(max(w, h))
            tortuosity = float(length / max(span, 1.0))
            cx, cy = (float(value) for value in centres[label])
            region = self._assign_region(cx, cy, valid)
            confidence = "HIGH" if high_confidence else "MEDIUM"
            radial = bool(branch_count > 0 and int(np.count_nonzero(endpoints)) >= 3)
            output[component] = 255
            rows.append({
                "id": len(rows) + 1,
                "bbox": [x, y, w, h],
                "centroid": [round(cx, 2), round(cy, 2)],
                "region": region,
                "confidence": confidence,
                "area_px": area,
                "skeleton_length_px": length,
                "p50_width_px": round(float(np.percentile(widths, 50)), 4) if widths.size else 0.0,
                "p90_width_px": round(p90_width, 4),
                "p50_redness": round(float(np.percentile(response, 50)), 4),
                "p90_redness": round(float(np.percentile(response, 90)), 4),
                "mean_cp_line_response": round(mean_cp_line, 6),
                "mean_red_line_response": round(mean_red_line, 6),
                "branch_points": branch_count,
                "tortuosity": round(tortuosity, 4),
                "radial_like": radial,
            })
        return output, rows

    def _metrics(
        self,
        face: np.ndarray,
        exclusion: np.ndarray,
        valid: np.ndarray,
        mask: np.ndarray,
        skeleton: np.ndarray,
        branch: np.ndarray,
        hemoglobin: np.ndarray,
        instances: list[dict[str, Any]],
        rgb_support_ratio: float | None,
    ) -> dict[str, Any]:
        valid_pixels = int(np.count_nonzero(valid))
        area = int(np.count_nonzero(mask))
        length = int(np.count_nonzero(skeleton))
        widths = VascularStructureAnalyzer._skeleton_widths(mask, skeleton)
        redness = hemoglobin[skeleton > 0]
        regions = self._region_metrics(valid, instances)
        tortuosities = np.asarray([row["tortuosity"] for row in instances], dtype=np.float32)
        network_length = sum(
            int(row["skeleton_length_px"])
            for row in instances
            if int(row["branch_points"]) > 0
        )
        left_length = sum(float(row["total_length_px"]) for key, row in regions.items() if key.startswith("left_"))
        right_length = sum(float(row["total_length_px"]) for key, row in regions.items() if key.startswith("right_"))
        left_red = [float(row["p90_redness"]) for row in regions.values() if row["region_id"].startswith("left_") and row["count"]]
        right_red = [float(row["p90_redness"]) for row in regions.values() if row["region_id"].startswith("right_") and row["count"]]
        outer_guard = cv2.subtract(
            self._binary(face),
            self._erode(face, self.OUTER_FACE_GUARD_PX),
        )
        inside_outer_guard = int(np.count_nonzero((mask > 0) & (outer_guard > 0)))
        return {
            "algorithm": self.ALGORITHM,
            "measurement_channel": "CP_M",
            "red_role": "CP_DERIVED_PROPOSAL_AND_DISPLAY_NOT_INDEPENDENT_EVIDENCE",
            "input_width": 1024,
            "input_height": 1024,
            "face_pixels": int(np.count_nonzero(face)),
            "valid_face_pixels": valid_pixels,
            "exclusion_pixels": int(np.count_nonzero(exclusion)),
            "vascular_count": len(instances),
            "vascular_area_px": area,
            "vascular_area_ratio": float(area / max(valid_pixels, 1)),
            "vascular_total_length_px": float(length),
            "vascular_line_density_per_10k_face_px": float(length * 10000.0 / max(valid_pixels, 1)),
            "p50_width_px": float(np.percentile(widths, 50)) if widths.size else 0.0,
            "p90_width_px": float(np.percentile(widths, 90)) if widths.size else 0.0,
            "p50_redness": float(np.percentile(redness, 50)) if redness.size else 0.0,
            "p90_redness": float(np.percentile(redness, 90)) if redness.size else 0.0,
            "branch_point_count": int(np.count_nonzero(branch)),
            "branch_point_density_per_10k_face_px": float(np.count_nonzero(branch) * 10000.0 / max(valid_pixels, 1)),
            "tortuosity_p50": float(np.percentile(tortuosities, 50)) if tortuosities.size else 0.0,
            "tortuosity_p90": float(np.percentile(tortuosities, 90)) if tortuosities.size else 0.0,
            "network_ratio": float(network_length / max(length, 1)),
            "radial_structure_count": sum(bool(row["radial_like"]) for row in instances),
            "max_continuous_network_length_px": max((int(row["skeleton_length_px"]) for row in instances), default=0),
            "cp_rgb_visible_support_ratio": rgb_support_ratio,
            "region_distribution": regions,
            "left_right_summary": {
                "left_total_length_px": left_length,
                "right_total_length_px": right_length,
                "length_asymmetry_ratio": abs(left_length - right_length) / max(left_length + right_length, 1.0),
                "left_p90_redness": max(left_red, default=0.0),
                "right_p90_redness": max(right_red, default=0.0),
                "redness_difference": abs(max(left_red, default=0.0) - max(right_red, default=0.0)),
            },
            "qc_passed": True,
            "finding_state": "DETECTED" if instances else "NOT_DETECTED",
            "self_check": {
                "outside_valid_pixels": int(np.count_nonzero((mask > 0) & (valid == 0))),
                "inside_exclusion_pixels": int(np.count_nonzero((mask > 0) & (exclusion > 0))),
                "inside_outer_face_guard_pixels": inside_outer_guard,
                "passed": bool(
                    inside_outer_guard == 0
                    and not np.any((mask > 0) & ((valid == 0) | (exclusion > 0)))
                ),
            },
            "limitations": [
                "仅表示正面二维图像中的血管样线状结构，不代表真实血管数量。",
                "像素宽度不是实际血管直径，不用于疾病诊断。",
            ],
        }

    def _region_metrics(self, valid: np.ndarray, instances: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        names = (
            "forehead", "nose_alar_nasal_side", "left_periocular", "right_periocular",
            "left_zygoma", "right_zygoma", "left_cheek", "right_cheek",
            "left_jaw", "right_jaw",
        )
        rows = {name: {"region_id": name, "valid_pixels": 0, "count": 0, "total_length_px": 0.0, "total_area_px": 0, "p90_redness": 0.0} for name in names}
        ys, xs = np.where(valid > 0)
        x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        nx = (xs.astype(np.float32) - x0) / max(x1 - x0, 1.0)
        ny = (ys.astype(np.float32) - y0) / max(y1 - y0, 1.0)
        left = nx < 0.5
        masks = {
            "forehead": np.zeros_like(ny, dtype=bool),
            "nose_alar_nasal_side": (nx >= 0.38) & (nx <= 0.62) & (ny >= 0.30) & (ny <= 0.68),
        }
        assigned = masks["forehead"] | masks["nose_alar_nasal_side"]
        masks.update({
            "left_periocular": (~assigned) & left & (ny < 0.43),
            "right_periocular": (~assigned) & (~left) & (ny < 0.43),
            "left_zygoma": (~assigned) & left & (ny >= 0.43) & (ny < 0.55),
            "right_zygoma": (~assigned) & (~left) & (ny >= 0.43) & (ny < 0.55),
            "left_cheek": (~assigned) & left & (ny >= 0.55) & (ny < 0.73),
            "right_cheek": (~assigned) & (~left) & (ny >= 0.55) & (ny < 0.73),
            "left_jaw": (~assigned) & left & (ny >= 0.73),
            "right_jaw": (~assigned) & (~left) & (ny >= 0.73),
        })
        for name, region_mask in masks.items():
            rows[name]["valid_pixels"] = int(np.count_nonzero(region_mask))
        redness_by_region: dict[str, list[float]] = {name: [] for name in names}
        for item in instances:
            row = rows[item["region"]]
            row["count"] += 1
            row["total_length_px"] += float(item["skeleton_length_px"])
            row["total_area_px"] += int(item["area_px"])
            redness_by_region[item["region"]].append(float(item["p90_redness"]))
        for name, row in rows.items():
            row["line_density_per_10k_px"] = float(row["total_length_px"] * 10000.0 / max(row["valid_pixels"], 1))
            row["area_ratio"] = float(row["total_area_px"] / max(row["valid_pixels"], 1))
            row["p90_redness"] = max(redness_by_region[name], default=0.0)
        return rows

    @staticmethod
    def _assign_region(x: float, y: float, valid: np.ndarray) -> str:
        ys, xs = np.where(valid > 0)
        x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        nx = (x - x0) / max(x1 - x0, 1.0)
        ny = (y - y0) / max(y1 - y0, 1.0)
        side = "left" if nx < 0.5 else "right"
        if 0.38 <= nx <= 0.62 and 0.30 <= ny <= 0.68:
            return "nose_alar_nasal_side"
        if ny < 0.43:
            return f"{side}_periocular"
        if ny < 0.55:
            return f"{side}_zygoma"
        if ny < 0.73:
            return f"{side}_cheek"
        return f"{side}_jaw"

    def _empty(
        self,
        image: np.ndarray,
        face: np.ndarray,
        exclusion: np.ndarray,
        valid: np.ndarray,
        red_display: np.ndarray | None,
        reason: str,
    ) -> ClinicVascularResult:
        zero = np.zeros_like(valid)
        base = self._aligned_optional(red_display, image.shape)
        if base is None:
            base = image
        metrics = {
            "algorithm": self.ALGORITHM,
            "measurement_channel": "CP_M",
            "face_pixels": int(np.count_nonzero(face)),
            "valid_face_pixels": int(np.count_nonzero(valid)),
            "vascular_count": None,
            "qc_passed": False,
            "finding_state": "NOT_ASSESSABLE",
            "reason_code": reason,
        }
        return ClinicVascularResult(face, exclusion, valid, zero.astype(np.float32), zero.astype(np.float32), zero, zero, zero, None, base.copy(), metrics, ())

    @staticmethod
    def _aligned_optional(image: np.ndarray | None, expected_shape: tuple[int, ...]) -> np.ndarray | None:
        if image is None:
            return None
        value = np.asarray(image)
        if value.shape[:2] != expected_shape[:2]:
            raise ValueError("RED display must already be aligned to CP 1024 space")
        return value

    @staticmethod
    def _robust_z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        samples = values[mask > 0]
        if samples.size < 100:
            return np.zeros_like(values, dtype=np.float32)
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        p90 = float(np.percentile(samples, 90.0))
        # Sparse positive-residual maps often have median==MAD==0.  A tiny
        # epsilon then saturates every non-zero JPEG fluctuation at z=8.  The
        # upper-quantile scale keeps real local colour drops separable from
        # compression texture without weakening the stated z>=1 gate.
        sigma = max(1.4826 * mad, (p90 - median) / 1.2816, 0.005)
        output = np.clip((values - median) / sigma, 0.0, 8.0).astype(np.float32)
        output[mask == 0] = 0.0
        return output

    @staticmethod
    def _robust_unit(values: np.ndarray, mask: np.ndarray, low: float, high: float) -> np.ndarray:
        samples = values[mask > 0]
        if samples.size < 100:
            return np.zeros_like(values, dtype=np.float32)
        lo, hi = (float(value) for value in np.percentile(samples, (low, high)))
        output = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
        output[mask == 0] = 0.0
        return output

    @staticmethod
    def _face_width(face: np.ndarray) -> int:
        _, xs = np.where(face > 0)
        return int(xs.max() - xs.min() + 1) if xs.size else 0

    @staticmethod
    def _endpoints(skeleton: np.ndarray) -> np.ndarray:
        binary = (skeleton > 0).astype(np.uint8)
        neighbours = cv2.filter2D(binary, cv2.CV_16S, np.ones((3, 3), np.uint8)) - binary.astype(np.int16)
        return ((binary > 0) & (neighbours == 1)).astype(np.uint8) * 255

    @staticmethod
    def _branch_points(skeleton: np.ndarray) -> np.ndarray:
        return VascularStructureAnalyzer._branch_points(skeleton)

    def _qualified_branch_points(self, mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        raw = self._branch_points(skeleton)
        qualified = np.zeros_like(raw)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        for label in range(1, count):
            component = labels == label
            if int(np.count_nonzero((skeleton > 0) & component)) < 20:
                continue
            branch_pixels = ((raw > 0) & component).astype(np.uint8)
            branch_count, _, _, centres = cv2.connectedComponentsWithStats(branch_pixels, 8)
            for branch_id in range(1, branch_count):
                x, y = np.rint(centres[branch_id]).astype(int)
                if 0 <= y < qualified.shape[0] and 0 <= x < qualified.shape[1]:
                    qualified[y, x] = 255
        return qualified

    @staticmethod
    def _binary(mask: np.ndarray) -> np.ndarray:
        return (np.asarray(mask) > 0).astype(np.uint8) * 255

    @staticmethod
    def _chromatic_colour_drop(
        local_r: np.ndarray,
        local_g: np.ndarray,
        local_b: np.ndarray,
    ) -> np.ndarray:
        """Keep red-selective darkening while rejecting achromatic lines."""

        return np.maximum(
            0.5 * (np.asarray(local_g) + np.asarray(local_b))
            - np.asarray(local_r),
            0.0,
        ).astype(np.float32)

    @classmethod
    def _has_extreme_chroma_occlusion(
        cls,
        image: np.ndarray,
        valid: np.ndarray,
    ) -> bool:
        """Reject heavily colour-painted consumer inputs for vascular scoring."""

        saturation = cv2.cvtColor(
            np.asarray(image),
            cv2.COLOR_BGR2HSV,
        )[:, :, 1]
        values = saturation[np.asarray(valid) > 0]
        if values.size < 100:
            return True
        p90 = float(np.percentile(values, 90.0))
        extreme_ratio = float(
            np.count_nonzero(values >= cls.EXTREME_SATURATION_THRESHOLD)
            / values.size
        )
        return bool(
            p90 >= cls.EXTREME_SATURATION_P90
            and extreme_ratio >= cls.EXTREME_SATURATION_AREA_RATIO
        )

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel)

    @staticmethod
    def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.erode((mask > 0).astype(np.uint8) * 255, kernel)
