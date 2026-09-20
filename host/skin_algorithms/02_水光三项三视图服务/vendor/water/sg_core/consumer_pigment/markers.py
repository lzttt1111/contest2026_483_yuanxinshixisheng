"""Compact marker rendering shared by consumer pigmentation modules."""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass(frozen=True, slots=True)
class CompactMarkerStyle:
    fill_bgr: tuple[int, int, int]
    outline_bgr: tuple[int, int, int]
    fill_alpha: float
RED_STYLE_MARKER = CompactMarkerStyle((0, 228, 255), (0, 228, 255), 1.0)
BROWN_STYLE_MARKER = CompactMarkerStyle((255, 220, 35), (120, 70, 0), 1.0)
UV_STYLE_MARKER = CompactMarkerStyle((0, 199, 255), (0, 199, 255), 1.0)
PORPHYRIN_STYLE_MARKER = RED_STYLE_MARKER

@dataclass(frozen=True, slots=True)
class ExistingMarkerFinding:
    centroid: tuple[int, int]
    source_area_px: int
    marker_radius_px: int
    peak_score: float

@dataclass(frozen=True, slots=True)
class ExistingMarkerResult:
    findings: tuple[ExistingMarkerFinding, ...]
    marker_mask: np.ndarray
    rejected_low_score: int
    rejected_exclusion: int
    rejected_nms: int

def consolidate_existing_marker_mask(candidate_mask: np.ndarray, score_map: np.ndarray, valid_mask: np.ndarray, exclusion_mask: np.ndarray, *, minimum_peak_score: float, minimum_distance_px: int, minimum_local_prominence_z: float=0.0, minimum_marker_edge_gap_px: int=0, large_component_prominence_relief: float=0.0, area_priority_weight: float=0.0, marker_radius_scale: float=0.55) -> ExistingMarkerResult:
    """Turn existing accepted components into sparse red-style circles."""
    candidate = (np.asarray(candidate_mask) > 0).astype(np.uint8)
    score = np.asarray(score_map, dtype=np.float32)
    valid = np.asarray(valid_mask) > 0
    exclusion = np.asarray(exclusion_mask) > 0
    (labels_count, labels) = cv2.connectedComponents(candidate, connectivity=8)
    proposed: list[ExistingMarkerFinding] = []
    rejected_low_score = 0
    rejected_exclusion = 0
    for label in range(1, labels_count):
        component = labels == label
        source_area = int(np.count_nonzero(component))
        component_scores = np.where(component, score, -1.0)
        (peak_y, peak_x) = np.unravel_index(int(np.argmax(component_scores)), component_scores.shape)
        peak_score = float(score[peak_y, peak_x])
        if peak_score < minimum_peak_score:
            rejected_low_score += 1
            continue
        (y0, y1) = (max(0, peak_y - 16), min(score.shape[0], peak_y + 17))
        (x0, x1) = (max(0, peak_x - 16), min(score.shape[1], peak_x + 17))
        (yy, xx) = np.ogrid[y0:y1, x0:x1]
        ring = ((yy - peak_y) ** 2 + (xx - peak_x) ** 2 >= 8 ** 2) & ((yy - peak_y) ** 2 + (xx - peak_x) ** 2 <= 16 ** 2) & valid[y0:y1, x0:x1] & ~exclusion[y0:y1, x0:x1] & (candidate[y0:y1, x0:x1] == 0)
        background = score[y0:y1, x0:x1][ring]
        if background.size:
            median = float(np.median(background))
            mad = float(np.median(np.abs(background - median)))
            prominence_z = (peak_score - median) / max(1.4826 * mad, 0.025)
            area_scale = max(float(source_area) / 25.0, 1.0)
            required_prominence = minimum_local_prominence_z / (1.0 + large_component_prominence_relief * math.log2(area_scale))
            if prominence_z < required_prominence:
                rejected_low_score += 1
                continue
        if exclusion[peak_y, peak_x] or not valid[peak_y, peak_x]:
            rejected_exclusion += 1
            continue
        radius = int(np.clip(round(marker_radius_scale * np.sqrt(source_area / np.pi)), 2, 7))
        proposed.append(ExistingMarkerFinding(centroid=(int(peak_x), int(peak_y)), source_area_px=source_area, marker_radius_px=radius, peak_score=round(peak_score, 6)))
    proposed.sort(key=lambda item: item.peak_score + area_priority_weight * min(math.log2(1.0 + item.source_area_px / 25.0), 3.0), reverse=True)
    accepted: list[ExistingMarkerFinding] = []
    rejected_nms = 0
    for finding in proposed:
        (x, y) = finding.centroid
        if any(((x - other.centroid[0]) ** 2 + (y - other.centroid[1]) ** 2 < max(minimum_distance_px, finding.marker_radius_px + other.marker_radius_px + minimum_marker_edge_gap_px if minimum_marker_edge_gap_px > 0 else minimum_distance_px) ** 2 for other in accepted)):
            rejected_nms += 1
            continue
        accepted.append(finding)
    marker_mask = np.zeros(candidate.shape, dtype=np.uint8)
    for finding in accepted:
        cv2.circle(marker_mask, finding.centroid, finding.marker_radius_px, 255, -1, cv2.LINE_AA)
    marker_mask[~valid | exclusion] = 0
    return ExistingMarkerResult(findings=tuple(accepted), marker_mask=marker_mask, rejected_low_score=rejected_low_score, rejected_exclusion=rejected_exclusion, rejected_nms=rejected_nms)

def render_compact_marker_mask(base_image: np.ndarray, marker_mask: np.ndarray, style: CompactMarkerStyle) -> np.ndarray:
    """Render one uniform filled style for every accepted finding."""
    output = np.asarray(base_image).copy()
    marker = np.asarray(marker_mask)
    marker_pixels = np.any(marker > 0, axis=2) if marker.ndim == 3 else marker > 0
    if np.any(marker_pixels):
        fill = np.asarray(style.fill_bgr, dtype=np.float32)
        output[marker_pixels] = np.clip((1.0 - style.fill_alpha) * output[marker_pixels].astype(np.float32) + style.fill_alpha * fill, 0.0, 255.0).astype(np.uint8)
    (contours, _) = cv2.findContours(marker_pixels.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, style.outline_bgr, 1, cv2.LINE_AA)
    return output
__all__ = ['BROWN_STYLE_MARKER', 'CompactMarkerStyle', 'ExistingMarkerFinding', 'ExistingMarkerResult', 'PORPHYRIN_STYLE_MARKER', 'RED_STYLE_MARKER', 'UV_STYLE_MARKER', 'consolidate_existing_marker_mask', 'render_compact_marker_mask']
