"""Stable VISIA-style single closed contour for report scope.

This module is deliberately limited to geometry.  It does not generate skin
features and it does not tune any detector.  The returned filled mask is used
as the final common scope after each detector has produced its native
candidates, so displayed and quantified targets share one boundary.
"""
from __future__ import annotations
import base64
import io
import zlib
from functools import lru_cache
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
FRONT_TEMPLATE_ANCHORS = np.asarray((10, 109, 338, 70, 300, 234, 454, 138, 367, 168, 6, 197, 4, 98, 327, 2, 209, 429), dtype=np.int32)
PORE_FOREHEAD_VERTICAL_SCALE = 1.5
FRONT_LEFT_CHIN_JOIN = 104
FRONT_RIGHT_CHIN_JOIN = 206
FRONT_JAW_LANDMARKS = (136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365)
FRONT_JAW_BASE_INSET_RATIO = 0.024
FRONT_JAW_CENTRE_INSET_RATIO = 0.027
_FRONT_TEMPLATE_B64 = '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'

@lru_cache(maxsize=1)
def _front_template() -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(base64.b64decode(_FRONT_TEMPLATE_B64))
    values = np.frombuffer(raw, dtype='<f4')
    landmark_values = 478 * 2
    return (values[:landmark_values].reshape(478, 2).copy(), values[landmark_values:].reshape(-1, 2).copy())

def _chaikin_closed(points: np.ndarray, iterations: int=1) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for _ in range(max(0, int(iterations))):
        following = np.roll(values, -1, axis=0)
        first = 0.75 * values + 0.25 * following
        second = 0.25 * values + 0.75 * following
        output = np.empty((len(values) * 2, 2), dtype=np.float32)
        output[0::2] = first
        output[1::2] = second
        values = output
    return values

def _deduplicate_closed(points: np.ndarray) -> np.ndarray:
    rounded = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
    if len(rounded) < 3:
        return rounded.reshape(-1, 1, 2)
    keep = np.ones(len(rounded), dtype=bool)
    keep[1:] = np.any(rounded[1:] != rounded[:-1], axis=1)
    values = rounded[keep]
    if len(values) >= 2 and np.array_equal(values[0], values[-1]):
        values = values[:-1]
    return np.vstack((values, values[0])).reshape(-1, 1, 2)

def _project_inside(points: np.ndarray, face_envelope: np.ndarray, *, unconstrained_above_y: float) -> np.ndarray:
    mask = (np.asarray(face_envelope) > 0).astype(np.uint8) * 255
    if not np.any(mask):
        return np.asarray(points, dtype=np.float32)
    inset = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if not np.any(inset):
        inset = mask
    boundary = cv2.subtract(inset, cv2.erode(inset, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))))
    boundary_yx = np.column_stack(np.where(boundary > 0))
    if not len(boundary_yx):
        return np.asarray(points, dtype=np.float32)
    boundary_xy = boundary_yx[:, ::-1].astype(np.float32)
    output = np.asarray(points, dtype=np.float32).copy()
    (height, width) = inset.shape
    rounded = np.rint(output).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    outside = (inset[rounded[:, 1], rounded[:, 0]] == 0) & (output[:, 1] >= float(unconstrained_above_y))
    for index in np.flatnonzero(outside):
        delta = boundary_xy - output[index]
        nearest = int(np.argmin(np.einsum('ij,ij->i', delta, delta)))
        output[index] = boundary_xy[nearest]
    return output

def _front_contour(landmarks: np.ndarray, shape: tuple[int, int], face_envelope: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    current = np.asarray(landmarks, dtype=np.float32)
    if current.shape[0] < 478:
        return (None, None)
    (reference, contour) = _front_template()
    (transform, _) = cv2.estimateAffine2D(reference[FRONT_TEMPLATE_ANCHORS], current[FRONT_TEMPLATE_ANCHORS], method=cv2.LMEDS)
    if transform is None or not np.isfinite(transform).all():
        return (None, None)
    mapped = cv2.transform(contour.reshape(-1, 1, 2), transform).reshape(-1, 2)
    pose_component = float(transform[0, 1])
    pose_strength = float(np.clip(abs(pose_component) / 0.095, 0.0, 1.5))
    mapped[:, 0] += float(np.sign(pose_component)) * pose_strength
    mapped[:, 1] += 3.0 * pose_strength
    brow_y = 0.5 * float(current[105, 1] + current[334, 1])
    current_top = float(np.min(mapped[:, 1]))
    desired_top = brow_y - 1.6 * (brow_y - float(current[10, 1]))
    desired_top = float(np.clip(desired_top, 42.0, brow_y - 24.0))
    if current_top < brow_y - 1.0 and abs(desired_top - current_top) > 0.5:
        forehead = mapped[:, 1] < brow_y
        blend = np.clip((brow_y - mapped[forehead, 1]) / (brow_y - current_top), 0.0, 1.0)
        mapped[forehead, 1] += (desired_top - current_top) * blend
    brow_indices = np.asarray((70, 63, 105, 66, 107, 55, 285, 336, 296, 334, 293, 300), dtype=np.int32)
    brow_points = current[brow_indices, :2]
    order = np.argsort(brow_points[:, 0])
    brow_points = brow_points[order]
    (unique_x, unique_at) = np.unique(brow_points[:, 0], return_index=True)
    unique_y = brow_points[unique_at, 1]
    if len(unique_x) >= 2:
        face_height = max(float(np.ptp(current[:, 1])), 1.0)
        clearance = max(10.0, 0.018 * face_height)
        for (start, stop) in ((3, 18), (136, 151)):
            segment = mapped[start:stop]
            limit = np.interp(segment[:, 0], unique_x, unique_y, left=float(unique_y[0]), right=float(unique_y[-1])) - clearance
            segment[:, 1] = np.minimum(segment[:, 1], limit)
            mapped[start:stop] = segment
    mapped[:, 0] = np.clip(mapped[:, 0], 1.0, float(shape[1] - 2))
    mapped[:, 1] = np.clip(mapped[:, 1], 1.0, float(shape[0] - 2))
    mapped = _project_inside(mapped, face_envelope, unconstrained_above_y=brow_y)
    stable = _chaikin_closed(mapped, 1).astype(np.float64)
    if len(stable) <= FRONT_RIGHT_CHIN_JOIN + 6:
        return (_deduplicate_closed(stable), None)
    separator = stable[FRONT_LEFT_CHIN_JOIN:FRONT_RIGHT_CHIN_JOIN + 1].copy()
    jaw_anchors = current[list(FRONT_JAW_LANDMARKS), :2].astype(np.float64)
    face_height = max(float(np.ptp(current[:, 1])), 1.0)
    weights = np.sin(np.linspace(0.18 * np.pi, 0.82 * np.pi, len(jaw_anchors)))
    jaw_anchors[:, 1] -= FRONT_JAW_BASE_INSET_RATIO * face_height + FRONT_JAW_CENTRE_INSET_RATIO * face_height * weights
    anchors = np.vstack((stable[FRONT_LEFT_CHIN_JOIN], jaw_anchors, stable[FRONT_RIGHT_CHIN_JOIN]))
    chord = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    parameter = np.r_[0.0, np.cumsum(np.maximum(chord, 1.0))]
    left_tangent = stable[FRONT_LEFT_CHIN_JOIN] - stable[FRONT_LEFT_CHIN_JOIN - 6]
    right_tangent = stable[FRONT_RIGHT_CHIN_JOIN + 6] - stable[FRONT_RIGHT_CHIN_JOIN]
    left_tangent /= max(float(np.linalg.norm(left_tangent)), 1e-06)
    right_tangent /= max(float(np.linalg.norm(right_tangent)), 1e-06)
    spline_x = CubicSpline(parameter, anchors[:, 0], bc_type=((1, float(left_tangent[0])), (1, float(right_tangent[0]))))
    spline_y = CubicSpline(parameter, anchors[:, 1], bc_type=((1, float(left_tangent[1])), (1, float(right_tangent[1]))))
    dense_parameter = np.linspace(parameter[0], parameter[-1], 280)
    jaw = np.column_stack((spline_x(dense_parameter), spline_y(dense_parameter)))
    jaw[:, 0] = np.clip(jaw[:, 0], 1.0, float(shape[1] - 2))
    jaw[:, 1] = np.clip(jaw[:, 1], 1.0, float(shape[0] - 2))
    outer = np.vstack((stable[:FRONT_LEFT_CHIN_JOIN + 1], jaw[1:-1], stable[FRONT_RIGHT_CHIN_JOIN:]))
    return (_deduplicate_closed(outer), _deduplicate_open(separator))

def _deduplicate_open(points: np.ndarray) -> np.ndarray:
    """Round an open display path without accidentally closing its endpoints."""
    rounded = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
    if not len(rounded):
        return rounded.reshape(-1, 1, 2)
    keep = np.ones(len(rounded), dtype=bool)
    keep[1:] = np.any(rounded[1:] != rounded[:-1], axis=1)
    return rounded[keep].reshape(-1, 1, 2)

def _nudge_open_path_inside_mask(path: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    """Move rounding-only outliers to the nearest filled pixel.

    The endpoints are shared with the closed contour and stay untouched.  A
    handful of adjacent separator points can otherwise round one pixel to the
    outside where the new jaw tangent diverges from the preserved nose path.
    """
    if path is None or len(path) < 3:
        return path
    points = np.asarray(path, dtype=np.int32).reshape(-1, 2).copy()
    (height, width) = mask.shape
    for index in range(1, len(points) - 1):
        (x, y) = (int(points[index, 0]), int(points[index, 1]))
        if 0 <= x < width and 0 <= y < height and (mask[y, x] > 0):
            continue
        best: tuple[int, int] | None = None
        best_distance = float('inf')
        for radius in range(1, 9):
            (x0, x1) = (max(0, x - radius), min(width - 1, x + radius))
            (y0, y1) = (max(0, y - radius), min(height - 1, y + radius))
            (local_y, local_x) = np.where(mask[y0:y1 + 1, x0:x1 + 1] > 0)
            if not len(local_x):
                continue
            absolute_x = local_x + x0
            absolute_y = local_y + y0
            distances = (absolute_x - x) ** 2 + (absolute_y - y) ** 2
            nearest = int(np.argmin(distances))
            candidate_distance = float(distances[nearest])
            if candidate_distance < best_distance:
                best = (int(absolute_x[nearest]), int(absolute_y[nearest]))
                best_distance = candidate_distance
            break
        if best is not None:
            points[index] = best
    return _deduplicate_open(points)

def _side_contour(landmarks: np.ndarray, shape: tuple[int, int], *, image_left: bool, face_width: float, face_height: float) -> np.ndarray | None:
    points = np.asarray(landmarks, dtype=np.float32)
    if image_left:
        (valley, outer_peak, outer_bottom) = (101, 116, 138)
        (lower_mid, inner_lower, nose_wing) = (187, 209, 220)
        (alar_outer, alar_bottom, alar_inner) = (98, 2, 327)
        sign = -1.0
    else:
        (valley, outer_peak, outer_bottom) = (330, 345, 367)
        (lower_mid, inner_lower, nose_wing) = (411, 429, 440)
        (alar_outer, alar_bottom, alar_inner) = (327, 2, 98)
        sign = 1.0
    required = (168, valley, outer_peak, outer_bottom, lower_mid, inner_lower, nose_wing, alar_outer, alar_bottom, alar_inner)
    if any((index >= len(points) for index in required)):
        return None
    path = points[list(required), :2].copy()
    path[0, 1] -= 0.03 * face_height
    path[1, 1] += 0.1 * face_height
    path[2, 0] -= sign * 0.02 * face_width
    path[2, 1] -= 0.23 * face_height
    path[3, 0] += sign * 0.1 * face_width
    path[3, 1] += 0.035 * face_height
    path[4, 0] -= sign * 0.045 * face_width
    path[4, 1] += 0.22 * face_height
    path[5, 1] -= 0.015 * face_height
    path[6:, 0] -= sign * 0.02 * face_width
    path[6, 1] -= 0.02 * face_height
    path[7, 1] -= 0.01 * face_height
    path[8, 1] -= 0.02 * face_height
    path[9, 1] -= 0.01 * face_height
    path[:, 0] = np.clip(path[:, 0], 1.0, shape[1] - 2.0)
    path[:, 1] = np.clip(path[:, 1], 1.0, shape[0] - 2.0)
    return _deduplicate_closed(_chaikin_closed(path, 4))

def build_unified_visia_scope(landmarks: np.ndarray, shape: tuple[int, int], face_envelope: np.ndarray, *, partial_face: bool, left_visible_area: int, right_visible_area: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Return the closed scope, its filled mask and display-only separator."""
    larger = max(left_visible_area, right_visible_area, 1)
    side_view = partial_face or min(left_visible_area, right_visible_area) < 0.28 * larger
    (ys, xs) = np.where(np.asarray(face_envelope) > 0)
    face_width = max(float(np.ptp(xs)) if xs.size else shape[1] * 0.6, 1.0)
    face_height = max(float(np.ptp(ys)) if ys.size else shape[0] * 0.7, 1.0)
    if side_view:
        contour = _side_contour(landmarks, shape, image_left=left_visible_area >= right_visible_area, face_width=face_width, face_height=face_height)
        separator = None
    else:
        (contour, separator) = _front_contour(landmarks, shape, face_envelope)
    mask = np.zeros(shape, dtype=np.uint8)
    if contour is not None and len(contour) >= 4:
        cv2.fillPoly(mask, [np.asarray(contour, dtype=np.int32)], 255, lineType=cv2.LINE_8)
    separator = _nudge_open_path_inside_mask(separator, mask)
    return (contour, mask, separator)

def draw_unified_visia_contour(image: np.ndarray, contour: np.ndarray | None, color: tuple[int, int, int], thickness: int, separator: np.ndarray | None=None) -> np.ndarray:
    output = np.asarray(image).copy()
    if contour is not None and len(contour) >= 4:
        cv2.polylines(output, [np.asarray(contour, dtype=np.int32)], True, color, thickness, lineType=cv2.LINE_AA)
    if separator is not None and len(separator) >= 2:
        cv2.polylines(output, [np.asarray(separator, dtype=np.int32)], False, color, thickness, lineType=cv2.LINE_AA)
    return output
__all__ = ['build_unified_visia_scope', 'draw_unified_visia_contour']
