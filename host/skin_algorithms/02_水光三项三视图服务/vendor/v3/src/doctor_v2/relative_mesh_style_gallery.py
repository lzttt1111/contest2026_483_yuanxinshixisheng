# -*- coding: utf-8 -*-
"""Debug-only 2.5D Face Mesh style gallery.

All renderings use the same display vertices, face scope and relative-z input.
They never change Module 11 measurements and must not be used as metric depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STYLE_LABELS = (
    ("01_clinical_emerald", "临床翡翠网格"),
    ("02_ice_blue_depth", "冰蓝深度网格"),
    ("03_viridis_surface", "青绿曲面"),
    ("04_warm_relief", "暖色起伏图"),
    ("05_graphite_relief", "银灰浮雕"),
    ("06_holographic_duotone", "全息双色网格"),
)


def _triangles(edges: Iterable[tuple[int, int]], vertex_count: int) -> tuple[tuple[int, int, int], ...]:
    normalized = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in edges
        if int(a) < vertex_count and int(b) < vertex_count and int(a) != int(b)
    }
    neighbors: dict[int, set[int]] = {index: set() for index in range(vertex_count)}
    for a, b in normalized:
        neighbors[a].add(b)
        neighbors[b].add(a)
    found: set[tuple[int, int, int]] = set()
    for a, b in normalized:
        for c in neighbors[a].intersection(neighbors[b]):
            triangle = tuple(sorted((a, b, c)))
            if len(set(triangle)) == 3:
                found.add(triangle)
    return tuple(sorted(found))


def _pose_corrected_depth(points: np.ndarray, relative_z: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float64)
    z = np.asarray(relative_z, dtype=np.float64)[: len(xy)]
    if z.shape != (len(xy),) or not np.all(np.isfinite(z)):
        raise ValueError("relative Face Mesh z is unavailable")
    scale = max(float(np.ptp(xy[:, 0])), 1.0)
    normalized_xy = xy / scale
    design = np.column_stack((np.ones(len(xy)), normalized_xy[:, 0], normalized_xy[:, 1]))
    coefficients, *_ = np.linalg.lstsq(design, z / scale, rcond=None)
    corrected = z / scale - design @ coefficients
    lower, upper = np.percentile(corrected, (2.0, 98.0))
    if upper - lower < 1e-8:
        raise ValueError("relative Face Mesh z has no usable variation")
    return np.clip((corrected - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _colormap(value: float, code: int) -> tuple[int, int, int]:
    scalar = np.asarray([[round(float(np.clip(value, 0.0, 1.0)) * 255.0)]], np.uint8)
    pixel = cv2.applyColorMap(scalar, code)[0, 0]
    return tuple(int(channel) for channel in pixel)


def _blend_face(base: np.ndarray, layer: np.ndarray, scope: np.ndarray, alpha: float) -> np.ndarray:
    output = np.asarray(base).copy()
    selected = (np.asarray(scope) > 0) & np.any(layer != 0, axis=2)
    output[selected] = np.clip(
        (1.0 - alpha) * output[selected].astype(np.float32)
        + alpha * layer[selected].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return output


def _draw_edges(
    canvas: np.ndarray,
    points: np.ndarray,
    edges: Iterable[tuple[int, int]],
    depth: np.ndarray,
    color_function,
    *,
    thickness: int = 1,
) -> np.ndarray:
    output = np.asarray(canvas).copy()
    for a, b in edges:
        if int(a) >= len(points) or int(b) >= len(points):
            continue
        color = color_function(float((depth[int(a)] + depth[int(b)]) * 0.5))
        cv2.line(
            output,
            tuple(np.rint(points[int(a)]).astype(np.int32)),
            tuple(np.rint(points[int(b)]).astype(np.int32)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return output


def _surface_layer(
    points: np.ndarray,
    depth: np.ndarray,
    triangles: tuple[tuple[int, int, int], ...],
    shape: tuple[int, int],
    color_function,
) -> np.ndarray:
    layer = np.zeros((shape[0], shape[1], 3), np.uint8)
    order = sorted(triangles, key=lambda tri: float(np.mean(depth[list(tri)])))
    xy_scale = max(float(np.ptp(points[:, 0])), 1.0)
    xyz = np.column_stack((points / xy_scale, depth[:, None]))
    light = np.asarray((-0.35, -0.45, 0.82), np.float64)
    light /= np.linalg.norm(light)
    for triangle in order:
        indices = np.asarray(triangle, np.int32)
        polygon = np.rint(points[indices]).astype(np.int32)
        if abs(float(cv2.contourArea(polygon))) < 0.5:
            continue
        a, b, c = xyz[indices]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        diffuse = 0.72 if norm < 1e-8 else 0.55 + 0.45 * abs(float(np.dot(normal / norm, light)))
        base_color = np.asarray(color_function(float(np.mean(depth[indices]))), np.float32)
        color = tuple(int(value) for value in np.clip(base_color * diffuse, 0, 255))
        cv2.fillConvexPoly(layer, polygon, color, lineType=cv2.LINE_AA)
    return layer


def render_holographic_duotone_mesh(
    image: np.ndarray,
    display_points: np.ndarray,
    face_scope: np.ndarray,
    relative_z: np.ndarray,
    tessellation: Iterable[tuple[int, int]],
    *,
    mesh_scope: np.ndarray | None = None,
) -> np.ndarray:
    """Render the selected cyan-to-magenta mesh for the formal Module 11 image.

    This is a display-only projection.  It reuses the frozen relative-z input
    and never changes the geometry metrics.
    """

    points = np.asarray(display_points, dtype=np.float32)[:468]
    scope = (np.asarray(face_scope) > 0).astype(np.uint8) * 255
    line_scope = (
        scope
        if mesh_scope is None
        else (np.asarray(mesh_scope) > 0).astype(np.uint8) * 255
    )
    if image.shape[:2] != scope.shape or points.shape != (468, 2):
        raise ValueError("relative mesh style input contract mismatch")
    if line_scope.shape != scope.shape:
        raise ValueError("relative mesh line scope shape mismatch")
    edges = tuple(
        sorted(
            {
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in tessellation
                if int(a) < 468 and int(b) < 468
            }
        )
    )
    depth = _pose_corrected_depth(points, np.asarray(relative_z)[:468])
    base = np.asarray(image).copy()
    darkened = np.clip(base.astype(np.float32) * 0.52, 0, 255).astype(np.uint8)
    rendered = _draw_edges(
        darkened,
        points,
        edges,
        depth,
        lambda value: (
            int(245 - 55 * value),
            int(225 - 180 * value),
            int(35 + 220 * value),
        ),
    )
    # Keep the face darkening continuous, but suppress only the coloured mesh
    # strokes where the preprocessor identified forehead/temple hair.
    rendered[line_scope == 0] = darkened[line_scope == 0]
    output = base.copy()
    selected = scope > 0
    output[selected] = rendered[selected]
    return output


def render_relative_mesh_style_gallery(
    image: np.ndarray,
    display_points: np.ndarray,
    face_scope: np.ndarray,
    relative_z: np.ndarray,
    tessellation: Iterable[tuple[int, int]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    points = np.asarray(display_points, dtype=np.float32)[:468]
    scope = (np.asarray(face_scope) > 0).astype(np.uint8) * 255
    if image.shape[:2] != scope.shape or points.shape != (468, 2):
        raise ValueError("relative mesh style input contract mismatch")
    edges = tuple(
        sorted(
            {
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in tessellation
                if int(a) < 468 and int(b) < 468
            }
        )
    )
    triangles = _triangles(edges, 468)
    depth = _pose_corrected_depth(points, np.asarray(relative_z)[:468])
    base = np.asarray(image).copy()

    emerald = _draw_edges(base, points, edges, depth, lambda _: (95, 220, 75))
    ice_base = np.clip(base.astype(np.float32) * 0.70, 0, 255).astype(np.uint8)
    ice = _draw_edges(
        ice_base,
        points,
        edges,
        depth,
        lambda value: (
            int(255 - 125 * value),
            int(215 - 80 * value),
            int(70 + 70 * value),
        ),
    )
    viridis_layer = _surface_layer(
        points,
        depth,
        triangles,
        scope.shape,
        lambda value: _colormap(value, cv2.COLORMAP_VIRIDIS),
    )
    viridis = _blend_face(base, viridis_layer, scope, 0.58)
    viridis = _draw_edges(viridis, points, edges, depth, lambda _: (70, 70, 70))

    warm_layer = _surface_layer(
        points,
        depth,
        triangles,
        scope.shape,
        lambda value: _colormap(value, cv2.COLORMAP_PLASMA),
    )
    warm_background = np.clip(base.astype(np.float32) * 0.22, 0, 255).astype(np.uint8)
    warm = _blend_face(warm_background, warm_layer, scope, 0.86)
    warm = _draw_edges(warm, points, edges, depth, lambda _: (105, 80, 125))

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    graphite = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    graphite_layer = _surface_layer(
        points,
        depth,
        triangles,
        scope.shape,
        lambda value: (int(105 + 115 * value),) * 3,
    )
    graphite = _blend_face(graphite, graphite_layer, scope, 0.48)
    graphite = _draw_edges(graphite, points, edges, depth, lambda _: (225, 225, 225))

    hologram = render_holographic_duotone_mesh(
        base,
        points,
        scope,
        relative_z,
        edges,
    )

    styles = {
        "01_clinical_emerald": emerald,
        "02_ice_blue_depth": ice,
        "03_viridis_surface": viridis,
        "04_warm_relief": warm,
        "05_graphite_relief": graphite,
        "06_holographic_duotone": hologram,
    }
    for key, rendered in tuple(styles.items()):
        clipped = base.copy()
        selected = scope > 0
        clipped[selected] = rendered[selected]
        styles[key] = clipped

    sheet = np.full((1024, 1024, 3), 245, np.uint8)
    positions = ((22, 32), (356, 32), (690, 32), (22, 522), (356, 522), (690, 522))
    labels: list[tuple[int, int, str]] = []
    for (key, label), (x, y) in zip(STYLE_LABELS, positions):
        thumb = cv2.resize(styles[key], (312, 420), interpolation=cv2.INTER_AREA)
        sheet[y : y + 420, x : x + 312] = thumb
        labels.append((x, y + 426, label))
    rgb = cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = None
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            font = ImageFont.truetype(candidate, 22)
            break
    for x, y, label in labels:
        draw.text((x + 4, y), label, fill=(20, 20, 20), font=font)
    sheet = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return styles, sheet


__all__ = [
    "STYLE_LABELS",
    "render_holographic_duotone_mesh",
    "render_relative_mesh_style_gallery",
]
