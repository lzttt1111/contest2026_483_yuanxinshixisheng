"""V3 local glabella observation domain, derived from native wrinkle geometry.

Each caller supplies landmarks in its own pixel frame. This domain describes
coarse local appearance; it does not establish physical wrinkle depth.
"""
import cv2
import numpy as np

VERSION = "v301-glabella-target-support-roi-3"
PARAMETERS = {"horizontal_brow_width_fraction": .06,
              "upper_brow_height_fraction": .35,
              "lower_brow_height_fraction": .80,
              "zone_dilate_px": 3, "eyebrow_dilate_px": 8,
              "eye_dilate_px": 12, "nose_dilate_px": 4,
              "fallback_halfwidth_fraction": .10, "fallback_min_halfwidth_px": 8,
              "target_horizontal_padding_px": 0, "target_dilate_px": 0,
              "target_lower_boundary_landmark": 168,
              "target_inner_boundary_guard_px": 8,
              "target_taper_to_nasal_root": True}
INDICES = {
    "image_left_brow": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "image_right_brow": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "image_left_eye": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                       157, 158, 159, 160, 161, 246, 469, 470, 471, 472),
    "image_right_eye": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398,
                        384, 385, 386, 387, 388, 466, 474, 475, 476, 477),
    "nose": (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 45, 220, 115,
             48, 64, 98, 97, 326, 327, 294, 278, 344, 440, 275),
}


def _dilate(mask, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def _build(valid, landmarks, *, target):
    """Return a skin-only native-brow domain without changing supplied masks.

    Uses the non-iris eye hull for a 468-point FaceMesh; iris points, when
    present, lie inside that hull. Missing/nonfinite face geometry fails closed.
    """
    skin = np.asarray(valid) > 0
    empty = np.zeros(skin.shape, dtype=bool)
    p = np.asarray(landmarks, dtype=float)
    if skin.ndim != 2 or p.ndim != 2 or p.shape[1] < 2 or len(p) < 468:
        return empty
    if not np.isfinite(p[:, :2]).all():
        return empty
    points = np.rint(p[:, :2]).astype(np.int32)
    brows = [points[list(INDICES[k])] for k in ("image_left_brow", "image_right_brow")]
    brows.sort(key=lambda group: float(group[:, 0].mean()))
    all_brows = np.vstack(brows)
    _, _, brow_width, brow_height = cv2.boundingRect(all_brows)
    if brow_width < 2 or brow_height < 2:
        return empty
    left, right = int(brows[0][:, 0].max()), int(brows[1][:, 0].min())
    inner_y = float(brows[0][np.argmax(brows[0][:, 0]), 1] +
                    brows[1][np.argmin(brows[1][:, 0]), 1]) / 2
    if right <= left:
        if target:
            return empty
        center = int(round(float(all_brows[:, 0].mean())))
        half = max(PARAMETERS["fallback_min_halfwidth_px"],
                   round(PARAMETERS["fallback_halfwidth_fraction"] * brow_width))
        left, right = center - half, center + half
    pad = PARAMETERS["target_horizontal_padding_px"] if target else round(PARAMETERS["horizontal_brow_width_fraction"] * brow_width)
    top = int(all_brows[:, 1].min()) - round(PARAMETERS["upper_brow_height_fraction"] * brow_height)
    bottom = int(all_brows[:, 1].max()) + round(PARAMETERS["lower_brow_height_fraction"] * brow_height)
    if target:
        bottom = min(bottom, int(points[PARAMETERS["target_lower_boundary_landmark"], 1]))
        guard = PARAMETERS["target_inner_boundary_guard_px"]
        left, right = left + guard, right - guard
        root_x, root_y = points[PARAMETERS["target_lower_boundary_landmark"]]
        if right <= left or root_y <= inner_y or not left < root_x < right:
            return empty
    height, width = skin.shape
    x0, x1 = max(0, left - pad), min(width, right + pad)
    y0, y1 = max(0, top), min(height, bottom)
    if x1 <= x0 or y1 <= y0:
        return empty
    zone = np.zeros(skin.shape, np.uint8)
    zone[y0:y1, x0:x1] = 1
    radius = PARAMETERS["target_dilate_px"] if target else PARAMETERS["zone_dilate_px"]
    zone = (_dilate(zone, radius) if radius else zone) > 0
    if target and PARAMETERS["target_taper_to_nasal_root"]:
        yy, xx = np.indices(skin.shape)
        progress = np.clip((yy - inner_y) / (root_y - inner_y), 0., 1.)
        # The lower brow-gap narrows onto the bridge, never into inner canthi.
        zone &= (xx >= left + progress * (root_x - left))
        zone &= (xx < right + progress * (root_x - right))
    for name, ids in INDICES.items():
        mask = np.zeros(skin.shape, np.uint8)
        hull = cv2.convexHull(points[[i for i in ids if i < len(points)]])
        cv2.fillConvexPoly(mask, hull, 1)
        kind = "eyebrow" if name.endswith("brow") else "eye" if name.endswith("eye") else "nose"
        zone &= _dilate(mask, PARAMETERS[kind + "_dilate_px"]) == 0
    return zone & skin


def build_glabella(valid, landmarks):
    """Count/coverage target: inner-brow gap, stopping above the nasal root.

    An 8px inner-edge safety margin keeps brow-edge fragments out. Below the
    inner brow endpoints the target tapers toward the nasal root, excluding
    rectangular lower corners next to the inner canthi. No lateral expansion
    or post-mask dilation restores excluded pixels.
    The number of sparse Z vertices does not determine this target's borders.
    """
    return _build(valid, landmarks, target=True)


def build_glabella_support(valid, landmarks):
    """Native wider neighboring skin used only for coarse-Z reference fitting."""
    return _build(valid, landmarks, target=False)
