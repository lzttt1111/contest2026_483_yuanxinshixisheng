"""Eight disjoint leaves / seven display groups, derived from the archived face-region mapper.

Only geometric methods are reused: no instrument model, new landmark model or frontal gate.
The observed skin mask remains authoritative; geometry cannot create measured skin.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .config import REGIONS

VERSION = "shuiguang_anatomy_v1"
LEAVES = tuple(REGIONS)
GROUPS = {
    "forehead": ("额头", ("forehead",)),
    "eyes": ("眼周", ("left_eye", "right_eye")),
    "nose": ("鼻部", ("nose",)),
    "left_cheek": ("左面颊", ("left_cheek",)),
    "right_cheek": ("右面颊", ("right_cheek",)),
    "perioral": ("口周", ("perioral",)),
    "chin": ("下巴", ("chin",)),
}
COLORS = {"forehead": "#e6d996", "left_eye": "#91c9de", "right_eye": "#91c9de",
          "nose": "#8bc685", "left_cheek": "#d98691", "right_cheek": "#ecaf87",
          "perioral": "#b3adce", "chin": "#648ed8"}
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
POLYGONS = {
    "forehead": [10,338,297,332,284,300,293,334,296,336,9,107,66,105,63,70,54,103,67,109],
    "right_cheek": [234,127,162,70,111,117,118,100,36,203,206,216,212,138,172,58,132,93],
    "left_cheek": [454,356,389,300,340,346,347,329,266,423,426,436,432,367,397,288,361,323],
    "chin": [61,91,84,17,314,321,291,367,397,365,379,378,400,377,152,148,176,149,150,136,172,138],
    "perioral": [203,206,216,212,57,43,106,182,83,18,313,406,335,273,287,432,436,426,423,2],
    "nose": [168,193,122,196,3,51,45,44,1,274,275,281,248,419,351,417],
    "right_eye": [33,70,63,105,66,107,133,112,26,22,23,24,110,25],
    "left_eye": [362,336,296,334,293,300,263,255,339,254,253,252,256,341],
}
EXCLUSIONS = ([33,160,158,133,153,144], [362,385,387,263,373,380],
              [61,40,37,0,267,270,291,321,314,17,84,91])

def expanded_oval(points, extension=.20):
    points = np.asarray(points, dtype=np.float64)
    oval = points[OVAL].copy()
    axis = points[152] - points[10]
    height = max(float(np.linalg.norm(axis)), 1.)
    direction = axis / height
    fraction = (oval - points[10]) @ direction / height
    weight = np.clip(1 - fraction / .38, 0, 1)
    return oval - weight[:, None] * height * extension * direction

def rings(mask):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, .8, True).reshape(-1, 2).tolist()
            for c in contours if cv2.contourArea(c) >= 2]

@dataclass
class Anatomy:
    labels: np.ndarray
    geometric_labels: np.ndarray
    face: np.ndarray
    excluded: np.ndarray
    hidden: np.ndarray
    coverage: dict

    def masks(self):
        return {name: (self.labels == i + 1).astype(np.uint8) * 255
                for i, name in enumerate(LEAVES)}

def partition(points, observed_skin, view="front", known_hair=None, for_statistics=False):
    p = np.asarray(points, dtype=np.float32)[:468]
    shape = observed_skin.shape
    empty = np.zeros(shape, np.uint8)
    if p.shape != (468, 2) or not np.isfinite(p).all():
        return Anatomy(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy(),
                       {n: {"state": "uncovered", "reason": "关键点证据不足"} for n in LEAVES})
    def hull(coords):
        mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(coords).astype(np.int32)), 255)
        return mask
    extended = expanded_oval(p)
    face = hull(np.concatenate([extended,p[POLYGONS['nose']]])) if for_statistics else hull(extended)
    geometric = np.zeros(shape, np.uint8)
    top_ids = [i for i, k in enumerate(OVAL) if k in [10,338,297,332,284,251,389,162,21,54,103,67,109]]
    for name, ids in POLYGONS.items():
        coords = p[ids]
        if name == "forehead":
            coords = np.concatenate([coords, extended[top_ids]])
        mask = hull(coords)
        if for_statistics and name=='chin':
            # Lower-lip plane prevents a profile convex hull swallowing lower cheek.
            axis=p[152]-p[10];axis=axis/max(float(np.linalg.norm(axis)),1.)
            yy,xx=np.indices(shape)
            below=(xx-p[17,0])*axis[0]+(yy-p[17,1])*axis[1]>=0
            mask[~below]=0
        geometric[(mask > 0) & (face > 0)] = LEAVES.index(name) + 1
    # Geometric gap fill is constrained to the geometric face, then observed skin.
    seeds = geometric > 0
    if seeds.any():
        _, nearest = cv2.distanceTransformWithLabels((~seeds).astype(np.uint8),
                    cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        values = geometric[seeds]
        gaps = (face > 0) & ~seeds
        geometric[gaps] = values[nearest[gaps] - 1]
    excluded = np.maximum.reduce([hull(p[ids]) for ids in EXCLUSIONS])
    geometric[excluded > 0] = 0
    hidden = np.zeros(shape, np.uint8)
    widths = {"right": abs(float(p[1, 0] - p[234, 0])),
              "left": abs(float(p[454, 0] - p[1, 0]))}
    near = max(widths, key=widths.get)
    far = "right" if near == "left" else "left"
    if view != "front" and widths[far] < .5 * max(widths[near], 1):
        for name in (far + "_cheek", far + "_eye"):
            hidden[geometric == LEAVES.index(name) + 1] = 255
    labels = geometric.copy()
    labels[(observed_skin == 0) | (hidden > 0)] = 0
    coverage = {}
    for i, name in enumerate(LEAVES, 1):
        n = int(np.count_nonzero(labels == i))
        geometric_area = int(np.count_nonzero(geometric == i))
        hair = int(np.count_nonzero((geometric == i) & (known_hair > 0))) if known_hair is not None else 0
        if n >= 128:
            state, reason = "observed", "已取得可见皮肤证据"
        elif hair > .5 * max(geometric_area, 1):
            state, reason = "occluded", "已有毛发遮挡证据"
        else:
            state, reason = "uncovered", "当前视角或缓存未覆盖有效皮肤"
        coverage[name] = {"state": state, "reason": reason, "observed_pixels": n,
                          "geometric_pixels": geometric_area, "known_hair_pixels": hair}
    return Anatomy(labels, geometric, face, excluded, hidden, coverage)

def classify(center, labels, boundary_radius=2):
    x, y = np.rint(center).astype(int)
    if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
        return None, "outside_image", []
    value = int(labels[y, x])
    if not value:
        return None, "unassigned_visible_observation", []
    r = boundary_radius
    patch = labels[max(0, y-r):y+r+1, max(0, x-r):x+r+1]
    neighbours = sorted(int(v) for v in np.unique(patch) if v)
    if len(neighbours) > 1:
        return None, "region_boundary_uncertain", [LEAVES[v-1] for v in neighbours]
    return LEAVES[value-1], "assigned", [LEAVES[value-1]]

def anchor(center, points):
    p = np.asarray(points)[:468]
    distances = np.linalg.norm(p - np.asarray(center), axis=1)
    indices = np.argsort(distances)[:3]
    return {"kind": "nearest_landmark_anchors_not_operation_points",
            "indices": indices.tolist(), "distances_analysis_px": distances[indices].round(3).tolist()}

def sync_mask_bundle(result, visible_mask):
    """Atomic consumer-mask binding without applying another hair dilation."""
    from sg_core.preprocess.analysis_mask_bundle import build_analysis_mask_bundle
    current = getattr(result, "mask_bundle", None)
    target = ((visible_mask > 0) & (result.skin_mask > 0)).astype(np.uint8) * 255
    if current is not None:
        result.mask_bundle = build_analysis_mask_bundle(
            valid_skin_mask=target, hair_mask=current.hair_mask,
            facial_hair_mask=current.facial_hair_mask,
            feature_exclusion_mask=current.feature_exclusion_mask,
            nostril_mask=current.nostril_mask, display_face_mask=current.display_face_mask,
            coordinate_space=current.coordinate_space)
        result.skin_mask = result.mask_bundle.algorithm_mask()
        assert np.array_equal(result.skin_mask, result.mask_bundle.algorithm_mask())
    else:
        result.skin_mask = target.copy()
