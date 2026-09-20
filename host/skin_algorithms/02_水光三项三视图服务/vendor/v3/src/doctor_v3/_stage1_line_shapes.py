"""Deterministic line morphology; original centerlines are never overwritten."""
import cv2
import numpy as np


def fragments(skeleton, split_junctions=True):
    line = np.asarray(skeleton, dtype=bool)
    padded = np.pad(line, 1)
    h, w = line.shape
    degree = np.zeros(line.shape, np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbour = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
        if dy and dx:
            # A raster stair-step has diagonal shortcuts but is not a branch.
            neighbour = neighbour & ~padded[1:1+h, 1+dx:1+dx+w] & ~padded[1+dy:1+dy+h, 1:1+w]
        degree += neighbour
    branch = line & (degree > 2)
    split = line & ~branch if split_junctions else line
    count, labels = cv2.connectedComponents(split.astype(np.uint8), 8)
    items = []
    for label in range(1, count):
        y, x = np.nonzero(labels == label)
        if len(x) < 3:
            continue
        points = np.column_stack((x, y)).astype(float)
        center = points.mean(axis=0)
        _, values, vectors = np.linalg.svd(points - center, full_matrices=False)
        direction = vectors[0]
        projection = (points - center) @ direction
        ends = points[[int(projection.argmin()), int(projection.argmax())]]
        items.append({"id": label, "xy": points, "length_px": int(len(x)),
                      "center": center, "direction": direction, "ends": ends,
                      "linearity": float(values[0] / max(values[-1], 1.0)),
                      "angle": float(np.arctan2(direction[1], direction[0]) % np.pi)})
    return items, branch


def merge_main(items, gap=8.0, cosine=0.94):
    """Join aligned endpoint gaps; bridge pixels never become measured length."""
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    links = []
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            if abs(float(a["direction"] @ b["direction"])) < cosine:
                continue
            distances = np.linalg.norm(a["ends"][:, None] - b["ends"][None, :], axis=2)
            u, v = np.unravel_index(np.argmin(distances), distances.shape)
            delta = b["ends"][v] - a["ends"][u]
            distance = float(np.linalg.norm(delta))
            if not 0 < distance <= gap:
                continue
            if min(abs(float(delta @ a["direction"])), abs(float(delta @ b["direction"]))) / distance < cosine:
                continue
            parent[root(j)] = root(i)
            links.append({"fragment_ids": [a["id"], b["id"]], "gap_px": distance})
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(root(i), []).append(item)
    return list(groups.values()), links


def mask_for(items, shape):
    mask = np.zeros(shape, bool)
    for item in items:
        xy = item["xy"].astype(int)
        mask[xy[:, 1], xy[:, 0]] = True
    return mask


def summaries(items):
    return [{"fragment_id": p["id"], "length_px": p["length_px"],
             "orientation_radians": p["angle"], "linearity": p["linearity"],
             "endpoints_xy": p["ends"].tolist()} for p in items]
