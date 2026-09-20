"""Private spatial evidence helpers for stage one; no detector reruns."""
import cv2
import numpy as np
from .phenotypes import warp_mask, PARAMETERS


def _good(metadata, project):
    meta = metadata.get(project, {})
    quality = meta.get("quality", {})
    status = meta.get("quality_status", quality.get("quality_status", quality.get("图像质量状态")))
    return str(status).upper() not in ("REJECT", "REJECTED", "FAIL", "FAILED", "UNASSESSABLE")


def pigment_support(data, metadata, profile, registration):
    spots = data.get("spots")
    if spots is None or "instances" not in spots:
        return None, "missing_visible_spot_instances", None
    supports, coverages = [], []
    registration = registration or {}
    for project, role in (("brown", "CP_M"), ("uv_spots", "365_M")):
        other = data.get(project)
        if other is None or "instances" not in other or not _good(metadata, project):
            continue
        if profile != "consumer" and registration.get(role) is not True:
            continue
        aligned = warp_mask(other, spots, other["instances"] > 0)
        valid = warp_mask(other, spots, other["valid"] > 0)
        if aligned is not None and valid is not None:
            supports.append(aligned & valid)
            coverages.append(valid)
    if not supports:
        return None, "missing_authorized_pigment_support", None
    coverage = np.logical_or.reduce(coverages)
    # A positive RGB candidate outside every qualified partner's evaluable
    # domain is unknown support, not a verified negative.
    unknown = (spots["instances"] > 0) & (spots["valid"] > 0) & ~coverage
    if np.count_nonzero(coverage & (spots["valid"] > 0)) < 100:
        return None, "insufficient_pigment_support_coverage", unknown
    support = np.logical_or.reduce(supports)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((spots["instances"] > 0).astype(np.uint8), connectivity=8)
    overlaps = np.bincount(labels[support], minlength=n)
    accepted = np.flatnonzero(overlaps / np.maximum(stats[:, cv2.CC_STAT_AREA], 1) >= .25)
    accepted = accepted[accepted != 0]
    return np.isin(labels, accepted), None, unknown


def exclusion_evidence(data, metadata, profile, registration):
    red = data.get("redness")
    if red is None or "instances" not in red:
        return None, "missing_focal_red_evidence"
    if profile != "consumer" and (registration or {}).get("CP_M") is not True:
        return None, "cross_channel_exclusion_not_authorized"
    excluded = red["instances"] > 0
    valid = red["valid"] > 0
    for project in ("vascular", "acne"):
        other = data.get(project)
        if other is None or "instances" not in other or not _good(metadata, project):
            return None, "missing_qualified_" + project + "_exclusion"
        aligned = warp_mask(other, red, other["instances"] > 0)
        coverage = warp_mask(other, red, other["valid"] > 0)
        if aligned is None or coverage is None:
            return None, "exclusion_coordinate_mismatch"
        # Pixels outside another detector's evaluable area cannot count as a
        # proven absence of vascular/focal candidates.
        excluded = excluded | aligned | (valid & ~coverage)
    excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return excluded & valid, None


def acne_evidence(data, metadata):
    a = data.get("acne")
    if a is None:
        return None, [], "missing_acne_evidence"
    result = dict(a)
    result["classification_mask"] = np.zeros(a["valid"].shape, np.uint8)
    raw_positive = (a["instances"] > 0) & (a["valid"] > 0) if "instances" in a else None
    if raw_positive is not None and raw_positive.any() and not metadata.get("acne", {}).get("instances"):
        result["classification_mask"][raw_positive] = 4
        _, _, _, centers = cv2.connectedComponentsWithStats(raw_positive.astype(np.uint8), connectivity=8)
        labels = [{"candidate_id": i, "centroid": [int(round(x)), int(round(y))],
                   "class": "unknown", "reason": "missing_candidate_evidence",
                   "classification_version": PARAMETERS["version"], "method": "unclassified_saved_support"}
                  for i,(x,y) in enumerate(centers[1:])]
        return result, labels, "missing_candidate_evidence"
    if "image" not in a or "instances" not in metadata.get("acne", {}):
        return result, [], "missing_acne_image_or_candidates"
    image = np.asarray(a["image"])
    if image.shape != (*a["valid"].shape, 3) or not np.all(np.isfinite(image)):
        return result, [], "invalid_acne_image"
    pores = data.get("pores")
    follicles = (warp_mask(pores, a, pores["instances"] > 0)
                 if pores is not None and "instances" in pores and _good(metadata, "pores") else None)
    coverage = warp_mask(pores, a, pores["valid"] > 0) if follicles is not None else None
    if coverage is not None:
        follicles &= coverage
    result["classification_valid"] = ((a["valid"] > 0) & coverage if coverage is not None else np.zeros(a["valid"].shape, bool))
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(float)
    valid = a["valid"] > 0
    candidates = [(o, False) for o in metadata["acne"]["instances"]]
    if follicles is not None:
        _, _, _, centers = cv2.connectedComponentsWithStats(follicles.astype(np.uint8), connectivity=8)
        candidates += [({"bbox_xyxy": [x-5, y-5, x+5, y+5]}, True) for x,y in centers[1:]]
    occupied = np.zeros(valid.shape, bool)
    labels = []
    invalid_candidates = 0
    for ordinal, (obj, follicle_candidate) in enumerate(candidates):
        box = obj.get("bbox_standardized_xyxy", obj.get("bbox_xyxy"))
        if box is None and obj.get("bbox") is not None:
            x,y,w,h = obj["bbox"]
            box = [x,y,x+w,y+h]
        if box is None or len(box) != 4 or not np.all(np.isfinite(box)):
            invalid_candidates += 1
            continue
        x0,y0,x1,y1 = [int(round(v)) for v in box]
        cx,cy = (x0+x1)//2,(y0+y1)//2
        if not (0 <= cy < valid.shape[0] and 0 <= cx < valid.shape[1]):
            invalid_candidates += 1
            continue
        if not valid[cy,cx] or occupied[cy,cx]:
            continue
        radius = max(3,min(x1-x0,y1-y0)//3)
        extent = int(np.ceil(radius*2.2))
        xa,xb = max(0,cx-extent),min(valid.shape[1],cx+extent+1)
        ya,yb = max(0,cy-extent),min(valid.shape[0],cy+extent+1)
        yy,xx = np.ogrid[ya:yb,xa:xb]
        distance = (xx-cx)**2+(yy-cy)**2
        domain = valid[ya:yb,xa:xb]
        center = (distance <= radius**2) & domain
        ring = (distance > (radius*1.3)**2) & (distance <= (radius*2.2)**2) & domain
        kind, reason, da, dl = "unknown", "insufficient_local_samples", None, None
        anchored = follicles is not None and bool(np.any(follicles[ya:yb,xa:xb][center]))
        if center.sum() >= 5 and ring.sum() >= 12:
            light, red = lab[ya:yb,xa:xb,0], lab[ya:yb,xa:xb,1]
            da = float(np.median(red[center])-np.median(red[ring]))
            dl = float(np.percentile(light[center],90)-np.median(light[ring]))
            if follicle_candidate and da < PARAMETERS["follicular_red_delta_a"]:
                continue  # normal pore proposal, not a model acne detection
            reason = "missing_follicle_support" if not anchored else "insufficient_class_evidence"
            if anchored and da >= PARAMETERS["follicular_red_delta_a"]:
                if follicle_candidate: kind = "erythema"
                elif dl >= PARAMETERS["white_center_delta_l"]: kind = "pustule"
                elif float(np.std(light[center])) >= 5: kind = "papule"
        labels.append({"candidate_id": ordinal, "centroid": [cx,cy], "class": kind,
                       "reason": reason if kind == "unknown" else None,
                       "red_delta_a": da, "center_delta_l": dl, "follicle_supported": anchored,
                       "classification_version": PARAMETERS["version"], "method": "image_phenotype_candidate"})
        occupied[ya:yb,xa:xb] |= distance <= (radius*2)**2
        local = result["classification_mask"][ya:yb,xa:xb]
        local[center & (local == 0)] = {"erythema":1,"papule":2,"pustule":3,"unknown":4}[kind]
    reason = ("invalid_acne_candidate_geometry" if invalid_candidates else
              "missing_follicle_support" if follicles is None else None)
    return result, labels, reason
