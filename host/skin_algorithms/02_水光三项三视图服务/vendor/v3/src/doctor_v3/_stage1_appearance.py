"""Regional extensions of documented V2 appearance products; not skin depth."""
import cv2
import numpy as np
from ._stage1_geometry_math import local_samples

VERSION = "v3-stage1-v2-regional-appearance-1"
GLABELLA_VERSION = "v301-glabella-target-support-appearance-1"


def texture_classes(texture, metadata):
    """Recover saved peak phenotypes only where their actual mask is present."""
    empty = np.zeros(texture["valid"].shape, bool)
    masks = {"raised": empty.copy(), "depressed": empty.copy()}
    count, labels = cv2.connectedComponents((texture["instances"] > 0).astype(np.uint8), 8)
    assigned, records = {}, []
    for item in metadata.get("instances", []):
        kind = {"raised_like": "raised", "depressed_like": "depressed"}.get(item.get("texture_type"))
        center = item.get("centroid")
        if kind is None or center is None or len(center) != 2:
            continue
        x, y = (int(round(v)) for v in center)
        if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
            continue
        label = int(labels[y, x])
        if label == 0:
            continue
        assigned.setdefault(label, set()).add(kind)
        records.append({"source_id": item.get("id"), "component_id": label, "kind": kind,
                        "centroid": [x, y], "source_area": item.get("measurement_area", item.get("area")),
                        "intensity": item.get("intensity", item.get("texture_score"))})
    ambiguous = []
    for label, kinds in assigned.items():
        if len(kinds) != 1:
            ambiguous.append(label)
            continue
        masks[next(iter(kinds))] |= labels == label
    return masks, records, ambiguous


def local_relief(rgb, domain):
    samples, reason = local_samples(rgb, domain)
    if samples is None:
        return None, {"reason": reason}
    residual = samples["relative_depth"]
    value = float(np.percentile(np.abs(residual), 90))
    return value, {"sample_ids": samples["ids"].tolist(),
                   "sample_xy": samples["pixel_xy"].tolist(),
                   "normalized_z_samples": samples["z"].tolist(),
                   "residual_z_samples": residual.tolist(),
                   "reference_plane": samples["plane"].tolist(),
                   "regional_relief_p90": value,
                   "source_formula": "V2 relative_face_geometry._midface_metrics: plane detrend + P90(abs(residual_z))"}


def line_appearance(rgb, domain, local_line, *, reference_domain=None):
    area = int(domain.sum())
    reference = domain if reference_domain is None else reference_domain
    relief, basis = local_relief(rgb, reference)
    local_line = (np.asarray(local_line) > 0) & (np.asarray(domain) > 0)
    coverage = float(local_line.sum()/max(area, 1))
    basis.update({"formula": "regional_relief_p90 * regional_original_line_coverage",
                  "line_pixels": int(local_line.sum()), "valid_area_px": area, "line_coverage": coverage,
                  "source_formula_id": "stable_wrinkles.relative_depth.mean_wrinkle_relative_depth",
                  "interpretation": "regional line and coarse facial-relief appearance product; not wrinkle depth",
                  "physical_depth": False})
    if reference_domain is not None:
        p = np.asarray(rgb["landmarks"], float)
        xy = np.rint(p[:, :2]).astype(int)
        valid_xy = ((xy[:, 0] >= 0) & (xy[:, 0] < domain.shape[1]) &
                    (xy[:, 1] >= 0) & (xy[:, 1] < domain.shape[0]))
        ids = np.flatnonzero(valid_xy)
        z = np.asarray(rgb.get("relative_z", rgb.get("landmarks_relative_z", []))).reshape(-1)
        if len(z) == len(p):
            ids = ids[np.isfinite(z[ids])]
        else:
            ids = np.array([], dtype=int)
        target_ids = ids[np.asarray(domain)[xy[ids, 1], xy[ids, 0]] > 0]
        support_ids = ids[np.asarray(reference)[xy[ids, 1], xy[ids, 0]] > 0]
        basis.update({"appearance_formula_version": GLABELLA_VERSION,
                      "reference_target_separated": True, "target_area_px": area,
                      "reference_support_area_px": int(np.count_nonzero(reference)),
                      "target_sample_ids": target_ids.tolist(),
                      "reference_support_sample_ids": support_ids.tolist(),
                      "minimum_reference_samples": 6,
                      "formula": "support_relief_p90 * target_original_line_pixels / target_valid_area_px",
                      "source_formula_id": "glabella.coarse_reference_relief_times_target_line_coverage",
                      "interpretation": "neighboring coarse-Z reference times inner-brow target coverage; not physical wrinkle depth"})
    if relief is None or area < 100:
        return None, basis
    return relief * coverage, basis


def surface_appearance(rgb, domain, class_mask, intensity_map):
    area = int(domain.sum())
    mask = domain & class_mask
    values = np.asarray(intensity_map)[mask]
    values = values[np.isfinite(values)]
    coverage = float(mask.sum()/max(area, 1))
    relief, basis = local_relief(rgb, domain)
    intensity = float(np.percentile(values, 90)) if len(values) else None
    basis.update({"formula": "area=class_pixels/valid_pixels; apparent_P90=regional_relief_P90*class_intensity_P90*class_area_ratio",
                  "valid_area_px": area, "class_area_px": int(mask.sum()), "class_area_ratio": coverage,
                  "class_intensity_samples": values.tolist(), "class_intensity_p90": intensity,
                  "source_formula_id": "smoothness.{raised,depressed}_irregularity.p90_{raised,depressed}_relative_{height,depth}",
                  "interpretation": "saved texture phenotype times coarse regional relief; not small skin height/depth",
                  "physical_depth": False})
    if area < 100:
        return None, None, basis
    burden = relief * intensity * coverage if relief is not None and intensity is not None else None
    return coverage, burden, basis
