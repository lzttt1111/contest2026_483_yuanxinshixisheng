"""V3 phase-one regional 2D line phenotypes, with reproducible raw evidence."""
import cv2
import numpy as np

from .registry import FINE_REGIONS, MODULES, measurement
from .regions import build
from .wrinkle_metrics import regions as stable_regions
from ._stage1_line_shapes import fragments, mask_for, merge_main, summaries

VERSION = "v3-stage1-lines-1"
PARAMETERS = {"grid_px": 32, "fine_max_length_px": 70, "fine_max_width_px": 5,
              "fine_min_fragments": 3, "fine_min_density": .035,
              "fine_high_density": .10, "stable_min_length_px": 10,
              "stable_min_linearity": 2.5, "merge_gap_px": 8,
              "merge_cosine": .94}


def _domains(arrays, module):
    # Explicit masks support validated caller-provided anatomical geometry.
    supplied = arrays.get("region_masks", {}).get(module)
    if supplied is not None:
        result = {k: (np.asarray(v) > 0) & (arrays["valid"] > 0) for k, v in supplied.items()}
        result["full_face"] = np.logical_or.reduce(list(result.values()))
        return result
    if module == "08":
        from ._stage1_glabella_roi import build_glabella
        result = stable_regions(arrays)
        if result:
            # The old report-level glabella is a supra-brow forehead strip.
            # Restore it to forehead, then give the new anatomical glabella
            # priority at the boundary so a diagonal line is not counted twice.
            full_forehead = result["forehead"] | result["glabella"]
            result["glabella"] = build_glabella(arrays["valid"], arrays["landmarks"])
            result["forehead"] = full_forehead & ~result["glabella"]
        return result
    return build(arrays["valid"], arrays["landmarks"], module)


def _fine_network(items, branch, valid, params):
    width, height = valid.shape[1], valid.shape[0]
    raw = mask_for(items, valid.shape)
    accepted = np.zeros_like(valid)
    high = np.zeros_like(valid)
    grid = int(params["grid_px"])
    cells = []
    for y in range(0, height, grid):
        for x in range(0, width, grid):
            roi = valid[y:y+grid, x:x+grid]
            area = int(roi.sum())
            if area < max(16, grid * grid // 8):
                continue
            local = [p for p in items if np.any(
                (p["xy"][:, 0] >= x) & (p["xy"][:, 0] < x+grid) &
                (p["xy"][:, 1] >= y) & (p["xy"][:, 1] < y+grid))]
            orientations = {int(p["angle"] * 4 / np.pi) % 4 for p in local}
            pixels = int((raw[y:y+grid, x:x+grid] & roi).sum())
            density = pixels / area
            # A short isolated line is insufficient: require concentrated
            # fragments AND multidirectional/junction network support.
            network = len(orientations) >= 2 or int(branch[y:y+grid, x:x+grid].sum()) >= 2
            keep = len(local) >= params["fine_min_fragments"] and network and density >= params["fine_min_density"]
            cells.append({"xyxy": [x, y, min(x+grid, width), min(y+grid, height)],
                          "valid_area_px": area, "fragment_count": len(local),
                          "orientation_bins": sorted(orientations), "density": density,
                          "phenotype_accepted": bool(keep)})
            if keep:
                accepted[y:y+grid, x:x+grid] = roi
                if density >= params["fine_high_density"]:
                    high[y:y+grid, x:x+grid] = roi
    return raw & accepted, accepted, high, cells


def _direction_ok(item, region, landmarks):
    direction = item["direction"]
    if region == "forehead":
        return abs(direction[0]) >= .70
    if region == "glabella":
        return abs(direction[1]) >= .55
    if region in ("left_crow_feet", "right_crow_feet"):
        anchor = np.asarray(landmarks)[33 if region.startswith("left") else 263, :2]
        radial = item["center"] - anchor
        norm = np.linalg.norm(radial)
        return norm > 0 and abs(float(direction @ radial)) / norm >= .65
    if region == "perioral":
        return abs(direction[1]) >= .50
    return True


def _emit(rows, basis, module, region, profile, values, context, reason=None):
    shared_key = "evidence:lines:" + module + ":" + region
    basis[shared_key] = context
    compact = {k: v for k, v in context.items() if k in (
        "valid_area_px", "line_pixels", "affected_area_px", "high_area_px",
        "main_groups", "quality_status", "high_precision", "physical_depth",
        "excluded_stable_pixels", "network_evidence_ref")}
    for name, value in values.items():
        unit = "条" if name == "main_count" else ("比例" if "area" in name else "标准化指数")
        unavailable_reason = reason or "missing_evidence"
        if value is None and context.get("valid_area_px", 0) >= 100 and context.get("quality_status") not in ("REJECT", "FAILED"):
            if name == "contrast_p50" and context.get("high_precision") is False:
                unavailable_reason = "high_precision_not_confirmed"
            elif context.get("line_pixels") == 0:
                unavailable_reason = "no_target"
        row = measurement(module, name, region, profile, value, unit,
                          unavailable_reason,
                          "rgb_line_appearance")
        row["definition_version"] = VERSION
        rows.append(row)
        basis[row["metric_id"] + ":" + region] = {
            **compact, "evidence_ref": shared_key, "formula": {"area": "affected_px/valid_px",
                "high_area": "high_density_px/valid_px",
                "density": "original_centerline_px*10000/affected_px",
                "contrast_p50": "P50(abs(gray-Gaussian(gray,sigma=3))/255 on original lines)",
                "main_count": "count(aligned endpoint-merged primary groups)",
                "length_burden": "sum(original centerline pixels)*10000/valid_px"}[name],
            "parameter_status": "engineering_candidate",
            "measurement_status": "measured" if value is not None else ("not_applicable" if unavailable_reason == "high_precision_not_confirmed" else "no_target" if unavailable_reason == "no_target" else "missing_evidence"),
            "reason": unavailable_reason if value is None else None}


def analyze(data, metadata, profile, registration=None, thresholds=None):
    arrays = data.get("wrinkle")
    if arrays is None:
        rows = [measurement(m, n, r, profile, reason="missing_wrinkle_evidence")
                for m in ("07", "08") for r in MODULES[int(m)-1].regions
                for n in (("area", "high_area", "density", "contrast_p50") if m == "07" or r not in ("forehead", "glabella") else ("main_count", "length_burden"))]
        return {"measurements": rows, "basis": {r["metric_id"]+":"+r["region"]: {"measurement_status": "missing_evidence"} for r in rows}, "arrays": {}}
    params = {**PARAMETERS, **(thresholds or {}).get("lines", {})}
    valid = np.asarray(arrays["valid"]) > 0
    skeleton = (np.asarray(arrays["skeleton"]) > 0) & valid
    gray = cv2.cvtColor(arrays["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast = np.abs(cv2.GaussianBlur(gray, (0, 0), 3)-gray)/255
    items, branch = fragments(skeleton)
    width = 2 * cv2.distanceTransform((arrays.get("instances", skeleton) > 0).astype(np.uint8), cv2.DIST_L2, 3)
    small = [p for p in items if p["length_px"] <= params["fine_max_length_px"] and
             float(np.median(width[p["xy"][:, 1].astype(int), p["xy"][:, 0].astype(int)])) <= params["fine_max_width_px"]]
    rows, basis, derived = [], {}, {"lines_original_skeleton": skeleton.astype(np.uint8)}
    stable = np.zeros_like(valid)
    stable_domains = _domains(arrays, "08")
    selected = {}
    for region, domain in stable_domains.items():
        if region in ("full_face", "crow_feet"):
            continue
        local, _ = fragments(skeleton & domain, split_junctions=False)
        candidates = [p for p in local if p["linearity"] >= params["stable_min_linearity"] and _direction_ok(p, region, arrays["landmarks"])]
        groups, links = merge_main(candidates, params["merge_gap_px"], params["merge_cosine"])
        groups = [group for group in groups if sum(p["length_px"] for p in group) >= params["stable_min_length_px"]]
        chosen = [p for group in groups for p in group]
        selected[region] = (chosen, groups, links)
        stable |= mask_for(chosen, valid.shape)
    # A bilateral radial summary reuses original pixels, never sums overlapping areas.
    if "crow_feet" in stable_domains:
        chosen = selected.get("left_crow_feet", ([], [], []))[0] + selected.get("right_crow_feet", ([], [], []))[0]
        selected["crow_feet"] = (chosen, [], [])
    quality = metadata.get("wrinkle", {})
    rejected = quality.get("quality_status") in ("REJECT", "FAILED")
    for region, (chosen, groups, links) in selected.items():
        domain = stable_domains[region]
        area = int(domain.sum())
        local = mask_for(chosen, valid.shape) & domain
        affected = (cv2.dilate(local.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & domain
        high = (cv2.boxFilter(local.astype(np.float32), -1, (16, 16)) >= .10) & affected
        context = {"valid_area_px": area, "line_pixels": int(local.sum()), "affected_area_px": int(affected.sum()),
                   "high_area_px": int(high.sum()), "fragments": summaries(chosen), "merge_links": links,
                   "main_groups": [[p["id"] for p in g] for g in groups], "parameters": params,
                   "quality_status": quality.get("quality_status", "UNKNOWN"),
                   "temporal_stability": "single_capture_not_verified", "physical_depth": False}
        if region in ("forehead", "glabella"):
            values = {"main_count": len(groups), "length_burden": float(local.sum()*10000/max(area, 1))}
        else:
            values = {"area": float(affected.sum()/max(area, 1)), "high_area": float(high.sum()/max(area, 1)),
                      "density": float(local.sum()*10000/affected.sum()) if affected.any() else None,
                      "contrast_p50": float(np.percentile(contrast[local], 50)) if local.any() else None}
        if area < 100 or rejected:
            values = dict.fromkeys(values)
        _emit(rows, basis, "08", region, profile, values, context, "rejected_quality" if rejected else "insufficient_valid_region")
    # Keep only fine fragments after the identified stable evidence is removed.
    small = [p for p in small if not np.any(stable[p["xy"][:, 1].astype(int), p["xy"][:, 0].astype(int)])]
    fine, affected, high, cells = _fine_network(small, branch, valid, params)
    network_key = "evidence:lines:07:network"
    basis[network_key] = {"cells": cells, "fragments": summaries(small), "parameters": params}
    high_precision = (thresholds or {}).get("imaging", {}).get("high_precision_texture") is True
    for region, domain in _domains(arrays, "07").items():
        area = int(domain.sum())
        local, spread, dense = fine & domain, affected & domain, high & domain
        values = {"area": float(spread.sum()/max(area, 1)), "high_area": float(dense.sum()/max(area, 1)),
                  "density": float(local.sum()*10000/spread.sum()) if spread.any() else None,
                  "contrast_p50": float(np.percentile(contrast[local], 50)) if local.any() and high_precision else None}
        if area < 100 or rejected:
            values = dict.fromkeys(values)
        context = {"valid_area_px": area, "line_pixels": int(local.sum()), "affected_area_px": int(spread.sum()),
                   "high_area_px": int(dense.sum()), "network_evidence_ref": network_key,
                   "excluded_stable_pixels": int((stable & domain).sum()),
                   "quality_status": quality.get("quality_status", "UNKNOWN"), "high_precision": high_precision}
        _emit(rows, basis, "07", region, profile, values, context,
              "rejected_quality" if rejected else ("high_precision_not_confirmed" if not high_precision else "insufficient_valid_region"))
    derived.update({"lines_fine_skeleton": fine.astype(np.uint8), "lines_fine_area": affected.astype(np.uint8),
                    "lines_fine_high": high.astype(np.uint8), "lines_stable_skeleton": stable.astype(np.uint8)})
    return {"measurements": rows, "basis": basis, "arrays": derived}
