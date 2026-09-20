"""V3 phase-one regional 2.5D evidence, with explicit spatial-resolution limits.

Landmark relief describes coarse anatomical appearance only. Local skin pits and
line depths require independently sampled dense surfaces; no landmark raster
interpolation is used to manufacture them.
"""
import cv2
import numpy as np

from .registry import MODULES, measurement
from .regions import build
from .stage1_lines import _domains
from ._stage1_geometry_math import local_samples, sparse_relief, curvature, dense_residual
from ._stage1_appearance import line_appearance, surface_appearance, texture_classes, VERSION as APPEARANCE_VERSION
from .geometry_regions import anatomical_groove_paths
from ._stage1_groove_measurement import measure_grooves

VERSION = "v3-stage1-regional25d-2"
PARAMETERS = {"relative_depth_threshold": .002, "dense_reference_sigma_px": 12,
              "groove_target_min_pixels": 5, "groove_assignment_distance_fraction": .08,
              "groove_support_halfwidth_px": 4}
JAW_IDS = {"left_jaw": [132, 58, 172, 136, 150, 149, 176, 148, 152],
           "right_jaw": [361, 288, 397, 365, 379, 378, 400, 377, 152]}


def _region(arrays, module):
    return _domains(arrays, module)


def _assign_grooves(line, domains, paths=None, parameters=None):
    """A connected image target belongs to one anatomical groove observation."""
    effective={**PARAMETERS,**(parameters or {})}
    assigned = {name: np.zeros_like(line) for name in domains if name != "full_face"}
    count, labels = cv2.connectedComponents(line.astype(np.uint8), 8)
    records = []
    for label in range(1, count):
        target = labels == label
        size = int(target.sum())
        if size < effective["groove_target_min_pixels"]:
            continue
        candidates = []
        for name, domain in domains.items():
            if name=="full_face":
                continue
            overlap=int((target & domain).sum())
            score=float(overlap)
            if paths is not None and overlap:
                y,x=np.nonzero(target&domain)
                points=np.column_stack((x,y))
                path=paths[name]
                delta=np.diff(path,axis=0)
                lengths=np.linalg.norm(delta,axis=1)
                distance=np.full(len(points),np.inf)
                for anchor,vector,length in zip(path,delta,lengths):
                    if length<1:
                        continue
                    t=np.clip(((points-anchor)*vector).sum(axis=1)/(length*length),0,1)
                    distance=np.minimum(distance,np.linalg.norm(points-(anchor+t[:,None]*vector),axis=1))
                score/=1+float(np.median(distance))/max(float(lengths.sum())*effective["groove_assignment_distance_fraction"],1.)
            candidates.append((score,overlap,name))
        _, overlap, name = max(candidates, default=(0, 0, None))
        if overlap < effective["groove_target_min_pixels"]:
            continue
        assigned[name] |= target & domains[name]
        records.append({"target_id": label, "region": name, "original_pixels": size, "in_region_pixels": overlap})
    return assigned, records


def _aligned(source, target, mask):
    if source is target:
        return np.asarray(mask) > 0
    if source is None:
        return None
    a, b = np.asarray(source.get("landmarks", [])), np.asarray(target.get("landmarks", []))
    if a.shape != b.shape or a.shape[0] < 400:
        return None
    if mask.shape == target["valid"].shape and np.allclose(a[:, :2], b[:, :2], atol=.5):
        return np.asarray(mask) > 0
    from .phenotypes import warp_mask
    return warp_mask(source, target, mask)


def _emit(rows, basis, module, region, profile, name, value, context, reason=None):
    unit = "比例" if name in ("extent", "raised_area", "depressed_area", "jaw_continuity") else "标准化2.5D指数"
    row = measurement(module, name, region, profile, value, unit, reason or "missing_evidence", "regional_25d_proxy")
    row["definition_version"] = VERSION
    if module=="09" and name=="extent":
        row.update(unit="解剖范围状态分",direction="higher_health",source_kind="anatomical_extent_state")
    if context.get("appearance_proxy"):
        row["definition_version"] = APPEARANCE_VERSION
        row["source_kind"] = "image_appearance_proxy"
        row["unit"] = "比例" if name.endswith("_area") else "标准化外观指数"
    if name == "jaw_continuity":
        row["direction"] = "higher_health"
    rows.append(row)
    group = name.split("_")[0] if module == "10" else (("surface" if name in ("smoothness", "turning") else name) if module == "11" else "common")
    evidence_key = "evidence:geometry:" + module + ":" + region + ":" + group
    basis[evidence_key] = context
    compact = {k: v for k, v in context.items() if k in (
        "valid_area_px", "original_region_area_px", "excluded_area_px", "texture_support_area_px",
        "unclassified_texture_area_px", "appearance_proxy", "strict_dense_measurement",
        "formula", "source_formula_id", "quality_status", "parameter_status", "geometry_sampling",
        "resolution_limit", "class_area_px", "class_area_ratio", "class_intensity_p90",
        "regional_relief_p90", "surface_area_px", "affected_area_px", "image_extent_area_px")}
    if module=="09" and name=="extent":
        compact["extent_classification"]=context.get("extent_classification")
    basis[row["metric_id"]+":"+region] = {**compact, "evidence_ref": evidence_key, "physical_3d": False,
        "measurement_status": "measured" if value is not None else ("pending_parameter" if reason and ("threshold_pending" in reason or "not_publishable" in reason) else "no_target" if reason == "no_target" else "missing_evidence"),
        "reason": reason if value is None else None}


def _jaw(arrays, region):
    p = np.asarray(arrays["landmarks"], float)
    if len(p) < 400:
        return None, {"reason": "missing_anatomical_jaw_path"}
    points = p[JAW_IDS[region], :2]
    if not np.isfinite(points).all():
        return None, {"reason": "invalid_jaw_path"}
    xy = np.rint(points).astype(int)
    shape = arrays["valid"].shape
    inside = (xy[:, 0]>=0) & (xy[:, 0]<shape[1]) & (xy[:, 1]>=0) & (xy[:, 1]<shape[0])
    if not inside.all():
        return None, {"reason": "jaw_path_outside_image"}
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0):
        return None, {"reason": "degenerate_jaw_path"}
    cross = np.cross(segments[:-1], segments[1:])
    dot = (segments[:-1]*segments[1:]).sum(axis=1)
    turns = np.abs(np.arctan2(cross, dot))
    value = float(np.clip(1 - np.percentile(turns, 90)/(np.pi/2), 0, 1))
    return value, {"sample_ids": JAW_IDS[region], "path_xy": points.tolist(),
        "segment_length_px": lengths.tolist(), "turn_angles_radians": turns.tolist(),
        "formula": "clip(1-P90(abs(turn_angle))/(pi/2),0,1)",
        "source_formula": "V2 relative_face_geometry._jaw_metrics, applied separately per side",
        "geometry_sampling": "2d_anatomical_path", "depth_required": False}


def _exclusion(data, rgb, line_result):
    excluded = np.zeros(rgb["valid"].shape, bool)
    missing = []
    for name, field in (("pores", "instances"),):
        source = data.get(name)
        mask = _aligned(source, rgb, source[field] > 0) if source is not None and field in source else None
        if mask is None:
            missing.append(name)
        else:
            excluded |= mask
    wrinkle = data.get("wrinkle")
    if wrinkle is None:
        missing.append("wrinkle")
    else:
        for key in ("lines_fine_skeleton", "lines_stable_skeleton"):
            mask = _aligned(wrinkle, rgb, line_result.get(key, np.zeros(wrinkle["valid"].shape)) > 0)
            if mask is None:
                missing.append(key)
            else:
                excluded |= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        # Remaining detected structural lines are excluded as well.
        mask = _aligned(wrinkle, rgb, wrinkle["skeleton"] > 0)
        if mask is not None:
            excluded |= cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return excluded, missing


def analyze(data, metadata, profile, registration=None, thresholds=None):
    params = {**PARAMETERS, **(thresholds or {}).get("geometry", {})}
    rgb = data.get("rgb")
    rows, basis, derived = [], {}, {}
    if rgb is None:
        for module in ("08", "09", "10", "11"):
            for region in (("forehead", "glabella") if module == "08" else ("full_face", *MODULES[int(module)-1].regions)):
                names = ("depth",) if module == "08" else tuple(MODULES[int(module)-1].weights)
                for name in names:
                    if module == "11" and region != "full_face" and ((region.endswith("jaw") and name != "jaw_continuity") or
                         (not region.endswith("jaw") and (name == "jaw_continuity" or (name == "jowl" and region.endswith("midface"))))):
                        continue
                    _emit(rows, basis, module, region, profile, name, None, {}, "missing_rgb_geometry")
        return {"measurements": rows, "basis": basis, "arrays": derived}
    meta = metadata.get("rgb", {})
    valid = np.asarray(rgb["valid"]) > 0
    rejected = meta.get("quality_status") in ("REJECT", "FAILED") or any(
        flag in meta.get("quality_flags", []) for flag in ("EXTREME_POSE", "PARTIAL_FACE", "INVALID_FACE_GEOMETRY"))
    dense, dense_reason = dense_residual(rgb, meta, params["dense_reference_sigma_px"])
    wrinkle = data.get("wrinkle")
    line = _aligned(wrinkle, rgb, wrinkle["skeleton"] > 0) if wrinkle is not None else None
    from .stage1_lines import analyze as line_analyze
    line_arrays = data.get("_stage1_line_arrays")
    if line_arrays is None:
        line_arrays = line_analyze(data, metadata, profile, registration, thresholds)["arrays"] if wrinkle is not None else {}
    context_base = {"parameters": params, "quality_status": meta.get("quality_status", "UNKNOWN"),
                    "parameter_status": "engineering_candidate", "geometry_source": "rgb"}
    # Forehead/ glabella skin-depth is below FaceMesh sampling resolution.
    stable_domains = _domains(rgb, "08")
    for region in ("forehead", "glabella"):
        domain = stable_domains.get(region, np.zeros_like(valid))
        local = (line & domain) if line is not None else np.zeros_like(valid)
        values = dense[local & np.isfinite(dense)] if dense is not None else np.array([])
        positive = values[values > params["relative_depth_threshold"]]
        reason = "rejected_quality" if rejected else dense_reason or ("no_target" if not len(positive) else None)
        value = float(np.percentile(positive, 90)) if len(positive) and not rejected else None
        extra = {}
        if dense is None and not rejected and line is not None:
            if region == "glabella":
                from ._stage1_glabella_roi import build_glabella_support
                from ._stage1_appearance import GLABELLA_VERSION
                reference_domain = domain if rgb.get("region_masks", {}).get("08") is not None else build_glabella_support(rgb["valid"], rgb["landmarks"])
                stable = line_arrays.get("lines_stable_skeleton")
                source_domain = _domains(wrinkle, "08").get("glabella")
                alignment_reason, source_pixels = None, None
                local = np.zeros_like(valid)
                if stable is None or np.asarray(stable).shape != wrinkle["valid"].shape:
                    alignment_reason = "missing_original_stable_line_evidence"
                elif source_domain is None:
                    alignment_reason = "missing_source_glabella_domain"
                else:
                    source_local = (np.asarray(stable) > 0) & source_domain
                    source_pixels = int(source_local.sum())
                    identified = _aligned(wrinkle, rgb, source_local)
                    if identified is None:
                        alignment_reason = "missing_glabella_image_geometry_correspondence"
                    else:
                        local = identified & domain
                        if source_pixels > 0 and not local.any():
                            alignment_reason = "glabella_targets_without_valid_aligned_correspondence"
                if alignment_reason:
                    value, extra = None, {"reason": alignment_reason,
                        "appearance_formula_version": GLABELLA_VERSION, "reference_target_separated": True,
                        "target_area_px": int(domain.sum()), "reference_support_area_px": int(reference_domain.sum()),
                        "physical_depth": False}
                else:
                    value, extra = line_appearance(rgb, domain, local, reference_domain=reference_domain)
                extra.update(source_target_line_pixels=source_pixels, aligned_target_line_pixels=int(local.sum()),
                             target_alignment_status="missing" if alignment_reason else "corresponding")
            else:
                identified = _aligned(wrinkle, rgb, line_arrays.get("lines_stable_skeleton", np.zeros_like(wrinkle["valid"])) > 0)
                local = identified & domain if identified is not None else np.zeros_like(valid)
                value, extra = line_appearance(rgb, domain, local)
            extra.update({"appearance_proxy": True, "strict_dense_measurement": {"value": None, "reason": dense_reason}})
            reason = extra.get("reason") if value is None else None
        elif region == "glabella" and dense is None and not rejected and line is None:
            value, reason = None, "missing_glabella_image_geometry_correspondence"
        _emit(rows, basis, "08", region, profile, "depth", value,
              {**context_base, "valid_area_px": int(domain.sum()), "line_pixels": int(local.sum()),
               "depth_samples": positive.tolist(), "formula": "P90(positive local z residual on identified lines)",
               "geometry_sampling": "dense_surface" if dense is not None else "sparse_landmarks",
               "resolution_limit": "sparse_face_landmarks_do_not_resolve_individual_wrinkle_depth", **extra}, reason)
        if region == "glabella" and extra.get("reference_target_separated"):
            rows[-1]["definition_version"] = extra["appearance_formula_version"]
    # Each anatomical groove obtains its own samples and baseline.
    groove_domains = _region(rgb, "09")
    groove_line = np.zeros_like(valid) if line is None else line.copy()
    if wrinkle is not None:
        for field in ("lines_fine_skeleton", "lines_stable_skeleton"):
            other = _aligned(wrinkle, rgb, line_arrays.get(field, np.zeros_like(wrinkle["valid"])) > 0)
            if other is not None:
                groove_line &= ~other
    paths = anatomical_groove_paths(rgb["landmarks"])
    assigned, assignment = _assign_grooves(groove_line, groove_domains, paths, params)
    groove_results = measure_grooves(rgb, groove_domains, assigned, assignment, paths,
                                    params, context_base, rejected, line is not None,
                                    (thresholds or {}).get("anatomical_extent"))
    for region, result in groove_results.items():
        for name, value in result["values"].items():
            _emit(rows, basis, "09", region, profile, name, value,
                  result["context"], result["reasons"][name])
        derived["geometry_groove_"+region] = result["mask"]
    excluded, missing_exclusions = _exclusion(data, rgb, line_arrays)
    derived["geometry_surface_exclusion"] = excluded.astype(np.uint8)
    texture = data.get("texture")
    texture_mask = _aligned(texture, rgb, texture["instances"] > 0) if texture is not None else None
    classes, class_records, ambiguous = ({}, [], [])
    intensity_map = None
    if texture is not None:
        original_classes, class_records, ambiguous = texture_classes(texture, metadata.get("texture", {}))
        classes = {name: _aligned(texture, rgb, mask) for name, mask in original_classes.items()}
        # Intensity is never coerced through a binary-mask warp.
        if (texture["valid"].shape == valid.shape and
                np.allclose(np.asarray(texture["landmarks"])[:, :2], np.asarray(rgb["landmarks"])[:, :2], atol=.5)):
            intensity_map = np.asarray(texture.get("score"), float)
        for name, mask in classes.items():
            if mask is not None:
                derived["geometry_texture_"+name] = (mask & ~excluded).astype(np.uint8)
    classified_mask = np.zeros_like(valid)
    for mask in classes.values():
        if mask is not None:
            classified_mask |= mask
    unclassified_mask = texture_mask & ~classified_mask if texture_mask is not None else np.zeros_like(valid)
    basis["evidence:geometry:texture_classes"] = {"instances": class_records, "ambiguous_components": ambiguous}
    for region, domain in _region(rgb, "10").items():
        target = domain & ~excluded
        if dense is None:
            target &= ~unclassified_mask
        if dense is not None:
            target &= np.isfinite(dense)
        support = target & texture_mask if texture_mask is not None else np.zeros_like(valid)
        reason = "rejected_quality" if rejected else dense_reason
        if missing_exclusions:
            reason = "missing_exclusion_evidence"
        elif texture_mask is None:
            reason = "missing_texture_correspondence"
        elif int(target.sum()) < 100:
            reason = "insufficient_valid_region"
        context = {**context_base, "valid_area_px": int(target.sum()),
                   "original_region_area_px": int(domain.sum()), "excluded_area_px": int((domain & excluded).sum()),
                   "texture_support_area_px": int(support.sum()), "missing_exclusions": missing_exclusions,
                   "unclassified_texture_area_px": int((domain & unclassified_mask).sum()),
                   "formula": "signed dense local z residual intersected with texture evidence after exclusions",
                   "resolution_limit": "landmark Z cannot distinguish small raised or depressed skin targets"}
        for kind, sign in (("raised", -1), ("depressed", 1)):
            samples = sign*dense[support] if dense is not None else np.array([])
            positive = samples[samples > params["relative_depth_threshold"]]
            context_kind = {**context, "height_depth_samples": positive.tolist(), "target_pixel_count": len(positive)}
            classification_known = "instances" in metadata.get("texture", {}) and (bool(class_records) or not np.any(texture["instances"]))
            appearance = dense is None and not rejected and not missing_exclusions and intensity_map is not None and classes.get(kind) is not None and classification_known
            appearance_area, appearance_burden = None, None
            if appearance:
                appearance_area, appearance_burden, evidence = surface_appearance(rgb, target, classes[kind], intensity_map)
                context_kind.update({**evidence, "appearance_proxy": True,
                    "texture_classification_ref": "evidence:geometry:texture_classes",
                    "texture_instance_ids": [r["source_id"] for r in class_records if target[r["centroid"][1], r["centroid"][0]]],
                    "strict_dense_measurement": {"value": None, "reason": dense_reason}})
            for name, value in ((kind+"_area", float(len(positive)/max(int(target.sum()), 1))),
                                (kind+"_p90", float(np.percentile(positive, 90)) if len(positive) else None)):
                row_reason = reason
                if appearance:
                    value = appearance_area if name.endswith("_area") else appearance_burden
                    row_reason = context_kind.get("reason") if value is None else None
                    if value is None and context_kind.get("class_area_px") == 0:
                        row_reason = "no_target"
                _emit(rows, basis, "10", region, profile, name, value if not row_reason else None,
                      context_kind, row_reason or ("no_target" if value is None else None))
    for region, domain in _region(rgb, "11").items():
        if region == "full_face":
            continue
        context = {**context_base, "valid_area_px": int(domain.sum())}
        if region.endswith("jaw"):
            value, jaw = _jaw(rgb, region)
            context.update(jaw)
            _emit(rows, basis, "11", region, profile, "jaw_continuity", None if rejected else value,
                  context, "rejected_quality" if rejected else jaw.get("reason"))
            continue
        samples, reason = local_samples(rgb, domain)
        curves = None
        if samples:
            curves, reason = curvature(samples)
            context.update({"sample_ids": samples["ids"].tolist(), "sample_xy": samples["pixel_xy"].tolist(),
                            "relative_z": samples["z"].tolist(), "scale_px": samples["scale_px"]})
        if curves:
            context.update(curves)
        context["formula"] = "P90(abs(adjacent original triangle slope-curvature minus same-mesh regional quadratic reference)); connected edge sign-change magnitude"
        if rejected:
            reason = "rejected_quality"
        values = {"smoothness": curves["p90_deviation"] if curves else None,
                  "turning": curves["turning"] if curves else None}
        if region.endswith("lower_face"):
            values["jowl"] = curves["convex"] if curves else None
        for name, value in values.items():
            _emit(rows, basis, "11", region, profile, name, value if not reason else None, context, reason)
    domains = _region(rgb, "11")
    surface = np.zeros_like(valid)
    lower = np.zeros_like(valid)
    for region, domain in domains.items():
        if region.endswith(("midface", "lower_face")):
            surface |= domain
        if region.endswith("lower_face"):
            lower |= domain
    for names, domain in ((("smoothness", "turning"), surface), (("jowl",), lower)):
        samples, reason = local_samples(rgb, domain)
        curves = None
        if samples is not None:
            curves, reason = curvature(samples)
        context = {**context_base, "valid_area_px": int(domain.sum()),
                   "formula": "recompute curvature over union of eligible anatomical domains; never average regional P90"}
        if curves:
            context.update(curves)
        for name in names:
            field = {"smoothness": "p90_deviation", "turning": "turning", "jowl": "convex"}[name]
            value = curves[field] if curves and not rejected else None
            _emit(rows, basis, "11", "full_face", profile, name, value, context,
                  "rejected_quality" if rejected else reason)
    left_value, left_jaw = _jaw(rgb, "left_jaw")
    right_value, right_jaw = _jaw(rgb, "right_jaw")
    turns = left_jaw.get("turn_angles_radians", []) + right_jaw.get("turn_angles_radians", [])
    jaw_value = float(np.clip(1-np.percentile(turns, 90)/(np.pi/2), 0, 1)) if left_value is not None and right_value is not None else None
    _emit(rows, basis, "11", "full_face", profile, "jaw_continuity", None if rejected else jaw_value,
          {**context_base, "formula": "clip(1-P90(concatenated left/right original turn angles)/(pi/2),0,1)",
           "turn_angles_radians": turns, "bilateral_complete": left_value is not None and right_value is not None,
           "geometry_sampling": "2d_anatomical_path"},
          "rejected_quality" if rejected else ("missing_bilateral_jaw_evidence" if jaw_value is None else None))
    return {"measurements": rows, "basis": basis, "arrays": derived}
