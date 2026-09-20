"""V3 stage-one measurements from saved detector evidence, without scoring.

Pixel areas refer to the detector's aligned coordinate system, never display
marker radii or physical mm. Distribution samples remain available for later
parameter evaluation even when a threshold-dependent measurement is pending.
"""
import cv2
import numpy as np

from .registry import measurement
from .regions import build
from .measurements import instance_samples, component_samples
from .stage1_two_d_phenotypes import acne_evidence, exclusion_evidence, pigment_support

VERSION = "doctor-v3-stage1-two-d-1"
PROJECTS = {"pores": "01", "surface_gloss": "02", "porphyrin": "02",
            "spots": "03", "brown": "03", "uv_spots": "03",
            "redness": "04", "vascular": "05", "acne": "06"}


def quality_reason(meta):
    quality = meta.get("quality", {})
    status = meta.get("quality_status", quality.get("quality_status", quality.get("图像质量状态")))
    return "rejected_quality" if str(status).upper() in ("REJECT", "REJECTED", "FAIL", "FAILED", "UNASSESSABLE") else None


def observed_quality_status(meta):
    quality = meta.get("quality") or {}
    status = meta.get("quality_status", quality.get("quality_status", quality.get("图像质量状态", "UNKNOWN")))
    return str(status).upper()


def _validate(arrays):
    if "valid" not in arrays or "landmarks" not in arrays:
        raise ValueError("missing coordinate evidence")
    shape = np.asarray(arrays["valid"]).shape
    if len(shape) != 2:
        raise ValueError("invalid coordinate evidence")
    points = np.asarray(arrays["landmarks"])
    if points.ndim != 2 or points.shape[1] < 2 or not np.all(np.isfinite(points)):
        raise ValueError("invalid landmark evidence")
    for key in ("valid", "instances", "continuous", "high", "score", "skeleton"):
        if key in arrays:
            value = np.asarray(arrays[key])
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError("invalid or mismatched evidence: " + key)


def _cutoff(thresholds, name, region):
    value = thresholds.get(name)
    if isinstance(value, dict):
        value = value.get(region)
    if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
        raise ValueError("invalid threshold: " + name)
    return value


def _pixel_summary(values):
    """Compact exact statistics; detailed NPZ remains the pixel replay source."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "sum": 0.0, "mean": None, "p50": None, "p90": None}
    return {"count": int(values.size), "sum": float(values.sum()),
            "mean": float(values.mean()), "p50": float(np.percentile(values,50)),
            "p90": float(np.percentile(values,90)), "min": float(values.min()), "max": float(values.max())}


def analyze(data, metadata, profile, registration=None, thresholds=None):
    """Return old measurement rows plus independently auditable per-row basis."""
    if profile not in ("consumer", "institution"):
        raise ValueError("explicit capture profile required")
    thresholds = thresholds or {}
    result = {"measurements": [], "basis": {}, "arrays": {}, "visual_data": {}}
    for project in PROJECTS.keys() & data.keys():
        _validate(data[project])
    acne, labels, acne_reason = acne_evidence(data, metadata)
    if acne is not None:
        result["arrays"]["acne_classification"] = acne["classification_mask"]
        result["visual_data"].update(classified_acne={**acne, "instances": (acne["classification_mask"] > 0).astype(np.uint8)*255}, acne_labels=labels)
    exclusion_data = dict(data)
    if acne is not None and acne_reason is None:
        exclusion_data["acne"] = result["visual_data"]["classified_acne"]
    exclusion, exclusion_reason = exclusion_evidence(exclusion_data, metadata, profile, registration)
    if exclusion is not None:
        result["arrays"]["redness_excluded"] = exclusion.astype(np.uint8)
        red = data["redness"]
        remaining = (red["valid"] > 0) & ~exclusion
        result["visual_data"]["diffuse_arrays"] = {**red, "valid": remaining.astype(np.uint8)*255,
            **{key: ((red[key] > 0) & remaining).astype(np.uint8)*255 for key in ("continuous", "high") if key in red}}
    supported, support_reason, unknown_support = pigment_support(data, metadata, profile, registration)
    if supported is not None:
        result["arrays"]["spots_supported"] = supported.astype(np.uint8)
    if unknown_support is not None:
        result["arrays"]["spots_unknown_support"] = unknown_support.astype(np.uint8)

    for project, module in PROJECTS.items():
        if project not in data:
            continue
        a, meta = data[project], metadata.get(project, {})
        masks = build(a["valid"], a["landmarks"], module)
        source = meta.get("source_kind", "rgb_measured" if profile == "consumer" or project in ("pores", "spots", "acne") else "spectral_measured")
        if profile == "consumer" and project in ("uv_spots", "porphyrin", "spots", "redness", "acne"):
            source = "rgb_proxy"
        objects = None
        if project == "porphyrin" and "instances" in a and "score" in a:
            objects = component_samples((a["instances"] > 0) & (a["valid"] > 0), a["score"], a["valid"] > 0)
        for region, domain in masks.items():
            area = int(domain.sum())
            common = {"algorithm_version": VERSION, "project": project,
                      "coordinate_system": meta.get("coordinate_system", "detector_aligned_pixels"),
                      "source_role": meta.get("pixel_source_role", "RGB_M" if profile == "consumer" or project in ("pores", "spots", "acne") else "365_M" if project in ("uv_spots", "porphyrin") else "PP_M" if project == "surface_gloss" else "CP_M"),
                      "effective_area_px": area, "original_valid_area_px": area,
                      "quality_status": observed_quality_status(meta),
                      "quality_reason": quality_reason(meta)}
            gate = quality_reason(meta) or ("insufficient_valid_region" if area < 100 else None)

            def emit(name, value, unit, formula, reason=None, **details):
                reason = gate or reason
                if gate:
                    value = None
                status = ("measured" if value is not None else "pending_parameter" if reason and "threshold" in reason
                          else "no_targets" if reason == "no_valid_targets" else "missing_evidence")
                if gate or (reason and reason.startswith(("insufficient_remaining", "insufficient_classification_domain"))):
                    status = "insufficient_quality"
                row = measurement(module, name, region, profile, value, unit, reason, source)
                row["measurement_status"] = status
                result["measurements"].append(row)
                result["basis"][row["metric_id"] + ":" + region] = {
                    **common, **details, "formula": formula, "measurement_status": status,
                    "reason": reason if value is None else None}

            mask = (a["instances"] > 0) & domain if "instances" in a else None
            signal = a.get("score")
            vals = signal[mask] if signal is not None and mask is not None else np.array([])
            if project == "pores":
                located, unlocated = [], []
                for obj in meta.get("instances", []):
                    center = obj.get("centroid", obj.get("center"))
                    if center is None or len(center) < 2 or not np.all(np.isfinite(center[:2])):
                        unlocated.append(obj)
                    else:
                        located.append(obj)
                selected = instance_samples(a, {**meta, "instances": located}, domain)
                sizes = [float(o.get("area", o.get("measurement_area", o.get("area_px")))) for o in selected
                         if o.get("area", o.get("measurement_area", o.get("area_px"))) is not None]
                sizes_ok = len(sizes) == len(selected) and all(np.isfinite(v) and v > 0 for v in sizes)
                if any(not np.isfinite(v) or v <= 0 for v in sizes):
                    raise ValueError("invalid pore instance area")
                evidence_ok = "instances" in meta and not unlocated and not (not meta["instances"] and mask is not None and mask.any())
                details = {"count": len(selected) if evidence_ok else None, "instance_areas_px2": sizes,
                           "unlocated_instance_count": len(unlocated),
                           "unlocated_instance_areas_px2": [o.get("area", o.get("measurement_area", o.get("area_px"))) for o in unlocated],
                           "instance_ids": [o.get("id", o.get("feature_id")) for o in selected],
                           "size_source": "detector_instance_area_not_marker_radius"}
                emit("density", len(selected) * 100000 / area if evidence_ok and area else None,
                     "个/10万标准化有效像素", "count * 100000 / effective_area_px", "missing_instance_geometry" if unlocated else "missing_instance_evidence", **details)
                emit("area_p50", float(np.median(sizes)) if evidence_ok and sizes_ok and sizes else None,
                     "标准化px²", "percentile(instance_areas_px2, 50)",
                     "no_valid_targets" if evidence_ok and not selected else "missing_instance_size", **details)
                cutoff = _cutoff(thresholds, "large_pore_area", region)
                value = sum(v >= cutoff for v in sizes) * 100000 / area if cutoff is not None and evidence_ok and sizes_ok and area else None
                emit("large_density", value, "个/10万标准化有效像素", "count(area >= threshold) * 100000 / effective_area_px",
                     "missing_instance_geometry" if unlocated else "large_pore_threshold_pending" if cutoff is None else "missing_instance_size", threshold=cutoff, **details)
            elif project == "surface_gloss":
                summary = _pixel_summary(vals)
                emit("gloss_area", float(mask.sum()/area) if mask is not None and area else None, "比例", "gloss_pixels / effective_area_px", "missing_gloss_mask")
                high = (a["high"] > 0) & domain if "high" in a else None
                emit("gloss_high_area", float(high.sum()/area) if high is not None and area else None, "比例", "high_gloss_pixels / effective_area_px", "missing_high_gloss_mask")
                for name, value in (("gloss_mean", float(vals.mean()) if vals.size else None),
                                    ("gloss_p90", float(np.percentile(vals, 90)) if vals.size else None)):
                    emit(name, value, "0-1", "mean(gloss_pixel_scores)" if name.endswith("mean") else "percentile(gloss_pixel_scores, 90)",
                         "missing_score_evidence" if signal is None else "no_valid_targets", pixel_score_summary=summary)
            elif project == "porphyrin":
                selected = instance_samples(a, meta, domain) if profile == "consumer" and meta.get("instances") else [o for o in objects or [] if domain[int(round(o["centroid"][1])), int(round(o["centroid"][0]))]]
                strengths = [float(o.get("mean_score", o.get("intensity"))) for o in selected if o.get("mean_score", o.get("intensity")) is not None]
                if any(not np.isfinite(v) for v in strengths):
                    raise ValueError("invalid porphyrin intensity")
                valid_samples = objects is not None and len(strengths) == len(selected)
                details = {"count": len(selected), "instance_intensities": strengths,
                           "sample_definition": "detector_mean_score" if profile == "consumer" and meta.get("instances") else "connected_component_mean_score"}
                cutoff = _cutoff(thresholds, "porphyrin_high_intensity", region)
                emit("porphyrin_high_density", sum(v >= cutoff for v in strengths)*100000/area if cutoff is not None and valid_samples and area else None,
                     "个/10万标准化有效像素", "count(intensity >= threshold) * 100000 / effective_area_px",
                     "porphyrin_threshold_pending" if cutoff is None else "missing_score_evidence", threshold=cutoff, **details)
                emit("porphyrin_p90", float(np.percentile(strengths, 90)) if valid_samples and strengths else None,
                     "0-1", "percentile(instance_intensities, 90)", "no_valid_targets" if valid_samples else "missing_score_evidence", **details)
            elif project in ("spots", "brown", "uv_spots"):
                prefix = {"spots": "spots", "brown": "brown", "uv_spots": "uv"}[project]
                support = supported if project == "spots" else a.get("continuous")
                reason = support_reason if project == "spots" else "missing_continuous_mask"
                if project == "spots" and unknown_support is not None and np.any(unknown_support & domain):
                    support, reason = None, "insufficient_pigment_support_coverage"
                local = (support > 0) & domain if support is not None else None
                scores = signal[local] if signal is not None and local is not None else np.array([])
                details = {"affected_area_px": int(local.sum()) if local is not None else None,
                           "continuous_definition": meta.get("continuous_definition"),
                           "continuous_source": meta.get("continuous_source"),
                           "pixel_score_summary": _pixel_summary(scores), "support_authorized": supported is not None if project == "spots" else None}
                emit(prefix+".area", float(local.sum()/area) if local is not None and area else None,
                     "比例", "continuous_supported_pixels / effective_area_px", reason, **details)
                emit(prefix+".p90", float(np.percentile(scores, 90)) if scores.size else None, "0-1", "percentile(continuous_supported_scores, 90)",
                     reason if local is None else "missing_score_evidence" if signal is None else "no_valid_targets", **details)
            elif project == "redness":
                remaining = domain & ~exclusion if exclusion is not None else None
                denominator = int(remaining.sum()) if remaining is not None else None
                details = {"excluded_area_px": area-denominator if denominator is not None else None,
                           "effective_area_px": denominator, "exclusion_components": ["focal_red", "vascular", "acne_candidates"]}
                local = (a["continuous"] > 0) & remaining if "continuous" in a and remaining is not None else None
                values = signal[local] if signal is not None and local is not None else np.array([])
                reason = exclusion_reason or ("insufficient_remaining_valid_region" if denominator is not None and denominator < 100 else None)
                summary = _pixel_summary(values)
                for name in ("area", "high_area", "mean", "p90"):
                    value = None
                    missing = reason
                    if not reason and local is not None:
                        if name == "area": value = float(local.sum()/denominator)
                        elif name == "high_area" and "high" in a: value = float(((a["high"]>0)&remaining).sum()/denominator)
                        elif name == "mean" and values.size: value = float(values.mean())
                        elif name == "p90" and values.size: value = float(np.percentile(values,90))
                    missing = missing or ("missing_continuous_mask" if local is None else "missing_high_mask" if name == "high_area" and "high" not in a else "missing_score_evidence" if signal is None else "no_valid_targets")
                    emit(name, value, "比例" if "area" in name else "0-1", {"area":"diffuse_pixels / effective_area_px", "high_area":"high_diffuse_pixels / effective_area_px", "mean":"mean(diffuse_pixel_scores)", "p90":"percentile(diffuse_pixel_scores, 90)"}[name], missing,
                         pixel_score_summary=summary, **details)
            elif project == "vascular":
                step = thresholds.get("vascular_grid_step_px", 16)
                if isinstance(step, bool) or int(step) != step or step < 1:
                    raise ValueError("invalid vascular grid step")
                step = int(step)
                affected = np.zeros_like(domain)
                cells = []
                if mask is not None:
                    for y in range(0, domain.shape[0], step):
                        for x in range(0, domain.shape[1], step):
                            block = np.s_[y:y+step, x:x+step]
                            if mask[block].any():
                                affected[block] = domain[block]
                                cells.append([x,y])
                count = int(affected.sum())
                skeleton = a.get("skeleton")
                length = int(((skeleton > 0) & affected).sum()) if skeleton is not None else None
                clusters = int(cv2.connectedComponents(affected.astype(np.uint8), connectivity=8)[0]-1)
                details = {"affected_area_px": count, "grid_step_px": step, "grid_origins_xy": cells,
                           "skeleton_length_px": length, "cluster_connectivity": 8}
                for name,value,unit in (("clusters",clusters,"处"),("roi_count",len(cells),"标准化网格"),
                                        ("affected_area",count/area if area else None,"比例"),
                                        ("local_density",length*10000/count if length is not None and count else None,"骨架px/万受累像素")):
                    emit(name, value if mask is not None else None, unit, {"clusters":"components(affected_grid_mask, 8)","roi_count":"len(grid_origins_xy)","affected_area":"affected_area_px / effective_area_px","local_density":"skeleton_length_px * 10000 / affected_area_px"}[name],
                         "missing_instance_mask" if mask is None else "missing_skeleton" if skeleton is None and name=="local_density" else "no_valid_targets", **details)
                if region == "full_face": result["arrays"]["vascular_affected_grid"] = affected.astype(np.uint8)
            elif project == "acne":
                selected = [o for o in labels if o.get("centroid") is not None and domain[o["centroid"][1],o["centroid"][0]]]
                unknown = sum(o["class"] == "unknown" for o in selected)
                reason = acne_reason or ("unknown_classification_targets" if unknown else None)
                classification_domain = acne.get("classification_valid") if acne is not None else None
                class_area = int((classification_domain & domain).sum()) if classification_domain is not None else 0
                reason = reason or ("insufficient_classification_domain" if class_area < 100 else None)
                details = {"classification_samples": selected, "unknown_count": unknown,
                           "effective_area_px": class_area, "excluded_area_px": area-class_area,
                           "denominator_definition": "acne_valid_intersect_registered_pore_valid",
                           "mutually_exclusive_classes": ["erythema", "papule", "pustule", "unknown"]}
                for kind in ("erythema", "papule", "pustule"):
                    count = sum(o["class"] == kind for o in selected)
                    for suffix,value,unit in (("count",count,"个"),("density",count*100000/class_area if class_area else None,"个/10万标准化有效像素")):
                        emit(kind+"_"+suffix, value if reason is None else None, unit,
                             "classified_count" if suffix == "count" else "classified_count * 100000 / effective_area_px",
                             reason, observed_classified_count=count, **details)
    return result
