"""Spatial measurements from internal arrays, never from rendered marker pixels."""
import json
from pathlib import Path
import cv2
import numpy as np
from .registry import MODULES, measurement
from .regions import build

def read_evidence(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    for key in ("valid", "landmarks"):
        if key not in arrays:
            raise ValueError("missing evidence " + key)
    shape = arrays["valid"].shape
    for key in ("instances", "continuous", "high", "score", "skeleton"):
        if key in arrays and arrays[key].shape != shape:
            raise ValueError("evidence coordinate mismatch")
    return arrays, meta

def instance_samples(arrays, meta, region):
    result = []
    for obj in meta.get("instances", []):
        center = obj.get("centroid", obj.get("center"))
        if center is None:
            continue
        x,y = [int(round(v)) for v in center[:2]]
        if 0 <= y < region.shape[0] and 0 <= x < region.shape[1] and region[y,x]:
            result.append(obj)
    return result

def component_samples(mask, score, region):
    n, labels, stats, centers = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), 8)
    sums = np.bincount(labels.ravel(),weights=np.asarray(score,dtype=float).ravel(),minlength=n)
    result = []
    for i in range(1,n):
        x,y = centers[i]
        if region[int(round(y)),int(round(x))]:
            result.append({"area": int(stats[i,cv2.CC_STAT_AREA]), "centroid":[float(x),float(y)],
                           "intensity":float(sums[i]/stats[i,cv2.CC_STAT_AREA])})
    return result

def measure_source(project, arrays, meta, profile, *, thresholds=None):
    thresholds = thresholds or {}
    module = {"pores":"01","surface_gloss":"02","porphyrin":"02","spots":"03","brown":"03","uv_spots":"03","redness":"04","vascular":"05"}.get(project)
    if module is None:
        return []
    regions = build(arrays["valid"], arrays["landmarks"], module)
    rows = []
    source = meta.get("source_kind", "rgb_measured" if profile == "consumer" or project=="pores" else "spectral_measured")
    if profile == "consumer" and project in ("porphyrin","uv_spots"):
        source = "rgb_proxy"
    mask = arrays.get("instances", np.zeros_like(arrays["valid"])) > 0
    signal = arrays.get("score", np.zeros_like(arrays["valid"], dtype=float))
    porphyrin_objects = component_samples(mask,signal,arrays["valid"]>0) if project=="porphyrin" else []
    for region_name, region in regions.items():
        area = int(np.count_nonzero(region))
        def emit(name, value, unit="标准化指数", reason=None):
            if area < 100 or meta.get("quality_status") == "REJECT":
                value, reason = None, "insufficient_valid_region_or_rejected_quality"
            rows.append(measurement(module, name, region_name, profile, value, unit, reason, source))
        local = mask & region
        values = signal[local]
        if project == "pores":
            objects = instance_samples(arrays, meta, region)
            sizes = [o.get("area",o.get("area_px")) for o in objects]
            sizes = [float(a) for a in sizes if a is not None]
            emit("density", len(objects)*100000/area if area else None, "个/10万标准化有效像素")
            emit("area_p50", float(np.median(sizes)) if sizes else None, "标准化px²", "no_valid_instances")
            cutoff = thresholds.get("large_pore_area", {}).get(region_name)
            emit("large_density", sum(a>=cutoff for a in sizes)*100000/area if cutoff is not None and area else None,
                 "个/10万标准化有效像素", "large_pore_threshold_uncalibrated")
        elif project == "surface_gloss":
            high = arrays.get("high")
            emit("gloss_area", float(local.sum()/area) if area else None, "比例")
            emit("gloss_high_area", float(((high>0)&region).sum()/area) if high is not None and area else None, "比例", "missing_high_gloss_mask")
            emit("gloss_mean", float(values.mean()) if values.size else None, "0-1", "no_valid_gloss")
        elif project == "porphyrin":
            objects = [o for o in porphyrin_objects if region[int(round(o["centroid"][1])),int(round(o["centroid"][0]))]]
            if profile=="consumer" and meta.get("instances"):
                objects = instance_samples(arrays,meta,region)
                strengths = [float(o["mean_score"]) for o in objects if o.get("mean_score") is not None]
            else:
                strengths = [float(o["intensity"]) for o in objects if o.get("intensity") is not None]
            cutoff = thresholds.get("porphyrin_high_intensity")
            emit("porphyrin_high_density", sum(x>=cutoff for x in strengths)*100000/area if cutoff is not None and area else None,
                 "个/10万标准化有效像素", "porphyrin_threshold_uncalibrated")
            emit("porphyrin_p90", float(np.percentile(strengths,90)) if strengths else None, "0-1", "no_valid_instances")
        elif project in ("spots","brown","uv_spots"):
            prefix = {"spots":"spots","brown":"brown","uv_spots":"uv"}[project]
            support = (arrays.get("continuous",mask)>0) & region
            emit(prefix+".area", float(support.sum()/area) if area else None, "比例")
            vals = signal[support]
            emit(prefix+".p90", float(np.percentile(vals,90)) if vals.size else None, "0-1", "no_valid_signal")
            if project == "spots" and not meta.get("multiview_verified", False):
                # A lone RGB mask is not evidence of multi-view support.
                for row in rows[-2:]:
                    row.update(value=None,status="unavailable",reason="multi_image_support_not_yet_verified")
        elif project == "redness":
            # Do not call a raw V2 red mask a V3 diffuse-only measurement.
            for name in ("area","high_area","mean","p90"):
                emit(name,None,reason="requires_focal_and_vascular_exclusion")
        elif project == "vascular":
            step = int(thresholds.get("vascular_grid_step_px",16))
            if step < 1:
                raise ValueError("invalid standardized grid size")
            skeleton = arrays.get("skeleton")
            affected = np.zeros_like(region)
            count = 0
            for y in range(0,region.shape[0],step):
                for x in range(0,region.shape[1],step):
                    sl = np.s_[y:y+step,x:x+step]
                    if np.any(local[sl]):
                        affected[sl] = region[sl]
                        count += 1
            clusters = max(cv2.connectedComponents(affected.astype(np.uint8),8)[0]-1,0)
            denominator = int(affected.sum())
            emit("clusters",clusters,"处")
            emit("roi_count",count,"标准化网格")
            emit("affected_area",denominator/area if area else None,"比例")
            emit("local_density",float(((skeleton>0)&affected).sum()*10000/denominator) if skeleton is not None and denominator else None,
                 "骨架px/万受累像素","no_affected_roi")
    return rows

def complete_measurements(measured, profile):
    indexed = {(m["metric_id"],m["region"]):m for m in measured}
    for module in MODULES:
        names = list(module.weights)
        if module.id == "03":
            names = [p+"."+n for p in ("spots","brown","uv") for n in ("area","p90")]
        for region in ("full_face",*module.regions):
            if module.id == "08":
                if region=="full_face":
                    continue
                names = ["main_count","length_burden","depth"] if region in ("forehead","glabella") else ["area","density","high_area","contrast_p50"]
            for name in names:
                key = (module.id+"."+name,region)
                if key not in indexed:
                    reason = "requires_calibrated_3d" if module.id in ("09","10","11") or (module.id=="08" and name=="depth") else "missing_v3_measurement_evidence"
                    indexed[key] = measurement(module.id,name,region,profile,reason=reason)
    return list(indexed.values())
