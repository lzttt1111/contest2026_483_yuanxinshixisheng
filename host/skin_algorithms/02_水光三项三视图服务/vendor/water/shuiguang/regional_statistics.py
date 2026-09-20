"""Versioned per-module/per-region primary observations. No cross-view lesion sum."""
import copy
import math
import cv2
import numpy as np
from .anatomy import LEAVES, classify, anchor
from .config import VIEWS, MODULES

VERSION = "regional_observation_statistics_v2.2"
MIN_PRIMARY_RELATIVE_AREA = .60
MIN_PIXELS = 128
# Explicit source definitions; never calculate scalar statistics from a colour heatmap.
INSTANCE_FIELDS = {
    "pores": ("measurement_area", "单体面积", "分析像素²"),
    "texture": ("measurement_area", "标注单体面积", "分析像素²"),
    "spots": ("mean_deltaE", "实例综合色差", "ΔE"),
    "brown": ("area", "单体标注面积", "分析像素²"),
    "redness": ("area", "单体标注面积", "分析像素²"),
    "vascular": ("skeleton_length_px", "实例骨架长度", "分析像素"),
    "acne": ("confidence", "候选置信度", "0–1"),
    "surface_gloss": ("area_px", "连续高光范围面积", "分析像素²"),
}
REJECT_FLAGS = {"NO_FACE", "MULTIPLE_FACES", "INVALID_IMAGE", "PREPROCESS_FAILED",
                "BLUR", "UNDEREXPOSED", "OVEREXPOSED", "SEVERE_LOCAL_LIGHTING"}

def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {"p50": float(np.percentile(values, 50)) if values.size else None,
            "p90": float(np.percentile(values, 90)) if values.size else None,
            "sample_count": int(values.size)}

def photo_quality(image, domain, flags=()):
    valid = domain > 0
    n = int(valid.sum())
    if n < MIN_PIXELS:
        return {"eligible": False, "tier": 0, "reason": "有效分析区域不足", "sharpness": 0}
    if REJECT_FLAGS.intersection(flags):
        return {"eligible": False, "tier": 0, "reason": "本视角采集质量不满足要求", "sharpness": 0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[valid]
    bright = float(np.mean(values > 248))
    dark = float(np.mean(values < 20))
    if bright > .25 or dark > .45:
        return {"eligible": False, "tier": 0, "reason": "本区域严重过曝或过暗", "sharpness": 0}
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(lap[interior])) if interior.any() else 0.
    # Sharpness ranks comparable observations, never defines disease or a skin score.
    tier = 1 if bright > .05 or dark > .10 else 2
    return {"eligible": True, "tier": tier, "reason": "局部皮肤证据有效",
            "sharpness": sharpness, "bright_fraction": bright, "dark_fraction": dark}

def module_observations(capture, view, module, item, evidence, anatomy, points, image, flags=(), boundary_radius=2):
    valid = np.asarray(evidence["valid_mask"]) > 0
    mask = (np.asarray(evidence["mask"]) > 0) & valid
    response = np.asarray(evidence["response"])
    rows, rejected = [], []
    for ordinal, raw in enumerate(item.get("instances", []), 1):
        point = list(raw["centroid"])
        x, y = np.rint(point).astype(int)
        oid = f"{capture}/{view}/{module}/{raw.get('id', ordinal)}"
        if not (0 <= y < valid.shape[0] and 0 <= x < valid.shape[1] and valid[y, x]):
            rejected.append({"observation_id": oid, "reason": "centre_outside_saved_effective_domain"})
            continue
        region, state, possible = classify(point, anatomy.labels,boundary_radius=boundary_radius)
        rows.append({"observation_id": oid, "local_id": raw.get("id", ordinal),
                     "region_id": region, "assignment": state, "candidate_regions": possible,
                     "centroid": point, "original_centroid": raw.get("original_centroid"),
                     "landmark_anchor": anchor(point, points),
                     "source_instance": copy.deepcopy(raw)})
    regions = {}
    for name in LEAVES:
        domain = valid & (anatomy.labels == LEAVES.index(name) + 1)
        n = int(domain.sum())
        selected = [row for row in rows if row["region_id"] == name]
        uncertain = [row for row in rows if name in row["candidate_regions"] and row["region_id"] is None]
        quality = photo_quality(image, domain.astype(np.uint8), flags)
        eligible = item.get("status") == "success" and quality["eligible"]
        if not eligible:
            regions[name] = {"status": "unavailable", "count": None, "coverage": None,
                "analysis_pixels": n, "density_per_100k_pixels": None,
                "reason": quality["reason"] if item.get("status") == "success" else "本检测项未完成",
                "quality": quality, "uncertain_count": len(uncertain)}
            continue
        field, label, unit = INSTANCE_FIELDS[module]
        samples = [r["source_instance"][field] for r in selected
                   if isinstance(r["source_instance"].get(field), (int, float))]
        source_values = (response[domain] if response.ndim == 2 and module not in {"acne", "spots"} else [])
        region = {"status": "available", "count": len(selected),
                  "analysis_pixels": n, "coverage": (None if item.get("mask_semantics") == "candidate_box"
                      else int(np.count_nonzero(mask & domain)) / n),
                  "coverage_status": ("not_measured_candidate_box" if item.get("mask_semantics") == "candidate_box"
                                      else "saved_support_mask_not_physical_area"),
                  "density_per_100k_pixels": len(selected) * 100000 / n,
                  "uncertain_count": len(uncertain), "quality": quality,
                  "observation_ids": [r["observation_id"] for r in selected],
                  "uncertain_observation_ids": [r["observation_id"] for r in uncertain],
                  "instance_distribution": {"field": field, "label": label, "unit": unit, **quantiles(samples)},
                  "response_distribution": {"source": "scalar_detector_response" if len(source_values) else "not_applicable",
                                            **quantiles(source_values)},
                  "mask_semantics": item.get("mask_semantics", "detected_support")}
        region["has_evidence"] = bool(region["count"] or (region["coverage"] or 0) > 0 or uncertain)
        regions[name] = region
    return {"status": item.get("status", "unavailable"), "observations": rows,
            "uncertain_observations": [r for r in rows if r["assignment"] != "assigned"],
            "rejected_observations": rejected, "regions": regions}

def select_primary(observed_views):
    result = {}
    for region in LEAVES:
        modules = {}
        for module in MODULES:
            candidates, excluded = [], []
            for view in VIEWS:
                m = observed_views.get(view, {}).get("modules", {}).get(module, {}).get("regions", {}).get(region)
                if not m or m.get("status") != "available" or not m.get("quality", {}).get("eligible", False):
                    excluded.append({"view": view, "reason": (m or {}).get("reason", "本视角无可用分区结果")})
                    continue
                candidates.append((view, m))
            if not candidates:
                modules[module] = {"primary_view": None, "metrics": None, "alternatives": [],
                                   "excluded_views": excluded, "statement": "本次无可评估观察"}
                continue
            # A tiny clear remnant cannot represent a broadly visible region.
            # Keep it as a supplementary observation; never expand its denominator.
            largest = max(m["analysis_pixels"] for _,m in candidates)
            primary_candidates=[(v,m) for v,m in candidates if m["analysis_pixels"]>=MIN_PRIMARY_RELATIVE_AREA*largest]
            best_sharp = max(m["quality"]["sharpness"] for _, m in primary_candidates)
            central = region in {"forehead", "nose", "perioral", "chin"}
            eye = region.endswith("_eye")
            def ranking(candidate):
                view, m = candidate
                q = m["quality"]
                clarity_band = int(q["sharpness"] >= .7 * best_sharp)
                near_side = observed_views.get(view,{}).get("near_anatomical_side")
                preferred = (int(view == "front") if central else
                    int(view != "front" and (not near_side or region.startswith(near_side))))
                return (q["tier"], clarity_band,
                        q["sharpness"] if eye else preferred,
                        m["analysis_pixels"], q["sharpness"], -VIEWS.index(view))
            view, metrics = max(primary_candidates, key=ranking)
            alternatives = [{"view": v, "metrics": copy.deepcopy(m),
                             "has_findings": bool(m["count"] or m.get("uncertain_count") or (m["coverage"] or 0) > 0),
                             "role_reason": ("有效覆盖较少，仅作补充观察" if m["analysis_pixels"]<MIN_PRIMARY_RELATIVE_AREA*largest
                                             else "本次采用其他质量或方向更合适的主观察")}
                            for v, m in candidates if v != view]
            supplementary = any(a["has_findings"] for a in alternatives)
            statement = ("主视角未检出；补充视角有发现" if not metrics.get("has_evidence") and supplementary
                         else "主视角有区域证据，未检出中心目标" if metrics["count"] == 0 and metrics.get("has_evidence")
                         else "主视角观察结果")
            modules[module] = {"primary_view": view, "metrics": copy.deepcopy(metrics),
                "selection_reason": "先排除质量不合格观察，再保证本项分区覆盖不少于可用最大域的60%，最后按质量、方向和清晰度选择；不按目标数选择",
                "ranking_policy": "min-relative-domain-0.60 / photo-tier / relative-clarity-0.7 / view-preference / effective-pixels",
                "alternatives": alternatives, "excluded_views": excluded, "statement": statement}
        result[region] = {"modules": modules}
    return {"version": VERSION, "regions": result,
            "counting_policy": "one_primary_view_per_module_per_leaf_region",
            "cross_view_lesion_deduplication": False, "whole_face_unique_count": None,
            "notice": "分区观察值不可直接相加为全脸唯一目标数；补充视角不计入主视角数量"}
