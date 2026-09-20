"""Anatomical groove basis and regional coarse-Z burden; no scoring references."""
import cv2
import numpy as np
from ._stage1_geometry_math import local_samples, sparse_relief, weighted_percentile
from ._stage1_anatomical_extent import extent_evidence, classify_extent_basis

VERSION = "v3-anatomical-groove-regional-mesh-2"
PARAMETERS = {"minimum_valid_area_px": 100, "support_halfwidth_px": 4}


def measure_grooves(rgb, domains, assigned, assignment, paths, parameters, context_base,
                    rejected=False, line_available=True, configurations=None):
    output = {}
    covered = np.zeros(rgb["valid"].shape,bool)
    pooled_depths, pooled_weights = [], []
    valid_union = np.zeros_like(covered)
    observations = {}
    for region, domain in domains.items():
        if region=="full_face":
            continue
        local = assigned[region]
        width=2*int(parameters.get("groove_support_halfwidth_px",PARAMETERS["support_halfwidth_px"]))+1
        raw_spread = (cv2.dilate(local.astype(np.uint8),np.ones((width,width),np.uint8))>0)&domain
        spread = raw_spread & ~covered
        covered |= spread
        reason = "rejected_quality" if rejected else "missing_image_geometry_correspondence" if not line_available else "insufficient_valid_region" if domain.sum()<100 else None
        kind = region.split("_",1)[1]
        extent = extent_evidence(paths[region],local,domain,kind,
                                 [r["target_id"] for r in assignment if r["region"]==region])
        context = {**context_base,"valid_area_px":int(domain.sum()),"image_line_pixels":int(local.sum()),
                   "image_extent_area_px":int(spread.sum()),"image_extent_area_ratio":float(spread.sum()/max(domain.sum(),1)),
                   "target_assignment":[r for r in assignment if r["region"]==region],
                   "anatomical_extent":extent,
                   "formula":"original regional vertex residual; triangle depth weighted by nonoverlapping observed image-support pixels",
                   "source_formula":"regional extension of V2 plane-detrended 2.5D relief; never physical skin depth"}
        samples, sample_reason = local_samples(rgb,domain)
        relief = None
        if samples is not None:
            context.update({"sample_ids":samples["ids"].tolist(),"sample_xy":samples["pixel_xy"].tolist(),
                            "relative_z":samples["z"].tolist(),"reference_plane":samples["plane"].tolist()})
            if local.any():
                relief,sample_reason = sparse_relief(samples,domain,parameters["relative_depth_threshold"],spread)
            else:
                relief={"mean_depth":None,"p90_depth":None,"volume":0.0,
                        "triangle_depth":[],"triangle_support_area_px":[],"affected_area_px":0.0,"surface_area_px":0.0}
        geometry_reason = reason or sample_reason
        if relief is not None:
            context.update(relief)
        values = {"mean_depth":None,"p90_depth":None,"volume":None}
        reasons = dict.fromkeys(values,geometry_reason)
        if not geometry_reason and relief is not None:
            values={k:relief[k] for k in values}
            reasons={k:None if v is not None else "no_target" for k,v in values.items()}
            valid_union |= domain
            raw_depth=np.asarray(relief["triangle_depth"])
            raw_weight=np.asarray(relief["triangle_support_area_px"])
            positive=raw_depth>parameters["relative_depth_threshold"]
            pooled_depths.extend(raw_depth[positive].tolist())
            pooled_weights.extend(raw_weight[positive].tolist())
        config = (configurations or {}).get(kind)
        if geometry_reason:
            extent["path_measurement_status"]=extent["measurement_status"]
            extent["measurement_status"]="missing_evidence"
            extent["reason"]=geometry_reason
        if local.any() and relief is not None and relief["mean_depth"] is None and not geometry_reason:
            extent["path_measurement_status"]=extent["measurement_status"]
            extent["measurement_status"]="missing_evidence"
            extent["reason"]="image_without_25d_depression_support"
        state = classify_extent_basis(extent,config) if not reason else {"status":"missing_evidence","extent_score":None,"extent_grade":None,"reason":reason}
        if state["status"]=="test_only":
            # Test configurations are callable from basis but cannot populate
            # measured rows or silently become a formal report classification.
            state={**state,"extent_score":None,"extent_grade":None,"status":"pending_parameter",
                   "reason":"test_anatomical_configuration_not_publishable"}
        context["extent_classification"]=state
        values["extent"]=state.get("extent_score")
        reasons["extent"]=state.get("reason")
        output[region]={"values":values,"reasons":reasons,"context":context,"mask":spread.astype(np.uint8)}
        observations[region]=extent
    # Pool actual original triangle observations. Per-region support masks above
    # are mutually exclusive, so the same observed area cannot be counted twice.
    depth=np.asarray(pooled_depths)
    weight=np.asarray(pooled_weights)
    total_area=int(valid_union.sum())
    global_values={"mean_depth":float(np.average(depth,weights=weight)) if len(depth) else None,
                   "p90_depth":weighted_percentile(depth,weight,.9) if len(depth) else None,
                   "volume":float(np.sum(depth*weight)/total_area) if total_area else None,
                   "extent":None}
    global_reason="rejected_quality" if rejected else "missing_image_geometry_correspondence" if not line_available else "insufficient_regional_geometry" if total_area<100 else None
    global_reasons={k:global_reason or ("no_target" if v is None else None) for k,v in global_values.items()}
    eligible={r:e for r,e in observations.items() if e.get("measurement_status")=="measured"}
    extent_numerator=sum(e["burden_numerator"] for e in eligible.values())
    extent_denominator=sum(e["burden_denominator"] for e in eligible.values())
    global_reasons["extent"]=global_reason or "anatomical_extent_threshold_pending"
    global_context={**context_base,"valid_area_px":total_area,"original_region_area_px":int(domains["full_face"].sum()),
                    "formula":"pool unique observed-support triangle samples; P90 on pooled depths, volume=sum(depth*support_area)/union_valid_area",
                    "triangle_depth":pooled_depths,"triangle_support_area_px":pooled_weights,
                    "anatomical_extent":{"definition_version":next(iter(observations.values()))["definition_version"],
                        "groove_type":"full_face","measurement_status":"measured" if eligible else "missing_evidence",
                        "structure_evidence":observations,"eligible_regions":list(eligible),
                        "burden_numerator":extent_numerator,"burden_denominator":extent_denominator,
                        "anatomical_extent_burden":extent_numerator/extent_denominator if extent_denominator else None,
                        "aggregation":"sum(path_length*actual_progress_weighted_coverage_numerator)/sum(path_length*valid_progress_weighted_denominator)"},
                    "extent_classification":{"status":"pending_parameter","reason":"anatomical_extent_threshold_pending",
                                             "extent_score":None,"extent_grade":None}}
    full_config=(configurations or {}).get("full_face")
    if full_config is not None and not global_reason:
        state=classify_extent_basis(global_context["anatomical_extent"],full_config)
        if state["status"]=="test_only":
            state={**state,"extent_score":None,"extent_grade":None,"status":"pending_parameter",
                   "reason":"test_anatomical_configuration_not_publishable"}
        global_context["extent_classification"]=state
        global_values["extent"]=state["extent_score"]
        global_reasons["extent"]=state["reason"]
    output["full_face"]={"values":global_values,"reasons":global_reasons,"context":global_context,"mask":covered.astype(np.uint8)}
    return output
