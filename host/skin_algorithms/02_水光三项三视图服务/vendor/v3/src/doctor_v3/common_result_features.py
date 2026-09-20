"""Reconstruct common scoring inputs from saved evidence, not original photographs."""
import csv
import json
from pathlib import Path
import numpy as np
from .bundle import load,contained
from .measurements import read_evidence

def scalar_csv(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        row=next(csv.DictReader(stream),{})
    result={}
    for key,value in row.items():
        name=key.split("-",1)[-1]
        try:result[name]=float(value)
        except (ValueError,TypeError):pass
    return result

def extras_from_source(source,cloud=False):
    source=Path(source)
    if not cloud:
        complete=json.loads((source/"十二项完整量化指标.json").read_text(encoding="utf-8"))
        return complete.get("detector_results",{})
    bundle=json.loads((source/"cloud_response_bundle.json").read_text(encoding="utf-8"))
    result={}
    for name,task in bundle["tasks"].items():
        response=task["response"]
        if response["status"]!="success":raise ValueError("failed cloud task")
        raw=response["raw_result"]
        project="acne" if name=="acne_v2" else name
        result[project]={"status":"success","metrics":raw.get("metrics",{})}
        if name=="wrinkle":
            key=raw["medical_report_csv_v2"]
            csv_path=contained(source,"simulated_oss/"+key)
            json_path=csv_path.parent/"summary_json.json"
            if json_path.is_symlink() or not json_path.is_file():raise ValueError("cloud wrinkle source JSON missing")
            result[project]["metrics"]=json.loads(json_path.read_text(encoding="utf-8"))
        if name=="contour_firmness":
            result[project]["metrics"]=scalar_csv(contained(source,"simulated_oss/"+raw["medical_report_csv_v2"]))
    return result

def common_features(bundle_root,source,cloud=False,additional_metrics=None):
    from src.utils.detailed_metrics import build_medical_payload,build_medical_metrics_v2
    from src.medical_v2_schema import to_english_document
    from src.nine_analysis.v2_proxy_projection import build_v2_proxy_modules
    payload=load(bundle_root)
    extras=extras_from_source(source,cloud)
    extra_path=Path(additional_metrics) if additional_metrics else Path(bundle_root)/"evidence/contour_complete.json"
    if extra_path.is_file():
        geometry=json.loads(extra_path.read_text(encoding="utf-8"))
        original=extras["contour_firmness"]["metrics"]
        for public,raw_key in (("中面部曲面连续性","midface_surface_continuity_ratio"),("下颌缘连续性","jaw_continuity_ratio"),("左右轮廓差异","jaw_arc_asymmetry_ratio")):
            expected=original.get(public,original.get(raw_key))
            if expected is not None and abs(float(expected)-float(geometry[raw_key]))>1e-6:
                raise ValueError("supplemental geometry differs from current result")
        extras["contour_firmness"]["metrics"]=geometry
    results={}
    five={"pores":"毛孔","redness":"红区","brown":"棕区","spots":"斑点","texture":"纹理"}
    for project,label in five.items():
        p=contained(bundle_root,"evidence/doctor_v3_"+project+".npz")
        arrays,meta=read_evidence(p)
        medical=build_medical_payload(project=project,project_label=label,analysis_mask=arrays["valid"],
             landmarks=arrays["landmarks"],instances=meta.get("instances",[]),score_map=arrays["score"],
             instance_mask=arrays["instances"],continuous_mask=arrays.get("continuous"),high_mask=arrays.get("high"),
             quality_control=meta.get("quality",{}),limitations=[])
        doc=to_english_document(build_medical_metrics_v2(medical,project))
        if project=="pores":
            from .regions import build as build_v3_regions
            regions=build_v3_regions(arrays["valid"],arrays["landmarks"],"01")
            for region in ("left_inner_cheek","right_inner_cheek","left_outer_cheek","right_outer_cheek"):
                mask=regions[region];area=int(np.count_nonzero(mask));areas=[]
                for instance in meta.get("instances",[]):
                    x,y=(int(round(v)) for v in instance["centroid"])
                    if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]:
                        areas.append(float(instance["area"]))
                scope={"feature_density_per_100k_skin_px":len(areas)*100000/area if area else None,
                       "p50_instance_area_px":float(np.percentile(areas,50)) if areas else None,
                       "p90_instance_area_px":float(np.percentile(areas,90)) if areas else None,
                       "max_instance_area_px":max(areas) if areas else None}
                doc.setdefault("region_metrics",[]).append({"analysis_region":region,"valid_skin_area_px":area,
                    "evaluation_status":"可评估" if area else "不可评估","core_metrics":{"scope_and_burden":scope}})
        raw=dict(extras.get(project,{}).get("metrics",{}))
        raw["medical_metrics_v2"]=doc
        results[project]={"status":"success","metrics":raw}
    for project in ("wrinkle","contour_firmness"):
        results[project]={"status":extras[project].get("status","success"),"metrics":extras[project]["metrics"]}
    # Explicit common measured inputs, computed identically for local/cloud.
    p=contained(bundle_root,"evidence/doctor_v3_surface_gloss.npz")
    arrays,_=read_evidence(p)
    valid=arrays["valid"]>0
    mask=(arrays["instances"]>0)&valid
    high=(arrays["high"]>0)&valid
    area=int(np.count_nonzero(valid))
    if area<=0:raise ValueError("empty gloss domain")
    values=arrays["score"][mask]
    full={"valid_skin_area_px":area,"gloss_area_ratio":float(mask.sum()/area),
          "high_gloss_area_ratio":float(high.sum()/area),
          "mean_gloss_intensity":float(values.mean()) if values.size else None,
          "p50_gloss_intensity":float(np.percentile(values,50)) if values.size else None,
          "p90_gloss_intensity":float(np.percentile(values,90)) if values.size else None}
    import cv2
    n,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    full["largest_gloss_component_area_ratio"]=float(stats[1:,cv2.CC_STAT_AREA].max()/area) if n>1 else 0.0
    core={"surface_gloss_coverage":{k:full[k] for k in ("gloss_area_ratio","high_gloss_area_ratio")},
          "surface_gloss_intensity":{k:full[k] for k in ("p50_gloss_intensity","p90_gloss_intensity")},
          "surface_gloss_continuity":{"largest_gloss_component_area_ratio":full["largest_gloss_component_area_ratio"]}}
    gloss_raw=dict(extras.get("surface_gloss",{}).get("metrics",{}))
    from .saved_gloss_regions import rebuild as rebuild_gloss_regions
    rgb_arrays,_=read_evidence(contained(bundle_root,"evidence/doctor_v3_rgb.npz"))
    if payload["capture_profile"]=="consumer":
        gloss_raw["region_metrics"]=rebuild_gloss_regions(arrays,rgb_arrays,gloss_raw.get("region_metrics"))
    gloss_raw.update(full_face=full,doctor_core_inputs=core)
    results["surface_gloss"]={"status":"success","metrics":gloss_raw}
    p=contained(bundle_root,"evidence/doctor_v3_vascular.npz")
    arrays,meta=read_evidence(p)
    valid=arrays["valid"]>0
    area=int(valid.sum());instances=meta.get("instances",[])
    results["vascular"]={"status":"success","metrics":{
        "valid_face_pixels":area,"vascular_count":len(instances),
        "vascular_area_ratio":float(((arrays["instances"]>0)&valid).sum()/area),
        "vascular_total_length_px":float(((arrays["skeleton"]>0)&valid).sum()),
        "vascular_line_density_per_10k_face_px":float(((arrays["skeleton"]>0)&valid).sum())*10000/area,
        "branch_point_density_per_10k_face_px":sum(i.get("branch_points",0) for i in instances)*10000/area,
        "branch_point_count":sum(i.get("branch_points",0) for i in instances)}}
    from src.engines.vascular_structure_engine import VascularStructureAnalyzer
    widths=VascularStructureAnalyzer._skeleton_widths(arrays["instances"],arrays["skeleton"])
    response=arrays["score"][arrays["skeleton"]>0]
    length=float((arrays["skeleton"]>0).sum())
    results["vascular"]["metrics"].update({
        "p50_width_px":float(np.percentile(widths,50)) if widths.size else None,
        "p90_width_px":float(np.percentile(widths,90)) if widths.size else None,
        "p50_redness":float(np.percentile(response,50)) if response.size else None,
        "p90_redness":float(np.percentile(response,90)) if response.size else None,
        "network_ratio":sum(i["skeleton_length_px"] for i in instances if i["branch_points"]>0)/max(length,1)})
    # Porphyrin/UV saved instance response and medical regions are recomputable.
    purple={}
    uv_legacy={}
    for project,internal in (("uv_spots","purple_uv_spots"),("porphyrin","purple_porphyrin")):
        arrays,meta=read_evidence(contained(bundle_root,"evidence/doctor_v3_"+project+".npz"))
        instances=[{**i,"intensity":i.get("mean_score",i.get("intensity"))} for i in meta.get("instances",[])]
        if not instances and np.any(arrays["instances"]):
            from .measurements import component_samples
            instances=component_samples(arrays["instances"],arrays["score"],arrays["valid"]>0)
        if any(i["intensity"] is None for i in instances):raise ValueError("instance response missing")
        medical=build_medical_payload(project=internal,project_label=project,analysis_mask=arrays["valid"],
             landmarks=arrays["landmarks"],instances=instances,score_map=arrays["score"],instance_mask=arrays["instances"],
             continuous_mask=arrays.get("continuous"),high_mask=arrays.get("high"),quality_control={},limitations=[])
        purple[project]=build_medical_metrics_v2(medical,internal)
        if project=="uv_spots":
            from .legacy_uv_statistics import calculate
            uv_legacy=calculate(medical)
    results["porphyrin"]={"status":"success","metrics":purple}
    results["uv_spots"]={"status":"success","metrics":purple,"legacy_metrics":uv_legacy}
    results["acne"]=extras.get("acne",{"status":"success","metrics":{}})
    registry=json.loads((Path(__file__).resolve().parents[2]/"calibration/metric_registry_v2_proxy_20260728.json").read_text(encoding="utf-8"))
    _,features=build_v2_proxy_modules(results,registry)
    q=json.loads(contained(bundle_root,"evidence/doctor_v3_rgb.json").read_text(encoding="utf-8"))
    return payload,{"aggregate":{"scoring_features":features},"projects":results,"quality":{"status":q.get("quality_status","UNKNOWN")}}
