"""Rebind true V3 counts, reject incompatible direct scores, then apply evidence gates."""
from copy import deepcopy
from pathlib import Path
import math
from .scoring import aggregate,grade,display_score
from .severity_guard import POLICY,VERSION,apply_score,absolute_count_score,finite
from .historical_metrics import normalize

EMPTY={"score_raw":None,"score":None,"grade":"不可评估","status":"unavailable"}
def invalidate(item,trace,reason):
 item.update(EMPTY)
 if "value" not in item:item["display"]="暂不评分"
 trace["rejected_reference"]=trace.pop("reference",None)
 trace.update(EMPTY,reference_contract_error=reason)

def protect(model,payload,data,catalog,bundle_root):
 from .measurements import read_evidence
 from .regions import build as build_regions
 from .clinical_scoring import evaluate
 model=deepcopy(model);original=deepcopy(model["modules"]);original_traces=deepcopy(model["score_trace"])
 traces=model["score_trace"]
 measurement={(x["module"],x["metric_id"][3:],x["region"]):x for x in payload["measurements"]}
 from .scoring_domains import areas
 valid_areas=areas(Path(bundle_root)/"evidence")
 rejected=[]
 # Direct field scores must not silently score a different measurement.
 for mid,module in model["modules"].items():
  for region,metrics in module["metrics"].items():
   for key,item in metrics.items():
    address=mid+"."+key+":"+region;trace=traces.get(address,{})
    from .reference_contracts import rejection
    reason=rejection(mid,key,region,trace,model["capture_profile"])
    if mid=="05":reason="grid_cluster_metric_is_not_v2_pixel_or_segment_metric"
    elif mid=="06":reason="classification_count_is_not_focal_redness_proxy"
    elif mid=="04":reason="diffuse_background_domain_is_not_legacy_red_domain"
    elif mid=="03" and key.startswith("spots."):reason="multi_supported_spot_domain_has_no_equivalent_legacy_distribution"
    elif reason:pass
    elif "value" in item and item["value"] is None and trace.get("method") not in ("no_target_burden_in_valid_measured_scope","observed_zero_target_anchor"):
     reason="no_measurement_to_support_this_metric_score"
    elif finite(item.get("value")) and finite(trace.get("score_input")) and not math.isclose(item["value"],trace["score_input"],rel_tol=1e-4,abs_tol=1e-6):
     if trace.get("method")!="held_out_validated_monotone_mapping":
      reason="displayed_measurement_differs_from_reference_input"
    if reason and item.get("score") is not None:
     rejected.append({"module":mid,"region":region,"metric":key,"reason":reason})
     invalidate(item,trace,reason)
     traces[address]=trace
 # Classification is scored from the observed class count; no old redness feature fallback.
 arrays,_=read_evidence(Path(bundle_root)/"evidence/doctor_v3_acne.npz")
 domains=build_regions(arrays["valid"],arrays["landmarks"],"06")
 acne=model["modules"]["06"]
 acne["compatibility_definition"]="observed_class_count_20_40_40_with_v2_absolute_count_curve"
 for region,metrics in acne["metrics"].items():
  area=int(domains[region].sum())
  if payload.get("stage1"):
   basis=payload["stage1"]["basis"].get("06.erythema_density:"+region,{})
   measured_area=basis.get("effective_area_px")
   if not finite(measured_area) or measured_area<0:
    raise ValueError("stage1 class effective domain evidence missing")
   area=int(measured_area)
  for kind in ("erythema","papule","pustule"):
   count=metrics[kind+"_count"].get("value")
   proof=measurement.get(("06",kind+"_density",region),{})
   if proof.get("status")=="measured" and finite(count) and area>=100:
    if not math.isclose(proof["value"],count*100000/area,rel_tol=1e-5,abs_tol=1e-6):
     raise ValueError("class count/density denominator does not match saved evidence")
   scored=absolute_count_score(count,area,proof.get("status")=="measured")
   for suffix in ("count",):
    metrics[kind+"_"+suffix].update(scored)
    traces["06."+kind+"_"+suffix+":"+region]={**scored,"source":"06."+kind+"_count:"+region,
      "reference_contract":"observed_class_count_absolute_v2_curve","value":count,"valid_area_px":area}
  score=aggregate({k:metrics[k] for k in ("erythema_count","papule_count","pustule_count")},
                  {"erythema_count":.2,"papule_count":.4,"pustule_count":.4})
  (acne if region=="full_face" else acne["regions"][region]).update(score)
 # Vascular compatibility is a separately named full-domain pixel model.
 normalized=normalize(data);components={}
 for key,field in (("pixel_count","vascular_count"),("pixel_coverage","vascular_area_ratio"),("pixel_line_density","vascular_line_density_per_10k_face_px")):
  entry=normalized.get("vascular|full_face|"+field,{})
  ref_key="development1000|vascular|full_face|"+field
  ref=catalog["references"].get(ref_key)
  score=evaluate(entry.get("value"),ref)
  components[key]={"name":field,"value":entry.get("value"),"unit":"V2_pixel_domain",**score}
  traces["05."+key+":full_face"]={"source":"vascular|full_face|"+field,"score_input":entry.get("value"),
    "reference":ref_key if ref else None,"method":"explicit_v2_full_domain_vascular_compatibility",**score}
 vascular=model["modules"]["05"]
 vascular["compatibility_components"]=components
 vascular["compatibility_definition"]="requires_replayed_v2_full_domain_vascular_state"
 vascular.update(EMPTY)
 for region,metrics in vascular["metrics"].items():
  zero=all(measurement.get(("05",k,region),{}).get("status")=="measured" and metrics[k].get("value")==0 for k in ("clusters","affected_area"))
  if zero:
   for k,item in metrics.items():
    item.update(score_raw=100.0,score=100,grade="未见明显",status="zero_target")
    traces["05."+k+":"+region]={"method":"valid_zero_v3_grid_domain","score_raw":100.0,"score":100,"grade":"未见明显"}
  if region!="full_face":
   vascular["regions"][region].update(aggregate({k:metrics[k] for k in ("clusters","affected_area","local_density")},
                                   {"clusters":.3,"affected_area":.4,"local_density":.3}))
  elif zero:vascular.update(score_raw=100.0,score=100,grade="未见明显",status="zero_target")
 # Recompute compound nodes after invalidating their inputs. No stale parent score.
 for mid,module in model["modules"].items():
  if mid not in ("05","06"):
   for region,metrics in module["metrics"].items():
    target=module if region=="full_face" else module["regions"][region]
    if mid=="03":
     for layer in ("spots","brown","uv"):
      metrics[layer].update(aggregate({n:metrics[layer+"."+n] for n in ("area","p90")},{"area":.55,"p90":.45}))
      metrics[layer]["display"]=metrics[layer]["grade"]
    if mid=="02":
     for name,weights in (("surface_gloss",{"gloss_area":.25/.6,"gloss_high_area":.2/.6,"gloss_mean":.15/.6}),
                          ("porphyrin",{"porphyrin_high_density":.7,"porphyrin_p90":.3})):
      metrics[name].update(aggregate({k:metrics[k] for k in weights},weights))
      metrics[name]["display"]=metrics[name]["grade"]
    weights=target.get("weights",{})
    if weights:target.update(aggregate({k:metrics.get(k,{}) for k in weights},weights))
  quality=data.get("quality",{}).get("status")=="PASS"
  for region,metrics in module["metrics"].items():
   raw=deepcopy(metrics)
   pixels=valid_areas.get(mid+":"+region)
   if pixels is not None and pixels<POLICY["minimum_region_valid_pixels"]:
    for key,item in metrics.items():
     item.update(EMPTY)
     if "value" not in item:item["display"]="暂不评分"
     traces.setdefault(mid+"."+key+":"+region,{}).update(EMPTY,reason="insufficient_region_for_scoring",valid_pixels=pixels)
    (module if region=="full_face" else module["regions"][region]).update(EMPTY,reason="insufficient_region_for_scoring",valid_pixels=pixels)
    continue
   for key,item in metrics.items():
    if item.get("score_raw") is None:continue
    fixed=apply_score(item,mid,raw,quality)
    fixed["severity_guard"]["valid_pixels"]=pixels
    item.update(fixed)
    if "value" not in item:item["display"]=item["grade"]
    address=mid+"."+key+":"+region
    trace=traces.setdefault(address,{})
    trace.update({k:fixed[k] for k in ("raw_statistical_score","score_raw","score","grade","severity_guard")})
   target=module if region=="full_face" else module["regions"][region]
   evidence=target.get("compatibility_components",raw)
   target.update(apply_score(target,mid,evidence,quality))
  for k,item in module.get("compatibility_components",{}).items():
   raw=deepcopy(module["compatibility_components"])
   # Only the overall score is public for the separate compatibility model.
   traces.setdefault(mid+"."+k+":full_face",{})["compatibility_only"]=True
 model["calibration_version"]="doctor_v3_guarded_20260908_v1"
 model["severity_policy"]=deepcopy(POLICY)
 model["reference_contract_rejections"]=rejected
 model["measurement_contract_version"]="doctor_v3_explicit_binding_20260908_v1"
 model["scoring_domain_valid_pixels"]=valid_areas
 if data.get("quality",{}).get("status")=="REJECT":
  for mid,module in model["modules"].items():
   module.update(EMPTY,reason="rejected_input_quality")
   for region,metrics in module["metrics"].items():
    for key,item in metrics.items():
     item.update(EMPTY,reason="rejected_input_quality")
     if "value" not in item:item["display"]="暂不评分"
     traces.setdefault(mid+"."+key+":"+region,{}).update(EMPTY,reason="rejected_input_quality")
   for item in module["regions"].values():item.update(EMPTY,reason="rejected_input_quality")
 return model
