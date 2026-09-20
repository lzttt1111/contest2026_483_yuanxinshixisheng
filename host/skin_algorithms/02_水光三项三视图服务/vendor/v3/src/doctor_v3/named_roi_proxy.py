"""User-authorized V2 named-region image estimates with matching values and references."""
from copy import deepcopy
from .historical_metrics import normalize
from .clinical_scoring import evaluate,formatted
from .scoring import aggregate
from .severity_guard import apply_score,finite
ALLOWED={"01","02","03","04","10"}
ALIASES={"left_jaw":"画面左下颌","right_jaw":"画面右下颌"}
def apply(model,original,data,catalog):
 if data.get("quality",{}).get("status") not in ("PASS","WARNING"):
  model["named_roi_proxy_restorations"]=[]
  return model
 normalized=normalize(data);restored=[]
 for mid,module in model["modules"].items():
  if mid not in ALLOWED:continue
  for region,metrics in module["metrics"].items():
   if region in ("left_inner_cheek","right_inner_cheek","left_outer_cheek","right_outer_cheek","left_eye","right_eye"):continue
   original_metrics=original["modules"][mid]["metrics"].get(region,{})
   updates={}
   for key,target in metrics.items():
    if "value" not in target:continue
    trace=original["score_trace"].get(mid+"."+key+":"+region,{})
    source=trace.get("source","");reference=trace.get("reference")
    if not reference:continue
    method=trace.get("method")
    if method in ("parent_region_reference_transfer","v3_measurement_compatible_reference_transfer"):continue
    entry=normalized.get(source)
    estimated=method=="quantile_piecewise_large_density_estimate"
    if not entry and not estimated:continue
    if entry:
     parts=source.split("|")
     if len(parts)!=3 or parts[1] not in (region,ALIASES.get(region)):continue
     project,source_region,field=parts
     value=entry["value"]
     area=next((normalized.get(project+"|"+source_region+"|"+f,{}).get("value") for f in ("valid_skin_area_px","valid_area_px","valid_pixels","有效皮肤面积（像素）") if normalized.get(project+"|"+source_region+"|"+f,{}).get("value") is not None),None)
     if finite(area) and area<2000:continue
    else:
     if source!="pores_size_summary:"+region:continue
     project,source_region,field="pores",region,"large_density_estimate"
     value=original_metrics[key].get("value");area=None
    if not finite(value):continue
    estimator=None
    if project=="surface_gloss" and field=="mean_gloss_intensity" and reference.endswith("|mean_gloss_intensity_estimate"):
     config=catalog.get("gloss_mean_estimator",{})
     p50=normalized.get(project+"|"+source_region+"|p50_gloss_intensity",{}).get("value")
     p90=normalized.get(project+"|"+source_region+"|p90_gloss_intensity",{}).get("value")
     if not config.get("accepted") or not finite(p50) or not finite(p90):continue
     a,b,c=config["coefficients"];value=max(0,min(1,a*p50+b*p90+c))
     estimator={"configuration":config,"p50":p50,"p90":p90}
     field="mean_gloss_intensity_estimate";source=project+"|"+source_region+"|"+field
    # Spectral inputs may use only the existing V2 historical proxy cohort,
    # never masquerade as a validated RGB-to-spectral population transfer.
    if model["capture_profile"]=="institution" and mid in ("02","03","04"):
     reference="legacy|"+project+"|"+source_region+"|"+field
     if reference not in catalog["references"] and mid=="03" and project=="uv_spots":
      alternative={"uv.area":"feature_area_ratio","uv.p90":"p90_intensity"}.get(key)
      if alternative:
       candidate="legacy|uv_spots|"+source_region+"|"+alternative
       entry=normalized.get("uv_spots|"+source_region+"|"+alternative,{})
       if candidate in catalog["references"] and finite(entry.get("value")):
        reference=candidate;field=alternative;value=entry["value"]
        source="uv_spots|"+source_region+"|"+alternative
     if reference not in catalog["references"]:continue
    elif method=="held_out_validated_monotone_mapping":
     # Keep the independently validated transformation and its original display value.
     pass
    ref=catalog["references"].get(reference)
    if not ref:continue
    if reference.rsplit("|",1)[-1]!=field and method!="held_out_validated_monotone_mapping":continue
    score_input=trace.get("score_input") if method=="held_out_validated_monotone_mapping" and reference==trace.get("reference") else value
    raw=evaluate(score_input,ref)
    if raw.get("score") is None:continue
    fixed=apply_score(raw,mid,{},data.get("quality",{}).get("status")=="PASS")
    unit=target.get("unit","图像指数")
    if key=="spots.p90":unit="ΔE"
    if key.endswith(".p90") and key!="spots.p90":unit="0-1"
    native={k:deepcopy(target.get(k)) for k in ("value","unit","score","grade")}
    node={**deepcopy(target),**fixed,"value":value,"unit":unit,"display":formatted(value,unit),
          "status":"compatible","measurement_basis":"historical_named_roi_image_estimate",
          "native_v3_measurement":native,"historical_roi":source_region}
    node.pop("no_score_reason",None)
    proxy_trace={**deepcopy(trace),**fixed,"method":"historical_named_roi_image_estimate",
       "underlying_method":method,"reference":reference,"score_input":score_input,"display_value":value,
       "source":source,"historical_project":project,"historical_field":field,
       "historical_roi":source_region,"measurement_region":region,"reference_region":source_region,
       "valid_pixels":area,"native_v3_measurement":native,
       "reference_boundary":"existing_v2_unstratified_image_estimate_not_clinical_norm"}
    if estimator:proxy_trace["gloss_mean_estimator"]=estimator
    updates[key]=(node,proxy_trace)
   if mid=="02":
    for key,project,zero_field in (("gloss_mean","surface_gloss","gloss_area_ratio"),("porphyrin_p90","porphyrin","特征数量（个）")):
     if key in updates:continue
     zero=normalized.get(project+"|"+region+"|"+zero_field,{}).get("value")
     area=next((normalized.get(project+"|"+region+"|"+f,{}).get("value") for f in ("valid_skin_area_px","valid_area_px","有效皮肤面积（像素）") if normalized.get(project+"|"+region+"|"+f,{}).get("value") is not None),None)
     if zero!=0 or not finite(area) or area<2000:continue
     value=normalized.get(project+"|"+region+"|"+("mean_gloss_intensity" if key=="gloss_mean" else "实例P90强度（0～1）"),{}).get("value")
     if finite(value) and value>0:continue
     target=metrics[key];native={k:deepcopy(target.get(k)) for k in ("value","unit","score","grade")}
     node={**deepcopy(target),"value":None,"display":"—","score_raw":100.0,"raw_statistical_score":100.0,
           "score":100,"grade":"未见明显","status":"zero_target",
           "measurement_basis":"historical_named_roi_image_estimate","native_v3_measurement":native}
     node.pop("no_score_reason",None)
     trace={"method":"valid_zero_historical_roi_state","source":project+"|"+region+"|"+zero_field,
            "target_count_or_area":0,"valid_pixels":area,"historical_roi":region,"value":None,"score_raw":100.0,"score":100,"grade":"未见明显"}
     updates[key]=(node,trace)
   if not updates:continue
   # A proxy scope is evaluated solely from proxy items, not mixed with a
   # newly measured V3 component from a different mask.
   for key,(node,trace) in updates.items():
    if metrics[key].get("score") is None:restored.append(mid+"."+key+":"+region)
    metrics[key]=node;model["score_trace"][mid+"."+key+":"+region]=trace
   target=module if region=="full_face" else module["regions"][region]
   weights=original["modules"][mid].get("weights",{}) if region=="full_face" else original["modules"][mid]["regions"][region].get("weights",{})
   values={k:dict(v,score_raw=v.get("raw_statistical_score",v.get("score_raw"))) for k,(v,t) in updates.items()}
   if mid=="02":
    for name,group_weights in (("surface_gloss",{"gloss_area":.25/.6,"gloss_high_area":.2/.6,"gloss_mean":.15/.6}),
                               ("porphyrin",{"porphyrin_high_density":.7,"porphyrin_p90":.3})):
     score=aggregate({k:values.get(k,{}) for k in group_weights},group_weights)
     if score.get("score") is not None:
      protected=apply_score(score,mid,{},data.get("quality",{}).get("status")=="PASS")
      metrics[name].update(protected,display=protected["grade"],status="compatible",
                           measurement_basis="historical_named_roi_image_estimate",
                           assessment_basis="historical_named_roi_image_estimate")
   if mid=="03":
    for layer in ("spots","brown","uv"):
     terms={k:values.get(layer+"."+k,{}) for k in ("area","p90")}
     score=aggregate(terms,{"area":.55,"p90":.45})
     if score.get("score") is not None:
      protected=apply_score(score,mid,{},data.get("quality",{}).get("status")=="PASS")
      metrics[layer].update(protected,display=protected["grade"],status="compatible",
                            assessment_basis="historical_named_roi_image_estimate")
      values[layer]=score
     else:values[layer]={}
   if weights and all(k in values for k in weights):
    score=aggregate({k:values[k] for k in weights},weights)
    if score.get("score") is not None:
     for stale in ("source_v2","source_interval","target_interval","method"):
      if stale in target:target.setdefault("previous_state_assessment",{})[stale]=target.pop(stale)
     target.update(apply_score(score,mid,{},data.get("quality",{}).get("status")=="PASS"),status="compatible",
                   method="historical_named_roi_proxy_weighted_state",
                   assessment_basis="historical_named_roi_image_estimate",historical_roi=region,
                   proxy_definition="V2 named-region quantities and V3-compatible report weights")
 model["named_roi_proxy_restorations"]=restored
 tidy(model)
 return model

def tidy(model):
 for module in model["modules"].values():
  nodes=[module,*module.get("regions",{}).values(),*(item for metrics in module.get("metrics",{}).values() for item in metrics.values())]
  for node in nodes:
   proxy=node.get("assessment_basis")=="historical_named_roi_image_estimate" or node.get("measurement_basis")=="historical_named_roi_image_estimate"
   if proxy and node.get("score") is not None:
    if node.get("missing"):node["native_v3_missing"]=node.pop("missing")
    node.pop("no_score_reason",None)
 return model
