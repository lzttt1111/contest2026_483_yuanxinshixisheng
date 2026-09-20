"""Explicit user-approved historical image estimates, never hidden value substitution."""
from copy import deepcopy
def apply(model):
 changes=[]
 module=model["modules"]["08"]
 for region,metrics in module["metrics"].items():
  for target_key,source_key,unit in (("main_count","count","条"),("length_burden","length","标准化px")):
   target=metrics.get(target_key,{});source=metrics.get(source_key,{})
   if target.get("score") is not None or source.get("score") is None or source.get("value") is None:continue
   native={k:deepcopy(target.get(k)) for k in ("value","unit","score","grade")}
   for key in ("value","score_raw","score","grade","raw_statistical_score","severity_guard"):
    if key in source:target[key]=deepcopy(source[key])
   target["unit"]=unit;target["display"]=str(source["value"])+(" 条" if source_key=="count" else " 标准化px")
   target["name"]=("主纹数量" if target_key=="main_count" else "纹路延伸量")+"（图像估计）"
   target["measurement_basis"]="user_approved_historical_v2_image_estimate"
   target["status"]="compatible"
   target["native_v3_measurement"]=native
   target.pop("no_score_reason",None)
   origin="08."+source_key+":"+region;address="08."+target_key+":"+region
   trace=deepcopy(model["score_trace"].get(origin,{}))
   trace.update(method="historical_v2_report_proxy",proxy_source_metric=source_key,
                proxy_source_address=origin,native_v3_measurement=native,display_value=source["value"])
   model["score_trace"][address]=trace;changes.append(address)
 model["historical_proxy_view_restorations"]=changes
 return model
