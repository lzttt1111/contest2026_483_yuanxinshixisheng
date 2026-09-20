"""Grade-preserving coordinate conversion for explicitly replayed V2 state scores."""
from .scoring import grade,display_score
from .severity_guard import finite
LABELS=["未见明显","轻度","中度","较明显","显著"]
def convert(row):
 value=row.get("score");label=row.get("grade")
 if not finite(value) or isinstance(value,bool) or not 0<=value<=100 or label not in LABELS:return None
 boundaries=[0,20,40,60,90,100] if row.get("method")=="历史参考分布评分" else [0,25,45,65,90,100]
 i=LABELS.index(label);lo,hi=boundaries[i:i+2]
 fraction=max(0,min(1,(value-lo)/(hi-lo)))
 upper,lower=[(100,81),(80,61),(60,41),(40,21),(20,0)][i]
 raw=upper-fraction*(upper-lower)
 # V2's old percentile-only severe category cannot open the new strict gate.
 raw=max(raw,41 if label in ("较明显","显著") and not row.get("high_evidence_verified",False) else 21)
 return {"score_raw":raw,"score":display_score(raw),"grade":grade(raw),"status":"compatible",
   "method":"v2_replayed_state_grade_preserving_health_coordinate","source_v2":row,
   "source_interval":[lo,hi],"target_interval":[upper,lower],"significant_enabled":False}

def supplement(model,v2):
 changes=[]
 if not v2.get("quality_proven"):
  model["v2_compatible_state_supplements"]=[]
  return model
 for mid,module in model["modules"].items():
  if module.get("score") is not None:continue
  if module.get("reason") in ("insufficient_region_for_scoring","rejected_input_quality"):continue
  converted=convert(v2["modules"].get(mid,{}))
  if converted is None:continue
  module["strict_v3_assessment"]={k:module.get(k) for k in ("score_raw","score","grade","weights","missing","status")}
  module.pop("weights",None);module.pop("missing",None)
  module.update(converted)
  module["assessment_basis"]="replayed_v2_visible_state_not_missing_v3_measurement"
  module["assessment_title"]={"03":"综合色素外观","04":"泛红外观","05":"血管样外观","07":"可见细纹外观","08":"可见稳定纹路外观"}.get(mid,module["title"])
  changes.append(mid)
 model["v2_compatible_state_supplements"]=changes
 from .severity_guard import apply_score
 pigment=next((r for r in v2.get("official_v011",{}).get("formal_dimension_scores",[]) if r["dimension_id"]=="combined_pigmentation"),{})
 for group in pigment.get("groups",[]):
  key={"visible_spots":"spots","brown":"brown","uv_spots":"uv"}.get(group["group_id"])
  if key is None or group.get("score") is None:continue
  item=model["modules"]["03"]["metrics"]["full_face"][key]
  if item.get("score") is not None:continue
  evidence={}
  for metric in group.get("metrics",[]):
   identity=metric.get("metric_id","")
   if "area_ratio" in identity:evidence[key+".area"]={"score_raw":100-metric["score"]}
   if "p90" in identity:evidence[key+".p90"]={"score_raw":100-metric["score"]}
  shown=apply_score({"score_raw":100-group["score"]},"03",evidence,v2.get("quality_proven") is True)
  item.update(shown,display=shown["grade"],assessment_basis="V2_same_imaging_layer_state")
  model["score_trace"]["03."+key+":full_face"]={**shown,"method":"V2_same_imaging_layer_state",
    "source_group":group,"reference_file_sha256":v2["official_profile_sha256"]}
 return model
