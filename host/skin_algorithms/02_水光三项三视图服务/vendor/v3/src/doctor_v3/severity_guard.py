"""Automatic conservative display policy; statistical rarity is not severe disease."""
import math
from .scoring import grade,display_score

VERSION="v3_evidence_guard_20260908_v1"
POLICY={"version":VERSION,"burden_scale":.72,"burden_offset":5.0,"ordinary_max_burden":79.0,
        "unconfirmed_max_burden":59.0,"group_high_burden":75.0,"minimum_groups":2,
        "minimum_group_fraction":.60,"minimum_region_valid_pixels":2000,
        "medium_only_without_absolute_rule":["05","09","10","11"],"significant_rules":{}}
GROUPS={
 "01":{"quantity":["density"],"size":["area_p50"]},
 "02":{"surface_coverage":["gloss_area","gloss_high_area"],"surface_intensity":["gloss_mean"],
       "follicular":["porphyrin_high_density","porphyrin_p90"]},
 "03":{"coverage":["spots.area","brown.area","uv.area"],"colour":["spots.p90","brown.p90","uv.p90"]},
 "04":{"coverage":["area","high_area"],"colour":["mean","p90"]},
 # Different measurements remain different references; this list only groups
 # burden evidence. Grid count/area/cluster count never create extra votes.
 "05":{"extent":["pixel_coverage","affected_area","roi_count","clusters"],
       "network":["pixel_line_density","local_density"]},
 "06":{"quantity":["erythema_count","papule_count","pustule_count"]},
 "07":{"extent":["area","high_area"],"appearance":["contrast_p50"]},
 "08":{"quantity":["main_count"],"extent":["length","maximum"],"appearance":["contrast_p50"]},
 "09":{"extent":["extent"],"appearance":["mean_depth","p90_depth"]},
 "10":{"extent":["raised_area","depressed_area"],"appearance":["raised_p90","depressed_p90"]},
 "11":{"continuity":["smoothness","jaw_continuity"],"shape":["turning","jowl"]},
}
POLICY["evidence_groups"]=GROUPS

def finite(value):
 return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)

def absolute_count_score(count,valid_area,classification_valid=True):
 if not finite(count) or count<0 or int(count)!=count or not finite(valid_area) or valid_area<100 or not classification_valid:
  return {"score_raw":None,"score":None,"grade":"不可评估","status":"unavailable"}
 # V2 uses the same absolute count saturation, not the cohort rank of redness.
 raw=100*math.exp(-count/5.0)
 return {"score_raw":raw,"score":display_score(raw),"grade":grade(raw),"status":"candidate",
         "method":"absolute_observed_class_count","count":count,"valid_area":valid_area}

def consensus(metrics,module,quality_ok,policy=None):
 policy=policy or POLICY
 values=[]
 for group,names in GROUPS[module].items():
  scores=[100-metrics[n]["score_raw"] for n in names if n in metrics and finite(metrics[n].get("score_raw"))]
  if scores:values.append((group,max(scores)))
 high=[g for g,v in values if v>=policy["group_high_burden"]]
 return quality_ok and len(high)>=policy["minimum_groups"] and len(high)/max(len(values),1)>=policy["minimum_group_fraction"],high

def apply_score(item,module,metrics,quality_ok,absolute=None,policy=None):
 policy=policy or POLICY
 raw=item.get("score_raw")
 if not finite(raw):return {**item,"score_raw":None,"score":None,"grade":"不可评估"}
 if not 0<=raw<=100:raise ValueError("raw score outside range")
 allow_high,groups=consensus(metrics,module,quality_ok,policy)
 significant=False
 rule=policy.get("significant_rules",{}).get(module)
 # No shipped module is opened without a validated, versioned absolute rule.
 if rule and rule.get("validated") is True and rule.get("validation_id") and rule.get("bounds") and allow_high:
  significant=all(finite((absolute or {}).get(k)) and (absolute or {})[k]>=limit
                  for k,limit in rule["bounds"].items()) and 100-raw>=90
 if module in policy.get("medium_only_without_absolute_rule",[]) and not significant:allow_high=False
 burden=100-raw
 if burden==0:shown=100.0
 elif significant:shown=raw
 else:
  bounded=min(policy["ordinary_max_burden"],policy["burden_offset"]+policy["burden_scale"]*burden)
  if not allow_high:bounded=min(bounded,policy["unconfirmed_max_burden"])
  shown=100-bounded
 return {**item,"raw_statistical_score":raw,"score_raw":shown,"score":display_score(shown),"grade":grade(shown),
         "severity_guard":{"version":policy["version"],"allow_high":allow_high,
           "allow_significant":significant,"supporting_groups":groups,"quality_ok":quality_ok,
           "reason":"absolute_rule_validated" if significant else "conservative_evidence_gate"}}
