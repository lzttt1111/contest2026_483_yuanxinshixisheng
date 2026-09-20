"""Restore only historical scores whose new ROI is exactly reconstructable."""
import math
from .merged_reference import merge,REGIONS
from .historical_metrics import normalize
from .clinical_scoring import evaluate
from .severity_guard import apply_score
RECIPES={
 ("01","forehead") : ("pores","forehead_glabella",{"density":"density"}),
 ("01","chin") : ("pores","chin_perioral",{"density":"density"}),
 ("02","forehead") : ("surface_gloss","forehead_glabella",{"gloss_area":"gloss_area_ratio","gloss_high_area":"high_gloss_area_ratio"}),
 ("02","chin") : ("surface_gloss","lower_face",{"gloss_area":"gloss_area_ratio","gloss_high_area":"high_gloss_area_ratio"}),
 ("03","forehead") : ("brown","forehead_glabella",{"brown.area":"continuous_brown_coverage_ratio"}),
 ("03","nose") : ("brown","nose_with_nasal",{"brown.area":"continuous_brown_coverage_ratio"}),
 ("03","jaw") : ("brown","lower_jaw_chin",{"brown.area":"continuous_brown_coverage_ratio"})}
EXTRA_RECIPES={
 ("03","forehead","uv.area"):("uv_spots","forehead_glabella","连续异常面积占比"),
 ("03","nose","uv.area"):("uv_spots","nose_with_nasal","连续异常面积占比"),
 ("03","jaw","uv.area"):("uv_spots","lower_jaw_chin","连续异常面积占比")}
def recipe_for(module,metric,region):
 if (module,region,metric) in EXTRA_RECIPES:return EXTRA_RECIPES[module,region,metric]
 row=RECIPES.get((module,region))
 if row and metric in row[2]:return row[0],row[1],row[2][metric]
 return None
def supplement(model,data,catalog,bundle_root):
 if data.get("quality",{}).get("status") not in ("PASS","WARNING"):return model
 if not catalog.get("additive_reference_version"):return model
 normalized=normalize(data);changes=[];domains={}
 from pathlib import Path
 from .measurements import read_evidence
 from .regions import build
 entries=[(mid,region,key,project,merged,field) for (mid,region),(project,merged,fields) in RECIPES.items() for key,field in fields.items()]
 entries += [(mid,region,key,*spec) for (mid,region,key),spec in EXTRA_RECIPES.items()]
 for mid,region,key,project,merged,field in entries:
  if model["capture_profile"]!="consumer" and mid!="01":continue
  current=merge(normalized,project,REGIONS[merged])
  if project not in domains:
   arrays,_=read_evidence(Path(bundle_root)/("evidence/doctor_v3_"+project+".npz"))
   domains[project]=build(arrays["valid"],arrays["landmarks"],mid)
  area=int(domains[project][region].sum())
  if not current or current["valid_pixels"]<2000:continue
  if area is not None and abs(area-current["valid_pixels"])>.1:continue
  module=model["modules"][mid];metrics=module["metrics"].get(region,{})
  for key,field in ((key,field),):
   item=metrics.get(key,{})
   if item.get("score") is not None or item.get("value") is None:continue
   value=current.get(field)
   if value is None or not math.isclose(value,item["value"],rel_tol=1e-4,abs_tol=1e-6):continue
   reference="additive|development1000|"+project+"|"+merged+"|"+field
   ref=catalog["references"].get(reference)
   if not ref:continue
   calculated=evaluate(value,ref)
   shown=apply_score(calculated,mid,{},data.get("quality",{}).get("status")=="PASS")
   item.update(shown,restored_additive=True)
   trace={"method":"exact_additive_roi_reference","reference":reference,"score_input":value,
          "source":"sum_of_saved_named_regions","reference_region":region,"measurement_region":region,
          "region_components":list(REGIONS[merged]),"merged_recipe":merged,"project":project,
          "historical_field":field,"valid_pixels":area,"population_n":ref["n"],**shown}
   model["score_trace"][mid+"."+key+":"+region]=trace
   changes.append(mid+"."+key+":"+region)
 model["additive_score_restorations"]=changes
 return model
