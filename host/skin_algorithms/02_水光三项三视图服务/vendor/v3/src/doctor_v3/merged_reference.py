"""Exact additive ROI statistics; quantiles are deliberately not mergeable."""
import math
REGIONS={
 "forehead_glabella":("forehead","glabella"),
 "chin_perioral":("chin","perioral"),
 "lower_jaw_chin":("left_jaw","right_jaw","chin"),
 "lower_face":("left_jaw","right_jaw","chin","perioral"),
 "nose_with_nasal":("nose","left_nasal","right_nasal")}
AREA_FIELDS=("valid_skin_area_px","有效皮肤面积（像素）","valid_area_px","valid_pixels")
COUNT_FIELDS=("feature_count","特征数量（个）")
RATIOS=("feature_area_ratio","gloss_area_ratio","high_gloss_area_ratio",
        "diffuse_red_area_ratio","high_intensity_area_ratio","continuous_brown_coverage_ratio",
        "连续异常面积占比","高强度异常面积占比")
def finite(v):
 return not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v)
def value(metrics,project,region,fields):
 aliases={"left_jaw":"画面左下颌","right_jaw":"画面右下颌"}
 for f in fields:
  v=metrics.get(project+"|"+region+"|"+f,{}).get("value")
  if v is None and region in aliases:v=metrics.get(project+"|"+aliases[region]+"|"+f,{}).get("value")
  if finite(v):return v
 return None
def merge(metrics,project,regions):
 areas=[value(metrics,project,r,AREA_FIELDS) for r in regions]
 if any(a is None or a<0 for a in areas) or sum(areas)<=0:return {}
 total=sum(areas);out={"valid_pixels":total}
 counts=[value(metrics,project,r,COUNT_FIELDS) for r in regions]
 if all(finite(c) and c>=0 for c in counts):
  out.update(count=sum(counts),density=sum(counts)*100000/total)
 for f in RATIOS:
  values=[value(metrics,project,r,(f,)) for r in regions]
  if all(a==0 or finite(v) and 0<=v<=1 for a,v in zip(areas,values)):
   out[f]=sum(a*(v or 0) for a,v in zip(areas,values))/total
 # Whole-domain mean is additive only for a whole-domain intensity definition.
 if project in ("brown","redness"):
  values=[value(metrics,project,r,("mean_intensity",)) for r in regions]
  if all(a==0 or finite(v) for a,v in zip(areas,values)):
   out["mean_intensity"]=sum(a*(v or 0) for a,v in zip(areas,values))/total
 return out
