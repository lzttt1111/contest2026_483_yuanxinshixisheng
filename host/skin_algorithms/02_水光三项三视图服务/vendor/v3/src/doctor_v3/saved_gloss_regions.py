"""Recover the existing V2 oil-region statistics from saved arrays, without detection."""
import numpy as np
def rebuild(gloss,rgb,previous=None):
 from src.engines.visia_regions import build_visia_regions
 from src.engines.surface_gloss_engine import build_oil_regions,SurfaceGlossAnalyzer
 base=build_visia_regions(rgb["image"],rgb["valid"],gloss["landmarks"],[],include_chin=True,mode="full",feature_margin_px=10)
 masks=build_oil_regions(base,gloss["landmarks"]).regions;out={}
 for name,mask in masks.items():
  row=SurfaceGlossAnalyzer._region_metrics(mask,gloss["valid"],gloss["instances"],gloss["high"],gloss["score"])
  local=(mask>0)&(gloss["valid"]>0)&(gloss["instances"]>0)
  row["mean_gloss_intensity"]=float(gloss["score"][local].mean()) if local.any() else None
  if not local.any():row["p50_gloss_intensity"]=row["p90_gloss_intensity"]=None
  old=(previous or {}).get(name,{})
  for key in ("valid_area_px","gloss_area_ratio","high_gloss_area_ratio","p50_gloss_intensity","p90_gloss_intensity"):
   value=old.get(key)
   if isinstance(value,(int,float)) and row.get(key) is not None and not np.isclose(value,row[key],rtol=1e-5,atol=1e-6):
    raise ValueError("saved oil regional reconstruction differs from original V2 metric: "+name+"."+key)
  out[name]=row
 return out
