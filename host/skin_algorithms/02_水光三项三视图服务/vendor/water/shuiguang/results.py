"""Single quantitative truth used by JSON, region tables and renderers."""
import cv2
import numpy as np
from .config import REGIONS, VIEWS
from .detectors import centroid

def measure(finding,prepared,original_shape):
 p=prepared.result
 valid=cv2.bitwise_and((finding.valid_mask>0).astype(np.uint8)*255,p.skin_mask)
 mask=cv2.bitwise_and((finding.mask>0).astype(np.uint8)*255,valid)
 response=np.asarray(finding.response,dtype=np.float32).copy(); response[valid==0]=0
 instances=[]; excluded=[]
 h,w=valid.shape
 for raw in finding.instances:
  item=dict(raw); x,y=centroid(item)
  ix,iy=int(round(x)),int(round(y))
  if not (0<=ix<w and 0<=iy<h and valid[iy,ix]):
   excluded.append({"centroid":[x,y],"reason":"centre_outside_effective_skin"})
   continue
  item["centroid"]=[x,y]
  xy=cv2.transform(np.array([[[x,y]]],np.float32),p.inverse_transform_matrix)[0,0]
  if not (0<=xy[0]<original_shape[1] and 0<=xy[1]<original_shape[0]):
   excluded.append({"centroid":[x,y],"reason":"centre_outside_original_image"})
   continue
  item["original_centroid"]=[float(xy[0]),float(xy[1])]
  item["region"]=next((n for n,m in prepared.regions.items() if m[iy,ix]>0),None)
  item["id"]=len(instances)+1
  instances.append(item)
 def metrics(domain,rows):
  domain=domain>0; area=int(domain.sum())
  if not area:return {"status":"unavailable","count":None,"analysis_pixels":0,"coverage":None,"density_per_100k_pixels":None}
  scalar=response.ndim==2 and finding.mask_semantics!="candidate_box"
  values=response[domain] if scalar else np.array([])
  return {"status":"available","count":len(rows),"analysis_pixels":area,
    "coverage":None if finding.mask_semantics=="candidate_box" else float(np.count_nonzero((mask>0)&domain)/area),
    "density_per_100k_pixels":len(rows)*100000/area,
    "mean_response":float(np.mean(values)) if values.size else None,
    "p90_response":float(np.percentile(values,90)) if values.size else None,
    "response_statistics_status":"scalar_response" if scalar else "not_a_scalar_measurement"}
 by_region={}
 for name,m in prepared.regions.items():
  by_region[name]=metrics(cv2.bitwise_and(valid,m),[i for i in instances if i["region"]==name])
 return {"status":"success","instances":instances,"excluded_instances":excluded,"metrics":metrics(valid,instances),
   "regions":by_region,"mask_semantics":finding.mask_semantics,
   "native_metrics":finding.native_metrics},mask,response,valid

def choose_regions(views):
 output={}
 for region in REGIONS:
  candidates=[]
  for view in VIEWS:
   data=views.get(view,{})
   if data.get("status") not in ("success","partial_success"): continue
   area=data.get("region_pixels",{}).get(region,0)
   if not area: continue
   # Prefer frontal central regions; select observed cheek/eye projection by actual usable area.
   central=region in {"forehead","nose","perioral","chin"}
   priority=2 if central and view=="front" else 1
   candidates.append((priority,area,view))
  if not candidates: output[region]={"view":None,"modules":{}}; continue
  _,_,view=max(candidates,key=lambda row:(row[0],row[1],-VIEWS.index(row[2])))
  output[region]={"view":view,"modules":{
    name:item.get("regions",{}).get(region,{"status":"unavailable"})
    for name,item in views[view].get("modules",{}).items()}}
 return output
