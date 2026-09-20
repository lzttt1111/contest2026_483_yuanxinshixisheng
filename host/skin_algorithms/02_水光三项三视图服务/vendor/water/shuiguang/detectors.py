"""Eight in-memory detectors. No report, cloud task, spectral synthesis or grading."""
from dataclasses import dataclass
import os, hashlib
import cv2
import numpy as np
from .config import ROOT, MODULES
from .io import plain, sha256

@dataclass
class Finding:
 mask: np.ndarray
 response: np.ndarray
 instances: list
 valid_mask: np.ndarray
 native_metrics: dict
 mask_semantics: str = "detected_support"
 base: np.ndarray | None = None

def centroid(item):
 for key in ("centroid","center_xy","center"):
  v=item.get(key)
  if isinstance(v,(list,tuple,np.ndarray)) and len(v)>=2: return float(v[0]),float(v[1])
 if "x" in item and "y" in item: return float(item["x"]),float(item["y"])
 box=item.get("bbox_xyxy") or item.get("bbox")
 if box and len(box)==4: return (float(box[0])+float(box[2]))/2,(float(box[1])+float(box[3]))/2
 raise ValueError("实例缺少可追溯中心点")

def components(mask):
 count,labels,stats,centers=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
 return [{"id":i,"centroid":centers[i].tolist(),"area_px":int(stats[i,4])} for i in range(1,count)]

class Detectors:
 def __init__(self,device):
  from sg_core.engines.rbx_engine import ErythemaAnalyzer
  from sg_core.engines.brown_engine import BrownAreaAnalyzer
  from sg_core.engines.spots_engine import SpotsEngine
  from sg_core.engines.pores_engine import PoresAnalyzer
  from sg_core.engines.texture_engine import TextureAnalyzer
  from sg_core.engines.surface_gloss_engine import SurfaceGlossAnalyzer
  from sg_core.engines.clinic_vascular_structure_engine import ClinicVascularStructureAnalyzer
  from sg_core.capture_profile import CaptureProfile
  from sg_core.acne.acne_detector import AcneDetector
  self.red=ErythemaAnalyzer(capture_profile=CaptureProfile.CONSUMER)
  self.brown=BrownAreaAnalyzer(CaptureProfile.CONSUMER)
  self.brown._face_mask_analyzer=self.red
  self.spots=SpotsEngine(); self.pores=PoresAnalyzer(); self.texture=TextureAnalyzer()
  self.gloss=SurfaceGlossAnalyzer(); self.lines=ClinicVascularStructureAnalyzer()
  weight=ROOT/"models/acne/detector/acne_candidate_baseline_v3_n_768.pt"
  if sha256(weight)!="50df3a1db9d2ae7888958c36758430b4c6aac82543c75e72519eef3277fa2b23":
   raise ValueError("痤疮模型SHA不匹配")
  self.acne=AcneDetector(weight,conf=.15,imgsz=768,device=device)
  self.red_base=None

 def detect(self,name,p):
  if name=="redness":
   image=p.analysis_image; skin=p.skin_mask
   face=self.red._semantic_face_mask(image,p.landmarks,skin)
   r=self.red._process_arrays(image,skin,p.landmarks,face,"","input","input",
       quality_score=p.quality_score,quality_status=p.quality_status,quality_flags=p.quality_flags)
   if r is None: raise ValueError("红区有效域不足")
   # V2 consumer vascular reads the saved JPEG RED base; preserve those exact pixels in memory.
   self.red_base=cv2.imdecode(cv2.imencode('.jpg',r['base'])[1],cv2.IMREAD_COLOR)
   metrics={"marker_count":len(r["instances"]),"display_mask_sha256":hashlib.sha256(r["display_mask"].tobytes()).hexdigest()}
   return Finding(r["mask"],r["response"],r["instances"],r["valid_mask"],metrics,"instance_support",r["base"])
  if name=="vascular":
   if self.red_base is None: raise ValueError("红区底图缺失，线状红纹不使用替代结果")
   r=self.lines.detect(p,red_display=self.red_base)
   if not r.metrics.get("qc_passed",False): raise ValueError(str(r.metrics.get("quality_reason",r.metrics.get("reason","线状结构有效域不足"))))
   return Finding(r.vascular_mask,r.vesselness,list(r.instances),r.valid_mask,
                  {"total_length_px":r.metrics["vascular_total_length_px"]},"line_support",self.red_base)
  if name=="surface_gloss":
   r=self.gloss.detect_surface_gloss(p)
   return Finding(r.gloss_mask,r.gloss_intensity_map,components(r.gloss_mask),r.gloss_analysis_mask,
                  {"area_ratio":r.gloss_area_ratio,"mean_intensity":r.mean_gloss_intensity})
  if name=="acne":
   from sg_core.acne.detector_postprocess import filter_detections
   r=self.acne.predict(p.analysis_image)
   instances=filter_detections(r["detections"],p.skin_mask,cv2.bitwise_not(p.skin_mask),p.analysis_image.shape[:2])
   mask=np.zeros_like(p.skin_mask); response=np.zeros(mask.shape,np.float32)
   for item in instances:
    x1,y1,x2,y2=map(lambda x:int(round(x)),item["bbox_xyxy"])
    mask[max(y1,0):y2,max(x1,0):x2]=255
    response[max(y1,0):y2,max(x1,0):x2]=item["confidence"]
   return Finding(mask,response,instances,p.skin_mask,{"model":"acne_candidate_baseline_v3_n_768","conf":.15},"candidate_box")
  mapping={
   "pores":(self.pores.detect_pores,"pore_mask","pore_score_map","pore_locations"),
   "texture":(self.texture.detect_texture,"feature_mask","texture_score_map","texture_feature_locations"),
   "spots":(self.spots.detect_spots,"filtered_mask","candidate_heatmap","spot_locations"),
   "brown":(self.brown.detect_brown,"instance_mask","brown_score_map","brown_spot_locations"),
  }
  fn,mask,response,instances=mapping[name]; r=fn(p)
  native={k:v for k,v in r.metrics().items() if k not in {"region_distribution","quality_score","quality_status","quality_flags"} and isinstance(v,(int,float))}
  return Finding(getattr(r,mask),getattr(r,response),getattr(r,instances),r._medical_analysis_mask,native,
                 "marker_footprint" if name in {"pores","texture"} else "instance_support",
                 getattr(r,"brown_rbx_image",None))

 def close(self):
  self.brown._face_mask_analyzer=None
  self.red.close()
  self.spots.close()
