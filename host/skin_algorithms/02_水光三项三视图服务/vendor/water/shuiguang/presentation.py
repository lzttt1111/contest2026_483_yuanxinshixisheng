"""V2-style presentation only. Inputs are saved masks/instances, never re-detected."""
from pathlib import Path
import json
from types import SimpleNamespace
import cv2
import numpy as np
from dataclasses import replace
from .io import write_image,write_json,sha256
from .config import ROOT

def contours_for(p):
 from sg_core.engines.visia_regions import build_visia_regions
 regions=build_visia_regions(p.analysis_image,p.skin_mask,p.landmarks,p.quality_flags,
                           include_chin=True,mode="full")
 if getattr(p,"_presentation_no_bounds",False):
  regions=replace(regions,display_regions={},display_contour=None,display_separator=None)
 return regions

def boundary(canvas,regions,thickness=3):
 from sg_core.engines.visia_regions import draw_region_boundaries,VISIA_BOUNDARY_COLOR
 return draw_region_boundaries(canvas,regions.display_regions,VISIA_BOUNDARY_COLOR,
  thickness=thickness,partial_face=regions.partial_face,
  closed_contour=regions.display_contour,separator_contour=regions.display_separator)

def _build_result_native(name,p,mask,response,instances,base):
 """Reuses V2 marker/outline functions with the existing accepted instance list."""
 regions=contours_for(p)
 if name=="redness":
  from sg_core.engines.rbx_engine import ErythemaAnalyzer
  painter=object.__new__(ErythemaAnalyzer)
  result,_=painter._render_red_feature_overlay(base,regions.display_regions,instances,
     regions.partial_face,regions.display_contour,regions.display_separator)
  return result
 if name=="brown":
  from sg_core.consumer_pigment.markers import render_compact_marker_mask,BROWN_STYLE_MARKER
  return boundary(render_compact_marker_mask(base,mask,BROWN_STYLE_MARKER),regions,2)
 if name=="spots":
  from sg_core.engines.spots_engine import _draw_overlay
  return _draw_overlay(base,regions.display_regions,regions.partial_face,mask,
                       regions.display_contour,regions.display_separator)
 if name=="pores":
  from sg_core.engines.pores_engine import Config
  result=base.copy()
  for point in instances:
   x,y=np.rint(point["centroid"]).astype(int)
   radius=max(1,min(2,int(point.get("radius_px",2))))
   cv2.circle(result,(x,y),radius,Config.FEATURE_COLOR,-1,cv2.LINE_AA)
  return boundary(result,regions)
 if name=="texture":
  from sg_core.engines.texture_engine import Config
  result=base.copy()
  for point in instances:
   x,y=np.rint(point["centroid"]).astype(int)
   radius=Config.FEATURE_RADIUS+int(point.get("texture_score",0)>=.82)
   color=Config.RAISED_COLOR if point.get("texture_type")=="raised_like" else Config.DEPRESSED_COLOR
   cv2.circle(result,(x,y),radius,color,-1,cv2.LINE_AA)
  return boundary(result,regions)
 if name=="vascular":
  from skimage.morphology import skeletonize
  from sg_core.engines.clinic_vascular_structure_engine import ClinicVascularStructureAnalyzer
  skeleton=skeletonize(mask>0).astype(np.uint8)
  degree=cv2.filter2D(skeleton,cv2.CV_16S,np.ones((3,3),np.uint8))-skeleton
  branch=((degree>=3)&(skeleton>0)).astype(np.uint8)
  return ClinicVascularStructureAnalyzer.render_overlay(base,skeleton,branch)
 if name=="surface_gloss":
  from sg_core.engines.surface_gloss_engine import SurfaceGlossAnalyzer,build_oil_regions,Config
  high=((response>=Config.HIGH_GLOSS_INTENSITY_THRESHOLD)&(mask>0)).astype(np.uint8)*255
  result,_=SurfaceGlossAnalyzer._render_overlay(base,mask,high,build_oil_regions(regions,p.landmarks))
  return result
 if name=="acne":
  result=base.copy()
  for point in instances:
   x,y=np.rint(point["centroid"]).astype(int)
   box=point.get("bbox_xyxy")
   radius=max(5,round(max(box[2]-box[0],box[3]-box[1])*.6)) if box else 8
   cv2.circle(result,(x,y),radius,(60,55,225),2,cv2.LINE_AA)
  return result
 raise ValueError("未知结果图类型")

LINED_MODULES=frozenset({"pores","texture","spots","brown","redness"})

def render_with_scope(name,p,mask,response,instances,base,valid_mask=None):
 valid=((p.skin_mask if valid_mask is None else valid_mask)>0).astype(np.uint8)*255
 for point in instances:
  x,y=np.rint(point["centroid"]).astype(int)
  if not (0<=y<valid.shape[0] and 0<=x<valid.shape[1] and valid[y,x]):
   raise ValueError("正式目标中心不在有效分析域内，需先同步修正实例与量化")
 if name not in LINED_MODULES:
  return _build_result_native(name,p,mask,response,instances,base),None,"none"
 marker_p=SimpleNamespace(analysis_image=p.analysis_image,skin_mask=p.skin_mask,
     landmarks=p.landmarks,quality_flags=p.quality_flags,_presentation_no_bounds=True)
 marked=_build_result_native(name,marker_p,mask,response,instances,base)
 regions=contours_for(p)
 scope=regions.scope_mask if regions.scope_mask is not None else regions.analysis_mask
 scope=(scope>0).astype(np.uint8)*255
 outside=any(not scope[int(round(i["centroid"][1])),int(round(i["centroid"][0]))] for i in instances)
 if outside:
  edges,_=cv2.findContours(valid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  scope=np.zeros_like(valid);cv2.drawContours(scope,edges,-1,255,-1)
  marked[(scope==0)|(valid==0)]=base[(scope==0)|(valid==0)]
  from sg_core.engines.visia_regions import VISIA_BOUNDARY_COLOR
  cv2.drawContours(marked,edges,-1,VISIA_BOUNDARY_COLOR,2,cv2.LINE_AA)
  return marked,scope,"effective_domain_outline"
 marked[(scope==0)|(valid==0)]=base[(scope==0)|(valid==0)]
 return boundary(marked,regions),scope,"v2_scope_verified"

def build_result(name,p,mask,response,instances,base,valid_mask=None):
 return render_with_scope(name,p,mask,response,instances,base,valid_mask)[0]

def save_item(root,view,name,p,mask,response,instances,base=None,valid_mask=None):
 root=Path(root);folder=root/"05_V2正式结果图"/view;folder.mkdir(parents=True,exist_ok=True)
 if name in {"redness","brown","vascular"} and base is None:
  raise ValueError("红棕/线状红纹缺少V2底图，不能回退原图染色")
 base=p.analysis_image if base is None else base
 role="red" if name in {"redness","vascular"} else "brown" if name=="brown" else "rgb"
 base_path=folder/(role+"_base.png")
 # Preserve a separate JPEG-decoded red input for the inherited vascular role.
 if name=="vascular":base_path=folder/"vascular_red_base.png"
 write_image(base_path,base)
 result,scope,outline_mode=render_with_scope(name,p,mask,response,instances,base,valid_mask)
 path=folder/(name+"_result.png");write_image(path,result)
 info={"base":base_path.relative_to(root).as_posix(),"result":path.relative_to(root).as_posix(),
         "base_sha256":sha256(base_path),"result_sha256":sha256(path),"analysis_space":"aligned_1024",
         "style":"V2 native base and item-specific display policy","outline_mode":outline_mode}
 if scope is not None:
  scope_path=folder/(name+"_display_scope.png");write_image(scope_path,scope)
  info.update(display_scope=scope_path.relative_to(root).as_posix(),display_scope_sha256=sha256(scope_path))
 return info

def save_index(root,views):
 path=Path(root)/"V2结果图索引.json"
 write_json(path,{"version":"v2_native_presentation_v2_item_scopes","views":views})
 return path

class MissingStyleBuilder:
 """One-time backfill for pre-upgrade result packages missing their RGB color bases."""
 def __init__(self):
  from sg_core.engines.rbx_engine import ErythemaAnalyzer
  from sg_core.engines.brown_engine import BrownAreaAnalyzer
  self.red=ErythemaAnalyzer()
  self.brown=BrownAreaAnalyzer()
  self.brown._face_mask_analyzer=self.red
 def bases(self,p,brown_response):
  from sg_core.consumer_pigment.brown_contrast import enhance_brown_score_contrast
  image=p.analysis_image
  face=self.red._semantic_face_mask(image,p.landmarks,p.skin_mask)
  foreground=self.red._person_foreground_mask(image,face)
  stats=cv2.bitwise_and(p.skin_mask,face)
  work=self.red._rbx_lab_color_transfer(image,stats,face)
  _,_,raw,_=self.red._build_redness_maps(work,stats,face,p.landmarks)
  score=self.red._apply_style_lut(raw);score[face==0]=0
  luminance=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)/255
  red=self.red._render_result(score,luminance,self.red._foreground_alpha(foreground),
                             self.red._foreground_alpha(face),image,"natural")
  regions=contours_for(p)
  brown=self.brown._render_brown(image,face,foreground,regions.analysis_mask)
  brown=enhance_brown_score_contrast(brown,brown_response,regions.analysis_mask)
  return {"redness":red,"brown":brown,"vascular":cv2.imdecode(cv2.imencode(".jpg",red)[1],cv2.IMREAD_COLOR)}
 def close(self):self.red.close()
