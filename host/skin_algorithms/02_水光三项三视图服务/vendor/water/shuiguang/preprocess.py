"""View-specific quality policy; original hair detector thresholds stay unchanged."""
from dataclasses import dataclass
import cv2
import numpy as np
from sg_core.preprocess.image_preprocessor import ImagePreprocessor
from sg_core.capture_profile import CaptureProfile
from .config import ROOT, REGIONS

class TripletPreprocessor(ImagePreprocessor):
 def __init__(self):
  super().__init__(str(ROOT),str(ROOT/"runtime"),str(ROOT/"data/reference.jpg"),capture_profile=CaptureProfile.CONSUMER)
  self.view="front"
 def _quality_assessment(self,img,points,face_count):
  flags=[]
  if face_count!=1: flags.append("MULTIPLE_FACES" if face_count>1 else "NO_FACE")
  face=self._face_hull_mask(img.shape[:2],points)>0
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  pixels=gray[face]
  if not pixels.size: return 0.0,"REJECT",["NO_FACE"]
  if np.mean(pixels<20)>.45: flags.append("UNDEREXPOSED")
  if np.mean(pixels>248)>.25: flags.append("OVEREXPOSED")
  x,y,w,h=cv2.boundingRect(face.astype(np.uint8))
  crop=gray[y:y+h,x:x+w]
  normalized=cv2.resize(crop,(512,max(64,round(512*h/max(w,1)))))
  blur=float(cv2.Laplacian(normalized,cv2.CV_32F).var())
  if min(w,h)<160 or blur<12: flags.append("BLUR")
  # These are inherited 2D geometry indicators, not physical head angles.
  yaw,pitch,roll=self._pose_proxies(points)
  if self.view!="front": flags.append("PARTIAL_FACE")
  elif abs(yaw)>28: flags.append("FRONT_VIEW_MISMATCH")
  if abs(roll)>22: flags.append("EXCESSIVE_TILT")
  critical=set(flags)-{"PARTIAL_FACE"}
  return (50.0,"REJECT",flags) if critical else (95.0,"PASS",flags)

@dataclass
class Prepared:
 result: object
 regions: dict
 quality: dict

def region_masks(result,view):
 from .anatomy import partition
 bundle=getattr(result,"mask_bundle",None)
 hair=bundle.hair_mask if bundle is not None else None
 anatomy=partition(result.landmarks,result.skin_mask,view,hair)
 result._anatomy=anatomy
 return anatomy.masks(),{name:entry["state"]=="observed" for name,entry in anatomy.coverage.items()}

def prepare(preprocessor,image,view):
 preprocessor.view=view
 r=preprocessor.preprocess_image(image)
 if r.mask_bundle is not None: r.skin_mask=r.mask_bundle.algorithm_mask().copy()
 r._observed_skin_mask=r.skin_mask.copy()
 regions,visibility=region_masks(r,view) if len(r.landmarks)>=468 else ({n:np.zeros_like(r.skin_mask) for n in REGIONS},{n:False for n in REGIONS})
 from .anatomy import sync_mask_bundle
 # Anatomical label gaps do not erase valid skin findings; only explicit hidden-side evidence does.
 hidden=r._anatomy.hidden if hasattr(r,"_anatomy") else np.zeros_like(r.skin_mask)
 allowed=cv2.bitwise_and(r.skin_mask,cv2.bitwise_not(hidden))
 sync_mask_bundle(r,allowed)
 # Every consumer sees the same allowed visible skin, never the hidden-side estimate.
 if getattr(r,"_debug_masks",None) is not None:
  r._debug_masks={k:v.copy() for k,v in r._debug_masks.items()}
 flags=list(r.quality_flags)
 valid=int(np.count_nonzero(r.skin_mask))
 if valid<4000: r.quality_status="REJECT"; flags.append("INSUFFICIENT_VISIBLE_SKIN")
 return Prepared(r,regions,{"status":r.quality_status,"flags":list(dict.fromkeys(flags)),
  "valid_pixels":valid,"region_visible":visibility,"view":view})
