"""Result-only effective-domain checks, reusing the V2 2000-pixel scoring floor."""
from pathlib import Path
from .measurements import read_evidence
from .regions import build
SOURCES={"01":"pores","02":"surface_gloss","03":"spots","05":"vascular","06":"acne",
         "07":"wrinkle","08":"wrinkle","09":"wrinkle","10":"texture","11":"rgb"}
def areas(root):
 root=Path(root);cache={};out={}
 for mid,project in SOURCES.items():
  if project not in cache:cache[project]=read_evidence(root/("doctor_v3_"+project+".npz"))[0]
  arrays=cache[project]
  if mid in ("09","11"):
   # V3 groove regions require 3D and have no 2D masks. Their absence is not
   # an observed zero-area face; only the saved V2 full-face scope is known.
   domains={"full_face":arrays["valid"]>0}
  elif mid=="08":
   from .wrinkle_metrics import regions
   domains=regions(arrays);domains["full_face"]=arrays["valid"]>0
  else:domains=build(arrays["valid"],arrays["landmarks"],mid)
  for name,mask in domains.items():out[mid+":"+name]=int((mask>0).sum())
 # Diffuse-background scopes are deliberately not substituted with the old red domain.
 return out
