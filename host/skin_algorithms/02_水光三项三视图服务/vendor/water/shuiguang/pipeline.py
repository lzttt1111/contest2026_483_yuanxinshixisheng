import json, shutil, time, traceback
from pathlib import Path
import cv2
import numpy as np
from .config import ROOT,VIEWS,MODULES,REGIONS,COLORS
from .capture import load_capture
from .io import write_image,write_json,sha256
from .preprocess import TripletPreprocessor,prepare
from .detectors import Detectors
from .results import measure,choose_regions
from .render import rebuild,display_layer
from .presentation import save_item,save_index
from .regional_package import enrich_result

ORDER=("redness","spots","brown","texture","pores","surface_gloss","vascular","acne")

class AnalysisSession:
 def __init__(self,device="cpu"):
  from .runtime import configure
  configure(device); self.device=device
  start=time.perf_counter()
  self.preprocessor=TripletPreprocessor()
  self.detectors=Detectors(device)
  self.cold_seconds=time.perf_counter()-start
 def close(self):
  self.detectors.close(); self.preprocessor.close()
 def analyze(self,manifest,output_dir,progress=None):
  start=time.perf_counter(); capture,inputs=load_capture(manifest)
  root=Path(output_dir).resolve()/capture["capture_id"]
  if root.exists(): raise FileExistsError("结果目录已存在，请换一个输出目录，避免覆盖历史结果")
  root.mkdir(parents=True)
  if capture.get('acquisition'):
   originals=root/'00_采集原始JPEG';originals.mkdir()
   evidence={'acquisition':capture['acquisition'],'views':{}}
   for view,inp in inputs.items():
    dst=originals/(view+'.jpg');shutil.copyfile(inp['source'],dst)
    evidence['views'][view]={k:v for k,v in inp.items() if k not in ('source','image','path')}
    evidence['views'][view].update(path=dst.relative_to(root).as_posix(),sha256=sha256(dst))
   write_json(root/'自动采集原图索引.json',evidence)
  write_json(root/"运行回执.json",{"status":"running","capture_id":capture["capture_id"]})
  data={"schema_version":"shuiguang_result_v1","product_version":"0.0.1",
        "capture_id":capture["capture_id"],"subject_id":str(capture.get("subject_id","")),
        "modality":"RGB","views":{},"module_labels":MODULES,"region_labels":REGIONS,
        "region_definition_version":"shuiguang_anatomy_v1"}
  timings={}; saved_capture=dict(capture);saved_capture["views"]={}; presentations={}
  for vi,view in enumerate(VIEWS):
   inp=inputs[view]; image=inp["image"]
   folder=root/f"0{vi+1}_{view}";folder.mkdir()
   inputpath=root/"00_输入图片"/f"{view}.png";write_image(inputpath,image)
   saved_capture["views"][view]={k:v for k,v in inp.items() if k not in {"source","image","path"}}
   saved_capture["views"][view].update({"path":inputpath.relative_to(root).as_posix(),"source_sha256":inp["sha256"],"sha256":sha256(inputpath),"mirrored":False})
   began=time.perf_counter()
   if progress: progress({"view":view,"module":"preprocess","completed":vi*8,"total":24})
   try:
    prepared=prepare(self.preprocessor,image,view); p=prepared.result
   except Exception as exc:
    data["views"][view]={"input":inputpath.relative_to(root).as_posix(),"status":"failed",
      "quality":{"status":"REJECT","flags":["PREPROCESS_FAILED"]},"region_pixels":{},
      "modules":{n:{"status":"unavailable","reason":"本视角预处理未完成"} for n in MODULES}}
    write_json(folder/"failure.json",{"error_type":type(exc).__name__,"message":str(exc)})
    timings[view]={"preprocess":time.perf_counter()-began,"modules":{}}
    continue
   self.detectors.red_base=None
   vdata={"input":inputpath.relative_to(root).as_posix(),"source_sha256":inp["sha256"],
    "original_size":[image.shape[1],image.shape[0]],"quality":prepared.quality,
    "transform":p.face_transform_matrix.tolist(),"inverse_transform":p.inverse_transform_matrix.tolist(),
    "region_pixels":{n:int(np.count_nonzero(m)) for n,m in prepared.regions.items()},"modules":{}}
   geometry=folder/"geometry.npz"
   bundle=getattr(p,"mask_bundle",None)
   hair=bundle.hair_mask if bundle is not None else np.zeros_like(p.skin_mask)
   np.savez_compressed(geometry,skin_mask=p.skin_mask,landmarks=p.landmarks,
     observed_skin_mask=getattr(p,"_observed_skin_mask",p.skin_mask),hair_mask=hair,**prepared.regions)
   vdata["geometry"]=geometry.relative_to(root).as_posix()
   timings[view]={"preprocess":time.perf_counter()-began,"modules":{}}
   presentations[view]={"modules":{}}
   for mi,name in enumerate(ORDER):
    t=time.perf_counter(); dst=folder/name; dst.mkdir()
    if progress: progress({"view":view,"module":name,"completed":vi*8+mi,"total":24})
    if prepared.quality["status"]=="REJECT":
     item={"status":"unavailable","reason":"图像质量不满足检测要求","quality_flags":prepared.quality["flags"]}
    else:
     try:
      finding=self.detectors.detect(name,p)
      item,mask,response,valid=measure(finding,prepared,image.shape)
      item["module"]=name
      item["effective_domain_origin"]="detector_medical_analysis_mask_v2"
      presentations[view]["modules"][name]=save_item(root,view,name,p,mask,response,item["instances"],finding.base,valid)
      original_mask=cv2.warpAffine(mask,p.inverse_transform_matrix,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
      original_valid=cv2.warpAffine(valid,p.inverse_transform_matrix,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
      write_image(dst/"mask.png",original_mask);write_image(dst/"valid.png",original_valid)
      scale=float(np.hypot(p.inverse_transform_matrix[0,0],p.inverse_transform_matrix[1,0]))
      layer=display_layer(original_mask,name,item["instances"],scale)
      write_image(dst/"overlay.png",layer)
      item["overlay_layer"]=(dst/"overlay.png").relative_to(root).as_posix()
      np.savez_compressed(dst/"evidence.npz",mask=mask,response=response,valid_mask=valid)
      item.update({"mask":(dst/"mask.png").relative_to(root).as_posix(),
       "valid_mask":(dst/"valid.png").relative_to(root).as_posix(),
       "evidence":(dst/"evidence.npz").relative_to(root).as_posix(),
       "mask_sha256":sha256(dst/"mask.png"),"evidence_sha256":sha256(dst/"evidence.npz")})
     except Exception as exc:
      item={"status":"failed","reason":"本项检测未完成"}
      # Technical diagnostics are private to evidence; no exception/path in visible results.
      write_json(dst/"failure.json",{"error_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    vdata["modules"][name]=item
    write_json(dst/"metrics.json",item)
    timings[view]["modules"][name]=time.perf_counter()-t
   passed=sum(x["status"]=="success" for x in vdata["modules"].values())
   vdata["status"]="success" if passed==8 else ("partial_success" if passed else "failed")
   data["views"][view]=vdata
   write_json(root/"水光检测完整结果.json",data)
  data["region_summary"]=choose_regions(data["views"])
  success=sum(i["status"]=="success" for v in data["views"].values() for i in v["modules"].values())
  data["status"]="success" if success==24 else ("partial_success" if success else "failed")
  data["completed_modules"]=success; data["expected_modules"]=24
  write_json(root/"水光检测完整结果.json",data)
  write_json(root/"三视图输入清单.json",saved_capture)
  save_index(root,presentations)
  stats_start=time.perf_counter();enrich_result(root)
  regional_seconds=time.perf_counter()-stats_start
  render_start=time.perf_counter();rebuild(root)
  write_json(root/"运行回执.json",{"status":data["status"],"completed_modules":success,"expected_modules":24,
   "device":self.device,"cold_start_seconds":self.cold_seconds,"timings":timings,
   "regional_statistics_seconds":regional_seconds,
   "render_seconds":time.perf_counter()-render_start,"wall_seconds":time.perf_counter()-start,
   "models_reused_across_views":True,"inference_executed":True})
  if progress: progress({"completed":24,"total":24,"status":data["status"]})
  return root,data["status"]
