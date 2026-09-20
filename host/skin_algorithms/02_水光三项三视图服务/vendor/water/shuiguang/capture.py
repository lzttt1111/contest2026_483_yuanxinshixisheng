from pathlib import Path
import json, re
from PIL import Image
from .config import VIEWS
from .io import sha256, read_image

def load_capture(manifest):
 path=Path(manifest).resolve()
 data=json.loads(path.read_text(encoding="utf-8"))
 if data.get("schema_version") != "shuiguang_capture_v1": raise ValueError("不支持的三视图清单版本")
 if data.get("modality","RGB") != "RGB": raise ValueError("本项目只支持RGB")
 if set(data.get("views",{})) != set(VIEWS): raise ValueError("必须提供左侧、正面、右侧三张图")
 sid=str(data.get("capture_id",""))
 if not re.fullmatch(r"[\w-]{1,80}",sid): raise ValueError("采集编号仅允许文字、数字、下划线和短横线")
 loaded={}
 for view in VIEWS:
  item=dict(data["views"][view])
  if item.get("modality","RGB")!="RGB": raise ValueError("不接受混合光源")
  source=(path.parent/item["path"]).resolve()
  if not source.is_file(): raise ValueError(f"{view}原图不存在")
  actual=sha256(source)
  if item.get("sha256") and item["sha256"]!=actual: raise ValueError(f"{view}原图SHA不一致")
  if type(item.get("mirrored",False)) is not bool: raise ValueError("mirrored必须为布尔值")
  with Image.open(source) as im:
   if im.mode not in {"RGB","RGBA"}: raise ValueError("需要彩色RGB照片，不接受灰度图")
  image=read_image(source,item.get("mirrored",False))
  loaded[view]={**item,"source":source,"sha256":actual,"image":image}
 if len({v["sha256"] for v in loaded.values()})!=3: raise ValueError("三视图不能重复使用同一图片")
 return data,loaded

def folder_manifest(folder,capture_id=None):
 folder=Path(folder).resolve()
 views={}
 for view,name in zip(VIEWS,("RGB_L.jpg","RGB_M.jpg","RGB_R.jpg")):
  p=folder/name
  if not p.is_file(): raise ValueError(f"缺少{name}")
  item={"path":name,"sha256":sha256(p),"mirrored":False,"capture_method":"manual_import"}
  sidecars=[p.with_suffix(".json"),p.with_suffix(p.suffix+".json")]
  present=[m for m in sidecars if m.is_file()]
  if len(present)>1: raise ValueError("同一照片存在两份拍照元数据，需明确保留一份")
  if present:
   metadata=json.loads(present[0].read_text(encoding="utf-8"))
   for key in ("frame_id","timestamp","capture_time","raw_pose","pose","config","stable","mirrored"):
    if key in metadata: item[key]=metadata[key]
   item["capture_method"]="auto_capture_metadata"
   item["capture_metadata_sha256"]=sha256(present[0])
  views[view]=item
 return {"schema_version":"shuiguang_capture_v1","capture_id":capture_id or folder.name,
         "subject_id":capture_id or folder.name,"modality":"RGB","views":views}
