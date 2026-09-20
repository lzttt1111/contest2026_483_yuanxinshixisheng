"""Deterministic triptychs and static viewer rebuilt solely from saved results."""
from pathlib import Path
import json, shutil, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import ROOT,VIEWS,VIEW_LABELS,MODULES,COLORS
from .io import read_image,write_image,contained,sha256,write_json

def display_layer(mask,name,instances,scale=1.0):
 layer=np.zeros((*mask.shape,4),np.uint8)
 layer[:,:,:3]=COLORS[name]
 factor=min(420/mask.shape[1],560/mask.shape[0])
 if name=="vascular":
  radius=max(1,int(round(.65/max(factor,.01))))
  alpha=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
 elif name=="acne":
  alpha=np.zeros_like(mask)
  for item in instances:
   x,y=map(lambda v:int(round(v)),item["original_centroid"])
   box=item.get("bbox_xyxy")
   radius=max(5,int(round(max(box[2]-box[0],box[3]-box[1])*scale*.6))) if box else max(5,int(5/factor))
   cv2.circle(alpha,(x,y),radius,255,max(1,int(round(1.3/factor))),cv2.LINE_AA)
 else: alpha=mask
 layer[:,:,3]=alpha
 return layer

def annotate(original,item,root,alpha=.6):
 image=original.copy()
 if item.get("status")!="success": return image
 layer=cv2.imdecode(np.fromfile(contained(root,item["overlay_layer"]),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
 amount=layer[:,:,3:4].astype(np.float32)/255*alpha
 return np.uint8(np.clip(image.astype(np.float32)*(1-amount)+layer[:,:,:3]*amount,0,255))

def triptych(images,labels,notes,path,panel_height=560):
 fontpath=ROOT/"assets/NotoSansCJK-Regular.ttc"
 font=ImageFont.truetype(str(fontpath),25)
 small=ImageFont.truetype(str(fontpath),18)
 width,height=420,panel_height
 canvas=Image.new("RGB",(width*3+64,height+80),"white")
 draw=ImageDraw.Draw(canvas)
 for i,(image,label,note) in enumerate(zip(images,labels,notes)):
  photo=Image.fromarray(image[:,:,::-1]); photo.thumbnail((width,height))
  x=16+i*(width+16); y=48+(height-photo.height)//2
  draw.rectangle((x,48,x+width,48+height),fill="#f3f5f7")
  canvas.paste(photo,(x+(width-photo.width)//2,y))
  draw.text((x,10),label,font=font,fill="#1f2937")
  draw.text((x,height+54),note[:23],font=small,fill="#64748b")
 canvas.save(path)

def rebuild(result_dir):
 root=Path(result_dir).resolve()
 data=json.loads((root/"水光检测完整结果.json").read_text(encoding="utf-8"))
 presentation_path=root/"V2结果图索引.json"
 if presentation_path.is_file():data["presentation"]=json.loads(presentation_path.read_text(encoding="utf-8"))
 for key,file in (("regional_statistics","区域统计V2.json"),("concern_map","面部问题地图.json"),("front_v2_scores","正面V2评分.json")):
  if (root/file).is_file():data[key]=json.loads((root/file).read_text(encoding="utf-8"))
 target=root/"04_三视图结果图";target.mkdir(exist_ok=True)
 originals=[read_image(contained(root,data["views"][v]["input"])) for v in VIEWS]
 names=[VIEW_LABELS[v] for v in VIEWS]
 triptych(originals,names,["","",""],target/"00_原图三视图.jpg")
 index={"original_triptych":"04_三视图结果图/00_原图三视图.jpg","modules":{}}
 for n,label in MODULES.items():
  images=[]; notes=[]
  for v,original in zip(VIEWS,originals):
   item=data["views"][v].get("modules",{}).get(n,{"status":"unavailable"})
   style=data.get("presentation",{}).get("views",{}).get(v,{}).get("modules",{}).get(n)
   if style:
    base_path=contained(root,style["base"])
    if sha256(base_path)!=style["base_sha256"]:raise ValueError("V2底图SHA不匹配")
    formal=contained(root,style["result"])
    if sha256(formal)!=style["result_sha256"]:raise ValueError("正式结果图SHA不匹配")
    images.append(read_image(formal))
   elif "presentation" in data and item.get("status")=="success":
    raise ValueError("缺少V2正式结果图；请补齐制图证据，不允许回退通用染色")
   else:images.append(annotate(original,{**item,"module":n},root))
   notes.append(("检出 "+str(item["metrics"]["count"])+" 个目标") if item.get("status")=="success" else "本视角暂不可评估")
  name=f"{list(MODULES).index(n)+1:02d}_{label}三视图.jpg"
  triptych(images,names,notes,target/name,panel_height=420 if "presentation" in data else 560)
  index["modules"][n]={"path":"04_三视图结果图/"+name,"sha256":sha256(target/name)}
 if "presentation" in data:index["presentation_index"]="V2结果图索引.json"
 for key,file in (("regional_statistics","区域统计V2.json"),("concern_map","面部问题地图.json"),("front_v2_scores","正面V2评分.json")):
  if key in data:index[key]={"path":file,"sha256":sha256(root/file)}
 index["original_sha256"]=sha256(root/index["original_triptych"])
 write_json(root/"结果索引.json",index)
 for name in ("app.js","style.css","map.js","workflow.js","camera.js","camera.css"):
  shutil.copyfile(ROOT/"web"/name,root/name)
 html=(ROOT/"web/index.html").read_text(encoding="utf-8")
 # Static viewer needs no fetch/file server and no algorithm imports.
 payload=json.dumps(data,ensure_ascii=False).replace("<","\\u003c")
 html=html.replace("<!-- RESULT_DATA -->","<script>window.SKIN_RESULT="+payload+";</script>")
 temp=root/"index.html.tmp"
 temp.write_text(html,encoding="utf-8")
 os.replace(temp,root/"index.html")
 return index
