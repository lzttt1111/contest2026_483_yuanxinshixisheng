"""Describe measured spatial extent independently of severity calibration."""
from .registry import REGION_LABELS
LAYERS={"spots":"色斑","brown":"棕色色素","uv":"UV色素"}
def ranked(model,layer,native=False):
 m=model["modules"]["03"];out=[]
 for region,metrics in m["metrics"].items():
  if region=="full_face":continue
  area=model.get("scoring_domain_valid_pixels",{}).get("03:"+region)
  if area is not None and area<2000:continue
  item=metrics.get(layer+".area",{})
  value=(item.get("native_v3_measurement",item) if native else item).get("value")
  if isinstance(value,(int,float)) and not isinstance(value,bool) and 0<=value<=1:out.append((region,value))
 return sorted(out,key=lambda x:-x[1])
def regions_text(model,layer,native=False):
 rows=ranked(model,layer,native);positive=[r for r,v in rows if v>0]
 if positive:return "、".join(REGION_LABELS.get(r,r) for r in positive[:2])
 return "已评估区域未检出" if rows else "暂无分布依据"
def describe(model,label):
 if "UV下额外" in label:return "暂无法判断UV独有分布（缺少跨图对应量化）"
 if "多图共同" in label:
  supported=any(im.get("role")=="spots_supported" for group in model.get("module_figures",{}).get("03",[]) for im in group)
  return regions_text(model,"spots",True) if supported else "暂无法判断共同分布（缺少跨图支持证据）"
 if "左右" in label:
  pairs=[]
  for layer in LAYERS:
   values=dict(ranked(model,layer))
   for left,a in values.items():
    if not left.startswith("left_"):continue
    right=left.replace("left_","right_",1)
    if right in values:pairs.append((abs(a-values[right]),layer,left,right,a,values[right]))
  if not pairs:return "暂无法比较（缺少左右对应量化）"
  _,layer,left,right,a,b=max(pairs)
  return f'{LAYERS[layer]}覆盖率：{REGION_LABELS[left]}{a*100:.2f}%，{REGION_LABELS[right]}{b*100:.2f}%'
 if "主要区域" in label or "主要受累区域" in label:
  for layer,prefix in (("brown","棕色"),("uv","UV"),("spots","色斑")):
   if label.startswith(prefix):return regions_text(model,layer)
  return "；".join(name+"："+regions_text(model,layer) for layer,name in LAYERS.items())
 return None
def regional_description(model,region):
 area=model.get("scoring_domain_valid_pixels",{}).get("03:"+region)
 if area is not None and area<2000:return "暂无量化结果（有效区域不足）"
 metrics=model["modules"]["03"]["metrics"].get(region,{})
 parts=[]
 for layer,name in LAYERS.items():
  value=metrics.get(layer+".area",{}).get("value")
  if isinstance(value,(int,float)) and 0<=value<=1:parts.append(f"{name}覆盖率{value*100:.2f}%")
 return "；".join(parts) if parts else "暂无量化结果（有效区域不足）"
