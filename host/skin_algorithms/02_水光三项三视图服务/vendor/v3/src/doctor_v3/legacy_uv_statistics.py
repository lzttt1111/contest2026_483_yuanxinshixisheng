"""Expose exact existing V2 medical statistics, not a new pixel-mask approximation."""
from .historical_metrics import REGION_ALIASES,number
def calculate(medical):
 out={}
 rows=[medical["总体指标"],*medical["分区指标"]]
 for index,row in enumerate(rows):
  name="full_face" if index==0 else REGION_ALIASES.get(row["检测范围"],row["检测范围"])
  if row.get("评估状态")=="不可评估":continue
  values={}
  for old,new in (("有效皮肤面积（像素）","valid_skin_area_px"),
                  ("特征面积占比","feature_area_ratio"),("P90强度（0～1）","p90_intensity")):
   value=number(row.get(old))
   if value is not None:values[new]=value
  if values:out[name]=values
 return out
