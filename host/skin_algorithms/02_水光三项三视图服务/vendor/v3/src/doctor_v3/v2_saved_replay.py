"""Replay V2 scalar scoring on saved outputs; never call its report/model runner."""
import json,hashlib,csv,math
from pathlib import Path
from functools import lru_cache
from src.nine_analysis.metrics import extract_item

@lru_cache(maxsize=1)
def assets():
 from src.aisia_medical_report.word_scoring import load_word_population_profile,load_word_wrinkle_2d_references,load_word_acne_2d_reference
 code=Path(__file__).resolve().parents[2]
 path=code/"00_runtime_assets/scoring_v011/全量历史ECDF评分配置.json"
 profile=json.loads(path.read_text(encoding="utf-8"))
 return profile,hashlib.sha256(path.read_bytes()).hexdigest(),{
  p:load_word_population_profile(capture_profile=p) for p in ("consumer","institution")
 },load_word_wrinkle_2d_references(),load_word_acne_2d_reference()

def replay(root,complete,quality_proven):
 from src.scoring_calibration.v011.scoring import score_observation
 from src.aisia_medical_report.word_scoring import apply_word_scores
 quality_proven=quality_proven and complete.get("quality_control",{}).get("status")!="REJECT"
 root=Path(root);idx=json.loads((root/"十二项检测结果索引.json").read_text(encoding="utf-8"))
 items=idx.get("items",idx.get("十二项结果"));features={}
 for project in ("redness","spots","brown","texture","pores","uv_spots","porphyrin","wrinkle","acne"):
  raw=json.loads((root/items[project]["量化JSON"]).read_text(encoding="utf-8"))
  if "status" in raw and isinstance(raw.get("metrics"),dict):raw=raw["metrics"]
  if project in raw and isinstance(raw[project],dict):raw=raw[project]
  _,features[project]=extract_item(project,raw)
 uv_path=root/items["uv_spots"]["医学V2CSV"]
 with uv_path.open(encoding="utf-8-sig",newline="") as stream:
  row=next((r for r in csv.DictReader(stream) if r.get("检测范围")=="全面部" and "UV" in r.get("检测项目","")),None)
 if row:
  for group,field,column in (("UV样色素范围","density","核心-单位面积密度（个/10万有效皮肤像素）"),
      ("UV样色素范围","area_ratio","核心-特征面积占比"),
      ("UV样色素强度","p90_intensity","核心-P90强度（0～1）"),
      ("UV样色素强度","high_intensity_ratio","核心-高强度目标比例")):
   if row.get(column) not in (None,"","不可评估"):
    value=float(row[column])
    if math.isfinite(value):features["uv_spots"].setdefault(group,{})[field]=value
 profile,digest,populations,wrinkles,acne=assets()
 from .historical_metrics import normalize
 normalized=normalize({"projects":complete.get("detector_results",{}),"aggregate":{}})
 uv_range=features["uv_spots"].setdefault("UV样色素范围",{})
 if "density" not in uv_range:
  for field in ("单位面积密度（个/10万有效皮肤像素）","feature_density_per_100k_skin_px"):
   entry=normalized.get("uv_spots|full_face|"+field)
   if entry:
    uv_range["density"]=entry["value"];break
 if "density" not in uv_range:
  count=normalized.get("uv_spots|full_face|feature_count",{}).get("value")
  area=normalized.get("uv_spots|full_face|valid_skin_area_px",{}).get("value")
  if count is not None and area is not None and area>0:
   uv_range["density"]=count*100000/area
 observation={"status":"success","quality":complete.get("quality_control",{}),"features":features}
 if quality_proven:observation["input_quality_gate"]={"status":"PASS","reason_codes":[],"evidence":"current_detection_quality"}
 official=score_observation(observation,profile["references"],normalization_profile_sha256=digest,
                            scoring_profile_version=profile["scoring_profile_version"])
 rows={x["dimension_id"]:x for x in official.get("formal_dimension_scores",[])}
 ids={"01":"visible_pores","03":"combined_pigmentation","04":"diffuse_redness","10":"surface_smoothness_decline"}
 payload={"检测模块":[{"模块编号":f"{i:02d}","综合得分":None,"程度等级":None} for i in range(1,12)]}
 for m in payload["检测模块"]:
  row=rows.get(ids.get(m["模块编号"],""),{})
  if row.get("score") is not None:
   m["综合得分"]=row["score"]
   from src.aisia_medical_report.aggregator import _burden_grade
   m["程度等级"]=_burden_grade(row["score"])
 capture=complete.get("provenance",{}).get("capture_profile","consumer")
 if quality_proven:apply_word_scores(payload,complete,populations[capture],wrinkles,acne)
 return {"modules":{m["模块编号"]:{"score":m.get("综合得分"),"grade":m.get("程度等级"),
    "method":m.get("评分Profile")} for m in payload["检测模块"]},
    "official_v011":official,"official_dimensions":list(rows),"quality_proven":quality_proven,
    "official_profile_sha256":digest,"new_inference":False}
