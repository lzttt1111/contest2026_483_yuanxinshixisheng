#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
required=[
 "pyproject.toml","uv.lock","cloud_three/service_scores.py","cloud_three/live_scoring.py",
 "cloud_three/public_scores.py","cloud_three/tasks.py","cloud_three/api.py",
 "vendor/v3/run.py","vendor/v3_reference.json","vendor/water/shuiguang/config.py",
 "vendor/aisia-contracts/pyproject.toml","preview/index.html",
 "examples/current_demo/result.json","examples/current_demo/result_internal.json",
 "cloud_three/scoring_evidence.py","cloud_three/independent_spots.py","run_diagnostic_json.py",
 "vendor/v3/src/preprocess/face_parser.py","vendor/v3/models/face_parsing_resnet18.onnx",
 "vendor/independent_spots_reference.json","third_party/face-parsing-LICENSE.txt",
 "examples/diagnostic_one/result.json","examples/diagnostic_two/result.json",
 "BiSeNet_Celery诊断部署与字段说明.md","模型资产与用途说明.md",
]
missing=[x for x in required if not (ROOT/x).is_file()]
for module in ("pores","spots","surface_gloss"):
 for view in ("left","front","right"):
  p=ROOT/"examples/current_demo/detections"/module/(view+".png")
  if not p.is_file():missing.append(str(p.relative_to(ROOT)))
for view in ("left","front","right"):
 p=ROOT/"examples/current_demo/inputs"/(view+".png")
 if not p.is_file():missing.append(str(p.relative_to(ROOT)))
if missing:
 print("MISSING",*missing,sep="\n");raise SystemExit(1)
reference_sha=hashlib.sha256((ROOT/"vendor/v3_reference.json").read_bytes()).hexdigest()
expected="8e27a91f193d6bb21587eb65c5d89003328da727152bcfe5951e788582095bdd"
if reference_sha!=expected:
 raise SystemExit(f"评分参考SHA错误: {reference_sha}")
from cloud_three.public_scores import ScoreResponse
model_sha=hashlib.sha256((ROOT/'vendor/v3/models/face_parsing_resnet18.onnx').read_bytes()).hexdigest()
if model_sha!='0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f':raise SystemExit('BiSeNet模型SHA不匹配')
for name in ('one','two'):
 diagnostic=json.loads((ROOT/f'examples/diagnostic_{name}/result.json').read_text())
 if diagnostic.get('schema_version')!='shuiguang_diagnostic_v1' or diagnostic.get('reference_matches_new_preprocessing') is not False:
  raise SystemExit('诊断用途标识缺失')
 contract=ScoreResponse.model_validate(diagnostic['scores'])
 if [len(getattr(contract,key).regions) for key in ('pores','spots','surface_gloss')]!=[9,6,9]:raise SystemExit('当前诊断分区合同不一致')
payload=json.loads((ROOT/"examples/current_demo/result.json").read_text(encoding="utf-8"))
parsed=ScoreResponse.model_validate(payload)
expected_values={"pores":(57.0,"中度",9),"spots":(None,None,8),"surface_gloss":(72.0,"轻度",9)}
for key,(score,severity,count) in expected_values.items():
 item=getattr(parsed,key)
 if (item.score,item.severity,len(item.regions))!=(score,severity,count):
  raise SystemExit(f"示例评分不一致: {key}")
patterns=(re.compile(r"/home/[^/]+/(?:DermaVision|workspace)"),)
scan_ext={".py",".toml",".yaml",".yml",".md",".json",".js",".css",".html"}
bad=[]
for top in ("cloud_three","vendor/water","vendor/v3","vendor/aisia-contracts","preview"):
 for path in (ROOT/top).rglob("*"):
  if not path.is_file() or path.suffix.lower() not in scan_ext or "runtime" in path.parts:continue
  try:text=path.read_text(encoding="utf-8")
  except UnicodeDecodeError:continue
  if any(p.search(text) for p in patterns):bad.append(str(path.relative_to(ROOT)))
if bad:
 print("发现开发机绝对路径:",*bad,sep="\n");raise SystemExit(1)
manifest_path=ROOT/"PACKAGE_MANIFEST.json"
if manifest_path.is_file():
 manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
 for row in manifest.get("files",[]):
  path=ROOT/row["path"]
  if not path.is_file() or path.stat().st_size!=row["size"] or hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
   raise SystemExit("交付清单校验失败: "+row["path"])
print(json.dumps({"status":"ok","v3_reference_sha256":reference_sha,
 "bisenet_model_sha256":model_sha,"diagnostic_examples":2,"formal_reference_ready":False,
 "historical_example":{"pores":57,"spots":None,"surface_gloss":72},
 "historical_regions":{"pores":9,"spots":8,"surface_gloss":9},
 "diagnostic_regions":{"pores":9,"spots":6,"surface_gloss":9}},ensure_ascii=False,indent=2))
