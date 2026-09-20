#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"vendor/aisia-contracts"))
sys.path.insert(0,str(ROOT))
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
required=[
 "README.md","V2单双输入十二项技术与部署说明_20260915.md","pyproject.toml","uv.lock",
 "run.py","run_cloud_worker.py","deploy/dermavision@.service","deploy/dermavision-workers.target",
 "src/worker.py","src/summary/worker.py","src/summary_scoring/entry.py",
 "calibration/scoring_input_field_list_v3.json","calibration/v2_explicit_formula_registry_v1.json",
 "00_runtime_assets/scoring_v011/SCORING_ASSET_MANIFEST.json",
 "00_runtime_assets/scoring_word_provisional/word_population_reference_1000.json",
 "00_runtime_assets/vendor_skin/VENDOR_RUNTIME_MANIFEST.json",
 "models/acne/grading/acne_lds_drive/lds-weights/model_fold_0.pth",
 "models/wrinkle/best.pt","templates/AISIA_面部多指标检测汇总模板_V2.docx",
 "vendor/aisia-contracts/pyproject.toml","SOURCE_SNAPSHOT.json","QA_RECEIPT.json"
]
missing=[x for x in required if not (ROOT/x).is_file()]
if missing:raise SystemExit("缺少关键文件:\n"+"\n".join(missing))
py=(ROOT/"pyproject.toml").read_text(encoding="utf-8")
lock=(ROOT/"uv.lock").read_text(encoding="utf-8")
if 'aisia-contracts = { path = "vendor/aisia-contracts" }' not in py or 'source = { directory = "vendor/aisia-contracts" }' not in lock:
 raise SystemExit("合同依赖未固定到包内")
import aisia_contracts
if not Path(aisia_contracts.__file__).resolve().is_relative_to((ROOT/"vendor/aisia-contracts").resolve()):
 raise SystemExit("aisia_contracts未从包内加载")
vendor=ROOT/"00_runtime_assets/vendor_skin/pyz/core/skin_generate.pyc"
if sha(vendor)!="fd590298569d80ef6b3dfd3d7043c653715b12fddb5f41dbf1fd0ff8417a1581":
 raise SystemExit("厂家sidecar主文件SHA不一致")
manifest=ROOT/"PACKAGE_MANIFEST.json"
if manifest.is_file():
 for row in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
  p=ROOT/row["path"]
  if not p.is_file() or p.stat().st_size!=row["size"] or sha(p)!=row["sha256"]:
   raise SystemExit("V2交付清单校验失败: "+row["path"])
forbidden=("lz"+"ttt8330","10."+"3.2.158","codeup."+"aliyun.com")
bad=[]
exts={".py",".md",".json",".toml",".yaml",".yml",".sh",".ps1",".cmd",".service",".target",".txt"}
for p in ROOT.rglob("*"):
 if not p.is_file() or p.suffix.lower() not in exts or p==Path(__file__).resolve() or p.name=="PACKAGE_MANIFEST.json":continue
 try:t=p.read_text(encoding="utf-8")
 except UnicodeDecodeError:continue
 if any(x in t for x in forbidden):bad.append(str(p.relative_to(ROOT)))
if bad:raise SystemExit("发现内部环境信息:\n"+"\n".join(bad))
print(json.dumps({"status":"ok","release":"v2-submission-20260915",
 "contract_source":str(Path(aisia_contracts.__file__).relative_to(ROOT)),
 "vendor_skin_sha256":sha(vendor),"dependency_source":"vendor/aisia-contracts"},ensure_ascii=False,indent=2))
