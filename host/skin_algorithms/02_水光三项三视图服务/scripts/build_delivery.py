#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
FILES=["README.md","pyproject.toml","uv.lock","request_scores_example.json",
       "run_scores_json.py","finalize_v3_runtime.py","preview_server.py","vendor/v3_reference.json","QA_RECEIPT.json","水光三项检测技术与部署说明_20260915.md","接口字段与JSON说明_20260915.md"]
DIRS=["cloud_three","preview","vendor/water","vendor/v3","vendor/aisia-contracts","examples","docs","scripts","delivery_tests","deploy"]
EXCLUDED={".git",".venv","__pycache__",".pytest_cache",".mypy_cache","runtime","outputs","07_运行依赖"}
def ignored(directory,names):
 return [n for n in names if n in EXCLUDED or n.endswith(".pyc") or n.endswith(".pyo")]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--destination",type=Path,required=True);args=p.parse_args()
 target=args.destination.resolve()
 if target.exists():raise SystemExit(f"目标已存在，拒绝覆盖: {target}")
 target.mkdir(parents=True)
 for item in FILES:shutil.copy2(SOURCE/item,target/item)
 for item in DIRS:shutil.copytree(SOURCE/item,target/item,ignore=ignored)
 manifest={"package":"水光三项检测服务","version":"0.1.0","created":"2026-09-15","files":[]}
 for path in sorted(target.rglob("*")):
  if path.is_file() and path.name!="PACKAGE_MANIFEST.json":
   manifest["files"].append({"path":path.relative_to(target).as_posix(),"size":path.stat().st_size,"sha256":sha(path)})
 (target/"PACKAGE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 archive=target.parent/(target.name+"_20260915.zip")
 with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for path in sorted(target.rglob("*")):
   if path.is_file():z.write(path,(target.name+"/"+path.relative_to(target).as_posix()))
 print(json.dumps({"directory":str(target),"archive":str(archive),
  "files":len(manifest["files"]),"bytes":sum(x["size"] for x in manifest["files"]),
  "archive_sha256":sha(archive)},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
