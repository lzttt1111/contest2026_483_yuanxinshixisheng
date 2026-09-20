#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
manifest=json.loads((ROOT/"PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
if not (ROOT/"README.md").is_file():raise SystemExit("缺少参赛项目README.md")
for row in manifest["files"]:
 p=ROOT/row["path"]
 if not p.is_file() or p.stat().st_size!=row["size"] or sha(p)!=row["sha256"]:
  raise SystemExit("统一提交清单校验失败: "+row["path"])
for name in ("01_V2单双输入十二项稳定版","02_水光三项三视图服务","03_目标检测与自动拍照"):
 if not (ROOT/name).is_dir():raise SystemExit("缺少组件目录: "+name)
archives=[p.name for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in {".zip",".tar",".gz",".tgz",".7z",".rar"}]
if archives:raise SystemExit("根目录存在不允许的外层压缩包: "+",".join(archives))
print(json.dumps({"status":"ok","files":len(manifest["files"]),
 "bytes":sum(x["size"] for x in manifest["files"]),"outer_archives":[]},ensure_ascii=False,indent=2))
