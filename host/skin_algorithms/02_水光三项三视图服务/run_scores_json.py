import argparse,json
from pathlib import Path
from cloud_three.service_scores import execute_scores
from cloud_three.service import atomic_json
p=argparse.ArgumentParser()
p.add_argument("--request",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
args=p.parse_args();result=execute_scores(json.loads(args.request.read_text(encoding="utf-8")))
args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,result)
print(result.get("status","success"));raise SystemExit(1 if result.get("status")=="failed" else 0)
