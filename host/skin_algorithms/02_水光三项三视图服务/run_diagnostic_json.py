"""Explicit diagnostic-only CLI; defaults to the selected BiSeNet pipeline."""
import argparse,json
from pathlib import Path
from cloud_three.service_scores import execute_scores
from cloud_three.service import atomic_json
parser=argparse.ArgumentParser(description='旧参考诊断JSON，不是正式新评分')
parser.add_argument('--request',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
result=execute_scores(json.loads(args.request.read_text(encoding='utf-8')),diagnostic=True)
args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,result)
print(result.get('purpose',result.get('status')))
raise SystemExit(1 if result.get('status')=='failed' else 0)
