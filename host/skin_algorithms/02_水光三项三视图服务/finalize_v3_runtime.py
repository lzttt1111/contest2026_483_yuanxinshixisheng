"""Create aggregate V3 score nodes from one completed stage-one scoring payload."""
import argparse
import json
import sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--code",type=Path,required=True)
p.add_argument("--payload",type=Path,required=True)
p.add_argument("--front",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
sys.path.insert(0,str(args.code.resolve()))
from src.doctor_v3.stage1_word import attach_stage1
from src.doctor_v3.severity_guard import POLICY
payload=json.loads(args.payload.read_text(encoding="utf-8"))
front_sha=__import__("hashlib").sha256(args.front.read_bytes()).hexdigest()
values=payload.get("input_sha256",{})
if front_sha not in (values.values() if isinstance(values,dict) else [values]):
    raise ValueError("V3 payload does not match front input")
model={"subject_id":payload["subject_id"],"capture_profile":"consumer",
 "modules":{f"{mid:02d}":{"metrics":{},"regions":{}} for mid in range(1,12)},
 "scoring_system_version":"V3.0.1","severity_policy":POLICY,
 "legacy_input_proof":{"input_sha256":[front_sha],"proof_kind":"current_input_and_quality"}}
result=attach_stage1(model,payload)
result["scoring_source"]="fresh_single_front_stage1_native"
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
