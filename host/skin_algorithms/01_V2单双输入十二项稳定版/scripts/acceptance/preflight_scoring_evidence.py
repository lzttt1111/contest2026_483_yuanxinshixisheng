from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.scoring_input.builder import build_scoring_input
from src.summary_scoring import score_report_from_evidence
from src.summary.request import merge_evidence,validate_request,recompute_fingerprint
from src.summary.release import RELEASE_ID
from src.summary.worker import run_aggregate_summary
from src.scoring_input.gate import compute_input_quality_gate
from aisia_contracts.overall_summary.v1 import SummaryRequestV1
from hashlib import sha256
ALGOS=("redness","spots","brown","texture","pores","purple","surface_gloss","vascular","contour_firmness","acne","wrinkle")
def main():
    p=argparse.ArgumentParser();p.add_argument("--local-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rows=[]
    for alias in ("clinic28-25","clinic28-09","clinic28-23"):
        root=a.local_root/alias
        complete=json.loads((root/"十二项完整量化指标.json").read_text())
        data=(root/"00_输入图像"/"RGB_M.jpg").read_bytes()
        gate=compute_input_quality_gate(data)
        evidence=complete["detector_results"]
        inputs={name:build_scoring_input(algorithm_name=name,detection_schema_version="1",detection_impl_version="1",
            report_id=alias,detection_attempt_id=alias+"-"+name,source_image_sha256=sha256(data).hexdigest(),
            capture_profile="institution",input_route="institution_queue",runtime={"detector_results":evidence},
            input_quality_gate=gate) for name in ALGOS}
        body={"schema_version":"overall_summary_request_v1","report_id":alias,"attempt_id":"preflight",
            "capture_profile":"institution","input_route":"institution_queue","expected_algorithms":list(ALGOS),
            "algorithm_outcomes":[{"algorithm_name":k,"status":"success"} for k in ALGOS],"scoring_inputs":inputs,
            "scoring_release":RELEASE_ID,"input_fingerprint":"0"*64}
        body["input_fingerprint"]=recompute_fingerprint(SummaryRequestV1.model_validate(body))
        result=run_aggregate_summary(body)
        direct=score_report_from_evidence(evidence,capture_profile="institution",input_route="institution_queue",quality=None,input_quality_gate=gate)
        differences=[]
        if result["status"]=="success":
            for module in result["raw_result"]["modules"]:
                key=module["module_no"]
                for system in ("word_display","production_proxy_v1"):
                    for field in ("score","grade","score_valid"):
                        left=direct["modules"][key][system][field];right=module[system][field]
                        if left!=right:differences.append({"module":key,"system":system,"field":field,"local":left,"summary":right})
        else:differences=[result]
        rows.append({"case":alias,"input_gate":gate,"input_states":{k:v["evidence_status"] for k,v in inputs.items()},
                     "missing":{k:v["missing_fields"] for k,v in inputs.items() if v["missing_fields"]},"differences":differences})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps([{"case":r["case"],"states":r["input_states"],"differences":r["differences"]} for r in rows],ensure_ascii=False,indent=2))
    assert not any(r["differences"] for r in rows),"scoring projection mismatch"
if __name__=="__main__":main()
