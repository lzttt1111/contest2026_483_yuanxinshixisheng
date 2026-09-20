TASK="publication"
"""Offline evidence checker. No model, network, training, deployment or input writes."""
import argparse,hashlib,json,math,sys
from pathlib import Path

def check(data,root):
    if data.get("task_kind")!=TASK:
        return {"status":"not_applicable","findings":[],"metrics":{}}
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError("non-finite number")
        if isinstance(value,dict):
            flags={'baseline_verified','calibration_required','calibrated','human_confirmed','valid','reset_all','cross_view_unique_claim','correspondence_verified','inference_requested','committed','artifacts_valid','point_overlay','registration_verified','clock_mapping_verified','mode_switch_clears_photos','document_claims_preserved'}
            for key,v in value.items():
                if key in flags and type(v) is not bool:raise ValueError('Boolean field required: '+key)
                finite(v)
        if isinstance(value,list):
            for v in value:finite(v)
    finite(data)
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    def contained(name):
        path=Path(name)
        if path.is_absolute():issue("path_escape",name);return None
        full=(root/path).resolve()
        if not full.is_relative_to(root):issue("path_escape",name);return None
        if not full.is_file():issue("missing_artifact",name);return None
        return full
    s=contained(d["source_file"]);r=contained(d["report_file"])
    if s and r:
        source=json.loads(s.read_text(encoding="utf-8"));report=json.loads(r.read_text(encoding="utf-8"))
        if json.dumps(source,sort_keys=True,allow_nan=False)!=json.dumps(report,sort_keys=True,allow_nan=False):issue("projection_mismatch",{"source":d["source_file"],"report":d["report_file"]})
    hashes={}
    for name in d["artifact_files"]:
        p=contained(name)
        if p:hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    if d.get("inference_requested"):issue("unexpected_inference","Publication-only work must consume existing results")
    bindings=d.get('artifact_sha256')
    if bindings is not None:
        if type(bindings) is not dict or not bindings:raise ValueError('nonempty artifact_sha256 mapping required')
        for name,expected in bindings.items():
            if type(expected) is not str or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('lowercase SHA256 required')
            if name not in hashes or hashes[name]!=expected:issue('artifact_hash_mismatch',name)
    metrics.update(artifact_hashes=hashes,files_checked=len(hashes),binding_verified=bool(bindings) and set(bindings)==set(hashes) and not findings)
    if bindings is None:metrics['binding_status']='not_checked: hashes computed only; projection claims are not asset bindings'

    return {"status":"fail" if findings else "pass","findings":findings,"metrics":metrics}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--root",type=Path,help="Explicit allowed artifact root; default is input parent")
    args=parser.parse_args()
    try:
        if args.input.stat().st_size>32*1024*1024:raise ValueError("Input too large; provide bounded snapshot")
        data=json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data,dict):raise ValueError("Input must be object")
        result=check(data,(args.root or args.input.parent).resolve())
        json.dumps(result,ensure_ascii=False,allow_nan=False)
    except (ValueError,TypeError,KeyError,IndexError,OSError,AttributeError,OverflowError) as exc:
        result={"status":"invalid_input","findings":[{"code":"invalid_input","detail":str(exc)}],"metrics":{}}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 2 if result["status"]=="invalid_input" else (1 if result["status"]=="fail" else 0)
if __name__=="__main__":sys.exit(main())
