TASK="spec"
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

    for key in ('available','requirements','candidates'):
        if type(d[key]) is not list:raise ValueError('list required: '+key)
    if any(type(v) is not str or not v.strip() for v in d['available']):raise ValueError('available must contain nonempty names')
    for r in d['requirements']:
        if type(r) is not dict or type(r.get('needs')) is not list or any(type(v) is not str or not v.strip() for v in r['needs']):raise ValueError('requirement object and needs list required')
    for row in d['requirements']+d['candidates']:
        if type(row) is not dict or type(row.get('id')) is not str or not row['id'].strip():raise ValueError('record with nonempty id required')
    available=set(d["available"]); requirements=d["requirements"]; candidates=d["candidates"]
    if not requirements or not candidates: raise ValueError("requirements/candidates must not be empty")
    for r in requirements:
        if not isinstance(r.get('unit'),str) or not r['unit'].strip():issue('missing_unit',r['id'])
        missing=set(r["needs"])-available
        if missing: issue("unsupported_measurement",{"id":r["id"],"missing":sorted(missing)})
        if r.get("calibration_required") and not r.get("calibrated"):issue("missing_calibration",r["id"])
    for c in candidates:
        if not c.get("license"):issue("unknown_license",c["id"])
    reuse=[c["id"] for c in candidates if c.get("baseline_verified") is True and c.get("license")]
    metrics.update(reuse_candidates=reuse,requirements_checked=len(requirements),recommendation="reuse_then_validate" if reuse and not findings else "collect_missing_evidence")

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
