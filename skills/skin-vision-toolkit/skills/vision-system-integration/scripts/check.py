TASK="integration"
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
    for key in ('current_revision','result_revision'):
        if type(data[key]) is not int or data[key]<0:raise ValueError('revision must be nonnegative integer')
    for key in ('source_frame','target_frame'):
        if isinstance(data[key],bool):raise ValueError('boolean frame identity not allowed')
    for key in ('source_frame','target_frame'):
        v=data[key]
        if not ((type(v) is str and v.strip()) or (type(v) is int and v>=0)):raise ValueError('nonempty frame identity required')
    for key in ('source_clock','target_clock'):
        if type(data[key]) is not str or not data[key].strip():raise ValueError('nonempty clock identity required')
    for key in ('producer_fields','consumer_required'):
        if type(data[key]) is not list or not data[key] or any(type(v) is not str or not v.strip() for v in data[key]):raise ValueError('nonempty field list required')
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    missing=set(d["consumer_required"])-set(d["producer_fields"])
    if missing:issue("missing_contract_fields",sorted(missing))
    if d["current_revision"]!=d["result_revision"]:issue("stale_revision",{"current":d["current_revision"],"result":d["result_revision"]})
    if d["point_overlay"] and d["source_frame"]!=d["target_frame"] and not d["registration_verified"]:issue("unregistered_point_transfer","Business linkage is not pixel registration")
    if d["source_clock"]!=d["target_clock"] and not d["clock_mapping_verified"]:issue("clock_domain_mismatch",{"source":d["source_clock"],"target":d["target_clock"]})
    if d["mode_switch_clears_photos"] and d["document_claims_preserved"]:issue("state_document_conflict","Implementation clears photos")
    metrics.update(contracts_checked=len(d["consumer_required"]),point_transfer_allowed=not findings,requires_live_verification=True)

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
