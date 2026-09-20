TASK="generalization"
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

    rows=d["records"]
    if not rows:raise ValueError("empty records")
    by_id={}; groups={k:{} for k in ("subject","session","sha256","cluster")}
    for r in rows:
        if r["id"] in by_id:issue("duplicate_id",r["id"])
        by_id[r["id"]]=r
        if r["split"] not in ("train","validation","test","reference"):raise ValueError("invalid split")
        for k,g in groups.items():
            if not r.get(k):issue("missing_group_identity",{"id":r["id"],"field":k});continue
            g.setdefault(r[k],set()).add(r["split"])
    for field,g in groups.items():
        for key,splits in g.items():
            if len(splits-{"reference"})>1:issue(field+"_leakage",{"group":key,"splits":sorted(splits)})
    if d["selection_split"]=="test":issue("test_selection","Checkpoint selection uses held-out test")
    elif d["selection_split"]!="validation":issue("unknown_selection_protocol",d["selection_split"])
    for name,code in (("calibration_ids","calibration_leakage"),("synthetic_sources","synthetic_leakage")):
        for ident in d.get(name,[]):
            if ident not in by_id or by_id[ident]["split"]!="train":issue(code,ident)
    metrics.update(records=len(rows),splits={s:sum(r["split"]==s for r in rows) for s in ("train","validation","test","reference")})

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
