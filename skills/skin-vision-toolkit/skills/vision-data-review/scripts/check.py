TASK="data"
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
    for row in data['records']:
        for key in ('revision','proposed_revision'):
            if key in row and (type(row[key]) is not int or row[key]<0):raise ValueError('revision must be nonnegative integer')
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    rows=d["records"]; seen=set(); hashes={}; protected=[]; rerun=set(d.get("rerun_ids",[])); eligible=set()
    if not rows:raise ValueError("empty records")
    for r in rows:
        ident=r["id"]
        if type(ident) is not str or not ident.strip():raise ValueError('nonempty record identity required')
        for key in ('mask_edited','human_edited','protected'):
            if key in r and type(r[key]) is not bool:raise ValueError('editing flags must be boolean')
        locked=any(r.get(key) is True for key in ('human_confirmed','mask_edited','human_edited','protected'))
        if ident in seen:issue("duplicate_id",ident)
        seen.add(ident)
        if not r.get("sha256") or not r.get("subject") or not r.get("session"):issue("missing_provenance",ident)
        if r.get("sha256") in hashes:issue("duplicate_image",ident)
        hashes[r.get("sha256")]=ident
        box=r.get("bbox")
        if box is not None:
            if len(box)!=4 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in box):raise ValueError("bbox requires four numbers")
            x,y,w,h=box
            if min(x,y)<0 or w<=0 or h<=0 or x+w>1 or y+h>1:issue("bbox_outside",ident)
        if r["label"]=="negative" and box is not None:issue("negative_has_box",ident)
        elif r["label"]=="positive" and box is None:issue("positive_missing_box",ident)
        elif r["label"] not in ("positive","negative","unknown"):raise ValueError("invalid label")
        if type(r.get('human_confirmed')) is not bool:
            issue('unknown_review_state',ident)
        elif r['human_confirmed'] is False and not locked:
            eligible.add(ident)
        if locked:
            protected.append(ident)
            if ident in rerun or r.get("proposed_revision",r.get("revision",0))<r.get("revision",0):issue("human_edit_overwrite",ident)
    for ident in sorted(rerun-seen):issue('unknown_rerun_id',ident)
    metrics.update(records=len(rows),protected_ids=sorted(set(protected)),safe_rerun_ids=sorted(rerun&eligible) if not findings else [])

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
