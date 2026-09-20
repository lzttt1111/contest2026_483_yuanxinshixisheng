TASK="capture"
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
    config=data['config']
    if type(config) is not dict or type(config.get('slots')) is not dict or not config['slots']:raise ValueError('config and nonempty slots objects required')
    if any(type(k) is not str or not k.strip() for k in config['slots']):raise ValueError('nonempty slot names required')
    if type(data['frames']) is not list or any(type(f) is not dict for f in data['frames']):raise ValueError('frame object list required')
    if type(config['yaw_sign']) is not int or config['yaw_sign'] not in (-1,1):raise ValueError('yaw_sign must be integer -1 or 1, not boolean')
    for key in ('yaw_tolerance','pitch_limit','roll_limit','stable_ms','max_gap_ms','max_age_ms'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]):raise ValueError('finite capture number required: '+key)
    for target in config['slots'].values():
        if type(target) not in (int,float) or not math.isfinite(target):raise ValueError('finite slot angle required')
    for frame in data['frames']:
        for key in ('t','age_ms','yaw','raw_yaw','pitch','roll'):
            if key in frame and (type(frame[key]) not in (int,float) or not math.isfinite(frame[key])):raise ValueError('finite frame number required: '+key)
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    c=d["config"]; slots=c["slots"]
    required=("yaw_tolerance","pitch_limit","roll_limit","stable_ms","min_frames","max_gap_ms","max_age_ms","yaw_sign")
    for k in required:
        if k not in c:raise ValueError("missing capture config: "+k)
    if not slots or type(c['min_frames']) is not int or c["yaw_sign"] not in (-1,1) or c["min_frames"]<1 or any(c[k]<0 for k in required if k!="yaw_sign"):raise ValueError("invalid capture configuration")
    for k in ("stable_ms","max_gap_ms","max_age_ms"):
        if c[k]<=0:raise ValueError("timing budgets must be positive")
    revision=d.get('capture_revision',0)
    if type(revision) is not int or revision<0:raise ValueError('capture_revision must be nonnegative integer')
    done={}; start=None; streak=0; pending=None; previous_t=None; face=None; traces=[]
    for i,f in enumerate(d["frames"]):
        t=f["t"]
        if previous_t is not None and t<=previous_t:raise ValueError("nonmonotonic replay timestamp")
        gap=previous_t is not None and t-previous_t>c["max_gap_ms"];previous_t=t
        if f.get('reset_all') or f.get('retake') is not None:revision+=1
        if f.get("reset_all"):done={};start=None;streak=0;pending=None;face=None
        if f.get("retake") is not None:
            slot=f["retake"]
            if slot not in slots:raise ValueError("unknown retake slot")
            done.pop(slot,None);start=None;streak=0;pending=None
        if not f.get("valid") or not f.get("face") or f["age_ms"]>c["max_age_ms"] or f["age_ms"]<0:
            issue("stale_frame" if f.get("valid") and f.get("face") else "invalid_face_or_pose",i)
            start=None;streak=0;pending=None;traces.append({"frame":i,"accepted":False});continue
        if face is not None and face!=f["face"]:
            issue("face_changed",i);start=None;streak=0;pending=None
            # Do not auto-switch identity mid inspection.
            traces.append({"frame":i,"accepted":False});continue
        face=f["face"]
        if gap:start=None;streak=0;pending=None
        yaw=f["yaw"]*c["yaw_sign"];raw=f["raw_yaw"]*c["yaw_sign"]
        options=[s for s,target in slots.items() if s not in done and abs(yaw-target)<=c["yaw_tolerance"]]
        if len(options)>1:issue("ambiguous_slot",i);options=[]
        slot=options[0] if options else None
        eligible=slot is not None and abs(raw-slots[slot])<=c["yaw_tolerance"] and abs(f["pitch"])<=c["pitch_limit"] and abs(f["roll"])<=c["roll_limit"]
        if slot is not None and abs(raw-slots[slot])>c["yaw_tolerance"]:issue("raw_pose_outside",i)
        if not eligible:start=None;streak=0;pending=None;traces.append({"frame":i,"accepted":False});continue
        if pending!=slot:start=t;streak=0;pending=slot
        streak+=1
        if t-start>=c["stable_ms"] and streak>=c["min_frames"]:
            done[slot]=i;start=None;streak=0;pending=None
        traces.append({"frame":i,"accepted":True})
    metrics.update(captured=done,complete=set(done)==set(slots),trace=traces,capture_revision=revision)

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
