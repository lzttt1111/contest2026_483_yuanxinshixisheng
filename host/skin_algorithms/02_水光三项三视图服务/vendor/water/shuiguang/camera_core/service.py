"""Persistent photos + short-lived pose state. No camera or skin model at import time."""
import hashlib, io, json, math, re, threading, time, uuid
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
from fastapi import HTTPException
from .state import CaptureGate,FaceLock,PoseFilter

VIEWS=("left","front","right")
DEFAULT={"yaw_target":45.,"yaw_tolerance":5.,"pitch_limit":12.,"roll_limit":12.,
         "stable_ms":500.,"max_gap_ms":350.,"max_frame_age_ms":700.,
         "min_face_px":80.,"blur_threshold":60.,"yaw_jitter":3.,"left_yaw_sign":-1}
EDITABLE={"yaw_target":(10,75),"yaw_tolerance":(1,15),"stable_ms":(300,3000)}
def checked_config(values):
    if not isinstance(values,dict) or set(values)-set(EDITABLE):raise HTTPException(400,"未知采集参数")
    c=dict(DEFAULT)
    for key,value in values.items():
        lo,hi=EDITABLE[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
            raise HTTPException(400,"采集参数超出范围")
        c[key]=float(value)
    if 2*c["yaw_tolerance"]>=c["yaw_target"]:raise HTTPException(400,"正面与侧面容差范围不能重叠")
    return c
def save_json(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(".tmp");temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class CaptureService:
    def __init__(self,root,busy,backend_factory=None):
        self.root_provider=root;self.skin_busy=busy;self.backend_factory=backend_factory
        self.backend=None;self.lock=threading.RLock();self.states={};self.active=None
    @property
    def root(self):return Path(self.root_provider())/"runtime/captures"
    def folder(self,sid):
        if not isinstance(sid,str) or not re.fullmatch("[a-f0-9]{32}",sid):raise HTTPException(404,"采集记录不存在")
        return self.root/sid
    def persist(self,s):save_json(self.folder(s["session_id"])/"session.json",s)
    def read(self,sid,owner):
        path=self.folder(sid)/"session.json"
        if not path.is_file():raise HTTPException(404,"采集记录不存在")
        s=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(owner,str) or hashlib.sha256(owner.encode()).hexdigest()!=s["owner_hash"]:
            raise HTTPException(403,"此采集属于另一个浏览器标签页")
        return s
    def view(self,s):
        return {k:v for k,v in s.items() if k!="owner_hash"}
    def available(self):
        if self.skin_busy():raise HTTPException(409,"皮肤检测正在运行，请完成后再采集")
    def claim(self,s):
        if self.active and self.active!=s["session_id"]:
            old=self.folder(self.active)/"session.json"
            if old.exists() and time.time()-json.loads(old.read_text(encoding="utf-8"))["updated_at"]<600:
                raise HTTPException(409,"另一标签页正在采集，请先在那里结束")
        self.active=s["session_id"]
    def init_state(self,s,reset_lock=True):
        old=self.states.get(s["session_id"])
        self.states[s["session_id"]]={"gate":CaptureGate(s["config"]),"filter":PoseFilter(),
             "face_lock":FaceLock() if reset_lock or not old else old["face_lock"],"last":-1,"clock":None}
        self.states[s["session_id"]]["gate"].done=set(s["photos"])
    def create(self,values):
        config=checked_config(values)
        with self.lock:
            self.available()
            s={"session_id":uuid.uuid4().hex,"owner_hash":"","revision":0,"config":config,
               "photos":{},"jobs":{},"status":"paused","stream_id":None,"updated_at":time.time(),"lost":False}
            self.claim(s)
            owner=uuid.uuid4().hex;s["owner_hash"]=hashlib.sha256(owner.encode()).hexdigest()
            self.persist(s);self.init_state(s)
            return {**self.view(s),"owner_token":owner}
    def get(self,sid,owner):
        with self.lock:return self.view(self.read(sid,owner))
    def action(self,sid,owner,data):
        with self.lock:
            self.available();s=self.read(sid,owner)
            if type(data.get("revision")) is not int or data["revision"]!=s["revision"]:raise HTTPException(409,"采集版本已变化，请刷新")
            action=data.get("action")
            if action not in ("resume","pause","retake","reset","close"):raise HTTPException(400,"未知采集操作")
            if action!="close":self.claim(s)
            if action in ("resume","retake") and s["lost"]:raise HTTPException(409,"已失锁，请开始新一轮")
            if action=="retake":
                view=data.get("view")
                if view not in VIEWS:raise HTTPException(400,"重拍槽位无效")
                s["photos"].pop(view,None);s["revision"]+=1
            if action=="reset":
                s["photos"]={};s["revision"]+=1;s["lost"]=False
            if action in ("resume","retake"):
                s["stream_id"]=uuid.uuid4().hex;s["status"]="review" if len(s["photos"])==3 else "capturing"
                self.init_state(s,reset_lock=True)
            else:
                s["status"]="closed" if action=="close" else "paused";s["stream_id"]=None;self.init_state(s)
            if action=="close" and self.active==sid:self.active=None
            s["updated_at"]=time.time();self.persist(s)
            return self.view(s)
    def frame(self,sid,owner,headers,payload):
        if not self.lock.acquire(blocking=False):raise HTTPException(409,"角度计算忙，本帧丢弃")
        try:
            self.available();s=self.read(sid,owner);self.claim(s)
            if s["status"]!="capturing" or headers.get("x-stream-id")!=s["stream_id"]:
                raise HTTPException(409,"视频流已暂停或过期，请恢复采集")
            if sid not in self.states:raise HTTPException(409,"服务已重启，请恢复摄像头")
            state=self.states[sid]
            try:
                fid=int(headers["x-frame-id"]);at=float(headers["x-capture-ms"]);initial=float(headers["x-frame-age-ms"])
                if fid<=state["last"] or not math.isfinite(at) or at<0 or not math.isfinite(initial) or not 0<=initial<=60000:raise ValueError()
                if not payload.startswith(b"\xff\xd8") or len(payload)>5*1024**2:raise ValueError()
                with Image.open(io.BytesIO(payload)) as im:
                    if im.width*im.height>16000000:raise ValueError()
                    im.verify()
                with Image.open(io.BytesIO(payload)) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            except (ValueError,KeyError,OSError):raise HTTPException(400,"无效JPEG帧、尺寸或时间信息")
            state["last"]=fid;received=time.monotonic()*1000
            offset=received-at-initial
            state["clock"]=offset if state["clock"] is None else min(state["clock"],offset)
            if self.backend is None:
                try:
                    if self.backend_factory:self.backend=self.backend_factory()
                    else:
                        from .backend import PoseBackend
                        self.backend=PoseBackend()
                except Exception as exc:raise HTTPException(503,"角度组件不可用，请使用手动导入；"+type(exc).__name__) from exc
            frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            try:
                face,raw,sharpness,size,reason=self.backend.observe(frame,state["face_lock"],at)
            except Exception as exc:
                state["filter"].clear();state["gate"].clear();s["status"]="paused"
                self.persist(s)
                raise HTTPException(503,"角度推理未完成，照片已保留，可恢复或手动导入") from exc
            now=time.monotonic()*1000
            age=max(initial,received-at-state["clock"])+(now-received)
            pose=state["filter"].step(raw,at) if age<=s["config"]["max_frame_age_ms"] else None
            if pose is None:state["filter"].clear()
            decision=state["gate"].step(pose,at,sharpness,size,age,raw_pose=raw)
            if reason:decision["reason"]=reason
            saved=None
            if state["face_lock"].lost:s["lost"]=True;s["status"]="paused";decision["reason"]="主脸失锁，请开始新一轮"
            if decision["trigger"]:
                view=decision["trigger"];name=f'{view}_r{s["revision"]}_{fid}_{uuid.uuid4().hex[:8]}.jpg'
                path=self.folder(sid)/name;temp=path.with_suffix(".tmp");temp.write_bytes(payload);temp.replace(path)
                saved={"view":view,"filename":name,"frame_id":fid,"stream_id":s["stream_id"],
                       "capture_method":"web_auto_pose","capture_session_id":sid,"capture_revision":s["revision"],
                       "capture_monotonic_ms":at,"saved_at_unix_ms":int(time.time()*1000),
                       "raw_pose":raw,"pose":pose,"stable":True,"sharpness":sharpness,"frame_age_ms":age,
                       "image_size":[int(rgb.shape[1]),int(rgb.shape[0])],"mirrored":False,
                       "coordinate_space":"unmirrored_original","sha256":hashlib.sha256(payload).hexdigest(),
                       "angle_model_sha256":dict(getattr(self.backend,"hashes",{})),
                       "angle_runtime_provider":"CPUExecutionProvider",
                       "config":s["config"],"image_url":f"/api/capture-sessions/{sid}/photos/{name}"}
                save_json(path.with_suffix(".json"),saved)
                s["photos"][view]=saved;state["gate"].commit(view)
                if len(s["photos"])==3:s["status"]="review"
            s["updated_at"]=time.time();self.persist(s)
            return {**self.view(s),"frame_id":fid,"face":face,"raw_pose":raw,"pose":pose,"decision":decision,
                    "saved":saved,"image_width":rgb.shape[1],"image_height":rgb.shape[0],
                    "runtime_provider":"YuNet=OpenCV_CPU;FSA-Net=CPUExecutionProvider",
                    "timings_ms":{"total":time.monotonic()*1000-received},"age_ms":age}
        finally:self.lock.release()
    def submit(self,data,enqueue,job_status):
        with self.lock:
            s=self.read(data.get("capture_session_id"),data.get("capture_owner"))
            revision=data.get("capture_revision")
            if type(revision) is not int or revision!=s["revision"]:raise HTTPException(409,"照片版本已变化，请重新确认")
            if data.get("confirmed") is not True:raise HTTPException(400,"请确认三张照片")
            previous=s["jobs"].get(str(revision))
            if previous:
                failed=job_status(previous)=="failed"
                if not (failed and data.get("retry") is True):return previous
            self.available()
            if set(s["photos"])!=set(VIEWS) or s["lost"]:raise HTTPException(400,"需要同一轮完整三视图")
            hashes=[]
            for photo in s["photos"].values():
                path=self.folder(s["session_id"])/photo["filename"]
                if not path.is_file() or digest(path)!=photo["sha256"]:raise HTTPException(400,"照片校验失败，请重拍")
                hashes.append(photo["sha256"])
            if len(set(hashes))!=3:raise HTTPException(400,"三张照片不能重复")
            jid=enqueue(s,self.folder(s["session_id"]))
            s["jobs"][str(revision)]=jid;s["status"]="submitted";s["stream_id"]=None
            s["updated_at"]=time.time();self.persist(s)
            if self.active==s["session_id"]:self.active=None
            return jid
