"""Single-machine camera lab. Same-origin UI; one in-flight inference globally."""
import asyncio
import hashlib
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lab.backend import Backend
from lab.state import CaptureGate, FaceLock, PoseFilter

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))


def create_app(backend=None, capture_root=None):
    captures = Path(capture_root or ROOT/'captures')
    sessions = {}
    guard = asyncio.Lock()
    diagnostics = deque(maxlen=300)

    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend or await asyncio.to_thread(Backend)
        try:
            yield
        finally:
            if backend is None:
                app.state.backend.regions.close()

    app = FastAPI(lifespan=lifespan)

    def get_session(sid):
        if sid not in sessions:
            raise HTTPException(409, '会话已失效，请重新开启摄像头')
        return sessions[sid]

    @app.get('/health')
    def health():
        return {**app.state.backend.health(), 'config': CONFIG}

    @app.get('/v1/debug/recent')
    def recent():
        # Memory-only numerical diagnostics; no image recording or uploads.
        return {'frames': list(diagnostics)}

    @app.post('/v1/session')
    async def new_session():
        if guard.locked():
            raise HTTPException(409, '推理忙，请稍后重试')
        async with guard:
            # A local single-operator lab: starting a new browser invalidates old sessions.
            sessions.clear()
            diagnostics.clear()
            sid = uuid.uuid4().hex
            sessions[sid] = {'gate': CaptureGate(dict(CONFIG)), 'filter':PoseFilter(), 'lock': FaceLock(), 'last': -1,
                             'mode': None, 'photos': {}, 'clock': None}
            return {'session_id': sid}

    @app.post('/v1/session/{sid}/settings')
    async def settings(sid: str, request: Request):
        limits={'yaw_target':(5,80),'yaw_tolerance':(1,20),'pitch_limit':(5,45),'roll_limit':(5,45),'stable_ms':(300,3000)}
        try:
            values=await request.json()
            if not isinstance(values,dict) or set(values)!=set(limits):raise ValueError()
            for k,(low,high) in limits.items():
                v=values[k]
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not low<=v<=high:raise ValueError()
            if values['yaw_tolerance']>=values['yaw_target'] or values['yaw_target']+values['yaw_tolerance']>89:raise ValueError()
        except (ValueError,TypeError):
            raise HTTPException(400,'参数无效：目标5–80°，容差1–20°且小于目标；目标+容差≤89°；倾斜上限5–45°，稳定300–3000ms')
        async with guard:
            s=get_session(sid)
            s['gate']=CaptureGate({**s['gate'].config,**values})
            s['filter'].clear();s['photos'].clear()
            return {'config':s['gate'].config,'message':'参数已应用，开始新一轮；历史照片保留在磁盘'}

    @app.post('/v1/session/{sid}/reset')
    async def reset(sid: str, side: str = 'all'):
        if side not in ('all', 'front', 'left', 'right'):
            raise HTTPException(400, 'invalid side')
        if guard.locked():
            raise HTTPException(409, '推理忙，请稍后重试')
        async with guard:
            s = get_session(sid)
            s['gate'].clear()
            s['filter'].clear()
            if side == 'all':
                s['gate'].done.clear()
                s['photos'].clear()
                s['lock'] = FaceLock()
            else:
                s['gate'].done.discard(side)
                s['photos'].pop(side, None)
            return {'ok': True}

    @app.get('/v1/photos/{sid}/{filename}')
    def photo(sid: str, filename: str):
        if len(sid) != 32 or any(c not in '0123456789abcdef' for c in sid):
            raise HTTPException(404)
        if Path(filename).name != filename or not filename.endswith(('.jpg', '.json')):
            raise HTTPException(404)
        path = captures/sid/filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    def infer(s, sid, mode, payload, fid, capture_ms, initial_age, received):
        b = app.state.backend
        if mode == 'device':
            result = b.device.infer_bytes(payload, fid)
            with Image.open(io.BytesIO(payload)) as raw:
                rgb=np.asarray(ImageOps.exif_transpose(raw).convert('RGB'))
            regions=b.regions.infer(rgb,result.get('contact_point'))
            result['timings_ms']['total_with_regions']=round(time.monotonic()*1000-received,1)
            return {**result, 'mode': mode, 'photos': s['photos'], 'face_regions':regions,
                    'contact_region':regions['contact_region'],
                    'model_lineage':{'device':'V5 REAL-ONLY YOLOX-S','contact':'V5 MobileNetV3-Small'}}
        with Image.open(io.BytesIO(payload)) as raw:
            if raw.width*raw.height > 16000000:
                raise ValueError('图像分辨率过大')
            image = ImageOps.exif_transpose(raw).convert('RGB')
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        face, reason = s['lock'].select(b.face.detect_all(frame), b.face, w, h)
        pose = None
        sharpness = size = 0
        if face:
            x1,y1,x2,y2 = b.face.crop_box(face,w,h)
            crop = frame[y1:y2,x1:x2]
            pose = b.pose.infer_bgr(crop).to_dict()
            sharpness = float(cv2.Laplacian(cv2.cvtColor(cv2.resize(crop,(160,160)),cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            size = min(face.width,face.height)
        now = time.monotonic()*1000
        # Browser clock offset is calibrated on first arrival. Rising transport delay
        # becomes positive age; wall clocks on Windows and WSL need not be synchronized.
        age = max(initial_age, received-capture_ms-s['clock'])+(now-received)
        raw_pose=pose
        if age>s['gate'].config['max_frame_age_ms']:
            s['filter'].clear()
        pose=s['filter'].step(raw_pose,capture_ms) if age<=s['gate'].config['max_frame_age_ms'] else None
        decision = s['gate'].step(pose, capture_ms, sharpness, size, age,raw_pose=raw_pose)
        if reason:
            decision['reason'] = reason
        diagnostics.append({'frame_id':fid,'pose':pose,'raw_pose':raw_pose,'sharpness':round(sharpness,1),
                            'face_size':round(size,1),'age_ms':round(age,1),
                            'decision':dict(decision)})
        saved = None
        if decision['trigger']:
            side = decision['trigger']
            folder = captures/sid
            folder.mkdir(parents=True,exist_ok=True)
            name = f'{side}_{fid}_{uuid.uuid4().hex[:8]}'
            saved = {'side': side,'frame_id':fid,'capture_monotonic_ms':capture_ms,
                     'saved_at_unix_ms':int(time.time()*1000),'pose':pose,'raw_pose':raw_pose,'sharpness':sharpness,
                     'frame_sha256':hashlib.sha256(payload).hexdigest(),
                     'coordinate_space':'unmirrored_original','config':dict(s['gate'].config),
                     'image_url':f'/v1/photos/{sid}/{name}.jpg',
                     'metadata_url':f'/v1/photos/{sid}/{name}.json'}
            # Store the exact incoming JPEG, not a later webcam frame.
            (folder/f'{name}.jpg').write_bytes(payload)
            (folder/f'{name}.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
            s['gate'].commit(side)
            s['photos'][side] = saved
        return {'mode':mode,'frame_id':fid,'image_width':w,'image_height':h,
                'face': face.to_dict() if face else None,'pose':pose,'raw_pose':raw_pose,'sharpness':round(sharpness,1),
                'decision':decision,'saved':saved,'photos':s['photos'], 'age_ms':round(age,1),
                'runtime_provider':'face=OpenCV_CPU;pose=CPUExecutionProvider',
                'timings_ms':{'total':round(time.monotonic()*1000-received,1)}}

    @app.post('/v1/stream/frame')
    async def frame(request: Request):
        if guard.locked():
            raise HTTPException(409,'推理忙，丢弃本帧')
        async with guard:
            s = None
            try:
                sid = request.headers.get('x-session-id','')
                s = get_session(sid)
                mode = request.headers.get('x-mode','face')
                fid = int(request.headers['x-frame-id'])
                capture_ms = float(request.headers['x-capture-ms'])
                initial_age = float(request.headers.get('x-frame-age-ms','0'))
                if mode not in ('face','device') or fid<=s['last'] or not np.isfinite(capture_ms) or not 0<=initial_age<=60000:
                    raise ValueError('帧参数无效或过期')
                payload = bytearray()
                async for chunk in request.stream():
                    payload.extend(chunk)
                    if len(payload)>5*1024*1024:
                        raise ValueError('图像超过5MB')
                payload = bytes(payload)
                if not payload.startswith(b'\xff\xd8'):
                    raise ValueError('流接口需要JPEG图像')
                with Image.open(io.BytesIO(payload)) as im:
                    if im.width*im.height > 16000000:
                        raise ValueError('图像分辨率过大')
                    im.verify()
                s['last'] = fid
                received = time.monotonic()*1000
                offset = received-capture_ms-initial_age
                s['clock'] = offset if s['clock'] is None else min(s['clock'],offset)
                if s['mode'] != mode:
                    s['gate'].clear()
                    s['filter'].clear()
                    s['lock'] = FaceLock()
                    s['mode'] = mode
                return await asyncio.to_thread(infer,s,sid,mode,payload,fid,capture_ms,initial_age,received)
            except HTTPException:
                raise
            except (ValueError, KeyError, OSError) as exc:
                if s:
                    s['gate'].clear()
                    s['filter'].clear()
                raise HTTPException(400,str(exc))
            except Exception as exc:
                if s:
                    s['gate'].clear()
                    s['filter'].clear()
                raise HTTPException(500,f'推理失败：{type(exc).__name__}')

    app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8876)
