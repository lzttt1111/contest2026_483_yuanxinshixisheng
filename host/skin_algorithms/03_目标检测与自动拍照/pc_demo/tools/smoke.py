"""Real models + API smoke; no changes to source images."""
import sys
import io
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from fastapi.testclient import TestClient
from server import create_app

with TestClient(create_app()) as client:
    print(json.dumps(client.get('/health').json(),ensure_ascii=False))
    sid=client.post('/v1/session').json()['session_id']
    buf=io.BytesIO();Image.new('RGB',(640,480)).save(buf,format='JPEG')
    frames=[('face',buf.getvalue()),('device',buf.getvalue())]
    if len(sys.argv)>1:
        with Image.open(sys.argv[1]) as im:
            im.thumbnail((1280,720))
            buf=io.BytesIO();im.convert('RGB').save(buf,format='JPEG');frames.append(('face',buf.getvalue()))
    for fid,(mode,data) in enumerate(frames):
        r=client.post('/v1/stream/frame',content=data,headers={'X-Session-Id':sid,'X-Mode':mode,'X-Frame-Id':str(fid),'X-Capture-Ms':str(fid*1000),'Content-Type':'image/jpeg'})
        assert r.status_code==200,r.text
        value=r.json()
        if fid==0:assert value['face'] is None and value['saved'] is None
        if fid==1:assert value['device_bbox'] is None
        if fid==2:assert value['pose'] and value['pose']['valid'],value
        print(json.dumps(value,ensure_ascii=False))
    r=client.post('/v1/stream/frame',content=b'bad',headers={'X-Session-Id':sid,'X-Frame-Id':'99','X-Capture-Ms':'9000'})
    assert r.status_code==400
    assert client.get('/').status_code==200
    print('REAL_MODEL_API_SMOKE_PASS')
