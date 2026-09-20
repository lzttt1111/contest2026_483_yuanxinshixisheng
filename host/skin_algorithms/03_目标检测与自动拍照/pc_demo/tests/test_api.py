"""Deterministic test doubles are test-only; production always loads real models."""
import io
import sys
import tempfile
import unittest
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server import create_app
from head_pose.face_detector import FaceDetection, YuNetFaceDetector

class FakeFace:
    def detect_all(self,frame):return [FaceDetection(10,10,150,150,.99,())]
    select_primary=staticmethod(YuNetFaceDetector.select_primary)
    def crop_box(self,*args):return (10,10,160,160)

class ApiTests(unittest.TestCase):
    def test_three_slots_exact_jpeg_and_front_retake(self):
        pose={'valid':True,'yaw':0.,'pitch':0.,'roll':0.}
        model=SimpleNamespace(face=FakeFace(),pose=SimpleNamespace(infer_bgr=lambda _:SimpleNamespace(to_dict=lambda:dict(pose))))
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(model,tmp)) as client, patch('server.cv2.Laplacian',return_value=SimpleNamespace(var=lambda:100.)):
            sid=client.post('/v1/session').json()['session_id']
            buf=io.BytesIO();Image.new('RGB',(320,240),'gray').save(buf,format='JPEG');payload=buf.getvalue()
            fid=0
            def send():
                nonlocal fid
                r=client.post('/v1/stream/frame',content=payload,headers={'X-Session-Id':sid,'X-Frame-Id':str(fid),'X-Capture-Ms':str(fid*100)})
                fid+=1;self.assertEqual(r.status_code,200,r.text);return r.json()
            for slot,yaw in [('front',0),('left',-45),('right',45)]:
                pose['yaw']=yaw
                for _ in range(20):value=send()
                self.assertIn(slot,value['photos'])
                saved=value['photos'][slot]
                self.assertEqual(client.get(saved['image_url']).content,payload)
                self.assertEqual(saved['frame_sha256'],hashlib.sha256(payload).hexdigest())
            self.assertEqual(set(value['photos']),{'front','left','right'})
            old=dict(value['photos'])
            self.assertEqual(client.post(f'/v1/session/{sid}/reset?side=front').status_code,200)
            pose['yaw']=0
            for _ in range(20):value=send()
            self.assertNotEqual(value['photos']['front']['frame_id'],old['front']['frame_id'])
            self.assertEqual(value['photos']['left'],old['left']);self.assertEqual(value['photos']['right'],old['right'])

    def test_exact_photo_once_retake_reset_and_old_session(self):
        model=SimpleNamespace(face=FakeFace(),pose=SimpleNamespace(infer_bgr=lambda _:SimpleNamespace(to_dict=lambda:{'valid':True,'yaw':-45.,'pitch':0.,'roll':0.})))
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(model,tmp)) as client, patch('server.cv2.Laplacian',return_value=SimpleNamespace(var=lambda:100.)):
            sid=client.post('/v1/session').json()['session_id']
            buf=io.BytesIO();Image.new('RGB',(320,240),'gray').save(buf,format='JPEG');payload=buf.getvalue()
            def send(fid):
                return client.post('/v1/stream/frame',content=payload,headers={'X-Session-Id':sid,'X-Frame-Id':str(fid),'X-Capture-Ms':str(fid*100)})
            for i in range(5):self.assertIsNone(send(i).json()['saved'])
            value=send(5).json();saved=value['saved'];self.assertIsNotNone(saved,value)
            content=client.get(saved['image_url']).content
            self.assertEqual(payload,content)
            meta=client.get(saved['metadata_url']).json()
            self.assertEqual(5,meta['frame_id']);self.assertEqual(hashlib.sha256(payload).hexdigest(),meta['frame_sha256'])
            self.assertIsNone(send(6).json()['saved'])
            self.assertEqual(400,send(6).status_code)
            self.assertEqual(200,client.post(f'/v1/session/{sid}/reset?side=left').status_code)
            self.assertIsNone(send(7).json()['saved'])
            client.post('/v1/session')
            self.assertEqual(409,send(8).status_code)

    def test_settings_validation_and_isolation(self):
        with TestClient(create_app(SimpleNamespace())) as client:
            sid=client.post('/v1/session').json()['session_id']
            values={'yaw_target':30,'yaw_tolerance':5,'pitch_limit':25,'roll_limit':12,'stable_ms':700}
            url=f'/v1/session/{sid}/settings'
            self.assertEqual(client.post(url,json=values).json()['config']['yaw_target'],30)
            for bad in [{**values,'yaw_tolerance':30},{**values,'yaw_target':True},{**values,'pitch_limit':90},{'yaw_target':30}]:
                self.assertEqual(client.post(url,json=bad).status_code,400)
            self.assertEqual(client.post('/v1/session/missing/settings',json=values).status_code,409)
            self.assertEqual(client.post(url,json=values).status_code,200)

if __name__=='__main__':unittest.main()
