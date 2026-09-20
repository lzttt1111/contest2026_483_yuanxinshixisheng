import sys
import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.state import CaptureGate, FaceLock, PoseFilter
sys.path.insert(0,str(ROOT/'vendor'))
from head_pose.face_detector import FaceDetection, YuNetFaceDetector
C=json.loads((ROOT/'config.json').read_text())

class GateTests(unittest.TestCase):
    def setUp(self):
        self.g=CaptureGate(C)
        self.p={'valid':True,'yaw':-45.,'pitch':0.,'roll':0.}
    def step(self,t,**kw):
        return self.g.step(kw.pop('pose',self.p),t,kw.pop('blur',100),kw.pop('size',150),kw.pop('age',20))
    def test_stability_and_once(self):
        for t in [0,100,200,300,400]: self.assertIsNone(self.step(t)['trigger'])
        self.assertEqual('left',self.step(500)['trigger'])
        self.g.commit('left')
        self.assertIsNone(self.step(600)['trigger'])
    def test_loss_blur_stale_small_and_tilt_reset(self):
        for bad in [{'pose':None},{'blur':0},{'age':701},{'size':20}, {'pose':{**self.p,'pitch':20}}, {'pose':{**self.p,'yaw':float('nan')}}]:
            self.g.clear();self.step(0);self.step(250)
            self.assertIsNone(self.step(400,**bad)['trigger'])
            self.assertIsNone(self.step(500)['trigger'])
    def test_gap_and_side_change(self):
        self.step(0);self.assertEqual(0,self.step(600)['progress'])
        self.assertEqual(0,self.step(700,pose={**self.p,'yaw':45})['progress'])
    def test_yaw_motion_resets(self):
        self.step(0);self.step(250)
        self.assertEqual(0,self.step(500,pose={**self.p,'yaw':-41})['progress'])
    def test_lost_face_never_switches_to_other(self):
        a=FaceDetection(10,10,100,100,.9,())
        b=FaceDetection(400,10,100,100,.9,())
        tracker=FaceLock()
        self.assertEqual(a,tracker.select([a],YuNetFaceDetector,640,480)[0])
        self.assertIsNone(tracker.select([b],YuNetFaceDetector,640,480)[0])
        self.assertIsNone(tracker.select([a],YuNetFaceDetector,640,480)[0])

    def test_brief_miss_recovers_but_long_loss_requires_reset(self):
        a=FaceDetection(10,10,100,100,.9,())
        tracker=FaceLock()
        tracker.select([a],YuNetFaceDetector,640,480,0)
        self.assertIsNone(tracker.select([],YuNetFaceDetector,640,480,100)[0])
        self.assertEqual(a,tracker.select([a],YuNetFaceDetector,640,480,200)[0])
        tracker.select([],YuNetFaceDetector,640,480,300)
        self.assertIsNone(tracker.select([a],YuNetFaceDetector,640,480,700)[0])

    def test_filter_noise_and_current_frame_guard(self):
        f=PoseFilter();g=CaptureGate(C);outputs=[];trigger=None
        for i in range(40):
            raw={**self.p,'yaw':-45+[-4,4,0][i%3]}
            sm=f.step(raw,i*33);outputs.append(sm['yaw'])
            decision=g.step(sm,i*33,100,150,20,raw_pose=raw)
            if decision['trigger']:trigger=decision['trigger']
        self.assertEqual(trigger,'left')
        self.assertLess(max(outputs[-15:])-min(outputs[-15:]),1)
        g=CaptureGate(C)
        for i in range(6):decision=g.step(self.p,i*100,100,150,20,raw_pose={**self.p,'yaw':-80})
        self.assertIsNone(decision['trigger'])
        self.assertIsNone(f.step(None,2000))
        self.assertEqual(f.step({**self.p,'yaw':30},2033)['yaw'],30)

    def test_filter_does_not_hide_sustained_turn(self):
        f=PoseFilter();f.step(self.p,0)
        for t in range(33,800,33):sm=f.step({**self.p,'yaw':45},t)
        self.assertGreater(sm['yaw'],43)

if __name__=='__main__':unittest.main()
