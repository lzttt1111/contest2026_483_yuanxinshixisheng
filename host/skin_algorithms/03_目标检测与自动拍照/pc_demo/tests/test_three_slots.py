import unittest
from test_state import C
from lab.state import CaptureGate

class ThreeSlots(unittest.TestCase):
    def test_front_and_both_sides_once(self):
        g=CaptureGate(C)
        for i,(slot,yaw) in enumerate([('front',0),('left',-45),('right',45)]):
            p=dict(valid=True,yaw=yaw,pitch=0,roll=0)
            for t in range(0,501,100):r=g.step(p,i*1000+t,100,150,0,raw_pose=p)
            self.assertEqual(r['trigger'],slot);g.commit(slot)
            self.assertIsNone(g.step(p,i*1000+600,100,150,0,raw_pose=p)['trigger'])
        self.assertEqual(g.done,{'front','left','right'})
        g.done.remove('front');g.clear()
        p=dict(valid=True,yaw=0,pitch=0,roll=0)
        for t in range(3000,3501,100):r=g.step(p,t,100,150,0,raw_pose=p)
        self.assertEqual(r['trigger'],'front')
        self.assertEqual(g.done,{'left','right'})

    def test_raw_front_mismatch_resets_stability(self):
        g=CaptureGate(C);p=dict(valid=True,yaw=0,pitch=0,roll=0)
        for t in (0,100,200):g.step(p,t,100,150,0,raw_pose=p)
        for raw in ({**p,'yaw':30},{'valid':False},{**p,'yaw':float('nan')},{**p,'pitch':20}):
            r=g.step(p,300,100,150,0,raw_pose=raw)
            self.assertIsNone(r['trigger']);self.assertEqual(g.samples,0)
        self.assertIsNone(g.step(p,400,100,150,0,raw_pose=p)['trigger'])

    def test_front_missing_blur_and_stale_never_trigger(self):
        for blur,size,age in [(0,150,0),(100,10,0),(100,150,701)]:
            g=CaptureGate(C);p=dict(valid=True,yaw=0,pitch=0,roll=0)
            for t in range(0,1000,100):self.assertIsNone(g.step(p,t,blur,size,age,raw_pose=p)['trigger'])

if __name__=='__main__':unittest.main()
