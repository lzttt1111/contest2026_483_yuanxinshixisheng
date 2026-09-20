import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.regions import classify,partition,REGIONS,expanded_oval,OVAL

class RegionTests(unittest.TestCase):
    def test_classification_and_exclusions(self):
        labels=np.ones((30,30),np.uint8);face=np.ones_like(labels);excluded=np.zeros_like(labels)
        self.assertEqual(classify(None,labels,face,excluded,1)[0],'unknown')
        self.assertEqual(classify((-2,0),labels,face,excluded,1)[0],'outside_face')
        self.assertEqual(classify((10,10),labels,face,excluded,1)[0],'forehead')
        excluded[10,10]=1
        self.assertEqual(classify((10,10),labels,face,excluded,1)[0],'excluded')
        labels[20,20]=2
        self.assertEqual(classify((20,20),labels,face,excluded,1)[0],'unknown')
    def test_mask_contract(self):
        rng=np.random.default_rng(8);points=rng.uniform(20,180,(468,2))
        labels,face,excluded,scale,regions=partition(points,200,200)
        self.assertEqual(len(regions),7)
        self.assertTrue(np.all(labels[excluded>0]==0))
        self.assertTrue(np.all(labels[face==0]==0))
        self.assertLessEqual(int(labels.max()),7)
        self.assertTrue(np.all(labels[(face>0)&(excluded==0)]>0))

    def test_forehead_extension_follows_head_axis(self):
        points=np.zeros((468,2));points[10]=[100,50];points[152]=[100,250]
        points[234]=[30,150];points[454]=[170,150]
        expanded=expanded_oval(points)
        self.assertTrue(np.allclose(expanded[OVAL.index(10)],[100,10]))
        self.assertTrue(np.allclose(expanded[OVAL.index(152)],[100,250]))
        self.assertTrue(np.allclose(expanded[OVAL.index(234)],[30,150]))
