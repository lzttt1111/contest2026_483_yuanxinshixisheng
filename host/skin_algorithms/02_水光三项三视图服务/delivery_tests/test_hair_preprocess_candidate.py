from pathlib import Path
import sys
from types import SimpleNamespace
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/v3'))
from src.preprocess.image_preprocessor import build_feature_exclusion_masks,LEFT_EYEBROW,RIGHT_EYEBROW,ImagePreprocessor
from src.preprocess.semantic_hair import build
from src.capture_profile import CaptureProfile

def test_eyebrow_30_means_radius_not_kernel():
 image=np.full((1024,1024,3),160,np.uint8);points=np.zeros((478,2),np.float32)
 polygon=np.array([[300,300],[320,300],[340,300],[350,305],[350,310],[340,315],[330,315],[320,315],[310,310],[300,305]])
 points[LEFT_EYEBROW]=polygon;points[RIGHT_EYEBROW]=polygon+(200,0)
 masks=build_feature_exclusion_masks(image,points,False,30)
 expected=cv2.dilate(masks['eyebrow_core_mask'],cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61)))
 assert np.array_equal(expected,masks['eyebrow_exclusion_mask'])
 assert masks['eyebrow_exclusion_mask'][270,320]>0
 assert masks['eyebrow_exclusion_mask'][269,320]==0

class Mask:
 def __init__(self,value):self.value=value
 def numpy_view(self):return self.value

def test_background_not_rescued_and_no_blanket_facial_hair():
 shape=(128,128);category=np.full(shape,3,np.uint8);category[:15]=0;category[15:30]=1
 result=SimpleNamespace(category_mask=Mask(category),confidence_masks=[Mask((category==i).astype(np.float32)) for i in range(6)])
 seg=SimpleNamespace(segment=lambda image:result)
 image=np.full((*shape,3),160,np.uint8);domain=np.full(shape,255,np.uint8);features=np.zeros(shape,np.uint8);features[60:65,40:90]=255
 skin,debug=build(image,domain,features,seg,np.zeros((478,2),np.float32))
 # The bundled semantic model treats category 3 as skin; only category 0
 # background and category 1 scalp/hair must be removed.
 assert not np.any(skin[:15])
 assert not np.any(skin[15:30])
 assert np.all(skin[35:55]>0)
 assert not np.any(skin[features>0])
 assert np.count_nonzero(debug['facial_hair_mask'])==0

def test_candidate_does_not_get_second_seven_pixel_expansion(monkeypatch):
 pre=object.__new__(ImagePreprocessor);pre.capture_profile=CaptureProfile.CONSUMER
 empty=np.zeros((128,128),np.uint8);hair=empty.copy();hair[40:45,40:45]=255
 pre.last_debug_masks={'hair_mask':hair,'facial_hair_mask':empty}
 result=SimpleNamespace(skin_mask=np.full_like(empty,255),landmarks=np.zeros((478,2)),analysis_image=np.zeros((128,128,3),np.uint8))
 monkeypatch.setenv('SHUIGUANG_PREPROCESS_MODE','bisenet_rgb_diagnostic_v1')
 pre._attach_mask_bundle(result)
 assert np.array_equal(result.mask_bundle.hair_mask,hair)
 assert not np.any(result.mask_bundle.algorithm_mask()[hair>0])
