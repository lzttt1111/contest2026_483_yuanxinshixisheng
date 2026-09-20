"""Publisher BiSeNet ONNX adapter for explicit local diagnostic mode.

Pre/post-processing follows yakhyo/face-parsing/onnx_inference.py.
No custom line/colour/component filtering. Weight distribution remains subject
to the separately recorded source/usage review; this module never downloads.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np

VERSION='bisenet_rgb_diagnostic_v1'
MODEL_SHA256='0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'
LABELS=('background','skin','left_brow','right_brow','left_eye','right_eye','eyeglasses','left_ear','right_ear','earring','nose','mouth','upper_lip','lower_lip','neck','necklace','cloth','hair','hat')

class FaceParser:
    def __init__(self,path,allowed_root):
        path=Path(path).resolve();allowed_root=Path(allowed_root).resolve()
        if not path.is_relative_to(allowed_root):raise ValueError('Face parser asset is outside the water sandbox')
        if not path.is_file():raise FileNotFoundError('Face parser asset missing; no fallback/download permitted')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=MODEL_SHA256:raise ValueError('Unverified face parser model SHA256')
        self.model_sha256=digest
        self.net=cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def categories(self,image):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Face parser requires uint8 BGR image')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        x=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        self.net.setInput(np.ascontiguousarray(x.transpose(2,0,1)[None]))
        logits=self.net.forward()
        if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():
            raise ValueError('Invalid face parser output shape/values')
        return cv2.resize(logits[0].argmax(0).astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)

    def build(self,image,feature_masks):
        categories=self.categories(image)
        raw_skin=np.isin(categories,(1,10))
        features=feature_masks['feature_exclusions']>0
        valid=(raw_skin&~features).astype(np.uint8)*255
        zero=np.zeros(categories.shape,np.uint8)
        hair=(categories==17).astype(np.uint8)*255
        debug={'semantic_skin_mask':raw_skin.astype(np.uint8)*255,
          'face_parser_categories':categories,'face_geometry_mask':raw_skin.astype(np.uint8)*255,
          'forehead_completion':zero.copy(),'semantic_candidate_before_hair':raw_skin.astype(np.uint8)*255,
          'bulk_hair_mask':hair,'hair_mask':hair,'color_hair_mask':zero.copy(),
          'strand_hair_mask':zero.copy(),'facial_hair_mask':zero.copy(),
          'hair_line_response':np.zeros(categories.shape,np.float32),'hair_scale_vote':zero.copy(),
          'display_face_mask':raw_skin.astype(np.uint8)*255,
          'parser_accessory_mask':np.isin(categories,(6,9,15,18)).astype(np.uint8)*255,
          **feature_masks}
        return valid,debug
