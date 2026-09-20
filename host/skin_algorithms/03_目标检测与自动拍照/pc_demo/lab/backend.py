import hashlib
import json
import sys
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
from head_pose.face_detector import YuNetFaceDetector
from head_pose.estimator import HeadPoseEstimator
from target_detection.pipeline import Pipeline
from lab.regions import FaceRegions


class Backend:
    def __init__(self):
        manifest = json.loads((ROOT/'models'/'SHA256.json').read_text(encoding='utf-8'))
        for name, expected in manifest.items():
            if hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Model SHA256 mismatch: {name}')
        cv2.setNumThreads(2)
        self.face = YuNetFaceDetector(ROOT/'models'/'face_detection_yunet.onnx')
        self.pose = HeadPoseEstimator(ROOT/'models'/'head_pose_fsanet_1x1.onnx')
        self.device = Pipeline(ROOT)
        self.regions = FaceRegions(ROOT/'models'/'face_landmarker.task')
        self.hashes = manifest
        # Complete lazy GPU initialization before accepting webcam frames.
        blank = io.BytesIO()
        Image.new('RGB', (640, 480)).save(blank, format='JPEG')
        self.device.infer_bytes(blank.getvalue())
        self.device.contact_session.run(None, {self.device.contact_input: np.zeros((1,3,256,256), dtype=np.float32)})

    def health(self):
        return {'ok': True, 'service': 'rk3576-camera-lab', 'execution': 'PC_SIMULATION',
                'providers': {'device': self.device.runtime_provider, 'face': 'OpenCV_CPU',
                              'pose': self.pose.session.get_providers()[0], 'regions':'MediaPipe_CPU'}, 'model_sha256': self.hashes,
                'model_lineage':{'device':'V5 REAL-ONLY YOLOX-S','contact':'V5 MobileNetV3-Small'}}
