"""Only YuNet + FSA-Net; models are immutable, CPU-only and loaded once."""
import hashlib, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[2]
HASHES = {"face_detection_yunet.onnx":"8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
          "head_pose_fsanet_1x1.onnx":"120fa107a2dd3be78c21c3a73a0db980590643a5372893e8878094898262f213"}

class PoseBackend:
    def __init__(self):
        dependencies=ROOT/"runtime_deps/ort_cpu"
        if not dependencies.is_dir():
            raise RuntimeError("缺少本项目隔离的角度推理依赖")
        # Dependency bootstrap only; never imports another project's business code.
        if str(dependencies) not in sys.path: sys.path.append(str(dependencies))
        import onnxruntime as ort
        if ort.__version__!="1.20.1": raise RuntimeError("角度推理依赖版本应为1.20.1")
        from .head_pose.face_detector import YuNetFaceDetector
        from .head_pose.estimator import HeadPoseEstimator
        folder=ROOT/"models/capture"
        for name,value in HASHES.items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=value:
                raise RuntimeError("角度模型校验失败："+name)
        self.face=YuNetFaceDetector(folder/"face_detection_yunet.onnx")
        self.pose=HeadPoseEstimator(folder/"head_pose_fsanet_1x1.onnx")
        self.hashes=dict(HASHES)
    def observe(self,frame,face_lock,now):
        height,width=frame.shape[:2]
        faces=self.face.detect_all(frame)
        if len(faces)>1:
            return None,None,0.,0.,"画面中有多个人脸，请仅保留一位受检者"
        face,reason=face_lock.select(faces,self.face,width,height,now)
        if not face: return None,None,0.,0.,reason
        x1,y1,x2,y2=self.face.crop_box(face,width,height)
        crop=frame[y1:y2,x1:x2]
        pose=self.pose.infer_bgr(crop).to_dict()
        sharpness=float(cv2.Laplacian(cv2.cvtColor(cv2.resize(crop,(160,160)),cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
        return face.to_dict(),pose,sharpness,min(face.width,face.height),reason
