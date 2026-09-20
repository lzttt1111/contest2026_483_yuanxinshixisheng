"""Unordered RGB import and orientation assignment. Uses the bundled landmark model only."""
import threading
import cv2
import numpy as np
from .io import read_image
_lock=threading.Lock()
_landmarker=None
VERSION="triplet_orientation_v1"

def classify_scores(scores):
    if len(scores)!=3 or not all(np.isfinite(scores)):
        raise ValueError("需要三张可识别人脸的照片")
    estimated=["front" if abs(v)<=.35 else "left" if v>0 else "right" for v in scores]
    confident=set(estimated)=={"left","front","right"}
    # A tentative ordering is offered for review, never claimed confident on ambiguous input.
    front=min(range(3),key=lambda i:abs(scores[i]))
    sides=sorted((i for i in range(3) if i!=front),key=lambda i:scores[i])
    assignment={"left":sides[-1],"front":front,"right":sides[0]}
    return {"assignment":assignment,"confident":confident,"estimated_views":estimated,
            "message":"已识别左侧、正面、右侧，请核对预览后开始检测。" if confident else
                      "正侧脸组合不明确，请手动确认三个槽位；若缺少某个视角，请重新选择照片。",
            "orientation_version":VERSION}

def inspect_photos(paths):
    global _landmarker
    import mediapipe as mp
    from sg_core.utils.model_loader import load_face_landmarker
    with _lock:
        if _landmarker is None:_landmarker=load_face_landmarker(num_faces=2)
        scores=[]
        for ordinal,path in enumerate(paths,1):
            image=read_image(path)
            factor=min(1.,1280/max(image.shape[:2]))
            if factor<1:image=cv2.resize(image,None,fx=factor,fy=factor)
            result=_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
            faces=result.face_landmarks or []
            if len(faces)!=1:raise ValueError(f"第{ordinal}张照片"+("未识别人脸" if not faces else "包含多个人脸")+"，请重新选择。")
            points=np.array([[v.x*image.shape[1],v.y*image.shape[0]] for v in faces[0]],np.float32)
            axis=points[454]-points[234];width=float(np.linalg.norm(axis))
            if width<40:raise ValueError(f"第{ordinal}张人脸过小，请使用清晰原图。")
            score=float(np.dot(points[1]-(points[234]+points[454])/2,axis/width)/(width/2))
            scores.append(score)
        return {**classify_scores(scores),"image_direction_ratios":scores,
                "semantics":"capture_slot_left_is_nose_towards_image_right; not_physical_pose_degrees"}

