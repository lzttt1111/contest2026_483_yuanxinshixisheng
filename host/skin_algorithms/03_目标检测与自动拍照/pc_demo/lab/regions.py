"""Seven geometric face ROIs, not treatment or skin-diagnosis recommendations."""
import time
import cv2
import numpy as np

OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
RIGHT_EYE = [33,160,158,133,153,144]
LEFT_EYE = [362,385,387,263,373,380]
LIPS = [61,40,37,0,267,270,291,321,314,17,84,91]
REGIONS = [
 ('forehead','额头','#e6d996',[10,338,297,332,284,300,293,334,296,336,9,107,66,105,63,70,54,103,67,109]),
 ('subject_right_cheek','右脸颊','#ecaf87',[234,127,162,70,111,117,118,100,36,203,206,216,212,138,172,58,132,93]),
 ('subject_left_cheek','左脸颊','#d98691',[454,356,389,300,340,346,347,329,266,423,426,436,432,367,397,288,361,323]),
 ('chin','下巴','#648ed8',[61,91,84,17,314,321,291,367,397,365,379,378,400,377,152,148,176,149,150,136,172,138]),
 ('mouth_area','嘴周','#b3adce',[203,206,216,212,57,43,106,182,83,18,313,406,335,273,287,432,436,426,423,2]),
 ('nose','鼻部','#8bc685',[168,193,122,196,3,51,45,44,1,274,275,281,248,419,351,417]),
 ('eye_area','眼周','#91c9de',None),
]

def expanded_oval(points, forehead_extension=.20):
    """Extend only the upper oval along the head axis, not the cheeks/background."""
    points=np.asarray(points,dtype=np.float64)
    oval=points[OVAL].copy()
    axis=points[152]-points[10]
    height=max(float(np.linalg.norm(axis)),1.)
    unit=axis/height
    relative=(oval-points[10])@unit/height
    weight=np.clip(1-relative/.38,0,1)
    oval-=weight[:,None]*height*forehead_extension*unit
    return oval

def fill_partition_gaps(labels,face):
    """Nearest existing region fills interior gaps; outside remains unassigned."""
    seeds=labels>0
    if not seeds.any():return labels
    _,nearest=cv2.distanceTransformWithLabels((~seeds).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    values=labels[seeds]
    gaps=(face>0)&(~seeds)
    labels[gaps]=values[nearest[gaps]-1]
    return labels

def partition(points, width, height):
    scale=min(1.,640/max(width,height))
    w,h=max(1,round(width*scale)),max(1,round(height*scale))
    p=np.rint(np.asarray(points)*scale).astype(np.int32)
    def hull(ids):
        m=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(m,cv2.convexHull(p[ids]),1)
        return m
    extended=np.rint(expanded_oval(points)*scale).astype(np.int32)
    face=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(face,cv2.convexHull(extended),1)
    labels=np.zeros((h,w),np.uint8)
    # Later regions have precedence. A single label map makes overlap impossible.
    for idx,(_,_,_,ids) in enumerate(REGIONS,1):
        region=hull(ids) if ids else np.maximum(hull([33,70,63,105,66,107,133,112,26,22,23,24,110,25]),hull([362,336,296,334,293,300,263,255,339,254,253,252,256,341]))
        if idx==1:
            # Preserve the lower forehead boundary, replace upper edge with extension.
            top_ids=[i for i,k in enumerate(OVAL) if k in [10,338,297,332,284,251,389,162,21,54,103,67,109]]
            cv2.fillConvexPoly(region,cv2.convexHull(np.concatenate([p[ids],extended[top_ids]])),1)
        labels[(region>0)&(face>0)]=idx
    labels=fill_partition_gaps(labels,face)
    # Eye openings and lips are explicitly excluded, not skin-treatment targets.
    excluded=np.maximum(np.maximum(hull(RIGHT_EYE),hull(LEFT_EYE)),hull(LIPS))
    labels[excluded>0]=0
    regions=[]
    for idx,(key,name,color,_) in enumerate(REGIONS,1):
        contours,_=cv2.findContours((labels==idx).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        rings=[(cv2.approxPolyDP(c,.8,True).reshape(-1,2)/scale).round(1).tolist() for c in contours if cv2.contourArea(c)>=2]
        regions.append({'id':key,'name':name,'color':color,'rings':rings})
    return labels,face,excluded,scale,regions

def classify(point,labels,face,excluded,scale):
    if point is None:return 'unknown','未提供有效接触点'
    x,y=np.rint(np.asarray(point)*scale).astype(int)
    if not (0<=x<labels.shape[1] and 0<=y<labels.shape[0]) or not face[y,x]:return 'outside_face','接触点在脸外'
    if excluded[y,x]:return 'excluded','接触点位于眼睛或嘴唇排除区'
    idx=int(labels[y,x])
    if not idx:return 'unknown','接触点位于未定义区域'
    # Do not label points immediately on a region boundary as certain.
    patch=labels[max(0,y-2):y+3,max(0,x-2):x+3]
    if np.any(patch!=idx):return 'unknown','接触点靠近分区边界'
    return REGIONS[idx-1][0],REGIONS[idx-1][1]

class FaceRegions:
    def __init__(self,path):
        import mediapipe as mp
        self.mp=mp
        options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),num_faces=2)
        self.model=mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self):self.model.close()

    def infer(self,rgb,contact=None):
        start=time.perf_counter();h,w=rgb.shape[:2]
        result=self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
        empty={'valid':False,'regions':[],'landmarks':[],'contact_region':'unknown','coordinate_space':'unmirrored_original','schema_version':'face-regions-7-v2','forehead_boundary':'geometric_extension_not_hairline'}
        if len(result.face_landmarks)!=1:
            return {**empty,'reason':'未检测到人脸' if not result.face_landmarks else '检测到多人，暂停区域归属'}
        points=np.array([(p.x*w,p.y*h) for p in result.face_landmarks[0][:468]])
        if not np.isfinite(points).all():return {**empty,'reason':'关键点无效'}
        # Conservative geometric side-view guard, not a calibrated yaw estimator.
        axis=points[454]-points[234];den=float(axis@axis)
        ratio=float((points[1]-points[234])@axis/max(den,1))
        if den<1600 or not .15<ratio<.85:return {**empty,'reason':'人脸过小或侧转过大，请尽量正面展示'}
        labels,face,excluded,scale,regions=partition(points,w,h)
        region,reason=classify(contact,labels,face,excluded,scale)
        return {**empty,'valid':True,'reason':reason,'regions':regions,'landmarks':points.round(1).tolist(),
                'contact_region':region,'landmark_count':468,'runtime_provider':'MediaPipe_CPU',
                'timing_ms':round((time.perf_counter()-start)*1000,1)}
