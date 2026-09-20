"""Type-specific 2D wrinkle measurements; 3D depth is deliberately not inferred."""
import cv2
import numpy as np
from .registry import measurement
from .regions import build

def regions(arrays):
    valid=arrays["valid"]>0
    p=arrays["landmarks"]
    if len(p)<400:
        return {}
    base=build(valid,p,"04")
    yy,xx=np.indices(valid.shape)
    result={"forehead":base["forehead"],"glabella":base["glabella"]}
    for side,index,inner in (("left",33,133),("right",263,362)):
        x,y=p[index,:2]; span=max(abs(float(x-p[inner,0])),12)
        outside=(xx<x) if side=="left" else (xx>x)
        result[side+"_crow_feet"]=valid&outside&(abs(xx-x)<span*.75)&(abs(yy-y)<span*.65)
    x0,x1=sorted([p[61,0],p[291,0]])
    top,bottom=sorted([p[0,1],p[17,1]])
    pad=max(12,(x1-x0)*.25)
    result["perioral"]=valid&(xx>x0-pad)&(xx<x1+pad)&(yy>top-pad)&(yy<bottom+pad)
    result["crow_feet"]=result["left_crow_feet"]|result["right_crow_feet"]
    return result

def measure(arrays,profile):
    domains=regions(arrays)
    line=arrays["skeleton"]>0
    gray=cv2.cvtColor(arrays["image"],cv2.COLOR_BGR2GRAY).astype(float)
    contrast=np.abs(cv2.GaussianBlur(gray,(0,0),3)-gray)/255
    rows=[]
    for region,domain in domains.items():
        local=line&domain; area=int(domain.sum())
        if region in ("forehead","glabella"):
            n,_,stats,_=cv2.connectedComponentsWithStats(local.astype(np.uint8),8)
            count=sum(int(s[cv2.CC_STAT_AREA])>=10 for s in stats[1:])
            values={"main_count":count if area>=100 else None,
                    "length_burden":float(local.sum()*10000/area) if area>=100 else None,"depth":None}
        else:
            affected=(cv2.dilate(local.astype(np.uint8),np.ones((9,9),np.uint8))>0)&domain
            high=(cv2.boxFilter(local.astype(np.float32),-1,(16,16))>=.1)&affected
            values={"area":float(affected.sum()/area) if area>=100 else None,
                    "density":float(local.sum()*10000/affected.sum()) if affected.any() else None,
                    "high_area":float(high.sum()/area) if area>=100 else None,
                    "contrast_p50":float(np.percentile(contrast[local],50)) if local.any() else None}
        for name,value in values.items():
            reason="requires_calibrated_3d" if name=="depth" else "no_valid_signal"
            unit="条" if name=="main_count" else ("比例" if "area" in name else "标准化指数")
            rows.append(measurement("08",name,region,profile,value,unit,reason))
    return rows
