"""Formal V3 result figures drawn from the same masks used for its measurements."""
from pathlib import Path
import cv2
import numpy as np
from .bundle import sha256

def generate(data, diffuse_arrays, lesions, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    products=[]
    def save(module,image,caption):
        path=output/("v3_"+module+".jpg")
        ok,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,92])
        if not ok:raise ValueError("cannot encode V3 result figure")
        encoded.tofile(str(path))
        products.append((path,{"path":"media/"+path.name,"modules":[module],"caption":caption,
                               "derived":True,"sha256":sha256(path)}))
    if diffuse_arrays is not None:
        valid=diffuse_arrays["valid"]>0
        base=np.full((*valid.shape,3),245,np.uint8)
        rgb=data.get("rgb")
        if rgb is not None and np.allclose(rgb["landmarks"],diffuse_arrays["landmarks"],atol=.05):
            base=rgb["image"].copy()
        support=diffuse_arrays["continuous"]>0
        tint=np.zeros_like(base); tint[:,:,2]=235
        base[support]=(.60*base[support]+.40*tint[support]).astype(np.uint8)
        save("04",base,"弥漫泛红分布图（已排除局灶目标及线状血管证据）")
    if "acne" in data:
        base=data["acne"]["image"].copy()
        palette={"erythema":(0,160,240),"papule":(40,40,235),"pustule":(180,60,170),"unknown":(130,130,130)}
        for lesion in lesions:
            x,y=lesion["centroid"]
            cv2.circle(base,(x,y),5,palette[lesion["class"]],1,cv2.LINE_AA)
        save("06",base,"毛囊及痤疮样目标分布：橙色为红斑样、红色为丘疹样、紫色为丘脓疱样，灰色为待进一步评估目标")
    if "wrinkle" in data:
        arrays=data["wrinkle"]
        from .phenotypes import PARAMETERS
        n,labels,stats,_=cv2.connectedComponentsWithStats((arrays["skeleton"]>0).astype(np.uint8),8)
        ids=np.flatnonzero((stats[:,cv2.CC_STAT_AREA]>=3)&(stats[:,cv2.CC_STAT_AREA]<=PARAMETERS["fine_max_length_px"]))
        ids=ids[ids!=0]
        from .regions import build
        fine=np.isin(labels,ids)&build(arrays["valid"],arrays["landmarks"],"07")["full_face"]
        from .wrinkle_metrics import regions
        domains=regions(arrays)
        stable=(arrays["skeleton"]>0)&np.logical_or.reduce(list(domains.values())) if domains else np.zeros_like(fine)
        saved=arrays.get("stage1_figures",{})
        if "lines_fine_skeleton" in saved:fine=saved["lines_fine_skeleton"]>0
        if "lines_stable_skeleton" in saved:stable=saved["lines_stable_skeleton"]>0
        for module,mask,caption in (("07",fine,"六区干燥样细纹分布图（二维图像表型）"),
                                    ("08",stable,"额纹、眉间纹、鱼尾纹及口周纹分布图（二维外观，不代表三维深度）")):
            base=arrays["image"].copy()
            display=cv2.dilate(mask.astype(np.uint8),np.ones((2,2),np.uint8))>0
            base[display]=(30,220,140)
            save(module,base,caption)
    return products
