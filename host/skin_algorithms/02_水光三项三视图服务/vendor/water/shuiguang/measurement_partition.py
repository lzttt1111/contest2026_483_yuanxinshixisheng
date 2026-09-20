"""Per-detector quantitative partitions; never expand the detector's valid domain."""
import numpy as np
import cv2
from .anatomy import Anatomy,LEAVES,partition
VERSION="detector_partition_v1"
V2_MODULES={"pores","texture","spots","brown","redness"}
def measurement_partition(module,image,skin,points,flags,valid,observed_skin,view,hair=None):
    if module not in V2_MODULES:
        return partition(points,observed_skin,view,hair,for_statistics=True),"anatomy_visible_v2",2
    from sg_core.engines.visia_regions import build_visia_regions
    margin=2 if module=="spots" else 10 if module=="redness" else 30
    work=image.copy()
    if module=="spots":work[skin==0]=0
    try:
        native=build_visia_regions(work,skin,points,flags,include_chin=True,mode="full",feature_margin_px=margin)
    except (ValueError,cv2.error):
        # Keep observations, but do not invent regions if geometry is unusable.
        empty=np.zeros(valid.shape,np.uint8)
        coverage={n:{'state':'uncovered','reason':'本项V2分区几何证据不足','observed_pixels':0,'geometric_pixels':0,'known_hair_pixels':0} for n in LEAVES}
        return Anatomy(empty,empty.copy(),empty.copy(),empty.copy(),empty.copy(),coverage),'v2_geometry_unavailable',0
    labels=np.zeros(valid.shape,np.uint8)
    # V2 names left/right mean image halves, not the subject's anatomy.
    rename={"left_cheek":"right_cheek","right_cheek":"left_cheek","forehead":"forehead","nose":"nose","chin":"chin"}
    for source,target in rename.items():
        mask=(native.regions[source]>0)&(valid>0)
        if np.any(labels[mask]):raise ValueError("V2 measurement regions overlap")
        labels[mask]=LEAVES.index(target)+1
    empty=np.zeros_like(labels)
    coverage={name:{"state":"observed" if np.count_nonzero(labels==i)>=128 else "uncovered",
        "reason":"本项V2分析分区与有效域交集","observed_pixels":int(np.count_nonzero(labels==i)),
        "geometric_pixels":int(np.count_nonzero(labels==i)),"known_hair_pixels":0} for i,name in enumerate(LEAVES,1)}
    return Anatomy(labels,labels.copy(),(labels>0).astype(np.uint8)*255,empty,empty.copy(),coverage),"v2_detector_regions_subject_sides",0
