import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.regions import FaceRegions,partition
from importlib.metadata import version
root=Path(__file__).resolve().parents[1]
model=FaceRegions(root/'models/face_landmarker.task')
try:
    with Image.open(sys.argv[1]) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((960,720));rgb=np.asarray(im)
    result=model.infer(rgb)
    assert result['valid'],result
    assert len(result['landmarks'])==468 and len(result['regions'])==7
    labels,_,_,_,regions=partition(result['landmarks'],rgb.shape[1],rgb.shape[0])
    for i,r in enumerate(regions,1):assert np.count_nonzero(labels==i)>0,r['id']
    overlay=rgb.copy()
    mask=cv2.resize(labels,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
    for i,r in enumerate(regions,1):
        color=tuple(int(r['color'][j:j+2],16) for j in (1,3,5));overlay[mask==i]=color
    out=cv2.addWeighted(rgb,.65,overlay,.35,0)
    Image.fromarray(out).save(sys.argv[2])
    print({'pass':True,'mediapipe':version('mediapipe'),'regions':[r['id'] for r in regions],'timing_ms':result['timing_ms']})
    assert not model.infer(np.zeros_like(rgb))['valid']
finally:model.close()
