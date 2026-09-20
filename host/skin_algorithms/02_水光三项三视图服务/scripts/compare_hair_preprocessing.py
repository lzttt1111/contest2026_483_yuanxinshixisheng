"""Six fixed images, before/after preprocessing only. Does not run detectors."""
from pathlib import Path
import os,sys,json,time,hashlib
import cv2,numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
os.environ['DERMAVISION_CLINICAL_SPEC']='v3'
sys.path.insert(0,str(ROOT/'vendor/v3'))
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.capture_profile import CaptureProfile

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 for name in ('face_landmarker.task','selfie_segmenter.tflite','selfie_multiclass_256x256.tflite'):assert (ROOT/'vendor/v3/models'/name).is_file()
 pre=ImagePreprocessor(capture_profile=CaptureProfile.CONSUMER);receipts=[]
 try:
  for sample in ('one','two'):
   for view in ('left','front','right'):
    source=args.inputs/sample/'inputs'/f'{view}.jpg';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as im:raw=cv2.cvtColor(np.array(ImageOps.exif_transpose(im).convert('RGB')),cv2.COLOR_RGB2BGR)
    tiles=[];arrays={};record={'sample':sample,'view':view,'input_sha256':digest}
    case=args.output/sample/view;case.mkdir(parents=True,exist_ok=True)
    for mode in ('legacy','semantic_hair_candidate_v1'):
     os.environ['SHUIGUANG_PREPROCESS_MODE']=mode;began=time.monotonic();r=pre.preprocess_image(raw);debug=pre.last_debug_masks;final=r.mask_bundle.algorithm_mask()
     arrays[mode]={'image':r.analysis_image.copy(),'valid':final.copy()}
     np.savez_compressed(case/(mode+'.npz'),image=r.analysis_image,valid=final,landmarks=r.landmarks,**debug)
     for key in ('eyebrow_core_mask','eyebrow_exclusion_mask','semantic_skin_mask','feature_exclusions','strand_hair_mask','bulk_hair_mask','facial_hair_mask'):
      if key in debug:
       mask=debug[key];paint=r.analysis_image.copy();paint[mask>0]=(paint[mask>0]*.5+np.array((0,0,255))*.5).astype(np.uint8)
       cv2.imwrite(str(case/(mode+'_'+key+'.png')),paint)
     if 'eyebrow_exclusion_mask' in debug:
      assert not np.any((final>0)&(debug['eyebrow_exclusion_mask']>0))
     record[mode]={'seconds':time.monotonic()-began,'final_skin_pixels':int(np.count_nonzero(final)),'quality_status':r.quality_status,'quality_flags':r.quality_flags}
     masks=[('Hair evidence',debug['bulk_hair_mask']|debug['strand_hair_mask']),("Hair safety",r.mask_bundle.hair_mask),("Final skin",final)]
     row=[]
     for label,mask in masks:
      img=r.analysis_image.copy();color=(0,255,0) if label=='Final skin' else (0,0,255);img[mask>0]=(img[mask>0]*.5+np.array(color)*.5).astype(np.uint8)
      cv2.imwrite(str(case/(mode+'_'+label.replace(' ','_')+'.png')),img)
      img=cv2.resize(img,(420,420));img=cv2.copyMakeBorder(img,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(img,('Old ' if mode=='legacy' else 'Candidate ')+label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1);row.append(img)
     tiles.append(np.hstack(row))
    assert np.array_equal(arrays['legacy']['image'],arrays['semantic_hair_candidate_v1']['image'])
    record['same_analysis_pixels']=True
    cv2.imwrite(str(case/'comparison.jpg'),np.vstack(tiles));receipts.append(record);print(json.dumps(record),flush=True)
  (args.output/'receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
 finally:
  pre.face_landmarker.close();pre.segmenter.close()
  if hasattr(pre,'semantic_hair_segmenter'):pre.semantic_hair_segmenter.close()
if __name__=='__main__':main()
