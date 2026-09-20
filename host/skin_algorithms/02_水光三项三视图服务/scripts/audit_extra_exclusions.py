"""Inspect downstream geometric exclusions without running pore detection."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/v3'))
from src.engines.visia_regions import build_visia_regions,build_nasolabial_shadow_mask,build_moustache_feature_exclusion_mask
from src.engines.pores_engine import Config

def main():
 source=ROOT/'hair_repair_research/layered_brow30_v1';target=ROOT/'hair_repair_research/extra_exclusions_v2';rows=[]
 flags={(r['sample'],r['view']):r['semantic_hair_candidate_v1']['quality_flags'] for r in json.loads((source/'receipt.json').read_text())}
 for sample in ('one','two'):
  for view in ('left','front','right'):
   out=target/sample/view;out.mkdir(parents=True,exist_ok=True)
   with np.load(source/sample/view/'semantic_hair_candidate_v1.npz') as a:
    image=a['image'];points=a['landmarks'];valid=a['valid']
    regions=build_visia_regions(image,valid,points,flags[sample,view],include_chin=True,mode='full',feature_margin_px=Config.FEATURE_EXCLUSION_MARGIN_PX)
    mask=regions.analysis_mask
    old=build_nasolabial_shadow_mask(image,points,mask,output_margin_px=4,restrict_frangi_to_corridor=True)
    new=build_nasolabial_shadow_mask(image,points,mask,output_margin_px=1,restrict_frangi_to_corridor=True,require_thin_component=True)
    strip=build_moustache_feature_exclusion_mask(mask.shape,points,mask,margin_px=12)
    np.savez_compressed(out/'masks.npz',analysis=mask,old_nasolabial=old,new_nasolabial=new,nasal_strip=strip)
    tiles=[]
    for title,m in [('Old dark exclusions',old),('Thin evidence only',new),('Unassessed nasal strip',strip)]:
     im=image.copy();im[m>0]=(im[m>0]*.5+np.array((0,0,255))*.5).astype(np.uint8);im=cv2.resize(im,(500,500));im=cv2.copyMakeBorder(im,32,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(im,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1);tiles.append(im)
    cv2.imwrite(str(out/'comparison.jpg'),np.hstack(tiles))
    row={'sample':sample,'view':view,'old_nasolabial_pixels':int(np.count_nonzero(old)),'new_nasolabial_pixels':int(np.count_nonzero(new)),'nasal_strip_pixels':int(np.count_nonzero(strip))};rows.append(row);print(json.dumps(row),flush=True)
 (target/'receipt.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
