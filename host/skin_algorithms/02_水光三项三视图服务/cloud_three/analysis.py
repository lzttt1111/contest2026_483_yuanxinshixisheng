"""Same detector/partition/count rules as the local web; three detectors only."""
import numpy as np
from .settings import bootstrap
MODULES=("pores","spots","surface_gloss")

def analyze_views(paths,task_id,progress):
    bootstrap()
    from shuiguang.preprocess import TripletPreprocessor,prepare
    from shuiguang.detectors import Finding,components
    from shuiguang.results import measure
    from shuiguang.io import read_image
    from shuiguang.anatomy import LEAVES
    from shuiguang.measurement_partition import measurement_partition
    from shuiguang.regional_statistics import module_observations,select_primary
    from shuiguang.observation_display import build_display_account
    from sg_core.engines.pores_engine import PoresAnalyzer
    from sg_core.engines.spots_engine import SpotsEngine
    from sg_core.engines.surface_gloss_engine import SurfaceGlossAnalyzer
    pre=TripletPreprocessor()
    pores,spots,gloss=PoresAnalyzer(),SpotsEngine(),SurfaceGlossAnalyzer()
    data={"views":{}}
    observed={}
    try:
        for view,path in paths.items():
            image=read_image(path)
            prepared=prepare(pre,image,view)
            if prepared.quality["status"]=="REJECT":
                raise ValueError("图像质量不满足检测要求: "+view)
            p=prepared.result
            bundle=getattr(p,"mask_bundle",None)
            hair=bundle.hair_mask if bundle is not None else None
            near=max(("left","right"),key=lambda side:int(np.count_nonzero(prepared.regions[side+"_cheek"])))
            data["views"][view]={"modules":{}}
            observed[view]={"modules":{},"near_anatomical_side":near}
            for name in MODULES:
                progress("detect",view=view,module=name)
                if name=="pores":
                    r=pores.detect_pores(p)
                    f=Finding(r.pore_mask,r.pore_score_map,r.pore_locations,r._medical_analysis_mask,{},"marker_footprint")
                elif name=="spots":
                    r=spots.detect_spots(p)
                    f=Finding(r.filtered_mask,r.candidate_heatmap,r.spot_locations,r._medical_analysis_mask,{},"instance_support")
                else:
                    r=gloss.detect_surface_gloss(p)
                    f=Finding(r.gloss_mask,r.gloss_intensity_map,components(r.gloss_mask),r.gloss_analysis_mask,{})
                item,mask,response,valid=measure(f,prepared,image.shape)
                measured,policy,radius=measurement_partition(name,p.analysis_image,p.skin_mask,p.landmarks,
                    prepared.quality["flags"],valid,p._observed_skin_mask,view,hair)
                values=module_observations(task_id,view,name,item,
                    {"mask":mask,"response":response,"valid_mask":valid},measured,p.landmarks,
                    p.analysis_image,prepared.quality["flags"],boundary_radius=radius)
                if values["rejected_observations"]:raise ValueError("检测数量与分区域不一致")
                data["views"][view]["modules"][name]=item
                observed[view]["modules"][name]=values
        statistics=select_primary(observed)
        statistics["views"]=observed
        statistics["display_reconciliation"]=build_display_account(data,statistics)
        return data,statistics
    finally:
        spots.close()
        pre.close()
