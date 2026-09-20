from pathlib import Path
from types import SimpleNamespace
import pytest
from src.detection_runtime.cloud_projection import project_clinic_result

@pytest.mark.parametrize("algorithm,extra",[("redness","red_areas_overlay"),("brown","brown_spots_overlay")])
def test_clinic_base_and_annotated_roles(tmp_path,algorithm,extra):
    base=tmp_path/"base.png";base.write_bytes(b"base")
    marker=tmp_path/"marker.png";marker.write_bytes(b"marker")
    metric=tmp_path/"metrics.json";metric.write_text("{}")
    item={"主结果图":str(marker),"附加结果图":[str(base)],"量化JSON":str(metric),
          "量化CSV":"metrics.csv","医学V2CSV":"medical.csv"}
    quality=SimpleNamespace(quality_score=1,quality_status="PASS",quality_flags=[])
    prepared=SimpleNamespace(preprocessed={"CP_M":quality})
    result=project_clinic_result(algorithm,{algorithm:item},prepared,tmp_path/"public",{"timing_seconds":{}})
    assert result["results"][algorithm]==str(base)
    assert result["results"][extra]==str(marker)
    item["附加结果图"]=[]
    with pytest.raises(ValueError,match="base image"):
        project_clinic_result(algorithm,{algorithm:item},prepared,tmp_path/"public",{"timing_seconds":{}})
