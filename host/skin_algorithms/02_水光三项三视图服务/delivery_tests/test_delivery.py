from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def test_public_contract_and_fixture():
    from cloud_three.public_scores import ScoreResponse
    data=json.loads((ROOT/"examples/current_demo/result.json").read_text(encoding="utf-8"))
    result=ScoreResponse.model_validate(data)
    assert set(data)=={"pores","spots","surface_gloss"}
    assert (result.pores.score,result.pores.severity,len(result.pores.regions))==(57.0,"中度",9)
    # Historical demo deliberately keeps the withdrawn mixed-pipeline spot
    # score as null; current three-item examples are in LATEST_CLOUD_RUN_*.json.
    assert (result.spots.score,result.spots.severity,len(result.spots.regions))==(None,None,8)
    assert (result.surface_gloss.score,result.surface_gloss.severity,len(result.surface_gloss.regions))==(72.0,"轻度",9)

def test_scoring_reference_and_runtime_roots(monkeypatch,tmp_path):
    expected="8e27a91f193d6bb21587eb65c5d89003328da727152bcfe5951e788582095bdd"
    assert hashlib.sha256((ROOT/"vendor/v3_reference.json").read_bytes()).hexdigest()==expected
    assert (ROOT/"vendor/v3/run.py").is_file()
    assert (ROOT/"vendor/water/shuiguang/config.py").is_file()
    monkeypatch.setenv("SHUIGUANG_RUNTIME_ROOT",str(tmp_path/"runtime"))
    monkeypatch.setenv("SHUIGUANG_INPUT_ROOT",str(tmp_path/"inputs"))
    import importlib
    from cloud_three import settings
    settings=importlib.reload(settings)
    assert settings.RUNTIME==tmp_path/"runtime"
    assert settings.input_root()==tmp_path/"inputs"

def test_preview_is_package_local():
    import preview_server
    data=preview_server.data("triplet")
    assert set(data["detections"])=={"pores","spots","surface_gloss"}
    for module in data["detections"].values():
        assert set(module)=={"left","front","right"}
    response=preview_server.download("triplet")
    public=json.loads(response.body)
    assert set(public)=={"pores","spots","surface_gloss"}
