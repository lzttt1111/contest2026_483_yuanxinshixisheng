import hashlib,json
from pathlib import Path
from src.scoring_input.builder import extract_evidence,load_pinned_field_list,build_scoring_input
from src.nine_analysis.metrics import extract_item
ROOT=Path(__file__).resolve().parents[1]
def test_original_field_list_is_preserved():
    assert hashlib.sha256((ROOT/"calibration/scoring_input_field_list_v1.json").read_bytes()).hexdigest()=="7d5814157df3cb262188d9d6e30a8a645e88b463e5cd9ebc95927f390f64bbad"
def test_institution_red_summary_survives_pruning():
    raw={"high_red_area_ratio":.7,"mean_redness":.8,"p90_redness":.9,
         "runtime_summary":{"high_red_area_ratio":.2,"mean_redness":.3,"p90_redness":.4},
         "medical_metrics_v2":{"overall_metrics":{"core_metrics":{"scope_and_burden":{"diffuse_red_area_ratio":.1}}}},
         "debug_private_path":"/private/should-not-be-published"}
    source={"redness":{"metrics":raw}}
    filtered=extract_evidence("redness",source,load_pinned_field_list())
    assert extract_item("redness",raw)[1]==extract_item("redness",filtered["redness"]["metrics"])[1]
    assert "debug_private_path" not in filtered["redness"]["metrics"]
def test_consumer_evidence_requirements_not_removed():
    old=json.loads((ROOT/"calibration/scoring_input_field_list_v1.json").read_text())
    new=json.loads((ROOT/"calibration/scoring_input_field_list_v2.json").read_text())
    for name,item in old["algorithms"].items():
        before={f["path"] for f in item["fields"] if f["required"] and f.get("profile_scope") in (None,"consumer")}
        after={f["path"] for f in new["algorithms"][name]["fields"] if f["required"] and f.get("profile_scope") in (None,"consumer")}
        assert before==after
