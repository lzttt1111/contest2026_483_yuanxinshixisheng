"""Read approved native V3 scores; never import detection or report rendering code."""
import hashlib
import json
import math
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from .v3_rules.scoring import grade as source_grade, display_score as source_display

VERSION="saved-v3-native-adapter-20260914-1"
EXPECTED_SCORING="0e4e990ab1f122ed639eb2e53af642888a79593842e3e93bd92535c783a8fab0"
REGIONS={
 "pores":["forehead","nose","left_nasal","right_nasal","left_inner_cheek","right_inner_cheek","left_outer_cheek","right_outer_cheek","chin"],
 "spots":["forehead","nose","left_zygoma","right_zygoma","left_cheek","right_cheek","perioral","jaw"],
 "surface_gloss":["forehead","nose","left_nasal","right_nasal","left_zygoma","right_zygoma","left_cheek","right_cheek","chin"],
}
LABELS={"forehead":"额部","nose":"鼻部","left_nasal":"画面左鼻旁","right_nasal":"画面右鼻旁",
 "left_inner_cheek":"画面左内侧面颊","right_inner_cheek":"画面右内侧面颊",
 "left_outer_cheek":"画面左外侧面颊","right_outer_cheek":"画面右外侧面颊",
 "left_cheek":"画面左面颊","right_cheek":"画面右面颊","left_zygoma":"画面左颧部",
 "right_zygoma":"画面右颧部","perioral":"口周","chin":"下巴","jaw":"下颌"}
NAMES={"pores":"毛孔","spots":"可见色斑","surface_gloss":"表面油光"}
BASES={"pores":"v3_visible_pores","spots":"v3_visible_spots","surface_gloss":"v3_surface_gloss"}

class EvidenceError(ValueError):
    def __init__(self,code,message):
        super().__init__(message);self.code=code;self.message=message

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def reject(value):raise ValueError("非有限JSON数字")
    return json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=reject)
def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()

def within(root,relative):
    root=Path(root).resolve();p=Path(relative)
    if p.is_absolute() or ".." in p.parts:raise EvidenceError("INVALID_REFERENCE","必须使用允许目录内的相对路径")
    path=(root/p).resolve()
    if not path.is_relative_to(root):raise EvidenceError("INVALID_REFERENCE","引用超出允许目录")
    return path

def missing(reason):
    return dict(score=None,severity=None,score_status="unavailable",score_reason=reason)

def display_grade(raw):
    return source_grade(raw)

def score_node(node):
    if not isinstance(node,dict):return missing("missing_region_score")
    value=node.get("score")
    if value is None:
        return missing(str(node.get("no_score_reason") or node.get("reason") or "missing_region_score"))
    if node.get("status") not in ("candidate","confirmed_zero"):
        return missing("unsupported_score_state")
    if node.get("method") not in ("stage1_complete_same_definition_aggregation","stage1_observed_negative_state"):
        return missing("unsupported_score_method")
    raw=node.get("score_raw")
    if (type(value) not in (int,float) or type(raw) not in (int,float)
            or not math.isfinite(value) or not math.isfinite(raw) or not 0<=value<=100 or not 0<=raw<=100):
        return missing("invalid_score_value")
    expected=source_display(raw)
    if value!=expected or node.get("grade")!=display_grade(raw):return missing("score_grade_mismatch")
    if node["method"]=="stage1_observed_negative_state" and not node.get("zero_state"):
        return missing("missing_zero_state_proof")
    return dict(score=value,severity=node["grade"],score_status="available",score_reason=None)

def version_of(report):
    versions=report.get("stage1_versions") or {}
    if (report.get("capture_profile")!="consumer" or report.get("scoring_system_version")!="V3.0.1"
        or versions.get("roi")!="doctor-v3-roi-4" or versions.get("scoring")!=EXPECTED_SCORING
        or not versions.get("reference") or (report.get("severity_policy") or {}).get("version")!="v3_evidence_guard_20260908_v1"):
        return None
    declared=report.get("score_direction")
    if declared is not None and declared!="higher_is_better":return None
    return "V3.0.1/"+versions["roi"]+"/"+versions["scoring"]+"/"+versions["reference"]

def report_input_hash(report):
    proof=report.get("legacy_input_proof") or {}
    values=proof.get("input_sha256")
    if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],str) or len(values[0])!=64:
        raise EvidenceError("INVALID_V3_IDENTITY","V3结果缺少唯一输入哈希证明")
    return values[0]

def extract(report,reason=None):
    version=version_of(report) if report else None
    reason=reason or (None if version else "unverified_scoring_version")
    result={}
    for key,names in REGIONS.items():
        mid={"pores":"01","spots":"03","surface_gloss":"02"}[key]
        module=(report or {}).get("modules",{}).get(mid,{})
        def get(region):
            if reason:return missing(reason)
            if key=="pores":
                node=module if region=="full_face" else module.get("regions",{}).get(region)
            else:
                node=module.get("metrics",{}).get(region,{}).get(key)
            return score_node(node)
        result[key]={**get("full_face"),"name":NAMES[key],"score_view":"front",
            "score_direction":"higher_is_better","scoring_version":version,"score_basis":BASES[key],
            "regional_scores":{"score_view":"front","score_direction":"higher_is_better",
                "region_schema":"v3_native","region_side_convention":"image_left_right",
                "scoring_version":version,
                "items":[dict(region=r,name=LABELS[r],**get(r)) for r in names]}}
    return result

def load_bundle(path):
    path=Path(path)
    manifest=read(path/"bundle.json")
    if manifest.get("schema_version")!="saved_v3_source_v1":raise EvidenceError("INVALID_V3_SOURCE","无效V3证据包")
    image=within(path,manifest["front_image"])
    report_path=within(path,"report.json")
    if sha(image)!=manifest["front_sha256"] or sha(report_path)!=manifest["report_sha256"]:
        raise EvidenceError("SOURCE_HASH_MISMATCH","V3证据文件哈希已变化")
    report=read(report_path)
    if report_input_hash(report)!=manifest["front_sha256"]:
        raise EvidenceError("V3_INPUT_MISMATCH","V3结果与输入图片不匹配")
    return report,manifest
