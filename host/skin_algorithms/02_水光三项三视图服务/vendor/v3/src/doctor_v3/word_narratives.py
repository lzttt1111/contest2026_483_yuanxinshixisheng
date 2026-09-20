"""Data-driven replacements for the physician's regional/result sentences."""
from .word_slots import ROWS,metric,score_text,grade_text
from .word_tables import description
GROUPS={
"双侧鼻旁":("left_nasal","right_nasal"),"内侧面颊":("left_inner_cheek","right_inner_cheek"),
"外侧面颊":("left_outer_cheek","right_outer_cheek"),"双侧眼下":("left_under_eye","right_under_eye"),
"双侧颧部":("left_zygoma","right_zygoma"),"双侧面颊":("left_cheek","right_cheek"),
"双侧下颌":("left_jaw","right_jaw"),"额部中央":("forehead",),"额部":("forehead",),
"鼻部":("nose",),"下巴":("chin",),"口周":("perioral",),"右眼下":("right_under_eye",),
"左眼下":("left_under_eye",),"右侧下脸":("right_lower_face",),"左侧下脸":("left_lower_face",)
}
GROUPS.update({
 "双侧木偶纹":("left_marionette","right_marionette"),
 "双侧法令纹":("left_nasolabial","right_nasolabial"),
 "双侧泪沟":("left_tear_trough","right_tear_trough"),
 "口角周围":("left_mouth_corner","right_mouth_corner"),
 "左侧法令纹":("left_nasolabial",),"右侧法令纹":("right_nasolabial",),
 "左侧颊中沟":("left_midcheek",),"右侧颊中沟":("right_midcheek",),
 "左侧面颊":("left_cheek",),"右侧面颊":("right_cheek",),
 "右侧颧部和右鼻旁":("right_zygoma","right_nasal"),
 "左侧颧部和左鼻旁":("left_zygoma","left_nasal"),
 "额部和下颌":("forehead","jaw"),"面颊":("left_cheek","right_cheek")})
def region_sentence(model,module,text):
    from .registry import REGION_LABELS
    normalized=text.lstrip("• ").removeprefix("目前")
    for label,regions in sorted(GROUPS.items(),key=lambda x:-len(x[0])):
        if not normalized.startswith(label):continue
        parts=[]
        for r in regions:
            value=metric(model,module,"_module",r)
            parts.append(REGION_LABELS.get(r,r)+"："+(grade_text(value) if value.get("score") is not None else score_text(value)))
        return "；".join(parts)+"。"
    return None

def mentioned_metric(model,module,text):
    candidates=[]
    for name,aliases in ROWS[module].items():
        for alias in aliases:
            if len(alias)>=4 and alias in text:candidates.append((len(alias),name,alias))
    if not candidates:return None
    _,name,alias=max(candidates)
    value=metric(model,module,name,"full_face")
    return alias+"状态评分为"+score_text(value)+"，"+grade_text(value)+"。"

def oil_combination(model):
    a=metric(model,"02","surface_gloss","full_face")
    b=metric(model,"02","porphyrin","full_face")
    return "表面油光为"+grade_text(a)+"，毛囊卟啉为"+grade_text(b)+"。"
