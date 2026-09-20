from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("left", "front", "right")
VIEW_LABELS = {"left": "左侧", "front": "正面", "right": "右侧"}
MODULES = {
 "pores": "毛孔", "texture": "表面纹理", "spots": "可见色斑",
 "brown": "棕色特征", "redness": "泛红", "vascular": "线状红纹",
 "acne": "疑似痤疮", "surface_gloss": "表面油光",
}
REGIONS = {
 "forehead":"额部", "nose":"鼻部", "left_eye":"左眼周", "right_eye":"右眼周",
 "left_cheek":"左面颊", "right_cheek":"右面颊", "perioral":"口周", "chin":"下巴",
}
COLORS = {"pores":(170,50,145),"texture":(215,125,20),"spots":(210,130,30),
 "brown":(30,120,210),"redness":(40,50,230),"vascular":(20,100,225),
 "acne":(210,70,200),"surface_gloss":(20,195,220)}

