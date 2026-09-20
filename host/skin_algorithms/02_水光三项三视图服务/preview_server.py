"""Local read-only preview of package-local JSON and marked images."""
from pathlib import Path
from io import BytesIO
from functools import lru_cache
from PIL import Image,ImageOps
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response
import json

ROOT=Path(__file__).resolve().parent
WEB=ROOT/"preview"
EXAMPLE_ROOT=ROOT/"examples"/"current_demo"
DETECTION_ROOT=EXAMPLE_ROOT/"detections"
SAMPLES={
 "triplet":{"title":"当前水光三视图","description":"包内真实示例：三视图检测图、三项整体评分及V3原生分区评分。",
    "file":"examples/current_demo/result_internal.json",
    "images":{"left":"examples/current_demo/inputs/left.png","front":"examples/current_demo/inputs/front.png","right":"examples/current_demo/inputs/right.png"}}
}
app=FastAPI(docs_url=None,redoc_url=None)
def sample(key):
    if key not in SAMPLES:raise HTTPException(404)
    return SAMPLES[key]
@app.get("/")
def index():return FileResponse(WEB/"index.html")
@app.get("/assets/{name}")
def asset(name:str):
    if name not in ("app.js","style.css","layout_fix.css"):raise HTTPException(404)
    return FileResponse(WEB/name)
@app.get("/data/{key}")
def data(key:str):
    s=sample(key)
    return {"title":s["title"],"description":s["description"],
      "images":{view:f"/images/{key}/{view}" for view in s["images"]},
      "detections":{module:{view:f"/detections/{module}/{view}" for view in ("left","front","right")}
                    for module in ("pores","spots","surface_gloss")},
      "payload":json.loads((ROOT/s["file"]).read_text(encoding="utf-8"))}
@app.get("/detections/{module}/{view}")
def detection(module:str,view:str):
    if module not in ("pores","spots","surface_gloss") or view not in ("left","front","right"):
        raise HTTPException(404)
    path=(DETECTION_ROOT/module/(view+".png")).resolve()
    if not path.is_relative_to(DETECTION_ROOT.resolve()) or not path.is_file():raise HTTPException(404)
    return FileResponse(path,media_type="image/png")
@lru_cache(maxsize=8)
def thumbnail(path,mtime):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1100,1100))
        stream=BytesIO();im.save(stream,"JPEG",quality=90)
        return stream.getvalue()
@app.get("/images/{key}/{view}")
def image(key:str,view:str):
    s=sample(key)
    if view not in s["images"]:raise HTTPException(404)
    p=ROOT/s["images"][view]
    return Response(thumbnail(str(p),p.stat().st_mtime_ns),media_type="image/jpeg")
@app.get("/download/{key}")
def download(key:str):
    sample(key)
    document=json.loads((EXAMPLE_ROOT/"result.json").read_text(encoding="utf-8"))
    return Response(json.dumps(document,ensure_ascii=False,indent=2),
        media_type="application/json",headers={"Content-Disposition":"attachment; filename=skin-scores.json"})
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8894)
