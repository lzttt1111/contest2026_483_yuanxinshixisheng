"""Camera routes are registered separately from the existing analysis workflow."""
from fastapi import Request,HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import re

def register(app,service):
    @app.post("/api/capture-sessions")
    async def create(request:Request):
        data=await request.json()
        if not isinstance(data,dict) or set(data)-{"config"}:raise HTTPException(400,"采集请求无效")
        return await run_in_threadpool(service.create,data.get("config",{}))
    @app.get("/api/capture-sessions/{sid}")
    def get(sid:str,request:Request):return service.get(sid,request.headers.get("x-capture-owner"))
    @app.post("/api/capture-sessions/{sid}/actions")
    async def action(sid:str,request:Request):
        data=await request.json()
        if not isinstance(data,dict):raise HTTPException(400,"采集请求无效")
        return await run_in_threadpool(service.action,sid,request.headers.get("x-capture-owner"),data)
    @app.post("/api/capture-sessions/{sid}/frames")
    async def frame(sid:str,request:Request):
        if request.headers.get("content-type","").split(";")[0]!="image/jpeg":raise HTTPException(415,"需要JPEG视频帧")
        chunks=bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks)>5*1024**2:raise HTTPException(413,"视频帧超过5MB")
        return await run_in_threadpool(service.frame,sid,request.headers.get("x-capture-owner"),dict(request.headers),bytes(chunks))
    @app.get("/api/capture-sessions/{sid}/photos/{name}")
    def photo(sid:str,name:str):
        if not re.fullmatch(r"(left|front|right)_r\d+_\d+_[a-f0-9]{8}\.jpg",name):raise HTTPException(404)
        folder=service.folder(sid);path=folder/name
        if not path.is_file() or path.is_symlink():raise HTTPException(404)
        return FileResponse(path)

