# -*- coding: utf-8 -*-
"""
DermaVision FastAPI 网关 — 本地调试用 HTTP 接口
════════════════════════════════════════════════════
注意：生产链路为 ai-skin-backend → Celery(dermavision.analyze_image)，
本网关仅供本地调试，直接调用 Pipeline（不走 Celery / OSS），
结果图通过 /static 静态文件服务返回。

启动命令：
  uv run uvicorn src.api_server:app --host 0.0.0.0 --port 8000
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles

from src.pipeline import DermaVisionPipeline

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="DermaVision API", description="医疗级面部皮肤检测服务（本地调试）")

# 挂载 output 目录为静态文件服务，让结果图片可通过 URL 直接访问
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

_pipeline = DermaVisionPipeline()


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    algorithms: str = Form("redness,spots"),
):
    """上传图片并同步执行皮肤检测（本地调试，不走 Celery / OSS）。

    Args:
        file: 用户上传的面部照片
        algorithms: 需要执行的算法，逗号分隔。支持 redness、spots、
            brown、texture、pores，例如 "redness,spots,brown"

    Returns:
        {"status": "success", "results": {"preprocessed": "/static/...", "redness": "/static/..."}}
    """
    algo_list = [a.strip() for a in algorithms.split(",") if a.strip()]
    if not algo_list:
        raise HTTPException(status_code=400, detail="algorithms 不能为空")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(status_code=400, detail="仅支持 JPG/PNG 格式")

    saved_path = os.path.join(UPLOAD_DIR, f"upload_{file.filename}")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = _pipeline.process_single(saved_path, algo_list)
    if result.get("status") != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "分析失败"))

    url_results = {}
    for key, path in result.get("results", {}).items():
        rel_path = os.path.relpath(path, OUTPUT_DIR)
        url_results[key] = f"/static/{rel_path}"
    return {"status": "success", "results": url_results}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "DermaVision API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
