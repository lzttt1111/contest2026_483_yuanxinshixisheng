"""AISIA单RGB十二项云端模拟结果的本地验收网页与Swagger文档。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from cloud_contracts import WORKER_ENVELOPE_TYPES, contract_catalog
from cloud.review_sanitize import sanitize_review_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_ROOT = PROJECT_ROOT / "output" / "cloud-pydantic-review-visia2-20260805"
REVIEW_ROOT = Path(os.getenv("CLOUD_REVIEW_ROOT", str(DEFAULT_REVIEW_ROOT))).resolve()
BUNDLE_PATH = REVIEW_ROOT / "cloud_response_bundle.json"

app = FastAPI(
    title="AISIA 单RGB十二项云端 Worker Pydantic 合同",
    version="1",
    description=(
        "本服务仅用于本地验收正式 Worker 六字段信封、中文字段说明与模拟OSS结果。"
        "不冒充 ai-skin-backend 的 to_response 生产接口。"
    ),
)

if REVIEW_ROOT.is_dir():
    app.mount("/assets", StaticFiles(directory=REVIEW_ROOT), name="review-assets")


def _load_bundle() -> dict[str, Any]:
    if not BUNDLE_PATH.is_file():
        raise HTTPException(status_code=503, detail=f"验收结果尚未生成: {BUNDLE_PATH}")
    return sanitize_review_bundle(
        json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/review")


@app.get("/review", include_in_schema=False)
def review() -> RedirectResponse:
    index = REVIEW_ROOT / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail=f"验收网页尚未生成: {index}")
    return RedirectResponse(url="/assets/index.html")


@app.get("/api/health", summary="验收服务健康检查")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if BUNDLE_PATH.is_file() else "waiting",
        "review_root": REVIEW_ROOT.name,
        "bundle_exists": BUNDLE_PATH.is_file(),
    }


@app.get("/api/contracts", summary="查看当前Worker的Pydantic JSON Schema")
def contracts() -> dict[str, Any]:
    return contract_catalog()


@app.get("/api/results", summary="查看单RGB完整云端任务模拟结果")
def results() -> dict[str, Any]:
    return _load_bundle()


def _register_algorithm_route(algorithm: str) -> None:
    @app.get(
        f"/api/results/{algorithm}",
        response_model=None,
        summary=f"查看{algorithm} Worker六字段信封",
        operation_id=f"get_{algorithm}_worker_envelope",
    )
    def algorithm_result() -> dict[str, Any]:
        tasks = _load_bundle().get("tasks", {})
        if algorithm not in tasks:
            raise HTTPException(status_code=404, detail=f"没有{algorithm}模拟结果")
        return tasks[algorithm]["response"]


for _algorithm in WORKER_ENVELOPE_TYPES:
    _register_algorithm_route(_algorithm)
