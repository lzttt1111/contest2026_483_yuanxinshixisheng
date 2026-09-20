from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import Pipeline
from .runtime import V5Runtime
from .version import SOFTWARE_VERSION


def create_app(runtime: V5Runtime | None = None) -> FastAPI:
    app = FastAPI(title="Target Detection", version=SOFTWARE_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8090", "http://localhost:8090"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Frame-Id", "X-Timestamp-Ms"],
    )
    app.state.runtime = runtime
    app.state.lock = threading.RLock()
    app.state.last_result = None

    def current_runtime() -> V5Runtime:
        if app.state.runtime is None:
            app.state.runtime = Pipeline()
        return app.state.runtime

    @app.get("/health")
    def health() -> dict[str, Any]:
        engine = current_runtime()
        return {**engine.metadata(), "status": "ok", "npu_status": "not_applicable"}

    async def infer_request(request: Request, frame_id: int, timestamp_ms: int | None) -> dict[str, Any]:
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="request body must contain JPEG or PNG bytes")
        try:
            with app.state.lock:
                result = current_runtime().infer_bytes(payload, frame_id=frame_id, timestamp_ms=timestamp_ms)
                app.state.last_result = result
                return result
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/infer/image")
    async def infer_image(request: Request) -> dict[str, Any]:
        return await infer_request(request, frame_id=0, timestamp_ms=None)

    @app.post("/v1/stream/frame")
    async def stream_frame(
        request: Request,
        x_frame_id: int = Header(...),
        x_timestamp_ms: int | None = Header(default=None),
    ) -> dict[str, Any]:
        return await infer_request(request, frame_id=x_frame_id, timestamp_ms=x_timestamp_ms)

    @app.post("/v1/stream/reset")
    def stream_reset() -> dict[str, Any]:
        with app.state.lock:
            app.state.last_result = None
        return {"reset": True, "reset_at_ms": int(time.time() * 1000)}

    return app


app = create_app()
