"""backend internal-api 图片传输封装。

worker 上传/下载全部经 backend internal-api（picture_service_base_url），
OSS 凭证仅保留于 backend，worker 不再直连 OSS（oss2 依赖已随直连实现一并移除）。
"""

from __future__ import annotations

import logging

import requests

from src.core.config import settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """OSS 存储操作失败。"""

    pass


def upload_via_internal_api(local_path: str, report_id: str, algo_name: str, file_name: str) -> str:
    """调 backend internal-api 上传结果图，返回 object_key。"""
    url = f"{settings.picture_service_base_url.rstrip('/')}/internal-api/picture/upload"
    try:
        with open(local_path, "rb") as f:
            resp = requests.post(
                url,
                data={"report_id": report_id, "algo_name": algo_name, "file_name": file_name},
                files={"file": (file_name, f)},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["object_key"]
    except requests.RequestException as exc:
        # 带上 backend 返回的响应体:4xx 的 body 里有具体错误消息(如"非法的算法名")
        body = exc.response.text if exc.response is not None else ""
        raise StorageError(f"internal-api 上传失败: {exc} body={body}") from exc


def download_via_internal_api(oss_key: str) -> bytes:
    """调 backend internal-api 下载图片，返回 bytes。"""
    url = f"{settings.picture_service_base_url.rstrip('/')}/internal-api/picture/{oss_key}"
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        body = exc.response.text if exc.response is not None else ""
        raise StorageError(f"internal-api 下载失败: {exc} body={body}") from exc
