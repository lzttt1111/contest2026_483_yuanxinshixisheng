"""DermaVision 配置 — 从环境变量 / .env 读取。

优先级：环境变量 > .env 文件 > 默认值。
与 ai-skin-backend / face-algorithm 共用同一套 Redis 与 OSS 凭证。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.capture_profile import CaptureProfile
from src.version import ALGORITHM_VERSION


class Settings(BaseSettings):
    """DermaVision 运行配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # 算法版本：默认取 src/version.py 常量，部署时可经 env ALGORITHM_VERSION 覆盖
    algorithm_version: str = ALGORITHM_VERSION
    capture_profile: CaptureProfile = Field(
        default=CaptureProfile.CONSUMER,
        validation_alias="DERMAVISION_CAPTURE_PROFILE",
    )

    # Celery / Redis（与 ai-skin-backend 共用：broker db0 / backend db1）
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"
    # 五个独立算法任务需要至少五个 Celery 执行槽才能真正并行。
    # 可由环境变量 CELERY_WORKER_CONCURRENCY 覆盖。
    celery_worker_concurrency: int = Field(default=5, ge=1, le=64)

    # backend 地址，如 http://backend:8000；worker 经 internal-api 上传/下载图片
    # （OSS 凭证仅保留于 backend，worker 不直连 OSS）
    picture_service_base_url: str = ""

    # 医学 V2 是对旧 raw_result 的可选旁路扩展。当前默认开启；紧急
    # 回滚时可通过环境变量关闭，旧 metrics 始终保持原样。
    enable_medical_metrics_v2: bool = True

    # Worker 超时（秒）：dermavision 预处理 + 红区 + 色斑约 5~20s，留余量
    task_soft_time_limit: int = 120
    task_time_limit: int = 150


settings = Settings()
