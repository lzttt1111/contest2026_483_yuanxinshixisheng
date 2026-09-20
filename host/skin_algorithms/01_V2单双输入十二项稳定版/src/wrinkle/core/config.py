"""服务配置 — 从环境变量 / .env 读取。

优先级：环境变量 > .env 文件 > 默认值。
与 ai-skin-backend / face-algorithm 共用同一套 Redis 与 OSS 凭证。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.wrinkle.version import ALGORITHM_VERSION


class Settings(BaseSettings):
    """运行配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # 算法版本：默认取 src/version.py 常量，部署时可经 env ALGORITHM_VERSION 覆盖
    algorithm_version: str = ALGORITHM_VERSION

    # 服务名是已与后端对齐的公开合同，不再使用模板占位符。
    service_name: str = "wrinkle"

    # Celery / Redis（与 ai-skin-backend 共用：broker db0 / backend db1）
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"

    # backend 地址，如 http://backend:8000；worker 经 internal-api 上传/下载图片
    # （OSS 凭证仅保留于 backend，worker 不直连 OSS）
    picture_service_base_url: str = ""

    # 当前默认开启医学 V2 旁路字段；紧急回滚时可由环境变量关闭。
    enable_medical_metrics_v2: bool = True

    # Worker 超时（秒）
    task_soft_time_limit: int = 120
    task_time_limit: int = 150

    # Wrinkle detection algorithm
    wrinkle_run_preset: str = "balanced"
    wrinkle_cpu_threads: int = 6
    wrinkle_device: str = "cuda:0"
    wrinkle_gpu_memory_fraction: float = 0.29
    wrinkle_output_dir: str = "output"
    wrinkle_subprocess_timeout: int = 300


settings = Settings()
