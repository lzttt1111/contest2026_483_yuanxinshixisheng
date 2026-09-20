"""Worker configuration loaded from environment variables and optional .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.acne.version import ALGORITHM_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACNE_DATA_ROOT = PROJECT_ROOT / "data" / "acne"
ACNE_MODEL_ROOT = PROJECT_ROOT / "models" / "acne"


class Settings(BaseSettings):
    """Runtime settings for the Celery worker and local pipeline entrypoint."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # 算法版本：默认取 src/version.py 常量，部署时可经 env ALGORITHM_VERSION 覆盖
    algorithm_version: str = ALGORITHM_VERSION

    service_name: str = "acne"

    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"

    # backend 地址，如 http://backend:8000；worker 经 internal-api 上传/下载图片
    # （OSS 凭证仅保留于 backend，worker 不直连 OSS）
    picture_service_base_url: str = ""

    # 当前默认开启医学 V2 旁路字段；紧急回滚时可由环境变量关闭。
    enable_medical_metrics_v2: bool = True

    task_soft_time_limit: int = 600
    task_time_limit: int = 720

    detector_profiles_path: str = str(ACNE_DATA_ROOT / "configs" / "detector_profiles.json")
    detector_profile: str = "v3_balanced"
    detector_device: str = "cuda:0"
    acne_gpu_memory_fraction: float = 0.20
    worker_output_dir: str = str(PROJECT_ROOT / "output" / "acne" / "worker")

    enable_grading: bool = True
    grading_checkpoint: str = str(ACNE_MODEL_ROOT / "grading" / "acne_lds_drive" / "lds-weights" / "model_fold_0.pth")
    grading_python: str = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    grading_timeout: int = 300


settings = Settings()
