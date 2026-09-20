# =======================================================
# DermaVision Celery Worker Dockerfile
# 基于 python:3.10.20-slim + uv 极速依赖安装
# =======================================================

FROM python:3.10.20-slim

# ---------- 1. 系统依赖 ----------
# libgl1 + libglib2.0-0: OpenCV 运行时依赖
# 替换 Debian 源为阿里云镜像加速
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tzdata \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- 2. 安装 uv ----------
# 用 pip 从阿里云 PyPI 安装 uv，避免拉 ghcr.io 镜像（国内极慢）
RUN pip install uv==0.11.27 -i https://mirrors.aliyun.com/pypi/simple/

# ---------- 3. 工作目录 ----------
WORKDIR /app
ENV TZ=Asia/Shanghai
ENV PYTORCH_NVML_BASED_CUDA_CHECK=1
# 依赖装到 .venv，把 .venv/bin 加到 PATH 让 CMD 的 celery 可直接解析
ENV PATH="/app/.venv/bin:$PATH"

# ---------- 4. 依赖安装 (利用 Docker 缓存层) ----------
# 先只拷贝依赖定义文件，利用缓存加速重建
COPY pyproject.toml uv.lock ./

# uv 安装依赖到 .venv (--frozen: 严格按 uv.lock, --no-dev: 不装开发依赖, --no-install-project: 只装依赖)
RUN uv sync --frozen --no-dev --no-install-project

# ---------- 5. 拷贝项目代码 ----------
COPY . .

# ---------- 6. 默认启动命令 ----------
# 配置（CELERY_*/OSS_*）通过运行时环境变量或挂载 .env 注入，不在镜像内硬编码。
CMD ["celery", "-A", "src.worker", "worker", "--loglevel=info", "--queues=redness,spots,brown,texture,pores,purple"]
