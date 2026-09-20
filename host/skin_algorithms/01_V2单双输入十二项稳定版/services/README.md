# 嵌套服务目录已停用

自 2026-08-05 起，痤疮和皱纹不再作为嵌套 Python 项目运行。

- 痤疮 Worker：`src/acne/worker.py`
- 皱纹 Worker：`src/wrinkle/worker.py`
- 八任务统一启动器：`run_cloud_worker.py`
- 八 Worker 并行部署：`deploy/README.md`
- 旧子项目版本记录：`docs/历史文档/来源快照/`

本目录仅保留该导航，不再包含独立 `pyproject.toml`、`uv.lock`、
Dockerfile 或同名 `src` 包，防止部署时误启动旧 Worker。
