# -*- coding: utf-8 -*-
"""算法版本号。

算法（含模型权重 / 阈值 / 后处理逻辑）迭代时 bump 本常量；
也可在部署时通过环境变量 ALGORITHM_VERSION 覆盖（见 src/core/config.py）。
版本号会经 metadata.algorithm_version 回传给 ai-skin-backend，
写入 skin_tasks.algorithm_snapshot 供跨版本对比。
"""

ALGORITHM_VERSION = "1.0.0"
