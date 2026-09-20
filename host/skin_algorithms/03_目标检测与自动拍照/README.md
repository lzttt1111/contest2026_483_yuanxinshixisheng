> 2026-09-20公开提交复核：请先阅读仓库 `contest_submission/skin-algorithms.md`。下文为历史快照，不能替代本次复核；水光固定样例分数回退未修复、未完成新参考标定，公开包不含真人测试照片。

# 目标检测与自动拍照

本模块提供电脑端摄像头采集、人物姿态引导、自动拍照以及旧仪器目标辅助检测能力。当前版本为PC功能基线，适用于Windows浏览器与WSL2 Ubuntu 22.04组合环境。

## 功能

- YuNet人脸检测与空间连续性锁定。
- FSA-Net头部Yaw/Pitch/Roll估计。
- 角度、清晰度、帧龄、稳定时间和失锁门禁。
- 左右侧目标角度自动拍照及单侧重拍。
- 旧仪器目标框与接触点估计。
- MediaPipe人脸七区及接触点落区显示。
- 本地网页、HTTP接口、CPU/CUDA依赖组和板端迁移参考。

## 文档

- [技术与部署说明](目标检测与自动拍照技术及部署说明_20260915.md)
- [接口与实现说明](TECHNICAL_HANDOFF.md)
- [验证记录](VALIDATION.md)
- [模型清单](MODEL_MANIFEST.json)

## 安装与启动

```bash
bash setup_uv.sh cpu
LAB_RUNTIME_GROUP=cpu bash start_uv.sh
```

CUDA环境使用 `bash setup_uv.sh cuda` 和 `LAB_RUNTIME_GROUP=cuda bash start_uv.sh`。浏览器打开 `http://127.0.0.1:8876/`。

## 校验

```bash
python3 scripts/verify_submission.py
```

## 边界

PC端代码、模型和网页已形成可运行基线。RK3576模型转换、NPU执行、摄像头/云台驱动、openvela调用和实板长稳尚未完成，不能宣称为已发布板端固件。人脸空间锁定不是身份认证；二维接触点不是物理接触证明；本模块不输出治疗建议。
