> 2026-09-20公开提交复核：请先阅读仓库 `contest_submission/skin-algorithms.md`。下文为历史快照，不能替代本次复核；水光固定样例分数回退未修复、未完成新参考标定，公开包不含真人测试照片。

# DermaVision V2 单RGB与四光源十二项检测

本模块提供普通用户单RGB和机构四光源两种输入方式，统一完成十二项皮肤图像检测、量化结果导出、十一模块评分以及可选报告生成，同时提供Celery Worker和CPU评分汇总服务。

## 主要能力

- consumer：单张RGB照片。
- institution：RGB、偏振、交叉偏振和UV四光源采集。
- 十二项结果：红区、可见色斑、Brown、纹理、毛孔、UV色斑、紫质、痤疮、皱纹、表面油光、血管样结构、轮廓紧致度。
- 输出检测图、简版CSV、医学V2 CSV、公开JSON、完整量化JSON、索引和运行回执。
- 可显式生成用户精简版和医生详细版Word。
- 云端为11个检测任务、12项结果；CPU summary独立汇总十一模块评分。

## 文档入口

- [完整技术与部署说明](V2单双输入十二项技术与部署说明_20260915.md)
- [Worker接口合同](WORKER_CONTRACT.md)
- [云端Worker部署](deploy/README.md)
- [双Profile说明](docs/机构与普通用户双Profile运行与部署说明_20260824.md)
- [评分输入字段矩阵](docs/十二项评分输入字段证据矩阵_20260912.md)
- [医学报告可选模块](docs/医学报告可选模块说明.md)

## 快速安装

```bash
uv python install 3.10
uv sync --frozen
python3 scripts/verify_submission_snapshot.py
```

本提交已附带精确版本的 `aisia-contracts` 源码，依赖从 `vendor/aisia-contracts` 安装，不需要访问专用源码服务器。

## 普通单RGB

```bash
uv run --frozen python run.py --input-dir /data/phone-rgb \
  --output-dir /data/results/consumer --capture-profile consumer --algorithms all
```

## 机构四光源

```bash
uv run --frozen python run.py --capture-manifest /data/clinic/capture_manifest.json \
  --capture-alias case-001 --output-dir /data/results/institution \
  --capture-profile institution --stop-on-error
```

四光源采集必须具有完整且可核验的RGB、PP、CP、UV角色，不能用同一图片替代不同光源。Word默认关闭，需要时显式增加 `--generate-medical-report`。

## 使用边界

输出为图像外观检测、量化和评分结果，不直接形成医疗诊断、注射点、治疗深度或剂量建议。正式部署还需配置对象存储、Redis、后端服务、数据权限和目标机器GPU环境。
