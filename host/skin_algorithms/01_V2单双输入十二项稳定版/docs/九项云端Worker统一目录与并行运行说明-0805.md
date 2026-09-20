# 九项云端 Worker 统一目录与并行运行说明-0805

## 结论

本版将痤疮和皱纹从嵌套的子项目移入根项目的
`src/acne`和`src/wrinkle`。同事不再需要进入 `services/*`
寻找 Worker、模型或单独的锁文件。

八个 Worker 任务保持独立，同一张图由后端并发提交八个任务；
`purple` 一个任务产生 UV 色斑和紫质两项结果。

## 新目录

```text
src/worker.py             八任务唯一注册表，同时是DermaVision六队列Worker
src/acne/worker.py        痤疮Worker
src/acne/                 痤疮算法与合同
src/wrinkle/worker.py     皱纹Worker
src/wrinkle/              皱纹算法与合同
models/acne/              痤疮模型
models/wrinkle/           皱纹模型
data/acne/                痤疮配置
run_cloud_worker.py       八目标统一启动器
deploy/dermavision@.service
deploy/dermavision-workers.target
```

根 `pyproject.toml`和 `uv.lock` 是唯一生产环境。

## 八个任务

| target | App | Queue | Task | 结果 |
|---|---|---|---|---|
| redness | `src.worker` | redness | `dermavision.analyze_image` | 红区 |
| spots | `src.worker` | spots | `dermavision.analyze_image` | 斑点 |
| brown | `src.worker` | brown | `dermavision.analyze_image` | 棕区 |
| texture | `src.worker` | texture | `dermavision.analyze_image` | 纹理 |
| pores | `src.worker` | pores | `dermavision.analyze_image` | 毛孔 |
| purple | `src.worker` | purple | `dermavision.analyze_image` | UV色斑+紫质 |
| acne | `src.acne.worker` | acne-detection-worker | `acne-detection-worker.analyze_image` | 痤疮 |
| wrinkle | `src.wrinkle.worker` | wrinkle | `wrinkle.analyze_image` | 皱纹 |

## 并行运行

安装后一次启动八个实例：

```bash
sudo systemctl enable --now dermavision-workers.target
```

其等价于同时启动：

```bash
sudo systemctl start dermavision@redness
sudo systemctl start dermavision@spots
sudo systemctl start dermavision@brown
sudo systemctl start dermavision@texture
sudo systemctl start dermavision@pores
sudo systemctl start dermavision@purple
sudo systemctl start dermavision@acne
sudo systemctl start dermavision@wrinkle
```

每个实例 `concurrency=1`。后端为同一图片并发发送八条消息，
不必等前一项完成。八项分别返回五字段信封，一项失败不改变其他任务的返回。

## 不变的对接合同

- 四参数顺序不变。
- Task、Queue和`meta_data.version="1"`不变。
- 成功五字段信封和失败结构不变。
- 痤疮17个OSS字段+10个结构化字段不变。
- 皱纹13个OSS字段+6个结构化字段不变。
- `run.py` 不因云端目录整理而改变用法。

字段细节见 `docs/九项云端Worker字段合同与前端对接说明-0805.md`，
运维命令见 `deploy/README.md`。
