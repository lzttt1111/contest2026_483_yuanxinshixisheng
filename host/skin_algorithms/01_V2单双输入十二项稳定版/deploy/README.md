> 2026-09-14：增加云端单图／四光源自动分流。原任务可选传入 `capture_images`：1张走consumer，4张RGB/PP/CP/UV走institution；原单图参数与返回合同保持兼容。详见[请求示例、模态分工及验收说明](../docs/CLOUD_FOUR_LIGHT_INPUT_20260914.md)。上线需重启对应Worker并由后端传完整图片列表；不需要修改前端结果字段。

# 单RGB十二项 Worker 部署与并行运行

更新时间：2026-08-20

## 一、直接结论

生产环境只使用一个模板：`deploy/dermavision@.service`。

模板可启动三套互斥的11进程集合，并保持dev原默认target名称：

- `dermavision-workers.target`：**默认V2**，使用`dermavision@acne_v2`。
- `dermavision-workers-v1.target`：显式回滚profile，只把上一项替换为
  `dermavision@acne`，其他10个实例不变。
- `dermavision-consumer-workers.target`：普通用户RGB profile，使用11个
  `consumer_*`实例；与前两套互斥。

任意两个target **不得同时启用或启动**，避免两种成像Profile或acne v1/v2同时占用模型资源。
每套检测集合均为11个独立 Celery Worker：

```text
redness / spots / brown / texture / pores / purple / acne（或 acne_v2）/ wrinkle
surface_gloss / vascular / contour_firmness
```

其中 `purple` 同时生成 UV 色斑和紫质，因此是“11个任务、12项结果”。
11个检测进程均为 `concurrency=1`，能够并行消费后端同时提交的11个任务；
没有新增检测 Task，也没有改变旧 Task、Queue、参数或返回信封。
生产默认不生成 Word；acne v2 也不改变该默认值。

默认 `dermavision-workers.target` 在上述11个检测实例之外，额外启动第12个
**独立 CPU 聚合实例** `dermavision@summary`（Queue `summary`，Task
`dermavision.aggregate_summary`，`concurrency=2`）。它不加载任何模型、不使用
GPU、不生成 Word，只消费后端在检测终态后派发的总体评分请求。详见第七节。
`dermavision-workers-v1.target` 与 `dermavision-consumer-workers.target`
本阶段**不包含** summary，`consumer_summary` 队列只声明不部署。

## 二、首次安装

发布门禁：`uv lock --check` 与 `uv sync --frozen` 必须先同时通过。
当前工作树的 `uv.lock` 仍受契约仓读权限/精确Git对象阻断；未解决前禁止
部署，也不得手工伪造lock更新。

厂家RED/BROWN sidecar也属于强制前置条件。当前项目仓库已随附
`00_runtime_assets/vendor_skin`完整8文件闭包和manifest；从项目环境拉取后无需另行补包或设置环境变量。先校验：

```bash
sha256sum 00_runtime_assets/vendor_skin/pyz/core/skin_generate.pyc
# fd590298569d80ef6b3dfd3d7043c653715b12fddb5f41dbf1fd0ff8417a1581
```

如运维选择仓外覆盖路径，额外执行：

```bash
sha256sum "${DERMAVISION_VENDOR_SKIN_MODULES_ROOT}/core/skin_generate.pyc"
```

路径、8文件闭包或SHA不匹配时禁止启动；不得删除校验或改为全盘搜索。只有运维明确把sidecar复制到仓外只读目录时，才在`.env`设置`DERMAVISION_VENDOR_SKIN_MODULES_ROOT`和`DERMAVISION_VENDOR_SKIN_MODULE_SHA256`覆盖默认路径。正式Word所需旧V0.1.1配置也已随仓位于`00_runtime_assets/scoring_v011`，缺失或SHA不符会在Word生成前失败关闭。

```bash
cd /opt/dermavision
uv lock --check
uv sync --frozen

sudo cp deploy/dermavision@.service /etc/systemd/system/
sudo cp deploy/dermavision-workers.target /etc/systemd/system/
sudo cp deploy/dermavision-workers-v1.target /etc/systemd/system/
sudo cp deploy/dermavision-consumer-workers.target /etc/systemd/system/
sudo systemctl daemon-reload

# 只有完成contracts -> backend -> worker发布门禁后，才启用默认V2
sudo systemctl enable --now dermavision-workers.target
```

如果服务器过去安装过已废弃的 `aisia-skin-worker@.service`，应先停止旧实例，
避免同一队列被重复消费：

```bash
sudo systemctl disable --now 'aisia-skin-worker@*' 2>/dev/null || true
```

## 三、启动、停止与重启

默认V2沿用dev原target名称和启停命令：

```bash
sudo systemctl start dermavision-workers.target
sudo systemctl stop dermavision-workers.target
sudo systemctl restart dermavision-workers.target
```

显式acne v1回滚profile：

```bash
sudo systemctl start dermavision-workers-v1.target
sudo systemctl stop dermavision-workers-v1.target
sudo systemctl restart dermavision-workers-v1.target
```

普通用户consumer profile：

```bash
sudo systemctl start dermavision-consumer-workers.target
sudo systemctl stop dermavision-consumer-workers.target
sudo systemctl restart dermavision-consumer-workers.target
```

consumer不调用厂家RED/BROWN sidecar；Profile由Queue显式决定，禁止按图片像素自动推断。同一GPU只允许启动一个聚合target。

从v1回滚profile切回默认v2时，先确认契约仓与后端已部署，然后执行：

```bash
sudo systemctl disable --now dermavision-workers-v1.target
sudo systemctl stop dermavision@acne.service
sudo systemctl enable --now dermavision-workers.target
```

`stop` 旧 target 未必会停止其依赖，所以必须显式停止
`dermavision@acne.service`。切换后应确认旧 target 是 disabled/inactive，
且 `dermavision@acne.service` 不在运行。

回滚到acne v1：

```bash
sudo systemctl disable --now dermavision-workers.target
sudo systemctl stop dermavision@acne_v2.service
# 先将backend投递切回 acne / acne.analyze_image，再执行：
sudo systemctl enable --now dermavision-workers-v1.target
```

完整发布顺序只能是
`aisia-contracts -> ai-skin-backend -> DermaVision Worker`。回滚时必须先停
v2 Worker，再切后端路由，最后启动v1聚合 target；已发布的v1契约不修改。

单独管理某一项：

```bash
sudo systemctl start dermavision@acne
sudo systemctl start dermavision@acne_v2
sudo systemctl restart dermavision@wrinkle
sudo systemctl stop dermavision@purple
```

查看状态和日志：

```bash
systemctl status dermavision-workers.target
systemctl status dermavision-workers-v1.target
systemctl status 'dermavision@*'
journalctl -u dermavision@acne -f
journalctl -u dermavision@acne_v2 -f
journalctl -u dermavision@wrinkle -f
journalctl -u 'dermavision@*' --since today
```

## 四、11个检测任务与12项结果（另加1个CPU聚合实例）

下表同时列出两个 profile；`dermavision@acne` 与
`dermavision@acne_v2` 是二选一，所以每个profile仍为11个检测任务；
`dermavision@summary` 是独立的 CPU 聚合实例，不是检测任务。

| systemd实例 | Celery App | Queue | Task | 检测结果 |
|---|---|---|---|---|
| `dermavision@redness` | `src.worker` | `redness` | `dermavision.analyze_image` | 红区 |
| `dermavision@spots` | `src.worker` | `spots` | `dermavision.analyze_image` | 斑点 |
| `dermavision@brown` | `src.worker` | `brown` | `dermavision.analyze_image` | 棕区 |
| `dermavision@texture` | `src.worker` | `texture` | `dermavision.analyze_image` | 纹理 |
| `dermavision@pores` | `src.worker` | `pores` | `dermavision.analyze_image` | 毛孔 |
| `dermavision@purple` | `src.worker` | `purple` | `dermavision.analyze_image` | UV色斑、紫质 |
| `dermavision@acne` | `src.acne.worker:celery_app` | `acne` | `acne.analyze_image` | 痤疮 |
| `dermavision@acne_v2` | `src.acne.worker:celery_app` | `acne_v2` | `dermavision.analyze_image` | 痤疮v2 |
| `dermavision@wrinkle` | `src.wrinkle.worker` | `wrinkle` | `wrinkle.analyze_image` | 皱纹 |
| `dermavision@surface_gloss` | `src.worker` | `surface_gloss` | `dermavision.analyze_image` | 油光 |
| `dermavision@vascular` | `src.worker` | `vascular` | `dermavision.analyze_image` | 血管样结构 |
| `dermavision@contour_firmness` | `src.worker` | `contour_firmness` | `dermavision.analyze_image` | 轮廓紧致度 |
| `dermavision@summary` | `src.summary.worker:celery_app` | `summary` | `dermavision.aggregate_summary` | 总体评分聚合（CPU，非检测结果） |

`dermavision@summary` 不产出检测项，因此不计入“12项结果”，也不进入
`NINE_ANALYSIS_ITEMS`；它是默认V2 target 的第12个进程，仅在
`dermavision-workers.target` 中作为依赖启动。

## 五、后端如何并行

后端对同一张图片创建11个不同 `record_id`，并同时向当前选中的
target 所对应11个 Queue
发送既有四参数任务。Celery 的11个常驻进程并行处理，各项独立成功或失败。
只有 `purple` 在一个任务里返回两个检测结果。

acne v1 的 Queue/Task 仍为 `acne`/`acne.analyze_image`；acne v2 的
Queue/Task 为 `acne_v2`/`dermavision.analyze_image`。后端必须按合同版本只投递
其中一个 Queue。

acne v2 `raw_result` 只允许
`overlay/metrics/quality_score/quality_status/quality_flags`。旧 acne v1 和其他旧任务
不得因v2切换而删字段、改名或改类型。

不要把十二项改成一个串行云端任务，也不要把九个 DermaVision queue 全部交给一个
`concurrency=1` 进程。模型在各自 Worker 子进程中仅冷启动一次，后续任务热启动复用。

## 六、启动前静态核对

不连接 Redis 即可检查每个实例实际解析到的 App、Queue 和 Task：

```bash
for target in redness spots brown texture pores purple acne acne_v2 wrinkle surface_gloss vascular contour_firmness summary; do
  uv run --frozen python run_cloud_worker.py --target "$target" --print-command
done
```

consumer静态核对：

```bash
for target in consumer_redness consumer_spots consumer_brown consumer_texture consumer_pores consumer_purple consumer_acne_v2 consumer_wrinkle consumer_surface_gloss consumer_vascular consumer_contour_firmness; do
  uv run --frozen python run_cloud_worker.py --target "$target" --print-command
done
```

必须确认 acne v1 与 acne v2 都显示 `src.acne.worker:celery_app`（但 Queue/Task 不同），
皱纹显示 `src.wrinkle.worker`。三者任一映射不符时禁止切换。
`summary` 必须显示 `src.summary.worker:celery_app`、Queue `summary`、
Task `dermavision.aggregate_summary`、`concurrency=2`，且该静态核对不加载
cv2/mediapipe/`src.worker` 等重型模块。

## 七、独立 CPU 总体评分聚合（summary）

`dermavision@summary` 消费后端在检测终态后派发的
`dermavision.aggregate_summary` 任务，把各算法持久化的 `scoring_input` 汇聚成
11 模块总体评分（`word_display` + `production_proxy_v1`）。

### 7.1 启动

```bash
# 随默认V2一起启动：
sudo systemctl enable --now dermavision-workers.target
# 单独管理：
sudo systemctl status dermavision@summary
journalctl -u dermavision@summary -f
```

启动前静态核对：

```bash
uv run --frozen python run_cloud_worker.py --target summary --print-command
```

### 7.2 CPU-only / 零模型声明

- App `src.summary.worker:celery_app` 的 import 链只含 celery、pydantic、
  settings 与 release 钉值；检测 `src.worker`/`src.pipeline`、torch、
  ultralytics、mediapipe、Word 渲染均**不在启动链**中（评分链在任务体内惰性加载）。
- 不读 GPU、不下载图片、不做推理、不生成 Word/CSV/图片；只读仓内冻结评分资产。
- `consumer_summary` 队列已声明但本阶段不部署。

### 7.3 scoring_release 与发布前置

- 本端 release id（`src/summary/release.py` 的 `RELEASE_ID`）：
  `word_v0.1.1+proxy_v1+fieldlist_v3@dermavision-0.2-four-light`
- 后端 `Settings.overall_summary_scoring_release` 必须配置为同一值；请求中的
  `scoring_release` 与之不一致时任务失败关闭（不悄悄换新评分）。
- 本轮整合保留最新dev的 `pyproject.toml`/`uv.lock`，真实契约固定为
  `9a0b9265d4622ac60c4d73fcc71219f230b05f78`（0.2.0.dev1），并已完成本地导入验证。
  后端和Worker须协调升级scoring_release与字段清单，旧输入不得混入新release。
  后续正式契约tag发布序列（独立部署任务，不是本轮已执行操作）：
  1. `aisia-contracts` 打 tag `v0.2.0` 并推送；
  2. `dermavision` 与 `ai-skin-backend` 的 `pyproject.toml` 锁
     `aisia-contracts @ v0.2.0`；
  3. 两仓 `uv sync` 生成/更新 `uv.lock`；
  4. 部署 backend 与 worker（先 contract 后两消费方）。

### 7.4 资产钉值与漂移门禁

`RELEASE_ID` 绑定的 5 个资产在每次任务评分前重新计算 SHA256，任一不符即
`status=failed / code=scoring_asset_mismatch`（完整 64 位值见
`src/summary/release.py`）：

| 资产 | SHA256（前缀） |
|---|---|
| `scoring_input_field_list_v3.json` | `e4f4cc69…ff` |
| `v2_explicit_formula_registry_v1.json` | `9ac2d51d…5c27` |
| `metric_registry_v2_proxy_20260728.json` | `5a6231ea…dbeb7` |
| `word_population_reference_1000.json` | `a6933505…894ea` |
| `全量历史ECDF评分配置.json` | `8cf292c2…5b808` |

### 7.5 实测资源（夹具级，测量环境=本机 CPU，无 GPU 参与）

夹具 `tests/fixtures/scoring/clinic28-25_r1.json`（consumer profile，11 算法全
`present`），直接调用 task 函数，不经 broker：

| 指标 | 值 |
|---|---|
| 应用启动（仅 import app）峰值 RSS | ~41.7 MB |
| 首次任务后峰值 RSS | ~289.4 MB |
| 稳定（二次任务）峰值 RSS | ~289.4 MB |
| `load_release` 资产校验 | 0.012 s |
| 评分资产首次加载（ECDF 11.3MB + Word profile 4.1MB） | 0.30 s |
| 端到端 cold（含惰性 import） | 0.37 s |
| 端到端 warm | 0.078 s |
| 请求消息字节 | 15033 B |
| 结果消息字节 | 6538 B |

数值随机器与文件缓存波动，仅作部署配置依据；不承诺固定小于 10 秒。

### 7.6 原图派生门禁 `input_quality_gate`（检测 Worker 产出，summary 消费）

- 正式评分要求 V0.1.1 原图派生门禁。检测 Worker（根/acne/wrinkle）在下载原图后、
  清理中间文件前，调用 `src.scoring_input.compute_input_quality_gate`，内部真实
  复用 V0.1.1 `evaluate_image_path`（同一函数、同一判定），把 `{status, reason_codes}`
  写入 `raw_result.scoring_input.quality.input_quality_gate` 随信封持久化。
- 该门禁为 **CPU-only**：使用仓内 `models/face_landmarker.task` 与
  `models/selfie_segmenter.tflite` 的 MediaPipe TFLite，不占 GPU、不引入 torch；
  仅对单算法信封任务各执行一次。模型缺失或函数抛错时门禁诚实置 `None`，聚合侧降级
  REVIEW，绝不伪造 PASS。
- summary 侧只消费，不重算：从各 `scoring_input` 提取门禁，要求所有携带者
  `{status, reason_codes}` 一致；缺失/不一致同样降级 REVIEW，不挑选、不拼接。
- 门禁结果仅映射契约 `InputQualityGate` 的 `status`/`reason_codes`；检测 Worker 自身的
  `quality_status/quality_flags` 不参与门禁判定。

## 八、目录定位

```text
src/worker.py          单RGB DermaVision九队列Worker + 目标注册表
src/summary/           独立CPU总体评分聚合app（worker/release/request/result）
src/acne/worker.py     痤疮Worker与算法代码
src/wrinkle/worker.py  皱纹Worker与算法代码
models/acne/           痤疮模型
models/wrinkle/        皱纹模型
run_cloud_worker.py    12实例统一启动器（11检测 + 1 CPU聚合）
deploy/                唯一systemd模板与v1/v2/consumer互斥target
```

完整字段含义见：
`docs/单RGB十二项运行与云端增量对接说明_20260820.md`；旧v1字段追溯见
`docs/九项云端Worker字段合同与前端对接说明-0805.md`。
