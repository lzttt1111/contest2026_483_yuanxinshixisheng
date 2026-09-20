# 技术交接：电脑基线到RK3576

> 归档说明：本文是2026-09-07稳定快照的原始技术交接。2026-09-15参赛源码的安装、清单和验收入口以根目录 `README.md`、`目标检测与自动拍照技术及部署说明_20260915.md`、`pyproject.toml` 和 `uv.lock` 为准；本文保留用于追溯算法和硬件边界。

## 1. 目标及当前结论

负责人目标：在独立RK3576板卡上实现旧仪器目标检测、人脸检测/锁定与角度估计、左右目标角度自动拍照；最终摄像头安装在云台上。电脑模拟先打通功能，再迁移板端。

截至本包生成：电脑端真实模型和网页可运行；自动拍照有用户实拍成功的历史，但后续反馈有角度跳动、难触发，因此新增滤波和可调参数，最新修复还需接收方实拍复测。不要将“自动测试通过”等价为所有实际场景稳定。

RK3576未连接验收；没有可直接运行的RK3576 RKNN模型、完整板端三功能二进制或云台协议实现。`rk3568_reference`仅是旧板已有工作保留，不代表3576已完成。

## 2. 电脑处理链

```text
Windows Edge/Chrome摄像头
  → 原始方向视频帧 → JPEG quality=0.90（最多1请求在途）
  → 本机127.0.0.1:8876 → WSL FastAPI，单操作会话、全局推理互斥
      ├─ face模式：EXIF校正 → YuNet → 空间锁定 → 扩大人脸ROI
      │            → FSA-Net → raw_pose → 中值+EMA → pose
      │            → 质量/角度/连续门禁 → 当前原始帧复核 → JPEG+JSON落盘
      └─ device模式：V5 YOLOX-S → NMS最高候选 → V5接触点
                   + MediaPipe468关键点 → 七区标签图 → 接触点落区
  → JSON → 未镜像坐标绘制；可选只镜像显示
```

模型执行位置：电脑WSL。仪器/接触点优先ORT CUDA，允许CPU回退并公开实际provider；YuNet、FSA-Net、MediaPipe目前CPU。网页不做神经网络推理，不发送云端。

## 3. 关键文件职责

| 文件 | 职责 |
|---|---|
| `pc_demo/server.py` | HTTP生命周期、会话、互斥、JPEG验证、时钟偏移估计、结果组织、照片和元数据保存 |
| `pc_demo/lab/backend.py` | 五个模型启动校验、实例化、设备/接触点预热、health信息 |
| `pc_demo/lab/state.py` | FaceLock、PoseFilter、CaptureGate；端侧移植必须对齐这些行为 |
| `pc_demo/lab/regions.py` | 七区标签、额头几何扩展、排除区、轮廓输出和接触点归属 |
| `pc_demo/vendor/target_detection/runtime.py` | V5设备和接触点ORT推理、阈值、解码、版本元数据 |
| `pc_demo/vendor/target_detection/preprocessing.py` | EXIF、letterbox、ROI及归一化 |
| `pc_demo/vendor/head_pose/face_detector.py` | YuNet检测和ROI裁剪 |
| `pc_demo/vendor/head_pose/preprocess.py` | FSA-Net BGR、64×64、NCHW及归一化 |
| `pc_demo/web/app.js` | 摄像头、单请求流、模式切换、会话参数、绘制及过期清空 |
| `pc_demo/web/angle-display.mjs` | 只限制数字展示频率，不改变推理频率 |
| `pc_demo/web/index.html / style.css` | 最新页面、拍照参数卡片、七区面板、响应式布局 |

## 4. 模型契约：禁止混淆模型输入或版本

### 4.1 PC设备V5

- `pc_demo/models/device/model.onnx`，V5 REAL-ONLY YOLOX-S，FP32。
- 输入BGR、NCHW `[1,3,640,640]`，左上对齐letterbox、填充值114；不是随意改成居中letterbox或RGB。
- PC ONNX输出已解码检测项；`runtime.py`按xywh/objectness/class概率处理，置信度0.25、NMS IoU0.65，当前只保留最高设备框。
- SHA256：`218e922d784795901bd5e165e11755898f6c29c4a5724715862aa8cdf1919a8c`。

### 4.2 PC接触点V5

- `pc_demo/models/contact/model.onnx`，V5 MobileNetV3-Small，FP32。
- 设备框扩大20%，RGB `[1,3,256,256]`，ImageNet归一化。
- 输出 `[1,1,64,64]` 热图和 `[1,3]` 可见性；类别 visible、occluded_inferable、not_annotatable。
- 半径3峰值邻域softmax加权质心反映射到原图；not_annotatable清空点。occluded_inferable是推断点，不是实际可见的物理接触证据。
- SHA256：`e4d9deba2eb3aaa9ec143045de305f7be38480124ac3ef577ac5e0eec122dae1`。

### 4.3 人脸、角度、分区

- YuNet：`face_detection_yunet.onnx`，检测器640×640 letterbox、置信度0.7；默认人脸ROI为max(width,height)×1.35，上移0.04×side。
- FSA-Net：`head_pose_fsanet_1x1.onnx`，输入 `[1,3,64,64]`，BGR不换RGB，`(pixel-127.5)/128`，输出 `[yaw,pitch,roll]`，度数，不是弧度。
- FSA-Net SHA256：`120fa107a2dd3be78c21c3a73a0db980590643a5372893e8878094898262f213`。不要把文档里的上游原始模型哈希误当本包已处理ONNX哈希。
- YuNet SHA256：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`。
- MediaPipe：`face_landmarker.task`，使用前468点，CPU；SHA256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。
- `.task`是MediaPipe模型包，不能作为普通ONNX直接交给RKNN转换；需独立选择板端实现或替代模型，并重新验证。

所有PC模型哈希在 `pc_demo/models/SHA256.json`，启动逐项校验；device/contact还各有详细manifest。模型文件随包传输，但默认不应直接提交普通Git仓库，使用组织允许的制品存储/LFS策略。

## 5. 拍照状态机及已修复问题

### 5.1 人脸锁定

首次选主要人脸，后续按IoU≥0.30匹配。短暂完全漏检在350ms内可以恢复原空间匹配，但漏检帧不拍照并清空滤波/稳定计时。超过350ms或匹配歧义会进入lost，需重置。空间连续性不能证明身份，存在交叉人物风险；不能改成“检测到任意脸就继承之前照片状态”。

### 5.2 稳定角度

`PoseFilter`：最近3帧中值，再按帧间时间用120ms时间常数EMA。推理不降频，前端每1000ms采样显示稳定值。无脸、无效数值、长间隔、过期、模式/参数重置会清空历史。

自动拍照连续门禁使用稳定值，但实际触发那帧的raw_pose还必须同侧、Yaw在容差内、Pitch/Roll在上限内。因此不会仅因滤波仍停在45°就保存实际已转至80°的图。保存元数据含两套角度，便于审计。

滤波引入响应延迟，缓解高频抖动而非模型精度改进。回放曾测相邻Yaw变化中位数2.10°→0.28°，采用近似33ms回放，不是摄像头端到端验收结果。

### 5.3 默认门禁与可调范围

默认：目标45°±5°；pitch/roll绝对值≤12°；稳定≥500ms且至少3帧；Yaw相对稳定起点变化≤3°；人脸最短边≥80px；160×160 ROI拉普拉斯方差≥60；帧龄≤700ms、有效间隔≤350ms。丢失、模糊、过期等会清空累计。

前端可调目标5–80°、容差1–20°且小于目标且总和≤89°、pitch/roll上限5–45°、稳定300–3000ms。后端独立校验，不信任前端范围。应用参数原子更新会话门禁、清空滤波/本轮照片状态；不删除已保存文件。刷新后回到默认值，尚无用户设置持久化。

未开放“关闭所有门禁”的按钮。CPU慢速导致帧间隔超限时，应该先评估性能和流水策略，不直接改成必拍。

## 6. 七区的实现与限制

七区：forehead、eye_area、nose、subject_left_cheek、subject_right_cheek、mouth_area、chin。内部最长边640的单标签图，覆盖连续、无重叠；眼睛开口及嘴唇剔除。上额头沿脸轴最多扩脸高20%，太阳穴渐弱；来自旧脸周扩张思路，并非神经网络皮肤分割。

分区版本 `face-regions-7-v2`。接触点和着色来自同一标签图，边界附近返回unknown；脸外outside_face，眼唇excluded；无点/无脸/多人/几何侧脸保护时unknown。无历史分区缓存，不在丢脸后继续显示旧分区。

对角度极大、遮挡、额头高低、刘海和深浅肤色未系统验收。画面区域只能作为后续皮肤算法ROI定义的工程基础，不承担诊断/治疗推荐。接入皮肤算法必须确保同一frame_id、同一原图坐标、输入分辨率与模型版本，另建指标/建议契约，不把示例颜色解释为皮肤状态。

## 7. HTTP接口（当前实现，不是旧板接口的直接复刻）

服务只绑定127.0.0.1:8876，同源静态页，无云端依赖，无跨网认证和生产访问控制。

| 方法与路径 | 输入 / 输出 |
|---|---|
| GET `/health` | 模型哈希、真实providers、model_lineage、默认config、PC_SIMULATION |
| POST `/v1/session` | 无body；返回随机session_id，新会话使旧会话失效 |
| POST `/v1/session/{sid}/settings` | JSON：yaw_target/yaw_tolerance/pitch_limit/roll_limit/stable_ms，五字段必须齐全；返回实际config |
| POST `/v1/session/{sid}/reset?side=all` | all/left/right；清空对应状态，保留磁盘照片 |
| POST `/v1/stream/frame` | 原始JPEG body；头X-Session-Id、X-Mode=face/device、X-Frame-Id、X-Capture-Ms、X-Frame-Age-Ms |
| GET `/v1/photos/{sid}/{filename}` | 会话目录内生成的JPEG或JSON，不允许任意路径 |
| GET `/v1/debug/recent` | 内存中最近300帧数值诊断，非图像；本机调试用 |

frame_id严格递增；capture_ms来自浏览器performance.now，单位毫秒。服务器估计时钟偏移判断额外传输延迟，不要求Windows与WSL时钟相同，但不能精确测量第一帧绝对网络延迟。JPEG≤5MiB、≤1600万像素，校验格式及解码；错误返回400，失效会话/忙返回409，模型异常500。最多一个推理任务，不建立无界帧队列。

face返回：face、pose（稳定）、raw_pose、decision（progress/reason/trigger）、saved、photos、age_ms、timings_ms。device返回：device_bbox(x1,y1,x2,y2)、device_score、contact_point(x,y)、可见性、contact_region、face_regions、模型版本哈希和耗时。坐标为EXIF校正后的未镜像原图像素。

注意：当前PC没有实现旧计划的 `/v1/infer/image` 和 `/v1/stream/reset` 路径，不能用旧接口测试直接判定本包失败。迁移时如需统一，应增加明确兼容适配和契约测试，不偷偷替换客户端路径。

## 8. RK3568历史参考的价值和不能复用的部分

来源工作树基线提交 `e8798373ad89af17f622f10ebc421a74d7a02b2c`，但rk3568_deploy目录包含未提交快照，因此以本包逐文件SHA256为准。

包含：独立C++14 backend/pipeline/service/probe；ONNX导出与RKNN转换脚本；校准/对拍/稳定性脚本；配置与部署参考；两个opset12 ONNX；历史对拍JSON和挂起观察。没有第三方SDK、编译器、sysroot、librknnrt、RGA库、板端二进制、校准图、checkpoint或RK3568专用RKNN文件。

尤其重要：板端导出设备ONNX是**原始三尺度输出**：stride8 `[1,6,80,80]`、stride16 `[1,6,40,40]`、stride32 `[1,6,20,20]`，网格/stride/exp等在C++解码；不可直接放入PC现有的已解码输出处理器。参考 `artifacts/onnx/export_manifest.json`。

历史旧板：Runtime/Toolkit2 1.5.2、Buildroot、RK3568。历史README记录100帧5.492FPS、P95往返192.205ms；设备单独约6.6FPS低于原8FPS目标；30分钟测试约10分钟出现HTTP与ADB shell无响应，原因未定。对拍通过不代表稳定性通过，更不代表RK3576性能。

`rk3568_reference/deploy`脚本有实际推送/停止进程等副作用和旧路径。默认只读参考，**禁止未核对目标就运行**。1.5.2、旧目标平台、工具链和旧板库不要原样用于RK3576。

官方项目列出RK3576支持，但具体版本必须由接收方按实际板卡SDK/驱动选择并锁定：[RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2/blob/master/README.md)、[发布记录](https://github.com/airockchip/rknn-toolkit2/releases)。本包不猜测维护人员板卡的驱动或运行库版本。

## 9. 维护人员自动化开发工具下一步执行顺序与验收

### P0：接手完整性和环境

确认包哈希、Windows/WSL名称、Ubuntu22.04、Python3.10、GPU、摄像头、可下载权限。先在包内独立环境跑13项回归和真实模型烟测；记录失败原文。补全clean-install依赖锁，区别原作者环境快照和本机验证结果。

### P1：电脑实拍闭环

单人正面到左右目标角，至少两套参数（如30°±5°及45°±5°）；验证原始/稳定角度、同帧保存、每侧一次、重拍、设置变更、新会话。验证遮挡、抖动、失锁、无脸、过期、模糊、多人切换。核对左右符号，记录俯仰偏置。不能以模拟模型或调到极宽阈值代替实拍通过。

旧仪器同时入镜，检查框/点/分区与镜像；找边界和眼唇排除区验证unknown/excluded；记录100帧吞吐、阶段耗时、P95和CPU/GPU占用。需要个人图像时先征得维护人员同意，不默认持续保存原始视频。

### P2：RK3576环境盘点

板卡接入后先只读记录型号、系统、架构、内核、NPU驱动、librknnrt、SDK、内存、存储、摄像头接口、云台协议、电源和散热；不要刷固件、改系统分区或更换驱动作为第一步。取得明确部署目录及设备访问授权。

### P3：模型迁移与对拍

创建独立RK3576工作区/转换环境，不污染PC环境。根据官方兼容矩阵选Toolkit2与Runtime，显式指定RK3576目标；先FP16正确性、后训练集校准INT8。设备/接触点优先从本包opset12 ONNX开始；YuNet/FSA-Net/MediaPipe逐个检查算子与输出契约，不能假定都已支持。

需要训练数据、校准图或checkpoint时向项目方申请；本包没有。历史50张金标准/100张训练校准图也没包含，不能声称直接复现历史对拍。若替换人脸关键点模型，必须重新映射索引和分区，不能继续硬套468索引。

### P4：板端服务与云台

板端C++接口与坐标契约保持可追溯。HTTP输入可先复用Windows摄像头+ADB链路，之后才换板载摄像头。迁移PoseFilter/CaptureGate、同帧保存、过期清空、参数校验，新增板端版本/provider/哈希。

云台单独抽象控制器：默认dry-run；确认协议、方向、机械限位、速度、停止/急停、丢脸停机、曝光和转动完成信号。模型Yaw是相对相机视角，不等同于电机编码器角。不要把头部Yaw直接不加限制地发给云台。实际转动需用户授权并确保设备安全。

### P5：正式验收

- 正确性：同一金标准对比PC/板端框、点、角度、区域及拍照决策；误差门禁事先约定，不临时调整过关。
- 初始参考指标：设备FP16 IoU≥0.98，INT8≥0.95且召回下降≤2个百分点；接触点归一化误差FP16≤2%、INT8≤4%，可见性一致率≥98%/95%。这是计划门禁，不是3576已达标。
- 角度/拍照新增标定图或人工复核标准，特别验证左右45°、俯仰偏差、滤波延迟和错拍率。
- 性能：100帧+30分钟持续运行；端到端延迟、实际FPS、内存、温度、NPU占用、请求积压和故障恢复。原参考目标设备8FPS、完整链路5FPS、P95≤250ms，但新增模型后需明确并发模式和采样策略。
- 完全离线运行、无需电脑执行模型，实际provider必须RKNN_NPU（混合CPU算子必须如实说明）。停止电脑推理仍能完成板卡功能才算板端闭环。

## 10. 发布纪律

先保留V0.0.1为交接基线，不给未验收板端打正式发布标签。后续改动记录原因、文件、模型哈希、测试、执行provider和未完成事项。禁止推送原作者非公开源码仓库，除非维护人员明确提供目标并授权。

最终对外发布另需许可证/模型授权复核、可复现依赖锁、使用手册和验收报告；不把个人照片、测试账号或环境目录放进发布包。
