# VelaVision

2026-09-20 皮肤算法补充：V2十二项/四光源、水光三项云服务、PC目标检测与拍照源码见 [皮肤算法提交说明](contest_submission/skin-algorithms.md)，包含视觉Skill和 lzttt1111 的真实AI日志。水光固定样例分数回退存在未修复问题，详见该说明的发布复核。

统一开发入口：`E:\openvela\VelaVision`。原生 openvela/NuttX、RK3576 BSP、视觉、无线、语音候选、前端与 AI 日志均在本仓库组织。

## 当前状态

**当前整机主线（2026-09-15，覆盖下方历史快照）：**自然聊天暂停；photo-upload 固件已通过 OTG 在 RAM 启动，App 配网后 IP 为 10.3.0.214，相机枚举、协商及保持停止的视频会话通过。26条阶段提示已在 Windows 网关预热，TTS 16 dB、短音24 dB。真实语音首轮上传150帧但识别为空，未启动云台，已恢复软件停止。正在构建真实命令提示音修复；三角度拍摄、阶段播报与真实后端受理尚未完成整机验收。后端开发 consentEvidenceRef 按用户要求使用 dev-consent。详见 [当前整机接续](docs/三角度姿态提示与上传接续_20260915.md)。

**2026-09-15 最新覆盖：**视觉/蓝牙分核版已上板，App 配网成功，40秒视觉/BLE/Wi-Fi 共存检查通过。真实后端设备登录、SSE中文问答和回答转云TTS已通过；原生 `chat-loop` 候选正在构建，尚未上板或真人连续对话验收。详见 [板端自由对话接续](docs/板端自由对话接入_20260915.md)。下面保留各阶段历史。

**整机整合最新方向（2026-09-15）：** 暂沿用 Windows 阿里云语音服务，整机控制向 openvela 板端迁移；host/whole_device 仅为测试参考。板端单轮命令固件已RAM加载，阶段TTS人工实听通过；连续轮次、RAM三图及阶段播报新候选已构建但未上板，相机尚未枚举。完整闭环未完成，见 [整机板端整合接续](docs/整机板端整合接续_20260915.md)。

**2026-09-15 最新覆盖：** BLE App 已配网，K7 IP `10.3.0.214`；分核固件已 RAM 运行。一次板端实时上传150帧、云端completed通过，前一轮仍失败，尚未验证真实人声或稳定性。TTS此前用户确认清楚；业务后端认证及聊天合同未提供，完整多轮对话未完成。详见 [板端联调记录](docs/板端云语音桥联调_20260914.md)。以下为历史阶段记录。

**最新流式ASR：**用户提供的阿里云 speech-service 已增加真实双向 WebSocket ASR。合成语音测试首个非空文字640ms、整句4281ms，TTS1219ms；11项隔离回归通过。新radio-status固件已ARM64构建，但K7仍待OTG恢复及麦克风WebSocket接入。见 [流式接入](docs/speech-service流式ASR接入_20260914.md)。

**2026-09-14 18:36 运行时接线候选：**`k7radio` 新增加锁只读的 DHCP 链路快照（单调 generation、保守 Wi-Fi/IPv4 状态和地址），云语音新增默认关闭的 radio bridge。断链、generation 回退/同代矛盾及非法 IPv4 均 fail closed，并清除旧 DNS/TLS/API 门；适配器只设置 Wi-Fi/IPv4 两门，其余六门不伪造。O0/O2 全套新增 radio bridge 6 组并通过，runtime sync 严格编译通过。该增量尚未重新做 ARM64 整包构建或上板，见 `evidence/k7cloud-radio-readiness-20260914/host-verification.json`。

**2026-09-14 18:23 推进：**云语音候选已增加离线 `k7cloud` CLI 和独立 `velavision_cloud_speech_local` 配置；22 个精确文件经 SDK 保护同步后检查为零差异，全新 ARM64 构建完成，MiMo/编排关键符号进入 ELF。`nuttx.bin` 为 2,778,832 字节，SHA256 `865390c1e9080fd22ca722f63cdd49f07146a87e1df998c86e39d95d42375ef7`。全量 SDK 检查仍只因未覆盖的 `voicelink/native_asr_runtime.cpp` 未知差异而拒绝。当前 CLI 仍不联网，真实 radio/DHCP、熵源、可信时间、CA、TCP、API 门禁与 Token 注入未接，尚未上板。证据见 `evidence/build/ble-wifi-cloud-speech-orchestrator-20260914/verification.json`。

**2026-09-14 18:10 接力入口：**已生成面向下一位模型的完整交接，冻结当前 BLE 手输密码→本地联网提示→八门禁→MiMo 云 ASR/TTS→业务后端主线，记录当前 RAM 板端状态、源码入口、测试、音频合同、OTG 恢复、禁止事项和前 15 分钟执行顺序。AI 日志部分明确了三个项目主会话、三个 hook 会话、Stop hook/30 秒 watcher、脱敏、手动刷新与原版校验流程；实时数量始终以日志 manifest 为准。见 [模型接力交接](docs/模型接力交接_20260914.md)。

**2026-09-14 17:58 更新：**产品配网主线已改为前端加密 BLE 扫网、选择 SSID 并手动输入密码；语音不再负责 Wi-Fi 密码。当前 RAM 镜像的无线四段和固件 CRC、NSH 启动通过，BLE host 与 Wi-Fi 共享服务已恢复，`k7voice probe` 返回 `native_network=1`；本轮尚未由手机重新提交网络，因此 Wi-Fi 当前未连接，本地 ASR 也因完整 encoder CRC 未复验而保持不可用。已生成 BLE→DHCP→联网回报→业务后端→云语音的前端适配器、当前 Swagger 的业务客户端、主机 MiMo ASR/TTS 桥和板端八门禁云语音编排候选。前端 26 项、主机语音桥 16 项、板端 DNS/HTTPS/MiMo/编排 O0/O2 回归均通过。业务 OpenAPI 当前没有 ASR/TTS 路径，设备会话凭据尚未提供，板端候选尚未编入新镜像或上板；完整闭环还未验收。说明见 [蓝牙配网与云语音整机接入](docs/蓝牙配网与云语音整机接入_20260914.md)，交付包为 `deliveries/整机蓝牙联网云端接口_20260914.zip`。

**2026-09-14 13:10 更新：**缓存式板端 TTS 冷启动和热启动均已实播，但保留 TTS Session 后创建 ASR Session 在 NuttX 系统堆的首个 960 字节分配处触发 data abort；当时独立 512 MiB 模型池仍有约 505 MiB，不能把故障归因于模型池容量。主线已改为预生成男声提示音 + 缓存在线 Paraformer ASR：20 条 16 kHz 提示音归一化到 90% 峰值，板端以单声道 S16 直接喂 SAI，DAC 0 dB 衰减且不分配临时播放缓冲。O0/O2 FIFO/边界回归通过；新固件 SHA256 `e9f48b37b769aa2ee6221ec344e37c242418ef2a35444a3252140afe9eef207f`，117,122,920 字节 RAM-only 单包 SHA256 `3018c2f0769faff89a7f32f50e0152f07ba372224eadb1fe6d69ebaacfbf3172`，距审计上限余 317,592 字节。Ubuntu 可用空间已从 4.6 GiB 恢复到 33 GiB。当前 VMware CH340 节点可见但打开返回 I/O error，新固件尚未上板，不能称提示音、ASR 或完整语音配网已验收。见 [离线 TTS 与语音配网](docs/离线TTS语音配网接入_20260913.md)。

**2026-09-14 09:37 更新：**修复版已通过一次 OTG 单包在 RAM 启动，Sherpa 已越过 VITS 文件路径校验并进入注册的 512 MiB 模型池，但 ONNX Session 初始化仍返回 `std::bad_alloc`；失败后模型池完整恢复到 536,870,144 字节，无泄漏。下一候选已把分配记录容量从 2048 扩到审计允许的 4096，并加入首次失败的精确统计；固件、ELF 和 116,525,024 字节 RAM-only 单包校验通过。当前 VMware 内 CH340 节点打开返回 I/O error，待串口重新挂接后可直接加载，不需重编或重传模型分段。见 [离线 TTS 与语音配网](docs/离线TTS语音配网接入_20260913.md)。

**2026-09-13 21:18 更新：**首个离线 TTS 固件已 RAM 启动并通过全部模型 CRC，但 VITS Session 因仍走系统堆而 `std::bad_alloc`。现已完成 ROMFS 直接模型读取、TTS ORT 模型池注册、ASR 512 MiB 边界重编和 ASR/TTS 生命周期共享锁；最终固件、ELF 及 116,524,928 字节 RAM-only OTG 单包审计通过。Ubuntu 的 CH340 节点当前打开返回 I/O error，因此修复版尚未上板，不能称 TTS 或全语音配网已通过。见 [离线 TTS 与语音配网](docs/离线TTS语音配网接入_20260913.md)。

**2026-09-13 18:00 更新：**按用户新要求已将唤醒规则改为‘你好’前缀，并保留‘开始联网’；20 组主机流程测试、独立 ARM64/ELF、OTG RAM 加载通过，提示音沿用已实听的最大档位。最新实录 ASR 输出‘哦’，未触发扫描，完整语音配网仍未通过。见 [中文前缀唤醒](docs/中文前缀语音唤醒_20260913.md)。

**2026-09-13 17:37 更新：**播放声道宏参数的优先级错误已修复，独立 ARM64 固件已 RAM 启动，寄存器恢复单路播放，用户确认短音重新可听。提示音复测使用用户指定的最大档位。无线共享服务已恢复；麦克风本轮识别出文字，但唤醒词识别错误，尚未进入语音扫网或完成 Wi-Fi 连接。见 [喇叭播放声道修复](docs/喇叭播放声道回归修复_20260913.md)。

**2026-09-13 17:11 更新：**OTG 通过板端软件复位恢复，尾段搬运后复位、CRC 复查和 Ubuntu 自动重连本轮通过。精确 `你好open` 唤醒兼容固件已通过 OTG 在 RAM 启动，完整模型及固件 CRC 通过，无线共享服务启动成功；正在进行实际麦克风唤醒与扫网测试，尚未完成 Wi-Fi 连接验收。联网后的云端播报按用户指定采用 `mimo-v2.5-tts`，接口待接入。见 [OTG 恢复与语音复测](docs/OTG恢复与语音复测_20260913.md)。下方尚未上板叙述为早前快照。

**2026-09-13：**规则语音配网的精确 `你好open` 唤醒兼容镜像已完成构建、ELF 审计和 116 MB OTG RAM 包校验；板端两个 ASR 模型暂存区 CRC 通过，当前停在 Fastboot，因 Windows VMware OTG 代理 Code 43 尚未加载。并行完成了正式 DNS/HTTPS 基础组件：主机 O0/O2 各 9 项 DNS 和 9 组 HTTPS 测试通过，独立 ARM64 固件编译链接及五个强符号门禁通过；尚未上板访问 DNS、HTTPS 或小米 API。见 [DNS 与 HTTPS 基础接入](docs/DNS与HTTPS基础接入_20260913.md) 与 `evidence/k7cloud-link-20260913/acceptance.json`。

**2026-09-12：**Ubuntu 虚拟机旧构建目录已先归档关键产物到 Windows E 盘，再删除 149 个可再生成目录；根分区从约 1.3 GiB 恢复到约 38 GiB 可用。正式 VoiceLink 已增加密码显式确认门，主机 O0/O2 19 项通过并编入 ARM64 固件。新固件已通过 OTG 在 RAM 启动，固定 WAV ASR 连续两轮成功且资源释放归零；按完整无线启动顺序后，正式 Controller 的真实扫描取得 `Lansee` 等 4 个去重网络并进入等待选择状态。麦克风 ASR 文字自动送入 Controller 和板载提示音尚未接通，因此完整语音配网仍未验收。见 [语音配网与虚拟机空间恢复](docs/语音配网与虚拟机空间恢复_20260912.md)。

**19:15 后更新：**audio-pause-20260911 已 RAM 运行；252 微秒暂停 RXDR 读取时四个 FIFO 均自行积累数据，音频问题继续定位。SDIO、无线 host 状态及 VoiceLink 真实扫描通过，当前未连接 Wi-Fi，未验收手机 GATT。诊断缓冲不可用于 ASR。见 docs/音频零样本定位_20260911.md。

**2026-09-11 最新主线：**voice-tls 已通过 OTG 在 RAM 启动，连续两个独立任务完成固定录音 ASR（42.250/42.074 秒），线程隔离、析构与模型池释放检查通过；随后麦克风 ASR 也完成且没有重启，但将“开始联网”识别为“泰车联网开车”，准确性仍未通过。周期性零采样尚未修复，语音配网尚未完成，当前无线未验收。见 docs/ASR重复调用TLS修复_20260911.md。

**近期优先级已调整为本地语音配网，联网后再接小米云端模型。** `app/voicelink` 已迁入异步控制核心并增加异步扫描接口，O0/O2各14个文字流程通过；音频和真实共享扫描后端尚未集成，不能直接对板子语音配网。离线llama模型不再前置，本地ASR/TTS库与模型仍需要准备。见 [本地语音配网实施](docs/本地语音配网优先实施_20260910.md)。

**板载MIC录音→板载喇叭回放基础功能已获实听确认，音质仍待修正。** `audio-input-20260910` 的PGA24回放用户确认“人声更大，但杂声仍明显”。后续 `audio-filter-20260910` 已通过板端200ms采集与一次MA4滤波回放、停止及恢复；没有人工实听，周期性零采样和正确16k格式尚未验收。同镜像串口私密提交完成WPA2/DHCP，网关前后各5/5；Windows BLE连接报设备未找到，不能称BLE回归通过。见 [音频滤波与夜间验证](docs/音频滤波与夜间验证_20260910.md) 与 [板载录音回放](docs/板载录音回放操作_20260910.md)。

- **eMMC：**原生低速初始化、单块重复读取、主备 GPT 和分区项 CRC 已真机通过；容量 61079552×512 字节，15 个现有分区。`emmc-block-20260910` 已真机注册16个只读块节点，块接口检查通过；普通文件打开因 BCH 未启用而返回 ENXIO。`emmc-vfs-20260910` 已 RAM 上板，BCH只读文件接口连续64KiB和跨扇区非对齐读取命令返回0。文件系统及模型文件尚未接入。没有格式化或写盘。见 [eMMC 接入](docs/eMMC原生只读接入_20260910.md)。
- **无线：**原生 BLE 真配网、WPA2/DHCP 与 IP 事件已接通。最后 VFS 版本恢复后网关 5/5、30秒保持通过，测试客户端主动断开；该轮结束时 Wi-Fi 在线、BLE客户端主动断开；最新恢复验证见 `evidence/smp-eight-20260910/recovery-acceptance.json`：真实WPA2/DHCP、30秒保持、网关5/5，本轮A-MSDU拒收0；历史异常仍保留。历史 reason 8 断连与长稳边界保留，不声称生产级稳定。见 `evidence/emmc-vfs-20260910/`。
- **DDR/CPU：**独立 1GiB CPU 模型池已在 model-arena 版本完成 384MiB 两轮稀疏读写及释放验证；后续镜像未重复该测试。系统堆仍约126MiB，此前无线版本为单核；最新独立诊断已验证四颗A53加四颗A72；未验证全部4GiB或八核外设并发。
- **视觉与 NPU：**保留已确认的人脸跟随、三视角拍照和固定 INT8 矩阵基线；完整 NPU 模型、仪器检测与整机联合验收未完成。当前任务不启动云台。
- **语音：**主机候选、共享 Wi-Fi broker 候选和 Windows 真实 ASR/TTS 模型运行已交接并核对哈希；未移植实际推理库或音频驱动到板端。语音编排、板载音频资料和codec候选共358份交付文件哈希已核对；完整codec录音配置仍待核实。见 [交接记录](docs/VoiceLink交接接收与集成顺序_20260910.md)。
- **Agent：**近期先完成本地规则语音配网，再接小米云端模型与板端工具校验。已完成的原生CPU llama.cpp基础作为后续离线增强保留，不再作为近期交付前提；完整Agent未部署。历史选型见 `docs/离线Agent选型与B0落地方案_20260910.md`。

最后已验证镜像、正在构建/加载的镜像分别记录在 `project-manifest.json` 的 `current_device`、`pending_firmware`；实测只对相应版本有效。历史叙述完整保存在 [README 历史快照](docs/README历史快照_20260910_块设备前.md)。

## 开发位置

## 比赛交付归类

本项目按 openvela AI 硬件开发者大赛要求保留真实构建路径，并用 `contest_submission/` 提供不破坏构建的交付索引：

- `contest_submission/README.md`：比赛交付入口、linkfile 映射和提交边界。
- `contest_submission/source-map.md`：源码按板级适配、应用、主机联调、测试证据和 AI 材料分类。
- `contest_submission/submission-checklist.md`：官方要求对应的提交前检查表。
- `contest_submission/submission-manifest.json`：可审计的 include/exclude、linkfile 和未完成项边界。

源码不复制到第二套目录，避免破坏 `contest2026_483_yuanxinshixisheng.xml` 的 `<linkfile>` 和构建入口。`private/`、历史归档和未完成候选只保留追溯用途，不进入公开提交包。

| 内容 | 位置 |
| --- | --- |
| 视觉、云台、NPU | `app/k7host/`、`app/gimbal/`、`app/k7npu/` |
| 无线、DDR、eMMC | `app/k7radio/`、`app/k7mem/`、`app/k7emmc/` |
| BSP 与构建配置 | `port/`、`board/kickpi_k7/` |
| 前端、主机、STM32 | `frontend/`、`host/`、`mcu/` |
| 未集成语音候选 | `work-in-progress/parallel-voicelink*`、`parallel-wifi-service` |
| 旧 PC 交付与证据 | `legacy/`、`baselines/`、`evidence/` |
| AI 日志 | **`logs/maomaojiang01/`** |

Ubuntu `/home/swl/openvela` 是外部 SDK，项目副本 `/home/swl/openvela/work/velavision-project`。先用 `tools/sync_sdk.py --check` 核对再同步；未知改动必须处理，不静默覆盖。独立版本构建记录在 `evidence/build/<revision>/verification.json`。已完成产物不覆盖；初始 `tools/build.sh` 不能代替最新版本构建记录。全新环境完整复现尚未验收。

## 日志与交付

队伍 `contest2026_483_yuanxinshixisheng`，GitHub 用户名 **maomaojiang01**。日志范围包含明确选定的旧主会话、当前主会话、VoiceLink 并行主会话及已采集的日志配置会话。`logs/maomaojiang01/manifest.json` 是数量、文件与哈希入口；按真实北京时间分日、跨日序号连续，保留脱敏、原生导出不完整及失败记录。

运行 `python tools/export_project_logs.py` 刷新显式会话，使用原版 `python tools/official-validator/tools/validate-log.py logs/` 校验。格式通过不代表比赛验收、完整无遗漏或已经上传。详见 [代码日志对应](docs/代码日志对应表.md) 与 [自动采集](docs/自动日志采集.md)。

Git 分支 `dev-ai-contest-2026`，来源为用户指定的 allenxun fork；当前本地整理未提交、未推送。最终正式源码、构建证据与日志统一交付，不把暂存候选或 Windows 运行结果冒充板端功能。
