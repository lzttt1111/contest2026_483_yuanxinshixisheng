# VelaVision 比赛交付索引

新增软件模块入口：[皮肤算法、采集源码与AI材料（2026-09-20）](skin-algorithms.md)。其中明确记录当前评分正确性、图片公开范围和部署验收边界。

本目录是 `远信实习生` 队伍的比赛交付入口。它只保存归类、提交边界和复现入口，不复制源码；真实构建路径仍由仓库根目录的 `contest2026_483_yuanxinshixisheng.xml` 和 `project-manifest.json` 管理。

## 交付内容

| 比赛材料 | 仓库位置 | 说明 |
| --- | --- | --- |
| 板级适配源码 | `board/kickpi_k7/`、`port/` | KICKPI K7 板级初始化、配置和平台适配 |
| openvela 应用 | `app/k7host/`、`app/gimbal/`、`app/k7radio/`、`app/k7npu/`、`app/k7usb/`、`app/k7agent/` | 视频、人脸跟随、云台、无线、NPU、USB、语音和后端任务 |
| 主机联调 | `host/`、`frontend/` | 后端协议、报告查询、TTS、验收工具和联调界面 |
| MCU 配套 | `mcu/` | 云台或外设控制器配套代码 |
| 构建与复现 | `config/`、`tools/`、`patches/` | 配置、构建、RAM 加载、校验和诊断工具 |
| 测试与证据 | `tests/`、`evidence/`、`docs/` | 自动化测试、脱敏证据、适配说明和限制 |
| AI 开发材料 | `skills/k7-openvela-driver-port/`、`logs/maomaojiang01/` | 自建 Skill 与 AI Coding 日志 |
| 作品材料 | `deliveries/` | 作品介绍文档、演示视频和提交前材料 |

## 构建映射

官方 manifest 通过 `<linkfile>` 把以下目录映射到 openvela 工作区：

```text
app/k7host       -> apps/examples/k7host
app/gimbal       -> apps/examples/gimbal
app/k7npu        -> apps/examples/k7npu
app/k7radio      -> apps/examples/k7radio
app/k7usb        -> apps/examples/k7usb
board/kickpi_k7  -> nuttx/boards/arm64/rk3576/kickpi_k7
```

不要把源码复制到第二套目录；提交时以这些真实路径为准，避免 linkfile、构建脚本和日志中的路径失效。

## 提交边界

应进入专属 GitHub 仓库的内容：源码、板级适配、defconfig、构建与复现说明、脱敏测试证据、`logs/maomaojiang01/` 和至少一个有效 Skill。

只作本地保护或历史归档的内容：`private/`、`legacy/`、`baselines/`、`vm-archive/`、`work-in-progress/`、未脱敏原图、设备 token、Wi-Fi 密码、临时网关输出和未完成候选。它们不应被打包上传，也不能作为当前版本通过证据。

当前已知边界：两小时 BLE/Wi-Fi/视频共存、十轮连续闭环和独立异常恢复仍未形成通过证据；后端报告讲解流的 mock 模式不能当作正式 AI 讲解。提交材料必须保留这些限制。

## 复现入口

```bash
# 在 Ubuntu SDK 工作区执行，先检查 SDK 差异
python tools/sync_sdk.py --check

# 使用项目构建入口
bash tools/build.sh /home/swl/openvela

# 刷新并校验 AI Coding 日志
python tools/export_project_logs.py
python tools/official-validator/tools/validate-log.py logs/
```

构建哈希、板端加载状态和真机观察结果分别放在 `evidence/build/`、对应版本证据目录和 `docs/` 中；编译通过、RAM 加载和真机验证必须分开表述。
