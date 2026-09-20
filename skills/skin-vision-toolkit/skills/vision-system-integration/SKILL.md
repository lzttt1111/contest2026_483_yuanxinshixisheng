---
name: vision-system-integration
description: "Integrate visual-AI modules through explicit capture/job/frame/version, clock and coordinate contracts, then verify reproducible delivery. Not hardware control, model training or automatic clinical operation planning."
---

# 产品集成与可复现交付

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 明确每个模块唯一基线、入口、模型和实际接口，核对文档与代码，不将过时草案当现行合同。
2. 定义Session/Capture/Job/Frame、重拍revision、坐标与时钟域；同次业务关联不等于几何配准。
3. 核对模式切换、换人、失锁、换镜像/相机、重拍和过期结果的状态生命周期；局部失败保留其余真实结果。
4. 在隔离副本连接一次端到端场景，验证冷启动、失败/恢复、结果索引和无模型重建；交付最小源码/模型/配置闭包与接手文档。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

接口适配表、状态生命周期、端到端验收、版本/依赖清单和可复现交付包。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

不将旧照片像素点直接当新帧点；不因设备点落脸上宣称物理接触或操作安全。不要把整个虚拟环境、私人图片或无关模型打包。

## 离线辅助检查

脚本检查合同字段、revision、点迁移、时钟域与文档/实现状态冲突；registration_verified与clock_mapping_verified必须由上游误差验证支持，不可随意设true。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
