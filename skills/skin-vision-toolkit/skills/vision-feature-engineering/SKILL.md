---
name: vision-feature-engineering
description: "Engineer or diagnose image detectors through input transforms, valid domain, candidate generation, filtering and instances. Use for visible false positives/negatives and mask/overlay discrepancies; not metric calibration or report restyling."
---

# 图像算法与有效域工程

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 锁定输入、基线和最小复现，不根据参考报告总数相同来调阈值。先区分显示变化与实际推理输入变化。
2. 逐层保存变换、前景/皮肤、排除域、候选和最终实例；统计各层面积与目标流失原因，核对Mask绑定和坐标还原。
3. 依据尺度、局部对比、噪声和几何证据选择传统算法/已有模型或组合；声明不足时先诊断，不自动修复。
4. 修改仅限已授权层；对困难与正常样本做定向回归，查看实际图像，不用通过单元测试替代效果验收。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

最小复现、阶段图与候选流失表、根因证据、变更范围和定向对照。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

绘图圆圈不是病灶真实面积；显示框不一致时不能删掉真实候选来修图。没有可信参考皮肤域时不能凭面积比断言误排。

## 离线辅助检查

脚本对提供的二值小栅格计算真实交集、包含关系与中心点归属；reference_skin是输入参考，不由工具推测。保留率门槛必须显式给出。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
