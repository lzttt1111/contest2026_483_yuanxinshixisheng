---
name: vision-spec-to-evidence
description: "Translate visual-AI product requirements into an evidence-backed capability matrix and validation plan. Use for feasibility, reuse-versus-training choices, and measurement boundaries; not deployment or cosmetic copy edits."
---

# 视觉需求与验证方案

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 先从用户目标建立能力表：输入模态、输出量、单位、允许误差、验证来源和是否需要标定。不要以模型会输出一个数代替该量能被测量。
2. 核对现有基线的许可证、模型来源与验证条件；复用、适配、训练分列，不默认重新训练或下载。
3. 用小型代表案例确定实验与停止条件；对缺少传感器/标尺/标注的要求明确不可评估或待验证。
4. 给出推荐路线、被排除方案和验收计划；计划不等于执行授权。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

能力矩阵、方案对照、输入输出要求、实验/验收计划及缺失证据。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

不推导临床疗效或操作处方；不能把RGB特征当作真实深度、含水量等未观测量。缺证据时不创建漂亮但无依据的总分。

## 离线辅助检查

脚本只检查提供的模态/标定/许可证清单，不会搜索文献或验证许可证真实性。pass仅指清单没有发现这些矛盾。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
