---
name: vision-data-review
description: "Audit image/video annotation manifests and plan safe auto-label plus human-review workflows. Use for frame deduplication, box/mask/point labels, negative semantics, and edit protection; not model training."
---

# 视觉数据与人工标注闭环

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 先核对图像哈希、匿名人物/Session、帧与坐标空间，原图保持只读。SHA完全重复与感知近重复分开，不能只靠文件名。
2. 自动初标作为建议，输出raw与人工终版两层；区分可见目标、遮挡可推断、无法标注和真实无目标。
3. bbox、Mask、关键点、预览及索引应来自同一标注revision。负样本含目标、空标签或坐标越界进入复核队列。
4. 重跑先识别人工确认/编辑项并跳过；只有显式的恢复或修订请求才可另存版本，不覆盖已确认结果。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

数据manifest、标签合同、待复核清单、保护/重跑列表和修订审计。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

人工标签才是真值；模型预测、界面叠加和预处理图不自动成为标签。检查工具不替代查看遮挡或含手Mask的人工审查。

## 离线辅助检查

脚本验证标准化xywh框、负样本和人工编辑保护，仅输出建议列表，不重写原图或标签。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
