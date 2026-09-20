---
name: vision-batch-recovery
description: "Audit resumable visual inference jobs, resource budgets, cache identity and completion records. Use for batch recovery and throughput reliability; not launching arbitrary workloads or changing detector parameters."
---

# 批量推理与可靠恢复

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 核对任务模式、输入分片、模型与配置身份，检查已有完成记录和实际产物，而非只读success字段。
2. 同配置且原子提交完成、产物有效才跳过；超时、半成品、错配置缓存进入明确重试/重算队列。
3. 按机器实测预算设置并发、模型常驻与I/O策略，健康记录区分计算、写盘、内存和服务超时；不盲目加worker。
4. 先在隔离小样例验证中断恢复/去重，再执行授权批量。保留原结果和失败证据，不静默更换输出模式。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

恢复计划、完成/重算列表、缓存身份检查、资源预算及健康回执。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

恢复计划不是启动许可；不重跑全部成功样本，不用仅量化模式替代需要图片的正式产物。declared artifacts_valid必须来自实际验证。

## 离线辅助检查

脚本只根据已核实状态快照生成skip/retry_or_recompute，不启动进程、不删除锁或重置服务；实际执行器须单独授权。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
