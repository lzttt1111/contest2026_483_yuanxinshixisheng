# 视觉AI产品全流程开发Skill包

版本：0.1.3。本包是开发方法与离线证据检查工具，不是新的皮肤模型、医疗操作方案或已部署设备固件。

## 按任务选用，不必依次执行全部Skill

| 阶段 | Skill | 负责什么 |
|---|---|---|
| 01 | [视觉需求与验证方案](skills/vision-spec-to-evidence/SKILL.md) | 能力矩阵、方案对照、输入输出要求、实验/验收计划及缺失证据。 |
| 02 | [视觉数据与人工标注闭环](skills/vision-data-review/SKILL.md) | 数据manifest、标签合同、待复核清单、保护/重跑列表和修订审计。 |
| 03 | [模型训练与跨场景泛化](skills/vision-generalization/SKILL.md) | 划分审计、基线实验表、分层指标、困难样本队列和模型选择依据。 |
| 04 | [姿态引导与自动采集](skills/pose-guided-capture/SKILL.md) | 状态机、参数、回放轨迹、拍照槽位及元数据合同、重拍测试。 |
| 05 | [图像算法与有效域工程](skills/vision-feature-engineering/SKILL.md) | 最小复现、阶段图与候选流失表、根因证据、变更范围和定向对照。 |
| 06 | [量化统计与评分标定](skills/vision-metrics-calibration/SKILL.md) | 指标字典、分母与单位检查、统计对账、参考分布、候选配置和差异解释。 |
| 07 | [结果图与报告同源交付](skills/vision-result-publication/SKILL.md) | 结果索引、字段/章节映射、文件哈希、同源检查、展示产物及重建回执。 |
| 08 | [批量推理与可靠恢复](skills/vision-batch-recovery/SKILL.md) | 恢复计划、完成/重算列表、缓存身份检查、资源预算及健康回执。 |
| 09 | [模型跨运行时迁移](skills/vision-runtime-parity/SKILL.md) | 模型合同、转换/对拍记录、误差表、运行provider与资源实测、可交付模型清单。 |
| 10 | [产品集成与可复现交付](skills/vision-system-integration/SKILL.md) | 接口适配表、状态生命周期、端到端验收、版本/依赖清单和可复现交付包。 |

研发：需求→数据/标注→实验/泛化→算法→量化→结果→运行/迁移→集成交付。
产品：发现人脸→稳定角度采集→有效域与检测→量化/结果→实时设备与业务会话关联。

相邻职责：01确定做什么和如何证明，10连接已选模块；02保障真值，03管理实验；04决定保存哪一帧，05检测图像；06计算量化，07只消费已有结果；08关注规模恢复，09关注运行时等价。

## 使用

在支持Skill的AI工具中选择需要的单个目录，显式要求使用对应Skill；本包未写入全局配置。SKILL.md标准Markdown/YAML可供兼容工具读取，agents/openai.yaml仅提供Codex界面提示，不是运行依赖。不承诺尚未实测的其他客户端会自动发现。

先提供目标、分析/修改/验证性质、输入与基线、允许目录、资源预算和输出要求。业务接口不必迁移到检查器的JSON格式；按references/contract.md整理一个限定证据快照即可。

示例：使用$vision-runtime-parity读取源/目标张量与provider记录，先做离线对拍诊断，不运行转换或部署。

每个scripts/check.py仅用Python 3.10+标准库，只读显式JSON和允许root下的文件，将结果输出stdout。它们不是训练器、相机程序或医学诊断器。缺证据、错误数据和不适用任务分别报告，不自动启动重计算。

## 验证

~~~text
python -B tests/run_checks.py --output evaluation/self_checks.json --relocation
~~~

此命令只在临时目录生成合成输入、复制必要样例，并写入显式指定的结果文件。30个正常/错误/不适用场景检验实际数值、状态和文件不变量；迁移再跑30次，不能代替模型真实行为测试。

独立模型对照见evaluation目录：先无Skill基线，再有Skill，保持模型medium；同代理先后两轮存在学习效应，不作因果效率宣称。最终发布状态见VALIDATION.md。

## 项目案例与证据

参考三个模块的实际输入合同和研发方法，而不是按日志体积分配技能数量。evidence/method_sources.json记录原会话ID、时间、派生日志seq、源文件/事件哈希及选用方法；evidence/source_code.json记录读取时的实际代码/文档哈希。摘要不冒充原始日志，新Skill的真实使用与历史方法来源分开保存。

本项目参考当前单RGB水光八项范围，不自动加四光源、综合评分或仪器操作点。不同客户端和板端模型能力要按实际版本核对，PC模拟不冒充实板。

## 隔离与发布

此包在水光目录的独立Skill沙箱内制作，未改业务算法、模型、V2/V3、Hook或全局Skill。发布包不带私人图像、业务模型、虚拟环境或大日志；只带技能、必要样例、检查器、可公开的证据索引与验证记录。新增代码采用Apache-2.0，第三方资料只链接和摘要，不转授其许可证。
