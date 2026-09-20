# 完整产物示例：恢复计划

当前cfg1，预算1个worker。a为success且committed=true、artifacts_valid=true、cfg1；b仍running，未提交且产物不完整。

| 样本 | 计划 | 执行前条件 |
|---|---|---|
| a | skip | 保留已验证成功产物 |
| b | retry_or_recompute | 原执行者已停或租约到期；临时输出隔离 |

健康记录至少区分最后完成样本、当前计算、写盘耗时、内存预算和超时。工具只输出计划，不删除锁、不重启服务。

若当前切换cfg2，a的旧缓存不能直接跳过；若b只写success但未原子提交，则继续视为半成品。JSON字符串false不是布尔False，必须拒绝。断点测试模拟进程中断后的状态，不需要实际启动大图库任务。
