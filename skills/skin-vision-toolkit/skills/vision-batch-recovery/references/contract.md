# 输入合同与结果解读

workers/max_workers必须为正整数，不接受布尔值或小数。committed/artifacts_valid必须为JSON布尔值。running样本在实际重试前必须确认原执行者停止或租约失效；工具仅输出此前提，不代替执行者存活检查。

本文件是只读检查器的适配合同，不是强制产品接口。

config_hash必须覆盖输入/模型/算法/输出模式等会影响结果的身份。jobs记录status、committed、artifacts_valid及原config_hash；workers与max_workers是当前机器预算，不写死历史机器参数。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "batch",
  "config_hash": "cfg1",
  "jobs": [
    {
      "id": "a",
      "status": "success",
      "committed": true,
      "artifacts_valid": true,
      "config_hash": "cfg1"
    },
    {
      "id": "b",
      "status": "running",
      "committed": false,
      "artifacts_valid": false,
      "config_hash": "cfg1"
    }
  ],
  "max_workers": 1,
  "workers": 1
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为batch时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

status 仅 success/failed/running/pending/queued/cancelled；未知状态拒绝生成恢复计划。running/pending/queued/cancelled 的建议执行前必须确认旧执行者停止或租约失效。id 为非空字符串；本检查器不执行计划、不检查真实租约。
