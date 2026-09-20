# 输入合同与结果解读

本文件是只读检查器的适配合同，不是强制产品接口。

available为已实际拥有的模态列表；requirements每项含id、needs、unit，可声明calibration_required/calibrated；candidates含id、license、baseline_verified。脚本不会对候选自动打综合分，缺license返回unknown_license。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "spec",
  "available": [
    "rgb"
  ],
  "requirements": [
    {
      "id": "spots",
      "needs": [
        "rgb"
      ],
      "unit": "pixels"
    }
  ],
  "candidates": [
    {
      "id": "existing",
      "license": "MIT",
      "baseline_verified": true
    }
  ]
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为spec时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

available、requirements、candidates 必须为列表；需求/候选为含非空 id 的对象，needs 为非空字符串元素的列表，不接受字符串替代列表。空 available/needs 可表达没有可用输入/无需传感输入，不等于完成真实测量论证。
