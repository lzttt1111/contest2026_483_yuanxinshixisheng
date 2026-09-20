# 输入合同与结果解读

数值参数不接受JSON布尔值或数字字符串；浮点量必须有限。整数型计数/版本/方向使用严格整数，yaw_sign只接受-1或1；物理标尺为正数、数值容差为非负数。二值Mask和明确的状态布尔字段仍按各自合同接受布尔值。

human_confirmed必须显式为布尔值。缺失状态或未知rerun ID不会获得safe_rerun_ids；存在清单问题时安全重跑列表为空。字符串false不能当作布尔False。

本文件是只读检查器的适配合同，不是强制产品接口。

records每项含id、sha256、subject、session、label(positive/negative/unknown)、bbox(归一化xywh或null)、human_confirmed、revision/proposed_revision；rerun_ids仅表示拟操作范围，不构成覆盖确认权限。重复id/hash、正例空框、负例有框都要处理。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "data",
  "records": [
    {
      "id": "a",
      "sha256": "a1",
      "subject": "p1",
      "session": "s1",
      "label": "positive",
      "bbox": [
        0.1,
        0.1,
        0.5,
        0.6
      ],
      "human_confirmed": true,
      "revision": 2,
      "proposed_revision": 2
    },
    {
      "id": "b",
      "sha256": "b1",
      "subject": "p2",
      "session": "s2",
      "label": "negative",
      "bbox": null,
      "human_confirmed": true
    }
  ],
  "rerun_ids": []
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为data时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

记录 id 为非空字符串。human_confirmed、mask_edited、human_edited、protected 任一为 true 都保护记录；可选编辑标志必须为布尔值。human_confirmed=false 不覆盖编辑保护。上游其他编辑字段须明确映射到这些标志，不得静默丢弃编辑状态。
