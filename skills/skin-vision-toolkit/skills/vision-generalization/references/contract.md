# 输入合同与结果解读

本文件是只读检查器的适配合同，不是强制产品接口。

records含id/subject/session/sha256/cluster/split；split为train/validation/test/reference。selection_split需说明验证协议；calibration_ids和synthetic_sources按id回查。检查器不会把空分组视作自动安全。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "generalization",
  "records": [
    {
      "id": "a",
      "subject": "p1",
      "session": "s1",
      "sha256": "a",
      "cluster": "c1",
      "split": "train"
    },
    {
      "id": "b",
      "subject": "p2",
      "session": "s2",
      "sha256": "b",
      "cluster": "c2",
      "split": "validation"
    },
    {
      "id": "c",
      "subject": "p3",
      "session": "s3",
      "sha256": "c",
      "cluster": "c3",
      "split": "test"
    }
  ],
  "selection_split": "validation",
  "calibration_ids": [
    "a"
  ],
  "synthetic_sources": [
    "a"
  ]
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为generalization时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
