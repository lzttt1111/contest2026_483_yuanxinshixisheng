# 输入合同与结果解读

投影使用规范化JSON文本比较，保留布尔与数值类型差异（true不等于1）。整数与小数表示不同也会报告差异；如业务合同允许数值归一化，调用方应先显式统一两份投影，再比较，不由检查器隐式强转。

本文件是只读检查器的适配合同，不是强制产品接口。

source_file/report_file为相对root的JSON；artifact_files是相对文件列表。绝对路径、..越界、软链接逃逸拒绝；文件缺失单列。对真实Word先提取其可比较字段和图片哈希形成projection，工具不解析任意office文档。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "publication",
  "source_file": "source.json",
  "report_file": "report.json",
  "artifact_files": [
    "evidence.txt"
  ],
  "inference_requested": false
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为publication时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

可选 artifact_sha256 为相对资产路径到小写64位SHA256的非空映射。显式提供时校验实际资产；仅在覆盖全部 artifact_files 且无错误时 binding_verified=true。未提供时明确 binding_status=not_checked，不能将任意投影内 image_sha256 当作已校验引用；该字段须由适配器提取到 artifact_sha256。
