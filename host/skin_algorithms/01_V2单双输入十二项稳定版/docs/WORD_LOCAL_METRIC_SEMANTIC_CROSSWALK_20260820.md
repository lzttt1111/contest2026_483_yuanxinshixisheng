# 冻结 Word 与单 RGB V2 代理指标语义交叉表

> 用途：工程对齐与审计，不是正式医疗报告，不会写入 Word。
> 日期：2026-08-20。本文只比较已有结构化结果和当前注册表，没有重新生成 Word、没有运行图像算法、没有修改 Word 模板或渲染链。

## 1. 对比边界和基线选择

正式 Word 骨架的冻结基线仍是 `VISIA2_医学Word同源统一线复核_20260813_最终门禁03`：

- 医生版 SHA256：`97fc266423e026c3beae96e7c6a9e7084a844ae84ec4c35e6430da79a9e8a2e1`。
- 用户版 SHA256：`cb5cb6c831f6fe15eaa7b730c7282cbf57e512a32a8fc4bff3f3e7a30f239c99`。
- 该基线结构化 JSON 有11个模块、123条核心指标，但旧字段没有 `source_path`，因此不能单独完成本次来源交叉审计。

本文用指定兄弟工作区 `unified-twelve-output-contract-lzt-0818` 内最后一组完整的三个 consumer + 三个 Clinic 正式 Word 结构化结果作为“带 `source_path` 的对比基线”：

- `test-output/unified-dag-v3-acceptance-lzt-0819/{consumer,clinic}/**/*_正式结构化数据.json`，共6份。
- 与首次完整 Word 同源门 `test-output/twelve-word-sync-r1/**/**_正式结构化数据.json` 相比，按 `模块+范围+名称+单位+说明+source_path` 取并集后均为434条，规范化集合 SHA256 均为 `280edf7d4b9fb253acad51448d403a415e1de7849ae11d6835a8dda2cb2802ab`。
- `unified-twelve-output-format-v2-lzt-0818/final-consumer` 和 `final-clinic` 本身只有十二项公开产物，没有 `正式报告/*_正式结构化数据.json`；所以不把它们虚构成 Word `source_path` 来源。

当前本地语义真源为 [`calibration/metric_registry_v2_proxy_20260728.json`](../calibration/metric_registry_v2_proxy_20260728.json)，SHA256 为 `5a6231ea936a38e1cbbfc28158defa66978b6ba03b7bc406bcfe8526365dbeb7`，共11个模块、137个必需评分指标 ID。中文术语和单位见 [`calibration/metric_labels_v2_zh_20260728.json`](../calibration/metric_labels_v2_zh_20260728.json)；当前工程取值源见 [`calibration/v2_explicit_formula_registry_v1.json`](../calibration/v2_explicit_formula_registry_v1.json)。

## 2. 判定规则

| 类别 | 含义 | 允许的结论 |
|---|---|---|
| `SAME` | Word 已显示同一医学/工程量，差异仅为英文 ID、中文名、比例显示或标准化像素单位。 | 可作语义延续；不代表已证明独立测量。 |
| `RELATED` | Word 有同类证据，但统计量、分母、百分位、阈值子集或表型归属不同。 | 只能作代理来源，不得宣称与 Word 指标等价。 |
| `NEW` | 带 `source_path` 的冻结 Word 中没有该构念。 | 必须保留2D/2.5D工程代理边界，不得回填 Word。 |

Word 的 `source_path` 是报告内导航/来源标记，不是本地指标 ID。六份对比文件合并后共142条不重复语义行：112条使用 `检测模块[...]`报告定位路径，26条只写 `formal_addition`，4条使用 `评分追溯.<group>.<metric>.raw_value`。尤其是 `formal_addition` 只证明增量适配，不能反推出引擎原始指标 ID。

## 3. 总体结论

| 模块 | 当前 ID 数 | `SAME` | `RELATED` | `NEW` |
|---|---:|---:|---:|---:|
| 01 可见毛孔 | 8 | 6 | 2 | 0 |
| 02 油脂分泌倾向 | 11 | 6 | 2 | 3 |
| 03 综合色素 | 12 | 2 | 3 | 7 |
| 04 弥漫性泛红 | 8 | 6 | 1 | 1 |
| 05 血管样结构 | 9 | 4 | 5 | 0 |
| 06 毛囊炎症/痤疮样活动 | 13 | 0 | 9 | 4 |
| 07 干燥性细纹 | 18 | 2 | 3 | 13 |
| 08 稳定性线性皱纹 | 14 | 4 | 4 | 6 |
| 09 结构性沟槽 | 13 | 0 | 1 | 12 |
| 10 面部皮肤光滑度 | 15 | 2 | 2 | 11 |
| 11 面部轮廓紧致度 | 16 | 3 | 2 | 11 |
| **合计** | **137** | **35** | **34** | **68** |

结论不是“Word 已覆盖137项”：只有35个当前 ID 与 Word 展示量语义相同，34个只有相关证据，68个是冻结 Word 中不存在的新代理构念。这个35/34/68是“语义交叉”，不是算法独立测量率、准确率或医学验证率。

## 4. 137 项逐模块交叉

### 01 可见毛孔 `pores`

- Word 锚点：`source_path=检测模块[01].指标.<指标名>`；包含单位面积密度、面积占比、P50/P90/最大单体面积、P50圆度、P50长宽比、低圆度比例和拉长样比例。
- `SAME` 6：`pore_density_per_100k_px`、`pore_area_ratio`、`p50_pore_area_px`、`p90_pore_area_px`、`low_circularity_pore_ratio`、`elongated_pore_ratio`。
- `RELATED` 2：`large_pore_ratio`、`large_pore_area_ratio`。Word 只有 P90/最大面积和整体面积占比，没有“大毛孔阈值子集”的个数或面积分母。
- Word 反向保留项：毛孔总数、最大单体面积、P50圆度和 P50长宽比仍是报告展示量，但不是137个评分目标 ID。

### 02 油脂分泌倾向 `oil_tendency`

- Word 锚点：核心油光/毛囊荧光指标的 `source_path=formal_addition`；油光分区使用 `检测模块[02].分区指标[...].<指标名>`。
- `SAME` 6：`gloss_area_ratio`、`high_gloss_area_ratio`、`p90_gloss_intensity`、`largest_gloss_component_area_ratio`、`porphyrin_area_ratio`、`p90_porphyrin_intensity`。
- `RELATED` 2：`porphyrin_density_per_100k_px`对应 Word 的毛囊荧光目标数量，但 Word 未显示标准化分母；`high_porphyrin_target_ratio`只有高置信数量锚点，没有比例。
- `NEW` 3：`p50_gloss_intensity`、`p50_porphyrin_intensity`、`high_porphyrin_area_ratio`。Word 只展示 P90，且没有高荧光子集面积占比。

### 03 综合色素 `pigmentation`

- Word 锚点：可见斑点和 Brown 数量使用 `检测模块[03].指标.*`；UV 数量/面积/P90使用 `formal_addition`；分区只展示 Visible+UV+Brown 联合区域数和联合面积占比。
- `SAME` 2：`uv_spot_area_ratio`、`p90_uv_spot_intensity`。
- `RELATED` 3：`visible_spot_density_per_100k_px`、`uv_spot_density_per_100k_px`、`brown_area_ratio`。前两项 Word 只有数量没有标准化密度；Brown 只有实例数量及三源联合面积，不是 Brown 自身面积占比。
- `NEW` 7：`visible_spot_area_ratio`、`p90_visible_spot_delta_e`、`visible_spot_point_patch_burden`、`high_uv_area_ratio`、`p90_brown_intensity`、`high_brown_area_ratio`、`brown_point_patch_burden`。

### 04 弥漫性泛红 `diffuse_redness`

- Word 锚点：四个核心量分别使用 `评分追溯.coverage.diffuse_red_area_ratio.raw_value`、`intensity.p90_redness`、`continuity.redness_continuity`和 `boundary_uniformity.uniformity`；平均强度、高强度面积等使用分区报告路径。
- `SAME` 6：`diffuse_red_area_ratio`、`high_red_area_ratio`、`mean_redness`、`p90_redness`、`redness_continuity`、`redness_uniformity`。
- `RELATED` 1：`max_continuous_red_area_ratio`。Word 的“红区连续性”说明用最大连通区占全部弥漫红区的比例，但没有将“最大连通面积占有效皮肤”作为独立量。
- `NEW` 1：`boundary_gradient`，Word 只有均匀度，无边界梯度。

### 05 血管样结构 `vascular`

- Word 锚点：9个核心量均为 `source_path=formal_addition`，包含结构数、骨架总长度、长度密度、面积占比、P90宽度、P90红色强度、分支点数、最大连续网络长度和白光可见支持率。
- `SAME` 4：`vascular_length_density_per_10k_px`、`vascular_area_ratio`、`p90_vascular_width_px`、`p90_vascular_redness`。
- `RELATED` 5：`vascular_density_per_100k_px`对应结构数；`p50_vascular_width_px`和 `p50_vascular_redness` 只有 P90 锚点；`branch_density_per_10k_px`只有分支点数；`network_ratio`只有最大网络长度/白光支持率，分母不同。

### 06 毛囊炎症/痤疮样活动 `acne_activity`

- Word 锚点：有目标时使用 `检测模块[06].指标.*`显示泛痤疮样特征数、密度、特征框面积/占比/P50/P90和分区数量；无目标时只有 `formal_addition` 检测结果文字。Word 没有毛囊红斑、丘疹、脓疱的独立归类。
- `RELATED` 9：`follicular_erythema_density`、`follicular_erythema_area_ratio`、`p90_follicular_redness`、`papule_density`、`papule_area_ratio`、`p90_papule_redness`、`pustule_density`、`pustule_area_ratio`、`p90_pustule_redness`。它们只能复用泛化的痤疮数/框面积与跨模块红度证据，不能证明已完成三表型独立测量。
- `NEW` 4：`erythema_halo_completeness`、`p90_papule_relief`、`p90_pustule_relief`、`p90_white_yellow_center_area_px`。
- `SAME` 为0：泛化痤疮框统计不能改名后冒充毛囊性红斑/丘疹/脓疱指标。

### 07 干燥性细纹 `dry_fine_lines`

- Word 锚点：`source_path=检测模块[07].指标.*` 及左右眼下分区路径；包含线段数、总长度、有效面积、长度密度、平均/最大线段长度、连续性、面积占比和视觉对比度。
- `SAME` 2：`fine_line_length_density`、`fine_line_network_coverage_ratio`。
- `RELATED` 3：`fine_line_density_per_100k_px`只有线段数；`p50_fine_line_length_px`只有平均/最大长度；`p50_fine_line_contrast`只有分区视觉对比度，未冻结为 P50。
- `NEW` 13：`short_fine_line_ratio`、`p50_fine_line_width_px`、`narrow_fine_line_ratio`、`small_scale_texture_ratio`、`high_density_fine_line_area_ratio`、`direction_dispersion`、`crossing_density`、`branch_density`、`multidirectional_interweave`、`network_texture_ratio`、`low_mid_contrast_ratio`、`short_discontinuous_ratio`、`large_groove_exclusion`。

### 08 稳定性线性皱纹 `stable_wrinkles`

- Word 锚点：`source_path=检测模块[08].指标.*` 及额头/眉间/鱼尾纹分区路径；包含线段数、长度、面积占比、长度密度、平均/最大长度、连续性和 P90 视觉对比度。
- `SAME` 4：`wrinkle_length_density`、`wrinkle_area_ratio`、`p90_wrinkle_contrast`、`wrinkle_continuity`。
- `RELATED` 4：`wrinkle_density_per_100k_px`只有线段数；`p50_wrinkle_length_px`/`p90_wrinkle_length_px`只有平均/最大长度；`p50_wrinkle_contrast`只有 P90/分区视觉对比度。
- `NEW` 6：`p50_wrinkle_width_px`、`p90_wrinkle_width_px`、`high_contrast_wrinkle_ratio`、`wrinkle_linearity`、`mean_wrinkle_relative_depth`、`p90_wrinkle_relative_depth`。后两项是2.5D相对深度，Word 冻结量是2D线纹统计。

### 09 结构性沟槽 `structural_grooves`

- Word 锚点：`source_path=检测模块[09].指标.*` 及法令纹/木偶纹分区路径；仅有2D线段数、中心线长度、平均/最大长度、连续性、面积占比和视觉对比度。
- `RELATED` 1：`groove_continuous_length_ratio`。Word 有中心线长度和连续性，但没有对“连续长度占比”冻结独立分母。
- `NEW` 12：`p50_groove_width`、`p90_groove_width`、`mean_groove_relative_depth`、`p90_groove_relative_depth`、`normalized_groove_volume`、`groove_volume_per_length`、`clear_valley_segment_ratio`、`valley_floor_continuity`、`groove_curvature`、`surface_transition`、`groove_bulge_height_difference`、`normalized_adjacent_bulge_volume`。
- `SAME` 为0：宽度、深度、体积、凹谷、曲率和邻近隆起不能由2D线长直接等价推出。

### 10 面部皮肤光滑度 `smoothness`

- Word 锚点：`source_path=检测模块[10].指标.*` 及13个分区路径；包含纹理总数/密度/面积，凸起样和凹陷样数量/比例/面积占比，强度、特征面积和聚集统计。
- `SAME` 2：`raised_area_ratio`、`depressed_area_ratio`。
- `RELATED` 2：`raised_density_per_100k_px`、`depressed_density_per_100k_px`。Word 有分类数量与整体纹理密度，但未显示分类后的标准化密度。
- `NEW` 11：`mean_raised_relative_height`、`p90_raised_relative_height`、`raised_volume_per_area`、`mean_depressed_relative_depth`、`p90_depressed_relative_depth`、`depressed_volume_per_area`、`p90_depressed_edge_slope`、`attachment_density_per_100k_px`、`attachment_area_ratio`、`p90_attachment_area_px`、`flake_lifted_ratio`。

### 11 面部轮廓紧致度 `contour_firmness`

- Word 锚点：5个核心量均为 `source_path=formal_addition`：下颌线连续比例、下颌弧线左右差异、下脸宽高比、下颌曲率波动 P90 和中面部曲面连续比例。
- `SAME` 3：`midface_surface_continuity`、`jawline_continuity`、`jawline_curvature_variation`。
- `RELATED` 2：`midface_transition_burden`只有中面部连续性锚点；`firmness_groove_range`只能关联09模块的2D沟纹长度，不是紧致度模块原生量。
- `NEW` 11：`midcheek_depression_burden`、`paranasal_depression_burden`、`lower_cheek_bulge_burden`、`firmness_groove_depth`、`firmness_groove_volume`、`firmness_groove_valley`、`firmness_adjacent_bulge`、`jowl_bulge_burden`、`prejowl_sulcus_burden`、`mouth_corner_depression_burden`、`other_lower_cheek_bulge`。
- Word 反向保留项：下颌弧线左右差异、下脸宽高比仍属冻结报告展示量，但未被当前137个目标 ID 直接接收。

## 5. 必须保留的差异原因

1. **用途不同**：Word 指标是已冻结的展示合同，包含目标数、有效面积、主要区域等解释性字段；137项是11个 V2 组合评分的输入合同，不会一对一保留所有 Word 展示字段。
2. **统计函数不同**：数量不等于密度，平均/最大不等于 P50/P90，高置信数不等于高置信比例，整体面积占比不等于阈值子集面积占比。
3. **表型归属不同**：泛化痤疮框不是毛囊性红斑/丘疹/脓疱的独立证据；整体纹理密度也不是凸起样/凹陷样分类密度。
4. **2D 与2.5D边界不同**：沟槽深度/体积、纹理高度/斜率、皱纹深度和紧致度结构负担均是当前新增相对代理；不得从 Word 的2D长度/面积字段宣称已获得毫米、立方毫米或真实组织位移。
5. **来源层级不同**：Word `source_path` 标记“指标在报告结构中的位置”；本地 `source_metrics`/公式注册表才标记工程取值。不得用 `formal_addition` 替代引擎级来源证据。
6. **语义类别与最终公式状态分开**：本表的 `SAME` 只说与冻结 Word 展示量语义一致。当前最终实现已为137项分别注册精确的 `module.group.metric` 定义，并137个不重复的 `source_metric_ids + formula` 有效签名；注册表不存在 `group_formulas`，公式操作符不包含 `mean_abs`，投影器不使用指标名、通配符或组级回退。精确来源缺失时输出 `value=null` 与 `availability=unavailable`，不会静默填0。这些门禁证明“每个 ID 有独立明示的工程取值定义”，但仍不会把 `RELATED`/`NEW` 自动变成 Word 等价量、病灶级独立测量或人群标定医学证据；取值证据应查对应 `source_metric_ids`、精确公式、`medical_rationale` 和运行追溯。
7. **已否决的历史状态（不适用于当前实现）**：早期临时版只有47个组级公式，84/137项使用组内 `mean_abs` 并出现多个语义不同 ID 共享来源/公式/数值；该版本已被独立复核退回并由上述137项精确定义取代，不得再用来描述最终评分链。

## 6. 冻结门验证

2026-08-20 在当前工作区执行了已有门禁，未新增 Word 测试：

```text
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_word_freeze_contract.py
1 passed in 0.19s
```

实际复算值与 [`tests/test_word_freeze_contract.py`](../tests/test_word_freeze_contract.py) 完全一致：

| 冻结对象 | SHA256 |
|---|---|
| `templates/AISIA_面部多指标检测汇总模板_V2.docx` | `9de65290471199a14690db5b0db1571257788a24c23168ccf077e3c3701ffd4c` |
| `src/aisia_medical_report/integration.py` | `0872dfee6279a29a9f778cb39f497fafbac5716a841932652ec574d6fd8ea43a` |
| `src/aisia_medical_report/aggregator.py` | `e6eb294a1effed2b6d4325efe22fd37303984bab719c7c078a216d972e757a24` |
| `src/aisia_medical_report/report.py` | `493bf07c929be67f0f3ec9e5c9697393bcc4e0329e98b0f86622e562922a5b78` |

因现有门已同时固定模板和三个正式生成链文件，本轮不再创建重复测试。本文是唯一新增产物。
