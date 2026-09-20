"""Registered native-template wording exceptions for image-only groove estimates."""
REPLACEMENTS={
 "整体三维凹陷负担":"整体凹陷外观负担",
 "标准化凹陷体积":"整体凹陷外观负担",
 "整体三维负担":"整体凹陷外观负担",
 "平均相对深度":"平均凹陷外观程度",
 "P90相对深度":"局部明显凹陷外观程度",
 "平均凹陷深度":"平均凹陷外观程度",
 "相对深度P90":"局部明显凹陷外观程度",
 "凹陷越深":"凹陷外观越明显",
}
def formal_label(text,module,stage1=False):
 if stage1 and module=="08":
  return text.replace("三维凹陷程度","凹陷外观估计").replace("三维凹陷","凹陷外观估计").replace("3D 凹陷","凹陷外观估计")
 if module!="09":return text
 for before,after in sorted(REPLACEMENTS.items(),key=lambda x:-len(x[0])):
  text=text.replace(before,after)
 return text
