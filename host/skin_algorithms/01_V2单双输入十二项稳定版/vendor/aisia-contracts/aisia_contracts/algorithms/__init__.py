"""各算法各版本契约。import 本包即触发所有版本注册到 REGISTRY。

各子包 __init__.py import 自己的 v1/v2 模块,模块末尾调 register()。
本文件由 fixer 在创建算法文件后填充 import。
"""
from aisia_contracts.algorithms import acne, brown, contour_firmness, pores, purple, redness, spots, surface_gloss, texture, vascular, wrinkle  # noqa: F401
