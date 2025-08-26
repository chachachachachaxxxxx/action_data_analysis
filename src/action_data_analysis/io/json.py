from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 避免运行时依赖，仅用于类型检查
  from action_data_analysis.datasets.schema import StandardDataset


def read_standard_json(json_path: str) -> "StandardDataset":
  """读取标准格式的 JSON 数据集（占位）。"""
  raise NotImplementedError("TODO: 读取并校验标准 JSON")


def write_standard_json(dataset: "StandardDataset", output_path: str) -> None:
  """写出标准格式的 JSON 数据集（占位）。"""
  raise NotImplementedError("TODO: 序列化并写出标准 JSON")