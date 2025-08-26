from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class StandardRecord(TypedDict, total=False):
  """标准化的单条样本记录。

  必填：
    - video_path: 视频文件路径或 URL
    - action_label: 动作类别（字符串标签）

  可选：
    - start_frame, end_frame: 用于时序裁剪的帧范围
    - metadata: 与具体数据源相关的扩展信息
  """

  video_path: str
  action_label: str
  start_frame: int
  end_frame: int
  metadata: Dict[str, Any]


class Metadata(TypedDict, total=False):
  """数据集级别的元信息，用于记录来源与处理过程。"""

  source_format: str
  created_by: str
  created_at: str
  num_items: int
  notes: str
  raw_summary: Dict[str, Any]


class StandardDataset(TypedDict):
  """标准化数据集的顶层结构。"""

  version: str
  records: List[StandardRecord]
  metadata: Metadata