from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
try:
  from typing import TypedDict  # py310+
except Exception:  # pragma: no cover
  from typing_extensions import TypedDict  # fallback
import json
import os

if TYPE_CHECKING:  # 避免运行时依赖，仅用于类型检查
  from action_data_analysis.datasets.schema import StandardDataset


def read_standard_json(json_path: str) -> "StandardDataset":
  """读取标准格式的 JSON 数据集（占位）。"""
  raise NotImplementedError("TODO: 读取并校验标准 JSON")


def write_standard_json(dataset: "StandardDataset", output_path: str) -> None:
  """写出标准格式的 JSON 数据集（占位）。"""
  raise NotImplementedError("TODO: 序列化并写出标准 JSON")


# --- LabelMe 简易读取工具 ---

class LabelmeShape(TypedDict, total=False):  # type: ignore[name-defined]
  label: str
  points: List[List[float]]
  group_id: Optional[int]
  shape_type: str
  flags: Dict[str, Any]
  attributes: Dict[str, Any]


class LabelmeRecord(TypedDict, total=False):  # type: ignore[name-defined]
  imagePath: str
  imageWidth: int
  imageHeight: int
  shapes: List[LabelmeShape]


def read_labelme_json(json_path: str) -> LabelmeRecord:
  """读取单个 LabelMe JSON。

  仅解析核心字段：imagePath/imageWidth/imageHeight/shapes。
  若字段缺失，做温和降级（返回默认值）。
  """
  with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  rec: LabelmeRecord = {
    "imagePath": data.get("imagePath", ""),
    "imageWidth": int(data.get("imageWidth", 0) or 0),
    "imageHeight": int(data.get("imageHeight", 0) or 0),
    "shapes": list(data.get("shapes", [])),
  }
  return rec


def iter_labelme_dir(dir_path: str, allow_exts: Tuple[str, ...] = (".json",)) -> Iterable[Tuple[str, LabelmeRecord]]:
  """遍历目录下的 LabelMe JSON 文件，yield (path, record)。"""
  for fname in sorted(os.listdir(dir_path)):
    if not fname.lower().endswith(allow_exts):
      continue
    fpath = os.path.join(dir_path, fname)
    try:
      yield fpath, read_labelme_json(fpath)
    except Exception:
      # 容忍个别坏文件，不中断整体流程
      continue


def extract_bbox_and_action(shape: LabelmeShape) -> Optional[Tuple[Tuple[float, float, float, float], str]]:
  """从 LabelMe shape 中提取 (x1,y1,x2,y2) 与动作名。

  - 支持 rectangle（四角点）与自定义点对（形如 [x1,y1,x2,y2] 的占位输入）。
  - 动作名优先从 attributes.action 获取；若无，尝试 flags.action。
  """
  pts = shape.get("points") or []
  if not pts:
    return None
  # rectangle: 4 corner points -> use min/max
  if len(pts) >= 2 and isinstance(pts[0], list) and len(pts[0]) == 2:
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
  # compact placeholder: [[x1,y1,x2,y2]]
  elif len(pts) == 1 and isinstance(pts[0], list) and len(pts[0]) == 4:
    x1, y1, x2, y2 = map(float, pts[0])
  else:
    return None

  action = ""
  attrs = shape.get("attributes") or {}
  if isinstance(attrs, dict):
    val = attrs.get("action")
    if isinstance(val, str):
      action = val
  if not action:
    flags = shape.get("flags") or {}
    if isinstance(flags, dict):
      val = flags.get("action")
      if isinstance(val, str):
        action = val

  return (x1, y1, x2, y2), action