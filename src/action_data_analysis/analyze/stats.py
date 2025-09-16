from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import os
import math

import numpy as np

from action_data_analysis.io.json import (
  iter_labelme_dir,
  extract_bbox_and_action,
)
def _parse_frame_index_from_path(path: str) -> Optional[int]:
  """从文件名中解析帧序号（取最后一段连续数字）。失败返回 None。"""
  base = os.path.basename(path)
  name, _ = os.path.splitext(base)
  digits = ""
  # 从后往前收集最后一段数字
  i = len(name) - 1
  while i >= 0 and name[i].isdigit():
    digits = name[i] + digits
    i -= 1
  if digits:
    try:
      return int(digits)
    except Exception:
      return None
  return None


def _autodetect_stride(frame_indices: List[int]) -> int:
  """根据帧序列数字自动估计采样间隔，返回最常见的正向差值，默认 1。"""
  if not frame_indices or len(frame_indices) < 2:
    return 1
  diffs: Counter[int] = Counter()
  prev = frame_indices[0]
  for idx in frame_indices[1:]:
    d = idx - prev
    if d > 0:
      diffs[d] += 1
    prev = idx
  if not diffs:
    return 1
  return max(diffs.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _detect_dataset_stride(path: str) -> int:
  """根据路径粗略判断数据集并返回既定 stride。

  - MultiSports/FineSports: 1
  - SportsHHI: 5
  默认 1。
  """
  s = path.lower()
  if "sportshhi" in s:
    return 5
  if "finesports" in s:
    return 1
  if "multisports" in s:
    return 1
  return 1


def _extract_track_id(shape: Dict[str, Any]) -> Optional[str]:
  """尽可能从 shape 中提取轨迹 id：attributes.id/track_id/player_id、flags.id 等，或 group_id。"""
  attrs = shape.get("attributes") or {}
  if isinstance(attrs, dict):
    for key in ("id", "track_id", "player_id"):
      val = attrs.get(key)
      if isinstance(val, (str, int)):
        return str(val)
  flags = shape.get("flags") or {}
  if isinstance(flags, dict):
    for key in ("id", "track_id", "player_id"):
      val = flags.get(key)
      if isinstance(val, (str, int)):
        return str(val)
  gid = shape.get("group_id")
  if isinstance(gid, (str, int)):
    return str(gid)
  return None



def _quantiles(values: List[float]) -> Dict[str, float]:
  if not values:
    return {"min": 0.0, "p25": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
  arr = np.asarray(values, dtype=float)
  return {
    "min": float(np.min(arr)),
    "p25": float(np.percentile(arr, 25)),
    "mean": float(np.mean(arr)),
    "p75": float(np.percentile(arr, 75)),
    "max": float(np.max(arr)),
  }


def compute_labelme_folder_stats(folder: str) -> Dict[str, Any]:
  """统计单个视频（帧目录）内的标注：类别分布、bbox 尺寸、异常等。"""
  action_counter: Counter[str] = Counter()
  widths: List[float] = []
  heights: List[float] = []
  areas: List[float] = []
  image_widths_px: List[float] = []
  image_heights_px: List[float] = []
  resolution_counter: Counter[str] = Counter()
  # 时空管计数
  tube_counter_by_action: Counter[str] = Counter()
  tubes_total = 0
  ignored_shapes_without_id = 0
  anomalies = {"coords_out_of_range": 0, "x1_ge_x2": 0, "y1_ge_y2": 0, "missing_values": 0}
  num_boxes = 0
  frames_with_ann = 0
  # 统计目录中的图片数量（jpg/jpeg）
  num_images = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg"))])

  # 遍历该目录下全部 LabelMe JSON
  # 先收集帧序列与 shape 元信息，便于时空管统计
  frames: List[Tuple[int, List[Tuple[str, str]]]] = []  # (frame_idx, [(track_id, action), ...])
  frame_indices_raw: List[int] = []
  for _json_path, rec in iter_labelme_dir(folder):
    W = max(1, int(rec.get("imageWidth", 0) or 0))
    H = max(1, int(rec.get("imageHeight", 0) or 0))
    if W > 0 and H > 0:
      image_widths_px.append(float(W))
      image_heights_px.append(float(H))
      resolution_counter[f"{W}x{H}"] += 1
    has_ann = False
    pairs: List[Tuple[str, str]] = []
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        anomalies["missing_values"] += 1
        continue
      (x1, y1, x2, y2), action = parsed
      has_ann = True
      if not (0 <= x1 <= W and 0 <= x2 <= W and 0 <= y1 <= H and 0 <= y2 <= H):
        anomalies["coords_out_of_range"] += 1
      if x1 >= x2:
        anomalies["x1_ge_x2"] += 1
      if y1 >= y2:
        anomalies["y1_ge_y2"] += 1
      w = max(0.0, float(x2 - x1))
      h = max(0.0, float(y2 - y1))
      a = w * h
      if W > 0 and H > 0:
        widths.append(w / W)
        heights.append(h / H)
        areas.append(a / (W * H))
      else:
        widths.append(0.0)
        heights.append(0.0)
        areas.append(0.0)
      action_counter[action or "__unknown__"] += 1
      num_boxes += 1
      tid = _extract_track_id(sh)
      if tid and action:
        pairs.append((tid, action))
      else:
        if not tid:
          ignored_shapes_without_id += 1
    if has_ann:
      frames_with_ann += 1
    fidx = _parse_frame_index_from_path(_json_path)
    if fidx is not None:
      frame_indices_raw.append(fidx)
    # 使用文件顺序作为后备顺序（若无法解析），用累积长度替代
    frames.append((fidx if fidx is not None else len(frames), pairs))

  # 按帧序号排序
  frames.sort(key=lambda t: t[0])
  # 使用数据集固定 stride 规则
  stride = _detect_dataset_stride(folder)
  # 统计时空管：相同 (track_id, action) 且帧间隔等于 stride 视为连续，否则新管
  last_seen_idx: Dict[Tuple[str, str], int] = {}
  for fidx, pairs in frames:
    if not pairs:
      continue
    for pair in pairs:
      if pair in last_seen_idx and (fidx - last_seen_idx[pair] == stride):
        # 延续，不计数
        last_seen_idx[pair] = fidx
      else:
        # 新管
        tubes_total += 1
        tube_counter_by_action[pair[1]] += 1
        last_seen_idx[pair] = fidx

  stats: Dict[str, Any] = {
    "folder": os.path.abspath(folder),
    "basic": {
      "num_images": num_images,
      "frames_with_annotations": frames_with_ann,
      "num_boxes": num_boxes,
      "num_actions": int(sum(action_counter.values())),
      "num_action_labels": len(action_counter),
    },
    "action_distribution": sorted(action_counter.items(), key=lambda kv: (-kv[1], kv[0])),
    "bbox_normalized": {
      "width": _quantiles(widths),
      "height": _quantiles(heights),
      "area": _quantiles(areas),
      "anomalies": anomalies,
    },
    "image_size": {
      "width_px": _quantiles(image_widths_px),
      "height_px": _quantiles(image_heights_px),
      "common_resolutions": sorted(resolution_counter.items(), key=lambda kv: (-kv[1], kv[0])),
    },
    "spatiotemporal_tubes": {
      "total": tubes_total,
      "per_action": sorted(tube_counter_by_action.items(), key=lambda kv: (-kv[1], kv[0])),
      "stride": stride,
      "ignored_shapes_without_id": ignored_shapes_without_id,
    },
  }
  return stats


def compute_aggregate_stats(folders: List[str]) -> Dict[str, Any]:
  """聚合多个视频目录的统计。"""
  action_counter: Counter[str] = Counter()
  widths: List[float] = []
  heights: List[float] = []
  areas: List[float] = []
  image_widths_px: List[float] = []
  image_heights_px: List[float] = []
  resolution_counter: Counter[str] = Counter()
  tubes_total = 0
  tubes_counter_by_action: Counter[str] = Counter()
  ignored_shapes_without_id_total = 0
  anomalies = {"coords_out_of_range": 0, "x1_ge_x2": 0, "y1_ge_y2": 0, "missing_values": 0}
  frames_with_ann = 0
  num_boxes = 0
  num_images_total = 0

  for folder in folders:
    s = compute_labelme_folder_stats(folder)
    for k, v in s.get("bbox_normalized", {}).get("anomalies", {}).items():
      anomalies[k] = anomalies.get(k, 0) + int(v)
    frames_with_ann += int(s.get("basic", {}).get("frames_with_annotations", 0))
    num_boxes += int(s.get("basic", {}).get("num_boxes", 0))
    num_images_total += int(s.get("basic", {}).get("num_images", 0))
    # 重新展开 action 分布
    for name, cnt in s.get("action_distribution", []):
      action_counter[name] += int(cnt)
    # 汇总时空管
    st = s.get("spatiotemporal_tubes", {})
    tubes_total += int(st.get("total", 0))
    for name, cnt in st.get("per_action", []):
      tubes_counter_by_action[name] += int(cnt)
    ignored_shapes_without_id_total += int(st.get("ignored_shapes_without_id", 0))
    # 无法从已聚合结果恢复原始样本，因此此处直接跳过合并宽高分布的精确值。
    # 为了近似，我们不再合并单目录的分位数，而是在聚合时重新遍历一次以收集值。

  # 为了获得正确的分布，重新二次遍历以收集 bbox 归一化尺寸
  for folder in folders:
    for _json_path, rec in iter_labelme_dir(folder):
      W = max(1, int(rec.get("imageWidth", 0) or 0))
      H = max(1, int(rec.get("imageHeight", 0) or 0))
      if W > 0 and H > 0:
        image_widths_px.append(float(W))
        image_heights_px.append(float(H))
        resolution_counter[f"{W}x{H}"] += 1
      for sh in rec.get("shapes", []) or []:
        parsed = extract_bbox_and_action(sh)
        if parsed is None:
          continue
        (x1, y1, x2, y2), _ = parsed
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        a = w * h
        widths.append(w / W if W > 0 else 0.0)
        heights.append(h / H if H > 0 else 0.0)
        areas.append(a / (W * H) if (W > 0 and H > 0) else 0.0)

  stats: Dict[str, Any] = {
    "folders": [os.path.abspath(f) for f in folders],
    "basic": {
      "num_folders": len(folders),
      "num_images": num_images_total,
      "frames_with_annotations": frames_with_ann,
      "num_boxes": num_boxes,
      "num_action_labels": len(action_counter),
    },
    "action_distribution": sorted(action_counter.items(), key=lambda kv: (-kv[1], kv[0])),
    "bbox_normalized": {
      "width": _quantiles(widths),
      "height": _quantiles(heights),
      "area": _quantiles(areas),
      "anomalies": anomalies,
    },
    "image_size": {
      "width_px": _quantiles(image_widths_px),
      "height_px": _quantiles(image_heights_px),
      "common_resolutions": sorted(resolution_counter.items(), key=lambda kv: (-kv[1], kv[0])),
    },
    "spatiotemporal_tubes": {
      "total": tubes_total,
      "per_action": sorted(tubes_counter_by_action.items(), key=lambda kv: (-kv[1], kv[0])),
      "ignored_shapes_without_id": ignored_shapes_without_id_total,
    },
  }
  return stats


def render_stats_markdown(stats: Dict[str, Any]) -> str:
  """将统计结果渲染为 Markdown 文本。"""
  lines: List[str] = []
  lines.append("# 统计结果")
  basic = stats.get("basic", {})
  lines.append("")
  lines.append("## 基本信息")
  for k in ["num_folders", "frames_with_annotations", "num_boxes", "num_action_labels"]:
    if k in basic:
      lines.append(f"- {k}: {basic[k]}")
  if "num_images" in basic:
    lines.append(f"- num_images: {basic['num_images']}")

  lines.append("")
  lines.append("## 类别分布（Top 50）")
  for i, (name, cnt) in enumerate(stats.get("action_distribution", [])[:50], 1):
    lines.append(f"{i}. {name}: {cnt}")

  lines.append("")
  lines.append("## 归一化 bbox 统计")
  for dim in ["width", "height", "area"]:
    q = stats.get("bbox_normalized", {}).get(dim, {})
    lines.append(f"- {dim}: min={q.get('min', 0):.4f}, p25={q.get('p25', 0):.4f}, mean={q.get('mean', 0):.4f}, p75={q.get('p75', 0):.4f}, max={q.get('max', 0):.4f}")
  an = stats.get("bbox_normalized", {}).get("anomalies", {})
  lines.append(f"- 异常: {an}")

  lines.append("")
  lines.append("## 图片尺寸统计")
  im = stats.get("image_size", {})
  for dim, label in [("width_px", "宽度(px)"), ("height_px", "高度(px)")]:
    q = im.get(dim, {})
    lines.append(
      f"- {label}: min={q.get('min', 0):.0f}, p25={q.get('p25', 0):.0f}, mean={q.get('mean', 0):.1f}, p75={q.get('p75', 0):.0f}, max={q.get('max', 0):.0f}"
    )
  res_list = im.get("common_resolutions", [])
  if res_list:
    lines.append("")
    lines.append("### 常见分辨率（Top 10）")
    for i, (name, cnt) in enumerate(res_list[:10], 1):
      lines.append(f"{i}. {name}: {cnt}")

  lines.append("")
  lines.append("## 时空管统计")
  tubes = stats.get("spatiotemporal_tubes", {})
  if "total" in tubes:
    lines.append(f"- total: {tubes.get('total', 0)}")
  if "stride" in tubes:
    lines.append(f"- stride(自动估计): {tubes.get('stride', 1)}")
  if "ignored_shapes_without_id" in tubes and int(tubes.get("ignored_shapes_without_id", 0)) > 0:
    lines.append(f"- 忽略无 id 的标注数: {tubes.get('ignored_shapes_without_id', 0)}")
  per_act = tubes.get("per_action", [])
  if per_act:
    lines.append("")
    lines.append("### 各动作的时空管数量（Top 50）")
    for i, (name, cnt) in enumerate(per_act[:50], 1):
      lines.append(f"{i}. {name}: {cnt}")

  return "\n".join(lines)


def compute_dataset_stats(json_or_csv_path: str) -> Dict[str, Any]:
  """保留原骨架函数名以兼容，但此实现仅做占位提示。"""
  raise NotImplementedError("请使用 compute_labelme_folder_stats 或 compute_aggregate_stats")


def compute_dataset_stats(json_or_csv_path: str) -> Dict[str, Any]:
  """统计分析：类别分布、时长统计等（占位）。"""
  raise NotImplementedError("TODO: 计算统计指标并返回字典结果")