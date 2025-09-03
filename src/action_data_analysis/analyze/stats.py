from __future__ import annotations

from typing import Any, Dict, List, Tuple
from collections import Counter
import os
import math

import numpy as np

from action_data_analysis.io.json import (
  iter_labelme_dir,
  extract_bbox_and_action,
)


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
  anomalies = {"coords_out_of_range": 0, "x1_ge_x2": 0, "y1_ge_y2": 0, "missing_values": 0}
  num_boxes = 0
  frames_with_ann = 0
  # 统计目录中的图片数量（jpg/jpeg）
  num_images = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg"))])

  # 遍历该目录下全部 LabelMe JSON
  for _json_path, rec in iter_labelme_dir(folder):
    W = max(1, int(rec.get("imageWidth", 0) or 0))
    H = max(1, int(rec.get("imageHeight", 0) or 0))
    has_ann = False
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
    if has_ann:
      frames_with_ann += 1

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
  }
  return stats


def compute_aggregate_stats(folders: List[str]) -> Dict[str, Any]:
  """聚合多个视频目录的统计。"""
  action_counter: Counter[str] = Counter()
  widths: List[float] = []
  heights: List[float] = []
  areas: List[float] = []
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
    # 无法从已聚合结果恢复原始样本，因此此处直接跳过合并宽高分布的精确值。
    # 为了近似，我们不再合并单目录的分位数，而是在聚合时重新遍历一次以收集值。

  # 为了获得正确的分布，重新二次遍历以收集 bbox 归一化尺寸
  for folder in folders:
    for _json_path, rec in iter_labelme_dir(folder):
      W = max(1, int(rec.get("imageWidth", 0) or 0))
      H = max(1, int(rec.get("imageHeight", 0) or 0))
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
  }
  return stats


def render_stats_markdown(stats: Dict[str, Any]) -> str:
  """将统计结果渲染为 Markdown 文本。"""
  lines: List[str] = []
  title = stats.get("folder") or ", ".join(stats.get("folders", [])) or "Dataset"
  lines.append(f"# 统计结果：{title}")
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

  return "\n".join(lines)


def compute_dataset_stats(json_or_csv_path: str) -> Dict[str, Any]:
  """保留原骨架函数名以兼容，但此实现仅做占位提示。"""
  raise NotImplementedError("请使用 compute_labelme_folder_stats 或 compute_aggregate_stats")


def compute_dataset_stats(json_or_csv_path: str) -> Dict[str, Any]:
  """统计分析：类别分布、时长统计等（占位）。"""
  raise NotImplementedError("TODO: 计算统计指标并返回字典结果")