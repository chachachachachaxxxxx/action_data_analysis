from __future__ import annotations

from typing import Any, Dict, List, Tuple
from collections import Counter
import os

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


def _iou_matrix(boxes_xyxy: np.ndarray) -> np.ndarray:
  """计算 NxN IoU 矩阵（对角线为 0）。

  参数: boxes_xyxy: (N,4) 数组，列为 [x1,y1,x2,y2]
  返回: (N,N) 数组，IoU 值，主对角线置 0
  """
  if boxes_xyxy.size == 0:
    return np.zeros((0, 0), dtype=float)
  x1 = boxes_xyxy[:, 0][:, None]
  y1 = boxes_xyxy[:, 1][:, None]
  x2 = boxes_xyxy[:, 2][:, None]
  y2 = boxes_xyxy[:, 3][:, None]

  xx1 = np.maximum(x1, x1.T)
  yy1 = np.maximum(y1, y1.T)
  xx2 = np.minimum(x2, x2.T)
  yy2 = np.minimum(y2, y2.T)

  inter_w = np.maximum(0.0, xx2 - xx1)
  inter_h = np.maximum(0.0, yy2 - yy1)
  inter = inter_w * inter_h

  areas = (np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) *
           np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1]))
  union = (areas[:, None] + areas[None, :] - inter)
  with np.errstate(divide='ignore', invalid='ignore'):
    iou = np.where(union > 0.0, inter / union, 0.0)
  # 清理数值与对角线
  np.fill_diagonal(iou, 0.0)
  iou = np.clip(iou, 0.0, 1.0)
  return iou


def compute_folder_overlaps(folder: str, thresholds: List[float] | None = None) -> Dict[str, Any]:
  """统计单个帧目录中检测框重合（IoU）情况。

  返回字段（摘要）：
  - basic: {num_images, frames_with_boxes, frames_with_overlaps, total_boxes, total_pairs}
  - iou: {quantiles, histogram, counts_ge{t}}
  - degree_per_threshold: {t: histogram of overlaps-per-box}
  - per_frame_max_iou: quantiles
  """
  if thresholds is None:
    thresholds = [0.1, 0.3, 0.5]

  frames_with_boxes = 0
  frames_with_overlaps = 0
  total_boxes = 0
  total_pairs = 0
  iou_values: List[float] = []
  per_frame_max_iou_vals: List[float] = []
  counts_ge = {float(t): 0 for t in thresholds}
  # 按动作类别统计：在同一帧内、同一动作名的成对框，IoU≥阈值的对数
  per_action_counts_ge: Dict[float, Counter[str]] = {float(t): Counter() for t in thresholds}
  # 按动作计每帧是否出现重合：帧级标记（同一帧只+1次）
  per_action_frames_with_overlaps_ge: Dict[float, Counter[str]] = {float(t): Counter() for t in thresholds}
  # 按动作对统计：在同一帧内，动作对 (a,b)（无序，a<=b）若 IoU≥阈值则 +1
  per_action_pair_counts_ge: Dict[float, Counter[Tuple[str, str]]] = {float(t): Counter() for t in thresholds}
  degree_hist_by_t: Dict[float, Counter[int]] = {float(t): Counter() for t in thresholds}

  num_images = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg"))])

  for _json_path, rec in iter_labelme_dir(folder):
    boxes: List[Tuple[float, float, float, float]] = []
    actions: List[str] = []
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      (x1, y1, x2, y2), act = parsed
      if x2 <= x1 or y2 <= y1:
        continue
      boxes.append((x1, y1, x2, y2))
      actions.append(act or "__unknown__")

    n = len(boxes)
    if n == 0:
      continue
    frames_with_boxes += 1
    total_boxes += n
    if n >= 2:
      total_pairs += n * (n - 1) // 2
      mat = _iou_matrix(np.asarray(boxes, dtype=float))
      # 只取上三角避免重复
      triu_idx = np.triu_indices(n, k=1)
      vals = mat[triu_idx]
      if vals.size > 0:
        iou_values.extend(vals.tolist())
        per_frame_max_iou_vals.append(float(np.max(vals)))
        # 是否有重合帧
        if np.any(vals > 0.0):
          frames_with_overlaps += 1
        # 阈值计数与度分布
        for t in thresholds:
          t = float(t)
          mask = (mat >= t)
          # 每行的度（与其他框 IoU≥t 的数量）
          deg = (np.sum(mask, axis=1) - 1).astype(int)
          for d in deg.tolist():
            if d >= 0:
              degree_hist_by_t[t][int(d)] += 1
          counts_ge[t] += int(np.sum(mask[triu_idx]))
          # 按动作统计（同一动作名）
          # 本帧内是否出现过该动作的重合，用于帧级计数
          seen_action_this_frame: set[str] = set()
          I, J = triu_idx
          for k in range(len(I)):
            i, j = int(I[k]), int(J[k])
            if actions[i] == actions[j] and mask[i, j]:
              per_action_counts_ge[t][actions[i]] += 1
              seen_action_this_frame.add(actions[i])
            # 动作对（无序）统计
            if mask[i, j]:
              a = actions[i] or "__unknown__"
              b = actions[j] or "__unknown__"
              key = tuple(sorted((a, b)))
              per_action_pair_counts_ge[t][key] += 1
          for act_name in seen_action_this_frame:
            per_action_frames_with_overlaps_ge[t][act_name] += 1
    else:
      per_frame_max_iou_vals.append(0.0)

  # 直方图（IoU 0.0~1.0，10 桶）
  hist_bins = [i / 10.0 for i in range(11)]
  hist_counts = [0 for _ in range(10)]
  for v in iou_values:
    idx = int(min(9, max(0, int(v * 10))))
    hist_counts[idx] += 1

  stats: Dict[str, Any] = {
    "folder": os.path.abspath(folder),
    "basic": {
      "num_images": num_images,
      "frames_with_boxes": frames_with_boxes,
      "frames_with_overlaps": frames_with_overlaps,
      "total_boxes": total_boxes,
      "total_pairs": total_pairs,
    },
    "iou": {
      "quantiles": _quantiles(iou_values),
      "histogram": {
        "bins": hist_bins,
        "counts": hist_counts,
      },
      "counts_ge": {str(k): int(v) for k, v in counts_ge.items()},
    },
    "per_action": {
      "pair_counts_ge": {
        str(t): sorted(per_action_counts_ge[t].items(), key=lambda kv: (-int(kv[1]), kv[0]))
        for t in sorted(per_action_counts_ge.keys())
      },
      "frames_with_overlaps_ge": {
        str(t): sorted(per_action_frames_with_overlaps_ge[t].items(), key=lambda kv: (-int(kv[1]), kv[0]))
        for t in sorted(per_action_frames_with_overlaps_ge.keys())
      },
      "pair_of_actions_counts_ge": {
        str(t): [
          (f"{a} || {b}", int(cnt))
          for (a, b), cnt in sorted(per_action_pair_counts_ge[t].items(), key=lambda kv: (-int(kv[1]), kv[0][0], kv[0][1]))
        ]
        for t in sorted(per_action_pair_counts_ge.keys())
      },
    },
    "degree_per_threshold": {
      str(t): sorted(((int(k), int(v)) for k, v in hist.items()), key=lambda kv: kv[0])
      for t, hist in degree_hist_by_t.items()
    },
    "per_frame_max_iou": _quantiles(per_frame_max_iou_vals),
  }
  return stats


def compute_aggregate_overlaps(folders: List[str], thresholds: List[float] | None = None) -> Dict[str, Any]:
  if thresholds is None:
    thresholds = [0.1, 0.3, 0.5]

  total_num_images = 0
  frames_with_boxes = 0
  frames_with_overlaps = 0
  total_boxes = 0
  total_pairs = 0
  all_iou_values: List[float] = []
  per_frame_max_iou_vals: List[float] = []
  counts_ge = {float(t): 0 for t in thresholds}
  per_action_counts_ge: Dict[float, Counter[str]] = {float(t): Counter() for t in thresholds}
  per_action_frames_with_overlaps_ge: Dict[float, Counter[str]] = {float(t): Counter() for t in thresholds}
  per_action_pair_counts_ge: Dict[float, Counter[Tuple[str, str]]] = {float(t): Counter() for t in thresholds}
  degree_hist_by_t: Dict[float, Counter[int]] = {float(t): Counter() for t in thresholds}

  for folder in folders:
    s = compute_folder_overlaps(folder, thresholds)
    total_num_images += int(s.get("basic", {}).get("num_images", 0))
    frames_with_boxes += int(s.get("basic", {}).get("frames_with_boxes", 0))
    frames_with_overlaps += int(s.get("basic", {}).get("frames_with_overlaps", 0))
    total_boxes += int(s.get("basic", {}).get("total_boxes", 0))
    total_pairs += int(s.get("basic", {}).get("total_pairs", 0))
    # 合并 IoU 值分布
    # 注：按动作的统计改为在重遍历阶段统一计算，避免重复叠加
    # 注意：单目录返回的是分位数，无法恢复原始值；因此这里重新遍历一次收集 IoU 值更可靠。
    for _json_path, rec in iter_labelme_dir(folder):
      boxes: List[Tuple[float, float, float, float]] = []
      actions: List[str] = []
      for sh in rec.get("shapes", []) or []:
        parsed = extract_bbox_and_action(sh)
        if parsed is None:
          continue
        (x1, y1, x2, y2), act = parsed
        if x2 <= x1 or y2 <= y1:
          continue
        boxes.append((x1, y1, x2, y2))
        actions.append(act or "__unknown__")
      n = len(boxes)
      if n == 0:
        continue
      if n == 1:
        per_frame_max_iou_vals.append(0.0)
        continue
      mat = _iou_matrix(np.asarray(boxes, dtype=float))
      triu_idx = np.triu_indices(n, k=1)
      vals = mat[triu_idx]
      if vals.size > 0:
        all_iou_values.extend(vals.tolist())
        per_frame_max_iou_vals.append(float(np.max(vals)))
      for t in thresholds:
        t = float(t)
        mask = (mat >= t)
        deg = (np.sum(mask, axis=1) - 1).astype(int)
        for d in deg.tolist():
          if d >= 0:
            degree_hist_by_t[t][int(d)] += 1
        counts_ge[t] += int(np.sum(mask[triu_idx]))
        # 按动作统计（同一动作名）
        seen_action_this_frame: set[str] = set()
        I, J = triu_idx
        for k in range(len(I)):
          i, j = int(I[k]), int(J[k])
          if actions[i] == actions[j] and mask[i, j]:
            per_action_counts_ge[t][actions[i]] += 1
            seen_action_this_frame.add(actions[i])
          # 动作对（无序）统计
          if mask[i, j]:
            a = actions[i] or "__unknown__"
            b = actions[j] or "__unknown__"
            key = tuple(sorted((a, b)))
            per_action_pair_counts_ge[t][key] += 1
        for act_name in seen_action_this_frame:
          per_action_frames_with_overlaps_ge[t][act_name] += 1

  hist_bins = [i / 10.0 for i in range(11)]
  hist_counts = [0 for _ in range(10)]
  for v in all_iou_values:
    idx = int(min(9, max(0, int(v * 10))))
    hist_counts[idx] += 1

  stats: Dict[str, Any] = {
    "folders": [os.path.abspath(f) for f in folders],
    "basic": {
      "num_folders": len(folders),
      "num_images": total_num_images,
      "frames_with_boxes": frames_with_boxes,
      "frames_with_overlaps": frames_with_overlaps,
      "total_boxes": total_boxes,
      "total_pairs": total_pairs,
    },
    "iou": {
      "quantiles": _quantiles(all_iou_values),
      "histogram": {
        "bins": hist_bins,
        "counts": hist_counts,
      },
      "counts_ge": {str(k): int(v) for k, v in counts_ge.items()},
    },
    "per_action": {
      "pair_counts_ge": {
        str(t): sorted(per_action_counts_ge[t].items(), key=lambda kv: (-int(kv[1]), kv[0]))
        for t in sorted(per_action_counts_ge.keys())
      },
      "frames_with_overlaps_ge": {
        str(t): sorted(per_action_frames_with_overlaps_ge[t].items(), key=lambda kv: (-int(kv[1]), kv[0]))
        for t in sorted(per_action_frames_with_overlaps_ge.keys())
      },
      "pair_of_actions_counts_ge": {
        str(t): [
          (f"{a} || {b}", int(cnt))
          for (a, b), cnt in sorted(per_action_pair_counts_ge[t].items(), key=lambda kv: (-int(kv[1]), kv[0][0], kv[0][1]))
        ]
        for t in sorted(per_action_pair_counts_ge.keys())
      },
    },
    "degree_per_threshold": {
      str(t): sorted(((int(k), int(v)) for k, v in hist.items()), key=lambda kv: kv[0])
      for t, hist in degree_hist_by_t.items()
    },
    "per_frame_max_iou": _quantiles(per_frame_max_iou_vals),
  }
  return stats


def render_overlaps_markdown(stats: Dict[str, Any]) -> str:
  lines: List[str] = []
  lines.append("# 检测框重合统计")
  lines.append("")
  lines.append("## 基本信息")
  basic = stats.get("basic", {})
  for k in [
    "num_folders",
    "num_images",
    "frames_with_boxes",
    "frames_with_overlaps",
    "total_boxes",
    "total_pairs",
  ]:
    if k in basic:
      lines.append(f"- {k}: {basic[k]}")

  lines.append("")
  lines.append("## IoU 分布")
  q = stats.get("iou", {}).get("quantiles", {})
  lines.append(
    f"- IoU: min={q.get('min', 0):.4f}, p25={q.get('p25', 0):.4f}, mean={q.get('mean', 0):.4f}, p75={q.get('p75', 0):.4f}, max={q.get('max', 0):.4f}"
  )
  counts_ge = stats.get("iou", {}).get("counts_ge", {})
  if counts_ge:
    items = ", ".join([f"IoU≥{k}: {v}" for k, v in sorted(counts_ge.items(), key=lambda kv: float(kv[0]))])
    lines.append(f"- 阈值对数: {items}")

  lines.append("")
  lines.append("## 每框重合度分布（按阈值）")
  deg = stats.get("degree_per_threshold", {})
  for t, hist in sorted(deg.items(), key=lambda kv: float(kv[0])):
    if not hist:
      continue
    lines.append(f"- 阈值 {t}：")
    preview = ", ".join([f"{k}:{v}" for k, v in hist[:15]])
    lines.append(f"  - 度直方图(预览): {preview}")

  lines.append("")
  lines.append("## 每帧最大 IoU 分布")
  q2 = stats.get("per_frame_max_iou", {})
  lines.append(
    f"- max IoU/帧: min={q2.get('min', 0):.4f}, p25={q2.get('p25', 0):.4f}, mean={q2.get('mean', 0):.4f}, p75={q2.get('p75', 0):.4f}, max={q2.get('max', 0):.4f}"
  )

  # 追加：按动作类别的重合对数 Top 列表
  per_act = stats.get("per_action", {})
  pair_counts_ge = per_act.get("pair_counts_ge", {})
  if pair_counts_ge:
    lines.append("")
    lines.append("## 按动作的重合对数（Top 20）")
    for t_str in sorted(pair_counts_ge.keys(), key=lambda s: float(s)):
      items = pair_counts_ge.get(t_str, [])
      if not items:
        continue
      lines.append(f"### IoU≥{t_str}")
      for i, (name, cnt) in enumerate(items[:20], 1):
        lines.append(f"{i}. {name}: {cnt}")

  # 追加：按动作对的重合对数 Top 列表
  pair_of_actions = per_act.get("pair_of_actions_counts_ge", {})
  if pair_of_actions:
    lines.append("")
    lines.append("## 按动作对的重合对数（Top 20）")
    for t_str in sorted(pair_of_actions.keys(), key=lambda s: float(s)):
      items = pair_of_actions.get(t_str, [])
      if not items:
        continue
      lines.append(f"### IoU≥{t_str}")
      for i, (pair_name, cnt) in enumerate(items[:20], 1):
        lines.append(f"{i}. {pair_name}: {cnt}")

  return "\n".join(lines)


