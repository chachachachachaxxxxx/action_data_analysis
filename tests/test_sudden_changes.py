from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pytest

from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
  ax1, ay1, ax2, ay2 = a
  bx1, by1, bx2, by2 = b
  inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
  inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
  inter = inter_w * inter_h
  if inter <= 0.0:
    return 0.0
  area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
  area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
  union = area_a + area_b - inter
  if union <= 0.0:
    return 0.0
  return float(inter / union)


def _read_frame_index(json_path: str) -> int:
  base = os.path.basename(json_path)
  stem = os.path.splitext(base)[0]
  # 允许常见命名: frame_000123, img_0010, 00000123 等；取末尾连续数字作为帧号
  i = len(stem) - 1
  digits = []
  while i >= 0 and stem[i].isdigit():
    digits.append(stem[i])
    i -= 1
  if not digits:
    return -1
  return int("".join(reversed(digits)))


def _load_frame_boxes(folder: str) -> List[Tuple[int, str, List[Tuple[Tuple[float, float, float, float], str, int]]]]:
  frames: List[Tuple[int, str, List[Tuple[Tuple[float, float, float, float], str, int]]]] = []
  for json_path, rec in iter_labelme_dir(folder):
    fidx = _read_frame_index(json_path)
    items: List[Tuple[Tuple[float, float, float, float], str, int]] = []
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      (x1, y1, x2, y2), act = parsed
      if x2 <= x1 or y2 <= y1:
        continue
      # tube id 来源：优先 attributes.group_id -> 其次 group_id -> 否则 -1
      gid = -1
      attrs = sh.get("attributes") or {}
      if isinstance(attrs, dict):
        v = attrs.get("group_id")
        if isinstance(v, int):
          gid = int(v)
      if gid == -1:
        v2 = sh.get("group_id")
        if isinstance(v2, int):
          gid = int(v2)
      items.append(((float(x1), float(y1), float(x2), float(y2)), str(act or ""), int(gid)))
    frames.append((fidx, json_path, items))
  frames.sort(key=lambda t: t[0])
  return frames


def _count_sudden_changes_in_sample(sample_dir: str, iou_threshold: float = -0.1) -> Dict[str, object]:
  frames = _load_frame_boxes(sample_dir)
  changes = 0
  disappear_only = 0
  appear_only = 0
  changes_detail: List[Dict[str, object]] = []

  for i in range(len(frames) - 1):
    fidx_a, json_a, items_a = frames[i]
    fidx_b, json_b, items_b = frames[i + 1]
    # 仅统计严格相邻的连续帧对；忽略未知帧号或存在跳帧的情况
    if fidx_a < 0 or fidx_b < 0 or (fidx_b - fidx_a) != 1:
      continue
    if not items_a and not items_b:
      continue
    
    gid_a_set = set()
    gid_b_set = set()
    for (box_a, act_a, gid_a) in items_a:
      gid_a_set.add(gid_a)
    for (box_b, act_b, gid_b) in items_b:
      gid_b_set.add(gid_b)
    
    change_a = list(gid_a_set - gid_b_set)
    change_b = list(gid_b_set - gid_a_set)
    if len(change_a) > 0 and len(change_b) > 0:
      for box_a, act_a, gid_a in items_a:
        for box_b, act_b, gid_b in items_b:
          if gid_a in change_a and gid_b in change_b:
            changes += 1
            changes_detail.append({
              "sample": os.path.basename(sample_dir),
              "frame_idx_a": int(fidx_a),
              "frame_idx_b": int(fidx_b),
              "json_a": os.path.abspath(json_a),
              "json_b": os.path.abspath(json_b),
              "action_a": str(act_a or  ""),
              "action_b": str(act_b or ""),
              "gid_a": int(gid_a),
              "gid_b": int(gid_b),
              "iou": _iou(box_a, box_b),
              "bbox_a": [float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])],
              "bbox_b": [float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])],
            })

  return {
    "changes": int(changes),
    "disappear_only": 0,
    "appear_only": 0,
    "changes_detail": changes_detail,
  }


@pytest.mark.parametrize("std_root", [
#   "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std",
#   "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted",
"/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step2_json",
])
def test_sudden_changes_std_dataset(std_root: str) -> None:
  if not os.path.isdir(std_root) or not os.path.isdir(os.path.join(std_root, "videos")):
    pytest.skip(f"std root not found: {std_root}")

  videos_root = os.path.join(std_root, "videos")
  sample_dirs = [os.path.join(videos_root, d) for d in sorted(os.listdir(videos_root)) if os.path.isdir(os.path.join(videos_root, d))]
  # 为了测试速度，只选前 N 个样本；若需要全量统计可调整 N
  N = len(sample_dirs)
  sample_dirs = sample_dirs[:N]

  total = {"changes": 0, "disappear_only": 0, "appear_only": 0}
  all_changes_detail: List[Dict[str, object]] = []
  for sdir in sample_dirs:
    stats = _count_sudden_changes_in_sample(sdir, iou_threshold=-0.1)
    for k in total.keys():
      total[k] += int(stats.get(k, 0))
    cd = stats.get("changes_detail", [])
    if isinstance(cd, list):
      all_changes_detail.extend(cd)

  # 将摘要输出为 JSON 便于查看
  out_dir = os.path.join(std_root, "stats")
  try:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sudden_changes_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump({
        "root": os.path.abspath(std_root),
        "samples": len(sample_dirs),
        **total,
        "changes_detail": all_changes_detail,
      }, f, ensure_ascii=False, indent=2)
  except Exception:
    # 非关键路径失败不影响测试
    pass

  # 基本断言：计数均为非负整数
  assert total["changes"] >= 0
  assert total["disappear_only"] >= 0
  assert total["appear_only"] >= 0


