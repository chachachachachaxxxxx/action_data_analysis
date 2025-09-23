from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

try:
  import cv2  # type: ignore
except Exception:
  cv2 = None

from action_data_analysis.io.json import (
  iter_labelme_dir,
  extract_bbox_and_action,
  read_labelme_json,
)
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def sample_per_class_examples(
  folders: List[str],
  per_class: int = 3,
) -> Dict[str, List[Tuple[str, str]]]:
  """在若干目录内，按动作类别随机抽取若干个 (json_path, image_path)。"""
  by_action: Dict[str, List[Tuple[str, str]]] = {}
  pool: Dict[str, List[Tuple[str, str]]] = {}
  for folder in folders:
    for json_path, rec in iter_labelme_dir(folder):
      img_path = os.path.join(folder, rec.get("imagePath", ""))
      actions: List[str] = []
      for sh in rec.get("shapes", []) or []:
        parsed = extract_bbox_and_action(sh)
        if parsed is None:
          continue
        _, act = parsed
        actions.append((act or "__unknown__").strip())
      # 将该帧加入该帧出现过的所有动作类别（去重）
      if actions:
        for act in sorted(set(actions)):
          pool.setdefault(act, []).append((json_path, img_path))

  rng = random.Random(0)
  for act, items in pool.items():
    if len(items) <= per_class:
      by_action[act] = items
    else:
      by_action[act] = rng.sample(items, per_class)
  return by_action


def _draw_boxes_on_image(image, shapes, color=(0, 255, 0)):
  for sh in shapes:
    parsed = extract_bbox_and_action(sh)
    if parsed is None:
      continue
    (x1, y1, x2, y2), act = parsed
    x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    cv2.rectangle(image, (x1i, y1i), (x2i, y2i), color, 2)
    if act:
      cv2.putText(image, act, (x1i, max(0, y1i - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def visualize_samples_with_context(
  folders: List[str],
  output_dir: str,
  per_class: int = 3,
  context_frames: int = 25,
) -> None:
  """为每个类别随机抽样若干样例，并导出该帧及其前后 N 帧（若存在）。

  约定：同目录下帧图像的文件名按照字典序近似时间顺序排列。
  """
  if cv2 is None:
    raise RuntimeError("需要 opencv-python 以生成可视化结果")

  _ensure_dir(output_dir)
  picked = sample_per_class_examples(folders, per_class=per_class)

  total_items = sum(len(v) for v in picked.values())
  pbar = tqdm(total=total_items, desc="visualize", unit="item")
  for act, items in picked.items():
    act_dir = os.path.join(output_dir, act or "unknown")
    _ensure_dir(act_dir)
    for idx, (json_center_path, img_center_path) in enumerate(items):
      folder = os.path.dirname(json_center_path)
      center_fname = os.path.basename(img_center_path)
      frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))])
      try:
        pos = frames.index(center_fname)
      except ValueError:
        pos = -1
      if pos >= 0:
        lo = max(0, pos - context_frames)
        hi = min(len(frames), pos + context_frames + 1)
        ctx = frames[lo:hi]
      else:
        ctx = [center_fname]

      sample_id = f"{os.path.basename(folder)}__{os.path.splitext(center_fname)[0]}"
      sample_dir = os.path.join(act_dir, sample_id)
      _ensure_dir(sample_dir)

      for f in ctx:
        src_img_path = os.path.join(folder, f)
        img = cv2.imread(src_img_path)
        if img is None:
          continue
        img_draw = img.copy()
        json_path = os.path.join(folder, os.path.splitext(f)[0] + ".json")
        if os.path.exists(json_path):
          try:
            rec = read_labelme_json(json_path)
            _draw_boxes_on_image(img_draw, rec.get("shapes", []) or [])
          except Exception:
            pass
        # 仅输出绘制后的图像，保持原文件名
        cv2.imwrite(os.path.join(sample_dir, f), img_draw)
      pbar.update(1)
  try:
    pbar.close()
  except Exception:
    pass


def visualize_dataset(json_or_csv_path: str, output_dir: str) -> None:  # 兼容旧骨架 API
  raise NotImplementedError("请使用 visualize_samples_with_context")