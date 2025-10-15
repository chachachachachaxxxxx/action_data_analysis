from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from action_data_analysis.analyze.export import _discover_std_sample_folders


@dataclass
class UpsampleSummary:
  output_root: str
  multiplier: int
  num_samples: int
  num_src_frames: int
  num_dst_frames: int
  annotations_updated: bool


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _list_sorted_images(folder: str) -> List[str]:
  allow_exts = (".jpg", ".jpeg", ".png")
  return sorted([f for f in os.listdir(folder) if f.lower().endswith(allow_exts)])


def _infer_index_width(name: str) -> int:
  stem, _ = os.path.splitext(name)
  # 连续数字宽度，例如 000001 -> 6
  if stem.isdigit():
    return len(stem)
  # 回退：保持至少 6 位
  return 6


def _format_index(idx: int, width: int) -> str:
  return str(idx).rjust(width, "0")


def _copy_and_rewrite_json(src_json: str, dst_json: str, new_image_name: str) -> None:
  try:
    with open(src_json, "r", encoding="utf-8") as f:
      rec = json.load(f)
  except Exception:
    # 读取失败则直接复制
    shutil.copy2(src_json, dst_json)
    return
  rec["imagePath"] = new_image_name
  with open(dst_json, "w", encoding="utf-8") as f:
    json.dump(rec, f, ensure_ascii=False, indent=2)


def _find_std_root_from_sample(sample_dir: str) -> str:
  # .../<std_root>/videos/<sample>
  videos_dir = os.path.dirname(os.path.abspath(sample_dir))
  return os.path.dirname(videos_dir)


def _collect_std_roots(sample_dirs: List[str]) -> List[str]:
  roots = []
  seen = set()
  for s in sample_dirs:
    r = _find_std_root_from_sample(s)
    if r not in seen:
      seen.add(r)
      roots.append(r)
  return roots


def _copy_annotations_if_any(std_root: str, out_root: str, multiplier: int) -> bool:
  """复制并更新 annotations 下的 CSV：
  - 若存在列名包含 frames 的第三列，则乘以 multiplier。
  - 否则按原样复制。
  返回是否发生了任何注释更新。
  """
  ann_dir = os.path.join(std_root, "annotations")
  if not os.path.isdir(ann_dir):
    return False
  out_ann = os.path.join(out_root, "annotations")
  _ensure_dir(out_ann)
  updated = False
  for name in os.listdir(ann_dir):
    if not name.lower().endswith(".csv"):
      continue
    src = os.path.join(ann_dir, name)
    dst = os.path.join(out_ann, name)
    try:
      with open(src, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    except Exception:
      shutil.copy2(src, dst)
      continue
    if not lines:
      shutil.copy2(src, dst)
      continue

    # 检测是否为三列表头包含 frames 的 CSV
    header = lines[0].split(",")
    is_header = ("frames" in [h.strip().lower() for h in header]) and (len(header) >= 3)
    out_lines: List[str] = []
    if is_header:
      out_lines.append(lines[0])
      for ln in lines[1:]:
        if not ln.strip():
          continue
        parts = ln.split(",")
        if len(parts) >= 3:
          try:
            frames_val = int(parts[2])
            parts[2] = str(frames_val * multiplier)
            updated = True
          except Exception:
            pass
        out_lines.append(",".join(parts))
    else:
      # 无 frames 或无表头 -> 原样复制
      out_lines = lines

    with open(dst, "w", encoding="utf-8") as f:
      f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
  return updated


def upsample_std_fps(
  inputs: List[str],
  out_root: Optional[str],
  src_fps: int,
  dst_fps: int,
  progress: str = "frames",  # frames | samples
) -> UpsampleSummary:
  if src_fps <= 0 or dst_fps <= 0:
    raise ValueError("src_fps 和 dst_fps 必须为正整数")
  if dst_fps % src_fps != 0:
    raise ValueError("当前实现仅支持整倍数上采样，例如 6→30 (x5)")
  multiplier = dst_fps // src_fps

  sample_dirs = _discover_std_sample_folders(inputs)
  if not sample_dirs:
    raise ValueError("未找到任何 std 样例目录，请传入 std 根目录 / videos 目录 / videos/* 样例目录")

  # 推断默认 out_root：基于第一个样例
  if not out_root:
    any_sample = sample_dirs[0]
    std_root = _find_std_root_from_sample(any_sample)
    out_root = std_root + f"_fpsx{multiplier}"

  videos_out_root = os.path.join(out_root, "videos")
  _ensure_dir(videos_out_root)

  total_src = 0
  total_dst = 0

  # 先收集每个样例的图片清单，以便计算总帧复制数
  sample_images: List[Tuple[str, List[str]]] = []
  for sample_dir in sample_dirs:
    imgs = _list_sorted_images(sample_dir)
    sample_images.append((sample_dir, imgs))
    total_src += len(imgs)
    total_dst += len(imgs) * multiplier

  from tqdm import tqdm as _tqdm
  if (progress or "frames").lower().startswith("frame"):
    pbar_total = total_src * multiplier
    unit = "frame"
  else:
    pbar_total = len(sample_dirs)
    unit = "sample"
  pbar = _tqdm(total=pbar_total, desc=f"upsample x{multiplier}", unit=unit)

  for sample_dir, images in sample_images:
    sample_name = os.path.basename(os.path.normpath(sample_dir))
    dst_sample_dir = os.path.join(videos_out_root, sample_name)
    _ensure_dir(dst_sample_dir)

    if not images:
      try:
        # 空样例按样例推进一次
        if unit == "sample":
          pbar.update(1)
      except Exception:
        pass
      continue
    width = _infer_index_width(images[0])

    # 逐帧复制并生成新索引
    new_index = 1
    for img_name in images:
      src_img_path = os.path.join(sample_dir, img_name)
      stem, ext = os.path.splitext(img_name)
      src_json_path = os.path.join(sample_dir, f"{stem}.json")

      for _ in range(multiplier):
        new_stem = _format_index(new_index, width)
        new_img_name = f"{new_stem}{ext}"
        new_json_name = f"{new_stem}.json"

        # 拷贝图片
        shutil.copy2(src_img_path, os.path.join(dst_sample_dir, new_img_name))
        # JSON：若存在则重写 imagePath
        if os.path.exists(src_json_path):
          _copy_and_rewrite_json(src_json_path, os.path.join(dst_sample_dir, new_json_name), new_img_name)

        new_index += 1
        # 按帧推进进度
        if unit == "frame":
          try:
            pbar.update(1)
          except Exception:
            pass

    if unit == "sample":
      try:
        pbar.update(1)
      except Exception:
        pass
  try:
    pbar.close()
  except Exception:
    pass

  # annotations：当且仅当所有样例来自同一个 std_root 时，复制并更新 frames
  std_roots = _collect_std_roots(sample_dirs)
  ann_updated = False
  if len(std_roots) == 1:
    ann_updated = _copy_annotations_if_any(std_roots[0], out_root, multiplier)

  return UpsampleSummary(
    output_root=out_root,
    multiplier=multiplier,
    num_samples=len(sample_dirs),
    num_src_frames=total_src,
    num_dst_frames=total_dst,
    annotations_updated=ann_updated,
  )


