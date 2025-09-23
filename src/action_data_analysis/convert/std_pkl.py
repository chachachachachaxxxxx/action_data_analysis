from __future__ import annotations

import os
import sys
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np  # type: ignore
from tqdm import tqdm

# 复用项目中现有的 std 发现与解析能力
from action_data_analysis.analyze.export import _discover_std_sample_folders
from action_data_analysis.analyze.export_tube_videos import (
  _build_tubes_from_folder,
  _split_by_separators,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _parse_frame_index_from_name(name: str) -> Optional[int]:
  base, _ = os.path.splitext(os.path.basename(name))
  digits = ""
  i = len(base) - 1
  while i >= 0 and base[i].isdigit():
    digits = base[i] + digits
    i -= 1
  if digits:
    try:
      return int(digits)
    except Exception:
      return None
  return None


def _iter_images(folder: str) -> Iterable[str]:
  try:
    for name in sorted(os.listdir(folder)):
      ext = os.path.splitext(name)[1].lower()
      if ext in IMG_EXTS:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
          yield path
  except Exception:
    return


def _probe_resolution(folder: str) -> Optional[Tuple[int, int]]:
  # 惰性导入 cv2，避免没用到时的依赖问题
  try:
    import cv2  # type: ignore
  except Exception:
    return None
  for img_path in _iter_images(folder):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      continue
    h, w = img.shape[:2]
    if h > 0 and w > 0:
      return (int(h), int(w))
  return None


def _count_frames(folder: str) -> int:
  max_idx = 0
  for img_path in _iter_images(folder):
    fi = _parse_frame_index_from_name(img_path)
    if fi is None:
      continue
    if fi > max_idx:
      max_idx = fi
  return int(max_idx)


def _build_pkl_from_std_samples(sample_dirs: List[str]) -> Dict[str, Any]:
  # 先用 action 名称收集，再统一映射到 class_id
  actions_seen: List[str] = []
  action_set: set[str] = set()
  # 暂存：gttubes_by_action[video_id][action] -> List[np.ndarray]
  gttubes_by_action: Dict[str, Dict[str, List[np.ndarray]]] = {}
  nframes: Dict[str, int] = {}
  resolution: Dict[str, Tuple[int, int]] = {}

  # 进度条：样例级 & tube 段级
  samples_bar = tqdm(total=len(sample_dirs), desc="samples", unit="sample")
  tubes_bar = tqdm(total=0, desc="build pkl", unit="tube")
  for folder in sample_dirs:
    video_id = os.path.basename(folder.rstrip(os.sep))
    # 分辨率与帧数
    res = _probe_resolution(folder)
    if res is not None:
      resolution[video_id] = (int(res[0]), int(res[1]))
    else:
      resolution[video_id] = (0, 0)
    nframes[video_id] = _count_frames(folder)

    # tubes: {(track_id, action) -> [(frame_idx, img_path, (x1,y1,x2,y2)) | (-1, ...)]}
    tubes = _build_tubes_from_folder(folder)
    for (tid, action), seq in tubes.items():
      if action not in action_set:
        action_set.add(action)
        actions_seen.append(action)
      # 切分连续段
      segments = _split_by_separators(seq)
      for seg in segments:
        rows: List[List[float]] = []
        for (fidx, _img, bbox) in seg:
          if fidx < 0:
            continue
          x1, y1, x2, y2 = [float(v) for v in bbox]
          rows.append([float(int(fidx)), x1, y1, x2, y2])
        if not rows:
          continue
        arr = np.asarray(rows, dtype=float)
        gttubes_by_action.setdefault(video_id, {}).setdefault(action, []).append(arr)
        try:
          tubes_bar.update(1)
        except Exception:
          pass
    try:
      samples_bar.update(1)
    except Exception:
      pass

  try:
    samples_bar.close()
    tubes_bar.close()
  except Exception:
    pass

  # 统一 labels 与 class_id
  labels: List[str] = list(sorted(action_set)) if action_set else []
  # 若更希望按出现顺序稳定，可替换上一行：labels = list(actions_seen)
  name_to_id: Dict[str, int] = {name: i for i, name in enumerate(labels)}

  gttubes: Dict[str, Dict[int, List[np.ndarray]]] = {}
  for vid, action_dict in gttubes_by_action.items():
    class_map: Dict[int, List[np.ndarray]] = {}
    for act, arr_list in action_dict.items():
      cls_id = name_to_id.get(act)
      if cls_id is None:
        # 未知动作（理论不该发生）
        continue
      class_map.setdefault(cls_id, []).extend(arr_list)
    gttubes[vid] = class_map

  data: Dict[str, Any] = {
    "version": "1.0",
    "labels": labels,
    "videos": list(sorted(nframes.keys())),
    "nframes": nframes,
    "resolution": resolution,
    "gttubes": gttubes,
  }
  return data


def std_to_pkl(std_root: str, out_root: Optional[str] = None, out_name: str = "gt.pkl") -> str:
  """从 std 根目录构建标准 PKL，并写入 <out_root>/annotations/out_name。

  - std_root: 标准数据集根目录，要求包含 videos/ 子目录
  - out_root: 输出根目录；默认与 std_root 相同
  - 返回：输出 pkl 的绝对路径
  """
  std_root = os.path.abspath(std_root)
  out_root = os.path.abspath(out_root or std_root)
  ann_dir = os.path.join(out_root, "annotations")
  _ensure_dir(ann_dir)

  sample_dirs = _discover_std_sample_folders([std_root])
  if not sample_dirs:
    raise ValueError(f"No std sample folders discovered under: {std_root}")

  data = _build_pkl_from_std_samples(sample_dirs)

  out_pkl = os.path.join(ann_dir, out_name)
  with open(out_pkl, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

  # 同步一份 labels.json 便于查看（可选）
  try:
    with open(os.path.join(ann_dir, "labels.json"), "w", encoding="utf-8") as f:
      json.dump(data.get("labels", []), f, ensure_ascii=False, indent=2)
  except Exception:
    pass

  return os.path.abspath(out_pkl)


def _list_images_in_video(videos_root: str, video_id: str) -> List[str]:
  folder = os.path.join(videos_root, video_id)
  if not os.path.isdir(folder):
    return []
  return [p for p in _iter_images(folder)]


def _shape_rect(x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
  return [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)], [round(x1, 3), round(y2, 3)]]


def _make_labelme_record(image_name: str, w: int, h: int, shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
  return {
    "version": "2.5.4",
    "flags": {},
    "shapes": shapes,
    "imagePath": image_name,
    "imageData": None,
    "imageHeight": int(h),
    "imageWidth": int(w),
  }


def pkl_to_std(
  pkl_path: str,
  videos_root: str,
  out_root: Optional[str] = None,
  inplace: bool = False,
  copy_images: bool = False,
) -> str:
  """将标准 PKL 重建为 std（LabelMe JSON）。

  - pkl_path: 含标准结构的 PKL（version/labels/videos/nframes/resolution/gttubes）
  - videos_root: 已存在的视频帧目录根（包含 <video_id>/ 帧图像）
  - out_root: 输出 std 根；若 None 且 inplace=False，则默认 pkl 同级目录追加 _std
  - inplace: 为 True 则直接在 videos_root/<video_id> 内写 JSON，不复制图片
  - copy_images: 若不 inplace 且为 True，会复制图片到 out_root/videos/
  - 返回：输出 std 根目录绝对路径
  """
  videos_root = os.path.abspath(videos_root)
  with open(pkl_path, "rb") as f:
    data: Dict[str, Any] = pickle.load(f)

  labels: List[str] = data.get("labels", []) or []
  id2label: Dict[int, str] = {i: name for i, name in enumerate(labels)}
  gttubes: Dict[str, Dict[int, List[Any]]] = data.get("gttubes", {}) or {}
  resolution: Dict[str, Tuple[int, int]] = data.get("resolution", {}) or {}
  videos: List[str] = data.get("videos", list(gttubes.keys()))

  if inplace:
    out_std_root = videos_root if videos_root.endswith("/videos") else os.path.join(os.path.dirname(videos_root), "")
    # 为保持一致，仍返回 videos_root 的上级作为 std 根
    out_std_root = os.path.abspath(os.path.join(videos_root, os.pardir))
  else:
    if out_root is None or str(out_root) == "":
      base = os.path.dirname(os.path.abspath(pkl_path))
      out_std_root = os.path.join(base, "..", Path(base).name + "_std")
    else:
      out_std_root = os.path.abspath(out_root)

  videos_out = videos_root if inplace else os.path.join(out_std_root, "videos")
  ann_out = os.path.join(out_std_root, "annotations")
  _ensure_dir(videos_out)
  _ensure_dir(ann_out)

  # 写一份 labels.json
  try:
    with open(os.path.join(ann_out, "labels.json"), "w", encoding="utf-8") as f:
      json.dump(labels, f, ensure_ascii=False, indent=2)
  except Exception:
    pass

  # 进度条：视频级 & 帧级
  videos_bar = tqdm(total=len(videos), desc="videos", unit="video")
  frames_bar = tqdm(total=0, desc="frames", unit="frame")

  # 遍历每个视频，合成 per-frame JSON
  for vid in videos:
    # 源图片目录
    src_dir = os.path.join(videos_root, vid)
    if not os.path.isdir(src_dir):
      # 不存在则跳过该视频
      continue
    # 目标图片目录
    dst_dir = src_dir if inplace else os.path.join(videos_out, vid)
    _ensure_dir(dst_dir)

    # 枚举源图片
    img_paths = _list_images_in_video(videos_root, vid)
    # 解析分辨率
    H, W = 0, 0
    if isinstance(resolution.get(vid), (list, tuple)) and len(resolution[vid]) >= 2:
      H = int(resolution[vid][0])
      W = int(resolution[vid][1])
    if H <= 0 or W <= 0:
      res = _probe_resolution(src_dir)
      if res is not None:
        H, W = int(res[0]), int(res[1])

    # 聚合：frame_idx -> [shapes]
    frame_to_shapes: Dict[int, List[Dict[str, Any]]] = {}
    per_class = gttubes.get(vid, {})
    for cls_id, tubes in per_class.items():
      action = id2label.get(int(cls_id), str(cls_id))
      for t_idx, arr in enumerate(tubes):
        # arr 可能是 list -> 转为 np.ndarray
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[1] < 5:
          continue
        # 约定：第一列为帧号（1-based），最后四列为 bbox
        x1c, y1c, x2c, y2c = a.shape[1] - 4, a.shape[1] - 3, a.shape[1] - 2, a.shape[1] - 1
        for row in a:
          try:
            fidx = int(row[0])
          except Exception:
            continue
          x1, y1, x2, y2 = float(row[x1c]), float(row[y1c]), float(row[x2c]), float(row[y2c])
          rect = _shape_rect(x1, y1, x2, y2)
          tid = f"c{int(cls_id)}_t{int(t_idx)}"
          shape = {
            "label": str(action),
            "description": None,
            "points": rect,
            "group_id": 0,
            "difficult": False,
            "direction": 0,
            "shape_type": "rectangle",
            "flags": {"class_id": int(cls_id), "track_id": tid},
            "attributes": {"action": str(action), "class_id": int(cls_id), "track_id": tid},
          }
          frame_to_shapes.setdefault(fidx, []).append(shape)

    # 写出每帧 JSON，并按需复制图片
    for src_img in img_paths:
      stem = os.path.splitext(os.path.basename(src_img))[0]
      dst_img = os.path.join(dst_dir, os.path.basename(src_img))
      # 复制图片（可选）
      if (not inplace) and copy_images:
        if os.path.isfile(src_img) and (not os.path.exists(dst_img)):
          try:
            import shutil
            shutil.copy2(src_img, dst_img)
          except Exception:
            pass
      # 生成 JSON（若对应帧有 shapes）
      fi = _parse_frame_index_from_name(stem)
      # 允许没有连续帧号：如果解析失败则跳过 JSON
      if fi is None:
        fi = _parse_frame_index_from_name(src_img)
      if fi is None:
        continue
      shapes = frame_to_shapes.get(int(fi), [])
      if not shapes:
        # 没有该帧标注可不写 JSON
        continue
      rec = _make_labelme_record(image_name=os.path.basename(src_img), w=W, h=H, shapes=shapes)
      dst_json = os.path.join(dst_dir, f"{stem}.json")
      try:
        with open(dst_json, "w", encoding="utf-8") as f:
          json.dump(rec, f, ensure_ascii=False, indent=2)
      except Exception:
        pass
      try:
        frames_bar.update(1)
      except Exception:
        pass
    try:
      videos_bar.update(1)
    except Exception:
      pass

  try:
    videos_bar.close()
    frames_bar.close()
  except Exception:
    pass

  return os.path.abspath(out_std_root)


def main(argv: Optional[List[str]] = None) -> int:
  import argparse

  parser = argparse.ArgumentParser(description="Convert between std dataset and standard PKL")
  sub = parser.add_subparsers(dest="cmd", required=True)

  # std -> pkl
  p_std2pkl = sub.add_parser("std-to-pkl", help="Build standard PKL from std dataset")
  p_std2pkl.add_argument("std_root", type=str, help="Std root directory (must contain videos/)")
  p_std2pkl.add_argument("--out", type=str, default=None, help="Output std root (default: std_root)")
  p_std2pkl.add_argument("--name", type=str, default="gt.pkl", help="Output PKL filename under annotations/")

  # pkl -> std
  p_pkl2std = sub.add_parser("pkl-to-std", help="Reconstruct std (LabelMe JSON) from standard PKL")
  p_pkl2std.add_argument("pkl_path", type=str, help="Path to standard PKL")
  p_pkl2std.add_argument("videos_root", type=str, help="Root containing <video_id>/ frame images")
  p_pkl2std.add_argument("--out", type=str, default=None, help="Output std root (default: sibling of PKL)")
  p_pkl2std.add_argument("--inplace", action="store_true", help="Write JSON next to existing images under videos_root")
  p_pkl2std.add_argument("--copy-images", action="store_true", help="When not inplace, copy images into out/videos/")

  args = parser.parse_args(argv)

  if args.cmd == "std-to-pkl":
    out_pkl = std_to_pkl(args.std_root, out_root=args.out, out_name=args.name)
    print(json.dumps({"out_pkl": out_pkl}, ensure_ascii=False))
    return 0

  if args.cmd == "pkl-to-std":
    out_std = pkl_to_std(args.pkl_path, args.videos_root, out_root=args.out, inplace=bool(args.inplace), copy_images=bool(args.copy_images))
    print(json.dumps({"out_std_root": out_std}, ensure_ascii=False))
    return 0

  return 1


if __name__ == "__main__":
  raise SystemExit(main())


