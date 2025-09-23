# 注意，针对短tube位于首尾帧的场景，这里采用了将窗口限制在合理范围内，而不是尝试使用img复制，如373-378所示
# 这个是数据集可以增强的地方
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
import json as _json
import bisect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from action_data_analysis.analyze.export import _discover_std_sample_folders
from action_data_analysis.io.json import read_labelme_json, iter_labelme_dir
from action_data_analysis.analyze.export_tube_videos import (
  _build_tubes_from_folder,
  _split_by_separators,
)
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _json_path_for_image(sample_dir: str, img_path: str) -> str:
  stem = os.path.splitext(os.path.basename(img_path))[0]
  return os.path.join(sample_dir, stem + ".json")


def _parse_frame_index_from_path(path: str) -> Optional[int]:
  base = os.path.basename(path)
  name, _ = os.path.splitext(base)
  digits = ""
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


def _read_json_raw(json_path: str) -> Dict[str, Any]:
  with open(json_path, "r", encoding="utf-8") as f:
    return _json.load(f)


def _image_path_for_json(folder: str, json_path: str, rec: Dict[str, Any]) -> Optional[str]:
  # 优先使用 JSON 的 imagePath
  img_name = str(rec.get("imagePath") or "").strip()
  if img_name:
    cand = os.path.join(folder, img_name)
    if os.path.exists(cand):
      return cand
  # 退化：用 json 文件名的 stem 去找图像
  stem = os.path.splitext(os.path.basename(json_path))[0]
  for ext in (".jpg", ".jpeg", ".png"):
    p = os.path.join(folder, stem + ext)
    if os.path.exists(p):
      return p
  # 兜底：搜索目录
  for name in os.listdir(folder):
    if os.path.splitext(name)[0] == stem:
      p = os.path.join(folder, name)
      if os.path.isfile(p):
        return p
  return None


def _filter_shapes_for_tube(rec: Dict[str, Any], track_id: str, action: str) -> Dict[str, Any]:
  shapes = rec.get("shapes", []) or []
  kept: List[Dict[str, Any]] = []
  for sh in shapes:
    attrs = sh.get("attributes") or {}
    flags = sh.get("flags") or {}
    tid: Optional[str] = None
    for key in ("id", "track_id", "player_id"):
      v = attrs.get(key)
      if isinstance(v, (str, int)):
        tid = str(v)
        break
    if tid is None:
      for key in ("id", "track_id", "player_id"):
        v = flags.get(key)
        if isinstance(v, (str, int)):
          tid = str(v)
          break
    if tid is None:
      v = sh.get("group_id")
      if isinstance(v, (str, int)):
        tid = str(v)
    if tid != track_id:
      continue
    kept.append(sh)
  out = dict(rec)
  out["shapes"] = kept
  return out


@dataclass
class SplitSummary:
  inputs: List[str]
  output_root: str
  samples_in: int
  tubes_found: int
  samples_out: int


def split_std_samples_by_tube(
  inputs: List[str],
  out_root: str | Path | None,
  min_len: int = 1,
) -> SplitSummary:
  """将 std 数据集样例按时空管拆分为新的 std 数据集。

  - inputs: std 根目录、videos 目录或 videos/* 样例目录；会自动发现样例。
  - out_root: 输出根目录；若为空，默认在首个 std 根旁创建 `<name>_by_tube`。
  - min_len: tube 最小帧数，过滤过短片段。
  输出结构：
    <out_root>/
      ├─ videos/
      │   └─ <orig_sample>__tid<id>__seg<k>/  # 只包含该 tube 的若干帧与对应 JSON（仅保留该 tube 的 shape）
      └─ stats/  # 预留
  """
  folders = _discover_std_sample_folders(inputs)
  if not folders:
    raise ValueError("No std sample folders discovered from inputs")

  # 推断默认 out_root
  if out_root is None or str(out_root) == "":
    p = Path(folders[0])
    std_root = p.parent.parent
    out_dir = std_root.parent / f"{std_root.name}_by_tube"
  else:
    out_dir = Path(out_root)
  videos_out = out_dir / "videos"
  stats_out = out_dir / "stats"
  _ensure_dir(str(videos_out))
  _ensure_dir(str(stats_out))

  samples_in = 0
  tubes_found = 0
  samples_out = 0

  total_samples = len(folders)
  samples_bar = tqdm(total=total_samples, desc="samples 0/%d" % total_samples, unit="sample")
  progress_bar = tqdm(total=0, desc="split by tube", unit="output")
  for si, sample_dir in enumerate(folders, start=1):
    try:
      samples_bar.set_description(f"samples {si}/{total_samples}")
    except Exception:
      pass
    samples_in += 1
    tubes = _build_tubes_from_folder(sample_dir)
    sample_name = os.path.basename(sample_dir.rstrip(os.sep))
    for (tid, action), seq in sorted(tubes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
      segments = _split_by_separators(seq)
      for seg_idx, seg in enumerate(segments):
        frames = [(p, f, b) for (f, p, b) in seg if f >= 0]
        if len(frames) < int(min_len):
          continue
        tubes_found += 1
        # 新样例目录
        new_sample = f"{sample_name}__tid{tid}__seg{seg_idx:02d}"
        new_dir = videos_out / new_sample
        _ensure_dir(str(new_dir))

        # 逐帧复制图片与重写 JSON（仅保留该 tube 对应的 shape）
        for img_path, _fidx, _bbox in frames:
          # 复制图片
          if os.path.isfile(img_path):
            dst_img = new_dir / os.path.basename(img_path)
            if not dst_img.exists():
              shutil.copy2(img_path, dst_img)
          # 读取并过滤 JSON（保留原始键，如 version/flags 等）
          src_json = _json_path_for_image(sample_dir, img_path)
          if os.path.isfile(src_json):
            try:
              rec_full = _read_json_raw(src_json)
            except Exception:
              # 读取失败则跳过该帧
              rec_full = None
            if rec_full is not None:
              out_rec = _filter_shapes_for_tube(rec_full, track_id=str(tid), action=str(action))
              # 重写 imagePath 指向新目录内的同名图片
              out_rec["imagePath"] = os.path.basename(img_path)
              dst_json = new_dir / os.path.basename(src_json)
              try:
                with open(dst_json, "w", encoding="utf-8") as f:
                  _json.dump(out_rec, f, ensure_ascii=False, indent=2)
              except Exception:
                pass
        samples_out += 1
        try:
          progress_bar.update(1)
        except Exception:
          pass
    try:
      samples_bar.update(1)
    except Exception:
      pass
  try:
    samples_bar.close()
    progress_bar.close()
  except Exception:
    pass

  return SplitSummary(
    inputs=[str(Path(p).resolve()) for p in inputs],
    output_root=str(out_dir.resolve()),
    samples_in=samples_in,
    tubes_found=tubes_found,
    samples_out=samples_out,
  )



def _build_all_frames_index(sample_dir: str) -> Dict[int, Tuple[str, str]]:
  """返回 {frame_idx: (img_path, json_path)}。

  - 以目录中的图片为基准（必须存在图片），解析其帧号；
  - 若对应 JSON 存在则使用，否则预测出应当的 JSON 路径（后续写入）。
  """
  images_by_idx: Dict[int, str] = {}
  # 扫描目录中的图片文件
  try:
    for name in os.listdir(sample_dir):
      ext = os.path.splitext(name)[1].lower()
      if ext not in IMG_EXTS:
        continue
      img_path = os.path.join(sample_dir, name)
      if not os.path.isfile(img_path):
        continue
      fi = _parse_frame_index_from_path(img_path)
      if fi is None:
        continue
      images_by_idx[int(fi)] = img_path
  except Exception:
    images_by_idx = {}

  # 扫描目录中的 JSON 文件（可选）
  json_by_idx: Dict[int, str] = {}
  try:
    for json_path, _rec in iter_labelme_dir(sample_dir):
      fi = _parse_frame_index_from_path(json_path)
      if fi is None:
        continue
      json_by_idx[int(fi)] = json_path
  except Exception:
    json_by_idx = {}

  index: Dict[int, Tuple[str, str]] = {}
  for fi, img_path in images_by_idx.items():
    if fi in json_by_idx:
      index[fi] = (img_path, json_by_idx[fi])
    else:
      # 预测 JSON 路径（按图片 stem）
      stem = os.path.splitext(os.path.basename(img_path))[0]
      pred_json = os.path.join(sample_dir, stem + ".json")
      index[fi] = (img_path, pred_json)
  return index


def _nearest_source_frame(target: int, annotated_frames_sorted: List[int]) -> int:
  if not annotated_frames_sorted:
    return target
  pos = bisect.bisect_left(annotated_frames_sorted, target)
  if pos <= 0:
    return annotated_frames_sorted[0]
  if pos >= len(annotated_frames_sorted):
    return annotated_frames_sorted[-1]
  left_val = annotated_frames_sorted[pos - 1]
  right_val = annotated_frames_sorted[pos]
  return left_val if (target - left_val) <= (right_val - target) else right_val


def split_std_samples_by_tube_fpswin(
  inputs: List[str],
  out_root: str | Path | None,
  fps: int,
  min_len: int = 1,
) -> SplitSummary:
  """按 fps 窗口与步长（win=fps, stride=fps//2）切分 tube，生成新的 std 数据集。

  - tube 长度 < win：以 tube 中心为窗口中心，向两侧均匀拓展到 win 帧；
    窗口内缺少该 tube 原始标注的帧，其 JSON 使用“最近的有标注帧”的框，并在 annotations/mask.csv 记录。
  - tube 长度 ≥ win：沿时间轴用滑窗（stride=fps//2）生成多个窗口；
    窗口内每一帧若无该 tube 的原始标注，同样用最近标注帧补全并记录 mask。
  输出：
    <out_root>/videos/<orig>__tidX__segYY__winZZZ/{frames+json}
    <out_root>/annotations/mask.csv（列：window,frame_index,source_frame；仅记录缺失原始标注的帧）
  """
  if int(fps) <= 0:
    raise ValueError("fps must be positive")
  win = int(fps)
  stride = max(1, int(fps // 2))

  folders = _discover_std_sample_folders(inputs)
  if not folders:
    raise ValueError("No std sample folders discovered from inputs")

  # 推断默认 out_root
  if out_root is None or str(out_root) == "":
    p = Path(folders[0])
    std_root = p.parent.parent
    out_dir = std_root.parent / f"{std_root.name}_by_tube_fpswin"
  else:
    out_dir = Path(out_root)

  videos_out = out_dir / "videos"
  ann_out = out_dir / "annotations"
  _ensure_dir(str(videos_out))
  _ensure_dir(str(ann_out))

  samples_in = 0
  tubes_found = 0
  samples_out = 0
  # 稀疏掩码：仅记录缺失帧 missing=1
  mask_rows: List[Tuple[str, int, int]] = []  # (window_name, frame_index, source_frame)

  total_samples = len(folders)
  samples_bar = tqdm(total=total_samples, desc="samples 0/%d" % total_samples, unit="sample")
  windows_bar = tqdm(total=0, desc="split fpswin", unit="window")
  for si, sample_dir in enumerate(folders, start=1):
    try:
      samples_bar.set_description(f"samples {si}/{total_samples}")
    except Exception:
      pass
    samples_in += 1
    # 全部帧索引（确保能找到对应图片/JSON）
    all_frames = _build_all_frames_index(sample_dir)
    if not all_frames:
      continue
    all_indices_sorted = sorted(all_frames.keys())
    min_all = all_indices_sorted[0]
    max_all = all_indices_sorted[-1]

    tubes = _build_tubes_from_folder(sample_dir)
    sample_name = os.path.basename(sample_dir.rstrip(os.sep))

    for (tid, action), seq in sorted(tubes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
      segments = _split_by_separators(seq)
      for seg_idx, seg in enumerate(segments):
        frames = [(f, p, b) for (f, p, b) in seg if f >= 0]
        if len(frames) < int(min_len):
          continue
        tubes_found += 1

        # 该段的注释帧索引及对应 bbox
        anno_fidxs = [f for (f, _p, _b) in frames]
        anno_fidxs_sorted = sorted(anno_fidxs)
        bbox_by_fidx: Dict[int, Tuple[float, float, float, float]] = {f: b for (f, _p, b) in frames}
        seg_min = min(anno_fidxs_sorted)
        seg_max = max(anno_fidxs_sorted)

        # 生成窗口起点列表（基于时间轴）
        starts: List[int] = []
        if (seg_max - seg_min + 1) >= win:
          s = seg_min
          last_start = seg_max - win + 1
          while s <= last_start:
            starts.append(s)
            s += stride
          if starts and starts[-1] != last_start:
            starts.append(last_start)
        else:
          # 居中扩展：尽量让窗口中心靠近 tube 中心
          center = int(round(sum(anno_fidxs_sorted) / float(len(anno_fidxs_sorted))))
          s0 = center - (win // 2)
          # 将起点限制在可用帧范围内
          s0 = max(min_all, min(s0, max_all - win + 1))
          starts.append(s0)

        # 逐窗口写出
        for w_idx, start in enumerate(starts):
          new_sample = f"{sample_name}__tid{tid}__seg{seg_idx:02d}__win{w_idx:03d}"
          new_dir = videos_out / new_sample
          _ensure_dir(str(new_dir))
          # 每个窗口的稀疏掩码（1-based 索引，范围 1..win）
          anno_mask_indices: List[int] = []
          img_mask_indices: List[int] = []
          for offset in range(win):
            fi = start + offset
            # 选择用于图像/JSON 的源帧（若不存在则用最近存在的图片帧）
            if fi in all_frames:
              img_path, json_path = all_frames[fi]
              img_src_idx = fi
            else:
              img_src_idx = _nearest_source_frame(fi, all_indices_sorted)
              img_path, json_path = all_frames[img_src_idx]
              img_mask_indices.append(offset + 1)
            # 标准化目标文件名（从 000001 开始）
            dest_stem = f"{offset + 1:06d}"
            src_ext = os.path.splitext(os.path.basename(img_path))[1]
            dest_img_name = f"{dest_stem}{src_ext}"
            dest_json_name = f"{dest_stem}.json"
            # 选择用于 bbox 的源帧（最近注释帧）
            src_f = fi if fi in bbox_by_fidx else _nearest_source_frame(fi, anno_fidxs_sorted)
            missing = 0 if fi in bbox_by_fidx else 1
            src_bbox = bbox_by_fidx.get(src_f)
            if missing == 1:
              anno_mask_indices.append(offset + 1)
            # 复制图片（使用目标文件名，避免与源文件名冲突）
            if os.path.isfile(img_path):
              dst_img = new_dir / dest_img_name
              if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            # 读取并重写 JSON
            # 读取完整 JSON（保留 version/flags 等），失败则构造最小骨架
            try:
              rec_full = _read_json_raw(json_path)
            except Exception:
              rec_full = {"version": "", "flags": {}, "imagePath": os.path.basename(img_path), "imageWidth": 0, "imageHeight": 0, "shapes": []}
            out_rec = _filter_shapes_for_tube(rec_full, track_id=str(tid), action=str(action))
            # 若该帧无该 tube 原始标注，则用最近源帧 bbox 合成一个 shape
            if (not out_rec.get("shapes")) and src_bbox is not None:
              x1, y1, x2, y2 = [float(v) for v in src_bbox]
              shape = {
                "label": str(action),
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "attributes": {"action": str(action), "track_id": str(tid), "id": str(tid)},
                "flags": {"action": str(action), "track_id": str(tid), "id": str(tid)},
              }
              out_rec["shapes"] = [shape]
            # 重写 imagePath
            out_rec["imagePath"] = dest_img_name
            dst_json = new_dir / dest_json_name
            try:
              with open(dst_json, "w", encoding="utf-8") as f:
                _json.dump(out_rec, f, ensure_ascii=False, indent=2)
            except Exception:
              pass
          # 写出图片（使用目标文件名，避免冲突；逐帧复制）
          for offset in range(win):
            fi = start + offset
            if fi in all_frames:
              img_path, _ = all_frames[fi]
            else:
              img_src_idx = _nearest_source_frame(fi, all_indices_sorted)
              img_path, _ = all_frames[img_src_idx]
            dest_stem = f"{offset + 1:06d}"
            src_ext = os.path.splitext(os.path.basename(img_path))[1]
            dest_img_name = f"{dest_stem}{src_ext}"
            if os.path.isfile(img_path):
              dst_img = new_dir / dest_img_name
              if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
          # 记录稀疏 mask（按窗口聚合，一行）
          if anno_mask_indices or img_mask_indices:
            anno_str = ",".join(str(i) for i in sorted(set(anno_mask_indices))) if anno_mask_indices else ""
            img_str = ",".join(str(i) for i in sorted(set(img_mask_indices))) if img_mask_indices else ""
            mask_rows.append((new_sample, anno_str, img_str))
          samples_out += 1
          try:
            windows_bar.update(1)
          except Exception:
            pass
    try:
      samples_bar.update(1)
    except Exception:
      pass

  # 写出 annotations/mask.csv（稀疏：仅当存在掩码才写一行；列为窗口内 1..win 的索引列表）
  try:
    import csv
    mask_csv = ann_out / "mask.csv"
    with open(mask_csv, "w", newline="", encoding="utf-8") as f:
      # 使用 QUOTE_NONNUMERIC：确保字符串字段（包括单个索引如 "5"）带双引号
      w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
      w.writerow(["window", "anno_mask", "img_mask"])  # 1-based 索引，逗号分隔；为空表示该类型无缺失
      for window_name, anno_str, img_str in mask_rows:
        w.writerow([window_name, anno_str, img_str])
  except Exception:
    pass

  try:
    samples_bar.close()
    windows_bar.close()
  except Exception:
    pass

  return SplitSummary(
    inputs=[str(Path(p).resolve()) for p in inputs],
    output_root=str(out_dir.resolve()),
    samples_in=samples_in,
    tubes_found=tubes_found,
    samples_out=samples_out,
  )

