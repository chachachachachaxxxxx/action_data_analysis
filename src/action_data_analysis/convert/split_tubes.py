# 注意，针对短tube位于首尾帧的场景，这里采用了将窗口限制在合理范围内，而不是尝试使用img复制，如373-378所示
# 这个是数据集可以增强的地方
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
import json as _json
import bisect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

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
  stride: Optional[int] = None,
  splits_dir: Optional[str] = None,
  single_out: bool = False,
  pad_edges_images: bool = False,
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
  stride_val = int(stride) if (stride is not None and int(stride) > 0) else max(1, int(fps // 2))

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
  # 稀疏掩码：记录缺失帧（nearest 填充）与边缘补全帧（edge pad）
  # 单数据集输出时附带 split 字段，否则不含
  mask_rows: List[Tuple[str, str, str, str, Optional[str]]] = []  # (window_name, anno_mask_csv, img_mask_csv, edge_pad_csv, split|None)

  # 若提供 splits_dir，则读取 train/val/test.csv 中的样例列表（第一列路径，形如 videos/<sample>）
  sample_splits: Optional[Dict[str, Set[str]]] = None
  if splits_dir:
    def _read_split_list(csv_path: str) -> Set[str]:
      names: Set[str] = set()
      try:
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
          reader = csv.reader(f)
          for row in reader:
            if not row:
              continue
            cell = row[0]
            # 跳过表头（兼容 "tube" 或 "path" 等）
            if isinstance(cell, str) and cell.strip().lower() in {"tube", "path"}:
              continue
            # 兼容未正确分列的情况
            if (len(row) == 1) and ("," in cell):
              cell = cell.split(",", 1)[0]
            # 期望开头为 videos/
            p = str(cell).strip()
            if p.startswith("videos/"):
              sample = p.split("/", 1)[1]
            else:
              sample = os.path.basename(p)
            if sample:
              names.add(sample)
      except Exception:
        names = set()
      return names

    sd = os.path.abspath(splits_dir)
    train_set = _read_split_list(os.path.join(sd, "train.csv"))
    val_set = _read_split_list(os.path.join(sd, "val.csv"))
    test_set = _read_split_list(os.path.join(sd, "test.csv"))
    sample_splits = {"train": train_set, "val": val_set, "test": test_set}
    # 单数据集输出时将为每个 split 生成索引
    split_index_rows: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
  else:
    split_index_rows = {}

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
    # 若指定了样例拆分，则仅处理属于任一 split 的样例
    assigned_splits: List[str] = []
    if sample_splits is not None:
      for sp in ("train", "val", "test"):
        if sample_name in (sample_splits.get(sp) or set()):
          assigned_splits.append(sp)
      if not assigned_splits:
        # 不在任何拆分中，跳过
        continue
    else:
      assigned_splits = []

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
            s += stride_val
          if starts and starts[-1] != last_start:
            starts.append(last_start)
        else:
          # 居中扩展：尽量让窗口中心靠近 tube 中心
          center = int(round(sum(anno_fidxs_sorted) / float(len(anno_fidxs_sorted))))
          s0 = center - (win // 2)
          if pad_edges_images:
            # 不再滑动到可用范围内，允许越界，由边缘补全
            starts.append(s0)
          else:
            # 将起点限制在可用帧范围内（传统做法）
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
          edge_pad_indices: List[int] = []
          # 该段内可用图片帧索引（用于在段内寻找最近图片）
          seg_img_indices = [i for i in all_indices_sorted if i >= seg_min and i <= seg_max]
          for offset in range(win):
            fi = start + offset
            # 选择用于图像/JSON 的源帧
            if fi in all_frames:
              img_path, json_path = all_frames[fi]
              img_src_idx = fi
            else:
              if pad_edges_images and (fi < seg_min or fi > seg_max):
                # 边界外：用段首/段尾
                edge_src = seg_min if fi < seg_min else seg_max
                # 若 edge_src 帧无图片，则就近取图片（全局）
                img_src_idx = edge_src if edge_src in all_frames else _nearest_source_frame(edge_src, all_indices_sorted)
                img_path, json_path = all_frames[img_src_idx]
                edge_pad_indices.append(offset + 1)
              else:
                # 段内或未启用边缘补全：取最近图片（优先段内）
                if seg_img_indices:
                  # 在 seg_img_indices 中找最近
                  pos = bisect.bisect_left(seg_img_indices, fi)
                  if pos <= 0:
                    img_src_idx = seg_img_indices[0]
                  elif pos >= len(seg_img_indices):
                    img_src_idx = seg_img_indices[-1]
                  else:
                    left_val = seg_img_indices[pos - 1]
                    right_val = seg_img_indices[pos]
                    img_src_idx = left_val if (fi - left_val) <= (right_val - fi) else right_val
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
            if fi in bbox_by_fidx:
              src_f = fi
            else:
              if pad_edges_images and (fi < seg_min or fi > seg_max):
                src_f = seg_min if fi < seg_min else seg_max
              else:
                src_f = _nearest_source_frame(fi, anno_fidxs_sorted)
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
          if anno_mask_indices or img_mask_indices or edge_pad_indices:
            anno_str = ",".join(str(i) for i in sorted(set(anno_mask_indices))) if anno_mask_indices else ""
            img_str = ",".join(str(i) for i in sorted(set(img_mask_indices))) if img_mask_indices else ""
            edge_str = ",".join(str(i) for i in sorted(set(edge_pad_indices))) if edge_pad_indices else ""
            # 若有拆分信息，则记录所属 split；若多个 split（理论不应发生），逐个记录
            if assigned_splits:
              for sp in assigned_splits:
                mask_rows.append((new_sample, anno_str, img_str, edge_str, sp if single_out else None))
            else:
              mask_rows.append((new_sample, anno_str, img_str, edge_str, None))

          # 若启用单数据集输出并提供了拆分，则为对应拆分写索引（path,label）
          if single_out and sample_splits is not None:
            window_rel = f"videos/{new_sample}"
            label_str = str(action)
            for sp in assigned_splits:
              split_index_rows[sp].append((window_rel, label_str))
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
      if single_out and splits_dir:
        w.writerow(["window", "anno_mask", "img_mask", "edge_pad", "split"])  # 附带所属拆分
        for window_name, anno_str, img_str, edge_str, sp in mask_rows:
          w.writerow([window_name, anno_str, img_str, edge_str, sp or ""])
      else:
        w.writerow(["window", "anno_mask", "img_mask", "edge_pad"])  # 1-based 索引，逗号分隔；为空表示该类型无缺失
        for window_name, anno_str, img_str, edge_str, _sp in mask_rows:
          w.writerow([window_name, anno_str, img_str, edge_str])
  except Exception:
    pass

  # 单数据集输出：写出 annotations/train.csv/val.csv/test.csv（无表头：path,label）
  if single_out and splits_dir and split_index_rows:
    try:
      import csv
      for sp in ("train", "val", "test"):
        rows = split_index_rows.get(sp) or []
        if not rows:
          # 若该拆分为空，也写出空文件保证一致性
          rows = []
        out_csv = ann_out / f"{sp}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
          w = csv.writer(f)
          for p, l in rows:
            w.writerow([p, l])
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

